"""Tasks for managing annotations."""

import random
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional
from prefect import task, get_run_logger
from arxiv_pipeline.data.schemas import (
    Annotation,
    AnnotationBatch,
    ProcessedPaper,
    Sentence
)
from arxiv_pipeline.utils.minio_client import (
    upload_json,
    download_json,
    ensure_bucket_exists,
    object_exists,
    list_objects
)
from arxiv_pipeline.utils.config import get_config, get_label_names


@task
def load_annotations(paper_id: Optional[str] = None) -> List[Annotation]:
    """
    Load annotations from MinIO.
    
    Args:
        paper_id: Optional paper ID to filter by
    
    Returns:
        List of Annotation objects
    """
    logger = get_run_logger()
    config = get_config()
    
    bucket_name = config.storage.bucket_name
    annotations_path = config.storage.annotations_path
    
    # List all annotation files
    prefix = f"{annotations_path}/"
    annotation_files = list_objects(bucket_name, prefix=prefix, recursive=True)
    
    all_annotations = []
    for file_path in annotation_files:
        if not file_path.endswith(".json"):
            continue
        
        try:
            annotations_dict = download_json(bucket_name, file_path)
            
            # Handle both single annotation and batch formats
            if isinstance(annotations_dict, dict):
                if "annotations" in annotations_dict:
                    # Batch format
                    batch = AnnotationBatch(**annotations_dict)
                    all_annotations.extend(batch.annotations)
                else:
                    # Single annotation
                    all_annotations.append(Annotation(**annotations_dict))
            elif isinstance(annotations_dict, list):
                # List of annotations
                all_annotations.extend([Annotation(**ann) for ann in annotations_dict])
        
        except Exception as e:
            logger.warning(f"Error loading annotations from {file_path}: {e}")
            continue
    
    # Filter by paper_id if provided
    if paper_id:
        all_annotations = [ann for ann in all_annotations if ann.paper_id == paper_id]
    
    logger.info(f"Loaded {len(all_annotations)} annotations")
    return all_annotations


@task
def save_annotations(annotations: List[Annotation], batch_name: Optional[str] = None) -> str:
    """
    Save annotations to MinIO.
    
    Args:
        annotations: List of Annotation objects
        batch_name: Optional name for the batch
    
    Returns:
        Path where annotations were saved
    """
    logger = get_run_logger()
    config = get_config()
    
    if batch_name is None:
        batch_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    object_name = f"{config.storage.annotations_path}/{batch_name}.json"
    
    batch = AnnotationBatch(annotations=annotations)
    
    bucket_name = config.storage.bucket_name
    ensure_bucket_exists(bucket_name)
    upload_json(bucket_name, object_name, batch.model_dump(mode="json"))
    
    logger.info(f"Saved {len(annotations)} annotations to {bucket_name}/{object_name}")
    return object_name


@task
def select_random_subset(
    processed_papers: List[ProcessedPaper],
    n: int,
    seed: Optional[int] = None
) -> List[ProcessedPaper]:
    """
    Select a random subset of papers.
    
    Args:
        processed_papers: List of ProcessedPaper objects
        n: Number of papers to select
        seed: Random seed
    
    Returns:
        List of selected ProcessedPaper objects
    """
    logger = get_run_logger()
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    n = min(n, len(processed_papers))
    selected = random.sample(processed_papers, n)
    
    logger.info(f"Selected {len(selected)} papers from {len(processed_papers)} total")
    return selected


@task
def select_unannotated_papers(
    processed_papers: List[ProcessedPaper],
    annotations: List[Annotation],
    n: Optional[int] = None
) -> List[ProcessedPaper]:
    """
    Select papers that haven't been annotated yet.
    
    Args:
        processed_papers: List of ProcessedPaper objects
        annotations: List of existing annotations
        n: Optional maximum number to return
    
    Returns:
        List of unannotated ProcessedPaper objects
    """
    logger = get_run_logger()
    
    annotated_paper_ids = {ann.paper_id for ann in annotations}
    unannotated = [
        paper for paper in processed_papers
        if paper.paper.arxiv_id not in annotated_paper_ids
    ]
    
    if n is not None:
        unannotated = unannotated[:n]
    
    logger.info(
        f"Selected {len(unannotated)} unannotated papers from {len(processed_papers)} total"
    )
    return unannotated


@task
def select_diverse_subset(
    processed_papers: List[ProcessedPaper],
    n: int,
    embeddings: Optional[np.ndarray] = None
) -> List[ProcessedPaper]:
    """
    Select a diverse subset of papers using embeddings.
    
    If embeddings are not provided, falls back to random selection.
    
    Args:
        processed_papers: List of ProcessedPaper objects
        n: Number of papers to select
        embeddings: Optional embeddings array (n_papers, embedding_dim)
    
    Returns:
        List of selected ProcessedPaper objects
    """
    logger = get_run_logger()
    
    if embeddings is None or len(embeddings) != len(processed_papers):
        logger.warning("Embeddings not provided or mismatched, using random selection")
        return select_random_subset(processed_papers, n)
    
    # Use k-means clustering to select diverse papers
    from sklearn.cluster import KMeans
    
    n = min(n, len(processed_papers))
    kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(embeddings)
    
    # Select one paper from each cluster (closest to centroid)
    selected_indices = []
    for cluster_id in range(n):
        cluster_mask = clusters == cluster_id
        if not np.any(cluster_mask):
            continue
        
        cluster_embeddings = embeddings[cluster_mask]
        cluster_indices = np.where(cluster_mask)[0]
        
        centroid = kmeans.cluster_centers_[cluster_id]
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        closest_idx = cluster_indices[np.argmin(distances)]
        selected_indices.append(closest_idx)
    
    # If we didn't get enough, fill with random
    if len(selected_indices) < n:
        remaining = [i for i in range(len(processed_papers)) if i not in selected_indices]
        needed = n - len(selected_indices)
        selected_indices.extend(random.sample(remaining, min(needed, len(remaining))))
    
    selected = [processed_papers[i] for i in selected_indices[:n]]
    
    logger.info(f"Selected {len(selected)} diverse papers from {len(processed_papers)} total")
    return selected


@task
def get_papers_for_annotation(
    selection_method: str = "random",
    n: int = 10,
    date: Optional[datetime] = None
) -> List[ProcessedPaper]:
    """
    Get papers ready for annotation using specified selection method.
    
    Args:
        selection_method: Method to use ("random", "unannotated", "diverse")
        n: Number of papers to select
        date: Date to load papers from (defaults to today)
    
    Returns:
        List of ProcessedPaper objects ready for annotation
    """
    from arxiv_pipeline.tasks.storage import load_processed_papers
    
    logger = get_run_logger()
    
    # Load processed papers
    processed_papers = load_processed_papers(date=date)
    
    if not processed_papers:
        logger.warning("No processed papers found")
        return []
    
    # Load existing annotations
    annotations = load_annotations()
    
    # Select based on method
    if selection_method == "random":
        selected = select_random_subset(processed_papers, n)
    elif selection_method == "unannotated":
        selected = select_unannotated_papers(processed_papers, annotations, n)
    elif selection_method == "diverse":
        # For diverse selection, we'd need embeddings
        # For now, fall back to random
        logger.warning("Diverse selection requires embeddings, using random instead")
        selected = select_random_subset(processed_papers, n)
    else:
        logger.warning(f"Unknown selection method: {selection_method}, using random")
        selected = select_random_subset(processed_papers, n)
    
    return selected

