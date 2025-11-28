"""Tasks for generating recommendations."""

from datetime import datetime
from typing import List, Dict, Optional
import torch
import numpy as np
from prefect import task, get_run_logger
from arxiv_pipeline.data.schemas import (
    ProcessedPaper,
    Recommendation,
    PaperRecommendations
)
from arxiv_pipeline.models.sentence_model import ArxivSentenceModel
from arxiv_pipeline.models.classifier import LabelClassifier
from arxiv_pipeline.utils.config import get_config, get_label_names
from arxiv_pipeline.utils.minio_client import (
    upload_json,
    ensure_bucket_exists
)
from arxiv_pipeline.tasks.training import load_model_from_minio
from arxiv_pipeline.tasks.storage import load_processed_papers


@task
def load_latest_models(
    embedding_model_id: Optional[str] = None
) -> tuple[ArxivSentenceModel, Dict[str, LabelClassifier]]:
    """
    Load the latest models from MinIO.
    
    Args:
        embedding_model_id: Optional specific embedding model ID
    
    Returns:
        Tuple of (embedding_model, classifiers_dict)
    """
    logger = get_run_logger()
    from arxiv_pipeline.utils.minio_client import list_objects
    
    config = get_config()
    bucket_name = config.storage.bucket_name
    label_names = get_label_names()
    
    # Load embedding model
    if embedding_model_id:
        logger.info(f"Loading specified embedding model: {embedding_model_id}")
        embedding_model = load_model_from_minio(
            embedding_model_id,
            "sentence_transformer"
        )
    else:
        # Find latest embedding model
        prefix = f"{config.storage.models_path}/sentence_transformer/"
        embedding_files = list_objects(bucket_name, prefix=prefix, recursive=False)
        
        if embedding_files:
            # Get the latest one (by name/timestamp)
            embedding_files.sort(reverse=True)
            latest_embedding_id = embedding_files[0].split("/")[-1].replace(".tar.gz", "")
            logger.info(f"Loading latest embedding model: {latest_embedding_id}")
            embedding_model = load_model_from_minio(
                latest_embedding_id,
                "sentence_transformer"
            )
        else:
            logger.warning("No embedding model found, using base model")
            embedding_model = ArxivSentenceModel()
    
    # Load classifiers for each label
    classifiers = {}
    for label_name in label_names:
        prefix = f"{config.storage.models_path}/classifier/{label_name}/"
        classifier_files = list_objects(bucket_name, prefix=prefix, recursive=False)
        
        if classifier_files:
            classifier_files.sort(reverse=True)
            latest_classifier_id = classifier_files[0].split("/")[-1].replace(".pth", "")
            logger.info(f"Loading classifier for {label_name}: {latest_classifier_id}")
            try:
                classifier = load_model_from_minio(
                    latest_classifier_id,
                    "classifier",
                    label_name=label_name
                )
                classifiers[label_name] = classifier
            except Exception as e:
                logger.warning(f"Failed to load classifier for {label_name}: {e}")
        else:
            logger.warning(f"No classifier found for {label_name}")
    
    logger.info(f"Loaded {len(classifiers)} classifiers")
    return embedding_model, classifiers


@task
def generate_predictions_for_paper(
    paper: ProcessedPaper,
    embedding_model: ArxivSentenceModel,
    classifiers: Dict[str, LabelClassifier],
    threshold: float = 0.5
) -> PaperRecommendations:
    """
    Generate predictions for a single paper.
    
    Args:
        paper: ProcessedPaper object
        embedding_model: Sentence transformer model
        classifiers: Dictionary of label classifiers
        threshold: Prediction threshold
    
    Returns:
        PaperRecommendations object
    """
    logger = get_run_logger()
    
    if not paper.sentences:
        logger.warning(f"No sentences in paper {paper.paper.arxiv_id}")
        return PaperRecommendations(
            paper_id=paper.paper.arxiv_id,
            recommendations=[]
        )
    
    # Encode all sentences
    sentence_texts = [sent.text for sent in paper.sentences]
    embeddings = embedding_model.encode(sentence_texts, show_progress_bar=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embeddings = embeddings.to(device)
    
    # Generate predictions for each label
    recommendations = []
    
    for label_name, classifier in classifiers.items():
        classifier = classifier.to(device)
        classifier.eval()
        
        with torch.no_grad():
            predictions = classifier(embeddings)
            predictions = predictions.cpu().numpy().flatten()
        
        # Find sentences above threshold
        above_threshold = np.where(predictions >= threshold)[0]
        
        if len(above_threshold) > 0:
            # Calculate average score
            avg_score = float(np.mean(predictions[above_threshold]))
            
            recommendation = Recommendation(
                paper_id=paper.paper.arxiv_id,
                label=label_name,
                score=avg_score,
                sentence_indices=[int(idx) for idx in above_threshold]
            )
            recommendations.append(recommendation)
    
    # Sort by score descending
    recommendations.sort(key=lambda x: x.score, reverse=True)
    
    logger.debug(
        f"Generated {len(recommendations)} recommendations for paper {paper.paper.arxiv_id}"
    )
    
    return PaperRecommendations(
        paper_id=paper.paper.arxiv_id,
        recommendations=recommendations
    )


@task
def generate_recommendations_for_papers(
    processed_papers: List[ProcessedPaper],
    embedding_model: ArxivSentenceModel,
    classifiers: Dict[str, LabelClassifier],
    threshold: float = 0.5
) -> List[PaperRecommendations]:
    """
    Generate recommendations for multiple papers.
    
    Args:
        processed_papers: List of ProcessedPaper objects
        embedding_model: Sentence transformer model
        classifiers: Dictionary of label classifiers
        threshold: Prediction threshold
    
    Returns:
        List of PaperRecommendations objects
    """
    logger = get_run_logger()
    
    all_recommendations = []
    
    for paper in processed_papers:
        try:
            recommendations = generate_predictions_for_paper(
                paper,
                embedding_model,
                classifiers,
                threshold
            )
            all_recommendations.append(recommendations)
        except Exception as e:
            logger.warning(f"Error generating recommendations for {paper.paper.arxiv_id}: {e}")
            continue
    
    logger.info(
        f"Generated recommendations for {len(all_recommendations)} papers"
    )
    return all_recommendations


@task
def store_recommendations(
    recommendations: List[PaperRecommendations],
    date: Optional[datetime] = None
) -> str:
    """
    Store recommendations in MinIO.
    
    Args:
        recommendations: List of PaperRecommendations objects
        date: Date for organizing storage (defaults to today)
    
    Returns:
        Path where recommendations were stored
    """
    logger = get_run_logger()
    config = get_config()
    
    if date is None:
        date = datetime.now()
    
    date_str = date.strftime("%Y-%m-%d")
    object_name = f"{config.storage.recommendations_path}/{date_str}/recommendations.json"
    
    # Convert to dict for JSON serialization
    recommendations_dict = [
        rec.model_dump(mode="json") for rec in recommendations
    ]
    
    bucket_name = config.storage.bucket_name
    ensure_bucket_exists(bucket_name)
    upload_json(bucket_name, object_name, recommendations_dict)
    
    logger.info(
        f"Stored {len(recommendations)} recommendations to {bucket_name}/{object_name}"
    )
    return object_name

