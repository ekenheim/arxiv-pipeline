"""Model training flow."""

from datetime import datetime
from typing import Dict, Optional
from prefect import flow, get_run_logger
from arxiv_pipeline.tasks.annotation import load_annotations
from arxiv_pipeline.tasks.storage import load_processed_papers
from arxiv_pipeline.tasks.training import (
    prepare_training_data,
    fine_tune_embeddings,
    train_classifier,
    save_model_to_minio
)
from arxiv_pipeline.models.sentence_model import ArxivSentenceModel
from arxiv_pipeline.models.classifier import LabelClassifier
from arxiv_pipeline.utils.config import get_config, get_label_names
from arxiv_pipeline.data.schemas import ModelMetadata


@flow(name="train-models")
def train_models(
    date: Optional[datetime] = None,
    fine_tune_embeddings_flag: bool = True,
    train_classifiers_flag: bool = True
) -> Dict[str, str]:
    """
    Main flow for training models.
    
    Args:
        date: Date to load papers from (defaults to today)
        fine_tune_embeddings_flag: Whether to fine-tune embeddings
        train_classifiers_flag: Whether to train classifiers
    
    Returns:
        Dictionary mapping model types to their storage paths
    """
    logger = get_run_logger()
    config = get_config()
    
    if date is None:
        date = datetime.now()
    
    logger.info(f"Starting model training flow for date: {date.strftime('%Y-%m-%d')}")
    
    # Load annotations and processed papers
    annotations = load_annotations()
    processed_papers = load_processed_papers(date=date)
    
    if not annotations:
        logger.warning("No annotations found, cannot train models")
        return {}
    
    if not processed_papers:
        logger.warning("No processed papers found, cannot train models")
        return {}
    
    # Prepare training data
    sentences, label_matrix, label_names = prepare_training_data(
        annotations,
        processed_papers
    )
    
    if len(sentences) == 0:
        logger.warning("No training data prepared")
        return {}
    
    model_paths = {}
    model_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Fine-tune embeddings if requested
    if fine_tune_embeddings_flag:
        logger.info("Fine-tuning embeddings...")
        base_model = ArxivSentenceModel()
        fine_tuned_model = fine_tune_embeddings(
            sentences,
            label_matrix,
            label_names
        )
        
        embedding_model_id = f"embedding_{model_id}"
        embedding_path = save_model_to_minio(
            fine_tuned_model,
            embedding_model_id,
            "sentence_transformer"
        )
        model_paths["embedding"] = embedding_path
        logger.info(f"Saved fine-tuned embedding model: {embedding_path}")
    else:
        # Use base model for classifiers
        fine_tuned_model = ArxivSentenceModel()
    
    # Train classifiers for each label
    if train_classifiers_flag:
        logger.info("Training classifiers...")
        classifiers = {}
        
        for label_idx, label_name in enumerate(label_names):
            logger.info(f"Training classifier for label: {label_name}")
            
            classifier = train_classifier(
                fine_tuned_model,
                sentences,
                label_matrix,
                label_name,
                label_idx
            )
            
            classifier_id = f"classifier_{label_name}_{model_id}"
            classifier_path = save_model_to_minio(
                classifier,
                classifier_id,
                "classifier",
                label_name=label_name
            )
            classifiers[label_name] = classifier_path
            model_paths[f"classifier_{label_name}"] = classifier_path
        
        logger.info(f"Trained {len(classifiers)} classifiers")
    
    logger.info("Model training flow completed")
    return model_paths


@flow(name="train-embeddings-only")
def train_embeddings_only(date: Optional[datetime] = None) -> str:
    """
    Flow to only fine-tune embeddings.
    
    Args:
        date: Date to load papers from (defaults to today)
    
    Returns:
        Path where embedding model was saved
    """
    model_paths = train_models(
        date=date,
        fine_tune_embeddings_flag=True,
        train_classifiers_flag=False
    )
    return model_paths.get("embedding", "")


@flow(name="train-classifiers-only")
def train_classifiers_only(
    date: Optional[datetime] = None,
    embedding_model_id: Optional[str] = None
) -> Dict[str, str]:
    """
    Flow to only train classifiers (using existing or new embedding model).
    
    Args:
        date: Date to load papers from (defaults to today)
        embedding_model_id: Optional ID of existing embedding model to use
    
    Returns:
        Dictionary mapping label names to classifier paths
    """
    from arxiv_pipeline.tasks.training import load_model_from_minio
    
    logger = get_run_logger()
    
    # Load or create embedding model
    if embedding_model_id:
        logger.info(f"Loading embedding model: {embedding_model_id}")
        embedding_model = load_model_from_minio(
            embedding_model_id,
            "sentence_transformer"
        )
    else:
        logger.info("Using base embedding model")
        embedding_model = ArxivSentenceModel()
    
    # Load annotations and processed papers
    annotations = load_annotations()
    processed_papers = load_processed_papers(date=date)
    
    if not annotations or not processed_papers:
        logger.warning("No data available for training")
        return {}
    
    # Prepare training data
    sentences, label_matrix, label_names = prepare_training_data(
        annotations,
        processed_papers
    )
    
    # Train classifiers
    model_paths = {}
    model_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for label_idx, label_name in enumerate(label_names):
        classifier = train_classifier(
            embedding_model,
            sentences,
            label_matrix,
            label_name,
            label_idx
        )
        
        classifier_id = f"classifier_{label_name}_{model_id}"
        classifier_path = save_model_to_minio(
            classifier,
            classifier_id,
            "classifier",
            label_name=label_name
        )
        model_paths[label_name] = classifier_path
    
    return model_paths

