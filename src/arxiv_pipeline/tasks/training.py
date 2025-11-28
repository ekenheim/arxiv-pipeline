"""Tasks for model training."""

import os
import shutil
import tarfile
import tempfile
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np
from prefect import task, get_run_logger
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from arxiv_pipeline.data.schemas import Annotation, ProcessedPaper, Sentence
from arxiv_pipeline.models.sentence_model import ArxivSentenceModel
from arxiv_pipeline.models.classifier import LabelClassifier, create_classifier_for_label
from arxiv_pipeline.utils.config import get_config, get_label_names
from arxiv_pipeline.utils.minio_client import (
    upload_file,
    download_file,
    ensure_bucket_exists,
    object_exists
)
from arxiv_pipeline.tasks.annotation import load_annotations
from arxiv_pipeline.tasks.storage import load_processed_papers


@task
def prepare_training_data(
    annotations: List[Annotation],
    processed_papers: List[ProcessedPaper]
) -> Tuple[List[str], np.ndarray, List[str]]:
    """
    Prepare training data from annotations.
    
    Args:
        annotations: List of annotations
        processed_papers: List of processed papers
    
    Returns:
        Tuple of (sentences, label_matrix, label_names)
        label_matrix has shape (n_sentences, n_labels) with NaN for missing labels
    """
    logger = get_run_logger()
    config = get_config()
    label_names = get_label_names()
    
    # Create mapping from paper_id to processed paper
    paper_map = {paper.paper.arxiv_id: paper for paper in processed_papers}
    
    # Collect all annotated sentences
    sentence_data = {}  # (paper_id, sentence_index) -> labels dict
    
    for ann in annotations:
        key = (ann.paper_id, ann.sentence_index)
        if key not in sentence_data:
            sentence_data[key] = {}
        sentence_data[key].update(ann.labels)
    
    # Build sentence list and label matrix
    sentences = []
    label_matrix = []
    
    for (paper_id, sent_idx), labels in sentence_data.items():
        if paper_id not in paper_map:
            continue
        
        paper = paper_map[paper_id]
        if sent_idx is None:
            # Abstract-level annotation - use all sentences
            for sentence in paper.sentences:
                sentences.append(sentence.text)
                row = [labels.get(label_name, np.nan) for label_name in label_names]
                label_matrix.append(row)
        else:
            # Sentence-level annotation
            if sent_idx < len(paper.sentences):
                sentence = paper.sentences[sent_idx]
                sentences.append(sentence.text)
                row = [labels.get(label_name, np.nan) for label_name in label_names]
                label_matrix.append(row)
    
    label_matrix = np.array(label_matrix, dtype=object)
    
    logger.info(
        f"Prepared training data: {len(sentences)} sentences, "
        f"{len(label_names)} labels"
    )
    
    return sentences, label_matrix, label_names


@task
def fine_tune_embeddings(
    sentences: List[str],
    label_matrix: np.ndarray,
    label_names: List[str],
    base_model: Optional[ArxivSentenceModel] = None,
    num_epochs: Optional[int] = None
) -> ArxivSentenceModel:
    """
    Fine-tune sentence transformer embeddings using setfit-like approach.
    
    Args:
        sentences: List of sentence strings
        label_matrix: Label matrix (n_sentences, n_labels) with NaN support
        label_names: List of label names
        base_model: Base model to fine-tune (defaults to new model)
        num_epochs: Number of training epochs (defaults to config)
    
    Returns:
        Fine-tuned ArxivSentenceModel
    """
    logger = get_run_logger()
    config = get_config()
    
    if base_model is None:
        base_model = ArxivSentenceModel()
    
    if num_epochs is None:
        num_epochs = config.model.num_epochs
    
    # Create training examples for each label
    # For each label, create positive and negative examples
    training_examples = []
    
    for label_idx, label_name in enumerate(label_names):
        # Get positive and negative examples for this label
        label_col = label_matrix[:, label_idx]
        positive_mask = label_col == 1
        negative_mask = label_col == 0
        
        positive_sentences = [sentences[i] for i in range(len(sentences)) if positive_mask[i]]
        negative_sentences = [sentences[i] for i in range(len(sentences)) if negative_mask[i]]
        
        # Create pairs: (positive, positive) and (positive, negative)
        for pos_sent in positive_sentences[:100]:  # Limit for efficiency
            # Positive pairs
            for other_pos in positive_sentences[:10]:
                if pos_sent != other_pos:
                    training_examples.append(
                        InputExample(texts=[pos_sent, other_pos], label=1.0)
                    )
            
            # Negative pairs
            for neg_sent in negative_sentences[:10]:
                training_examples.append(
                    InputExample(texts=[pos_sent, neg_sent], label=0.0)
                )
    
    if not training_examples:
        logger.warning("No training examples created, returning base model")
        return base_model
    
    logger.info(f"Created {len(training_examples)} training examples")
    
    # Fine-tune the model
    train_dataloader = DataLoader(
        training_examples,
        shuffle=True,
        batch_size=config.model.batch_size
    )
    
    train_loss = losses.CosineSimilarityLoss(base_model.model)
    
    base_model.model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=num_epochs,
        warmup_steps=100,
        show_progress_bar=True
    )
    
    logger.info("Fine-tuned embeddings completed")
    return base_model


@task
def train_classifier(
    model: ArxivSentenceModel,
    sentences: List[str],
    label_matrix: np.ndarray,
    label_name: str,
    label_idx: int,
    num_epochs: int = 10,
    learning_rate: float = 1e-3
) -> LabelClassifier:
    """
    Train a classifier for a specific label.
    
    Args:
        model: Sentence transformer model
        sentences: List of sentence strings
        label_matrix: Label matrix (n_sentences, n_labels)
        label_name: Name of the label to train
        label_idx: Index of the label in label_matrix
        num_epochs: Number of training epochs
        learning_rate: Learning rate
    
    Returns:
        Trained LabelClassifier
    """
    logger = get_run_logger()
    
    # Get labels for this specific label
    labels = label_matrix[:, label_idx]
    
    # Filter out NaN values
    valid_mask = ~np.isnan(labels)
    valid_sentences = [sentences[i] for i in range(len(sentences)) if valid_mask[i]]
    valid_labels = labels[valid_mask].astype(float)
    
    if len(valid_sentences) == 0:
        logger.warning(f"No valid training data for label {label_name}")
        # Return untrained classifier
        return create_classifier_for_label(
            label_name,
            model.get_embedding_dim()
        )
    
    logger.info(
        f"Training classifier for {label_name}: "
        f"{len(valid_sentences)} examples"
    )
    
    # Encode sentences
    embeddings = model.encode(valid_sentences, show_progress_bar=False)
    embeddings = embeddings.cpu()
    
    # Create classifier
    classifier = create_classifier_for_label(
        label_name,
        model.get_embedding_dim()
    )
    
    # Training setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = classifier.to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()
    
    # Convert labels to tensor
    labels_tensor = torch.FloatTensor(valid_labels).unsqueeze(1).to(device)
    embeddings_tensor = embeddings.to(device)
    
    # Training loop
    classifier.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        predictions = classifier(embeddings_tensor)
        loss = criterion(predictions, labels_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 5 == 0:
            logger.debug(
                f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}"
            )
    
    logger.info(f"Trained classifier for {label_name}")
    return classifier


@task
def save_model_to_minio(
    model: ArxivSentenceModel | LabelClassifier,
    model_id: str,
    model_type: str,
    label_name: Optional[str] = None
) -> str:
    """
    Save a model to MinIO.
    
    Args:
        model: Model to save (ArxivSentenceModel or LabelClassifier)
        model_id: Unique model identifier
        model_type: Type of model ("sentence_transformer" or "classifier")
        label_name: Label name (for classifiers)
    
    Returns:
        Path where model was saved
    """
    logger = get_run_logger()
    config = get_config()
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = os.path.join(temp_dir, model_id)
        
        if isinstance(model, ArxivSentenceModel):
            model.save(model_path)
        else:
            # Save PyTorch classifier
            torch.save(model.state_dict(), model_path + ".pth")
        
        # Upload to MinIO
        if label_name:
            object_name = f"{config.storage.models_path}/{model_type}/{label_name}/{model_id}"
        else:
            object_name = f"{config.storage.models_path}/{model_type}/{model_id}"
        
        bucket_name = config.storage.bucket_name
        ensure_bucket_exists(bucket_name)
        
        if isinstance(model, ArxivSentenceModel):
            # Upload entire directory
            archive_path = model_path + ".tar.gz"
            shutil.make_archive(model_path, "gztar", model_path)
            upload_file(bucket_name, object_name + ".tar.gz", archive_path)
        else:
            upload_file(bucket_name, object_name + ".pth", model_path + ".pth")
        
        logger.info(f"Saved {model_type} model to {bucket_name}/{object_name}")
        return object_name


@task
def load_model_from_minio(
    model_id: str,
    model_type: str,
    label_name: Optional[str] = None
) -> ArxivSentenceModel | LabelClassifier:
    """
    Load a model from MinIO.
    
    Args:
        model_id: Model identifier
        model_type: Type of model ("sentence_transformer" or "classifier")
        label_name: Label name (for classifiers)
    
    Returns:
        Loaded model
    """
    logger = get_run_logger()
    config = get_config()
    
    if label_name:
        object_name = f"{config.storage.models_path}/{model_type}/{label_name}/{model_id}"
    else:
        object_name = f"{config.storage.models_path}/{model_type}/{model_id}"
    
    bucket_name = config.storage.bucket_name
    
    with tempfile.TemporaryDirectory() as temp_dir:
        if model_type == "sentence_transformer":
            # Download and extract
            download_file(bucket_name, object_name + ".tar.gz", temp_dir + "/model.tar.gz")
            with tarfile.open(temp_dir + "/model.tar.gz", "r:gz") as tar:
                tar.extractall(temp_dir)
            model = ArxivSentenceModel.load(temp_dir + "/model")
        else:
            # Load classifier
            download_file(bucket_name, object_name + ".pth", temp_dir + "/model.pth")
            from arxiv_pipeline.models.classifier import create_classifier_for_label
            config_obj = get_config()
            classifier = create_classifier_for_label(
                label_name or "default",
                config_obj.model.embedding_dim
            )
            classifier.load_state_dict(torch.load(temp_dir + "/model.pth"))
            model = classifier
        
        logger.info(f"Loaded {model_type} model from {bucket_name}/{object_name}")
        return model

