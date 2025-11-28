"""Configuration loading and management."""

import os
from pathlib import Path
from typing import Dict, List, Any
import yaml
from pydantic import BaseModel, Field


class LabelConfig(BaseModel):
    """Label definition."""
    name: str
    description: str


class ModelConfig(BaseModel):
    """Model configuration."""
    base_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    max_length: int = 512
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3


class TrainingConfig(BaseModel):
    """Training configuration."""
    test_size: float = 0.2
    validation_size: float = 0.1
    min_samples_per_label: int = 5


class StorageConfig(BaseModel):
    """Storage configuration."""
    bucket_name: str = "arxiv-pipeline"
    raw_data_path: str = "raw"
    processed_data_path: str = "processed"
    annotations_path: str = "annotations"
    models_path: str = "models"
    recommendations_path: str = "recommendations"


class DataConfig(BaseModel):
    """Data processing configuration."""
    max_papers_per_day: int = 1000
    sentence_separators: List[str] = Field(default_factory=lambda: ["\n\n", ". ", "! ", "? "])
    min_sentence_length: int = 10


class PipelineConfig(BaseModel):
    """Main pipeline configuration."""
    labels: List[LabelConfig]
    model: ModelConfig
    training: TrainingConfig
    storage: StorageConfig
    data: DataConfig


_config: PipelineConfig | None = None


def load_config(config_path: str | None = None) -> PipelineConfig:
    """Load configuration from YAML file, with environment variable overrides."""
    import os
    global _config
    
    if _config is not None:
        return _config
    
    if config_path is None:
        # Default to configs/config.yml relative to project root
        project_root = Path(__file__).parent.parent.parent.parent
        config_path = project_root / "configs" / "config.yml"
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    
    # Override storage bucket_name from environment variable if set
    if "MINIO_BUCKET_NAME" in os.environ:
        if "storage" not in config_dict:
            config_dict["storage"] = {}
        config_dict["storage"]["bucket_name"] = os.environ["MINIO_BUCKET_NAME"]
    
    _config = PipelineConfig(**config_dict)
    return _config


def get_config() -> PipelineConfig:
    """Get the loaded configuration."""
    if _config is None:
        return load_config()
    return _config


def get_labels() -> List[LabelConfig]:
    """Get label definitions."""
    return get_config().labels


def get_label_names() -> List[str]:
    """Get list of label names."""
    return [label.name for label in get_labels()]

