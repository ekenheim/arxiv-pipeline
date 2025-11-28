"""Data schemas for arXiv pipeline."""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class Author(BaseModel):
    """Author information."""
    name: str
    affiliation: Optional[str] = None


class Paper(BaseModel):
    """ArXiv paper metadata."""
    arxiv_id: str
    title: str
    authors: List[Author]
    abstract: str
    categories: List[str]
    published: datetime
    updated: Optional[datetime] = None
    pdf_url: Optional[str] = None
    primary_category: Optional[str] = None


class Sentence(BaseModel):
    """Sentence from a paper abstract."""
    text: str
    sentence_index: int
    paper_id: str
    start_char: Optional[int] = None
    end_char: Optional[int] = None


class ProcessedPaper(BaseModel):
    """Processed paper with sentences."""
    paper: Paper
    sentences: List[Sentence]
    processed_at: datetime = Field(default_factory=datetime.now)


class Annotation(BaseModel):
    """Annotation for a sentence or abstract."""
    paper_id: str
    sentence_index: Optional[int] = None  # None for abstract-level annotations
    labels: Dict[str, bool]  # label_name -> bool
    annotated_at: datetime = Field(default_factory=datetime.now)
    annotator: Optional[str] = None


class AnnotationBatch(BaseModel):
    """Batch of annotations."""
    annotations: List[Annotation]
    created_at: datetime = Field(default_factory=datetime.now)


class ModelMetadata(BaseModel):
    """Metadata for a trained model."""
    model_id: str
    model_type: str  # e.g., "sentence_transformer", "classifier"
    label_name: Optional[str] = None  # For per-label classifiers
    base_model: str
    training_date: datetime
    metrics: Dict[str, float] = Field(default_factory=dict)
    config: Dict[str, Any] = Field(default_factory=dict)


class Recommendation(BaseModel):
    """Recommendation for a paper."""
    paper_id: str
    label: str
    score: float
    sentence_indices: List[int]  # Sentences that contributed to this recommendation
    generated_at: datetime = Field(default_factory=datetime.now)


class PaperRecommendations(BaseModel):
    """Recommendations for a paper."""
    paper_id: str
    recommendations: List[Recommendation]
    generated_at: datetime = Field(default_factory=datetime.now)

