"""Classifier models for label prediction."""

import torch
import torch.nn as nn
from typing import Optional
from arxiv_pipeline.utils.config import get_config


class LabelClassifier(nn.Module):
    """Binary classifier for a single label."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        """
        Initialize classifier.
        
        Args:
            input_dim: Input embedding dimension
            hidden_dim: Hidden layer dimension (defaults to input_dim)
            dropout: Dropout rate
        """
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = input_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            embeddings: Input embeddings (batch_size, input_dim)
        
        Returns:
            Predictions (batch_size, 1)
        """
        return self.classifier(embeddings)


class MultiLabelClassifier(nn.Module):
    """Multi-label classifier for all labels."""
    
    def __init__(
        self,
        input_dim: int,
        num_labels: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        """
        Initialize multi-label classifier.
        
        Args:
            input_dim: Input embedding dimension
            num_labels: Number of labels
            hidden_dim: Hidden layer dimension (defaults to input_dim)
            dropout: Dropout rate
        """
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = input_dim
        
        self.num_labels = num_labels
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
            nn.Sigmoid()
        )
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            embeddings: Input embeddings (batch_size, input_dim)
        
        Returns:
            Predictions (batch_size, num_labels)
        """
        return self.classifier(embeddings)


def create_classifier_for_label(
    label_name: str,
    input_dim: int,
    hidden_dim: Optional[int] = None,
    dropout: float = 0.1
) -> LabelClassifier:
    """
    Create a classifier for a specific label.
    
    Args:
        label_name: Name of the label
        input_dim: Input embedding dimension
        hidden_dim: Hidden layer dimension
        dropout: Dropout rate
    
    Returns:
        LabelClassifier instance
    """
    return LabelClassifier(input_dim, hidden_dim, dropout)

