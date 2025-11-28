"""Sentence transformer model for arXiv papers."""

from typing import List, Optional
import torch
from sentence_transformers import SentenceTransformer
from arxiv_pipeline.utils.config import get_config


class ArxivSentenceModel:
    """Sentence transformer model for arXiv paper processing."""
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize sentence transformer model.
        
        Args:
            model_name: Name of the base model (defaults to config)
            device: Device to use (cuda/cpu, defaults to auto)
        """
        config = get_config()
        
        if model_name is None:
            model_name = config.model.base_model
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model_name = model_name
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        self.config = config
    
    def encode(
        self,
        sentences: List[str],
        batch_size: Optional[int] = None,
        show_progress_bar: bool = False
    ) -> torch.Tensor:
        """
        Encode sentences into embeddings.
        
        Args:
            sentences: List of sentence strings
            batch_size: Batch size (defaults to config)
            show_progress_bar: Whether to show progress bar
        
        Returns:
            Tensor of embeddings (n_sentences, embedding_dim)
        """
        if batch_size is None:
            batch_size = self.config.model.batch_size
        
        embeddings = self.model.encode(
            sentences,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_tensor=True,
            device=self.device
        )
        
        return embeddings
    
    def encode_single(self, sentence: str) -> torch.Tensor:
        """
        Encode a single sentence.
        
        Args:
            sentence: Sentence string
        
        Returns:
            Embedding tensor (embedding_dim,)
        """
        return self.encode([sentence])[0]
    
    def get_embedding_dim(self) -> int:
        """Get the embedding dimension."""
        return self.config.model.embedding_dim
    
    def save(self, path: str) -> None:
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model
        """
        self.model.save(path)
    
    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "ArxivSentenceModel":
        """
        Load a model from disk.
        
        Args:
            path: Path to load the model from
            device: Device to use
        
        Returns:
            ArxivSentenceModel instance
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        model = SentenceTransformer(path, device=device)
        instance = cls.__new__(cls)
        instance.model = model
        instance.device = device
        instance.model_name = path
        instance.config = get_config()
        
        return instance

