"""
Embedding Engine for converting text to vector embeddings.

This module uses sentence-transformers to generate embeddings for resume chunks
and job descriptions, enabling semantic similarity search.
"""

from typing import List, Optional
from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingEngine:
    """
    Handles text-to-vector embedding generation using sentence-transformers.
    
    Uses the all-MiniLM-L6-v2 model which is lightweight (384 dimensions)
    and provides good performance for semantic similarity tasks.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding engine with a specified model.
        
        Args:
            model_name: Name of the sentence-transformers model to use.
                       Default is "all-MiniLM-L6-v2" (lightweight, 384 dims).
        """
        self.model_name = model_name
        self._model: Optional[SentenceTransformer] = None
    
    @property
    def model(self) -> SentenceTransformer:
        """
        Lazy-load the model on first use to save memory.
        
        Returns:
            Loaded SentenceTransformer model.
        """
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text string.
        
        Args:
            text: Input text to embed.
        
        Returns:
            List of floats representing the embedding vector.
        """
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")
        
        # Generate embedding and normalize for cosine similarity
        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        return embedding.tolist()
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts efficiently.
        
        Args:
            texts: List of text strings to embed.
        
        Returns:
            List of embedding vectors, one for each input text.
        """
        if not texts:
            raise ValueError("Input texts list cannot be empty")
        
        if any(not text or not text.strip() for text in texts):
            raise ValueError("All input texts must be non-empty")
        
        # Batch encoding is more efficient than individual calls
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 10,  # Show progress for large batches
            batch_size=32
        )
        
        return embeddings.tolist()
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimensionality of the embedding vectors.
        
        Returns:
            Integer dimension of the embedding space.
        """
        return self.model.get_sentence_embedding_dimension()
