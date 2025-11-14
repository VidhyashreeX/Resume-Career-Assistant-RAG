"""
Retrieval Engine for finding relevant resume chunks.

This module provides functionality to retrieve the most relevant resume sections
based on a job description query using semantic similarity search.
"""

from typing import List, Dict, Optional
import logging
from src.vector_store import VectorStoreManager
from src.embedding_engine import EmbeddingEngine

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RetrievalEngine:
    """
    Finds relevant resume chunks for a given job description.
    
    Uses the embedding engine to convert job descriptions to vectors,
    then queries the vector store to find semantically similar resume chunks.
    """
    
    def __init__(self, vector_store: VectorStoreManager, embedding_engine: EmbeddingEngine):
        """
        Initialize the Retrieval Engine.
        
        Args:
            vector_store: VectorStoreManager instance for querying stored chunks
            embedding_engine: EmbeddingEngine instance for generating query embeddings
        """
        self.vector_store = vector_store
        self.embedding_engine = embedding_engine
        logger.info("Initialized RetrievalEngine")
    
    def retrieve_relevant_chunks(self, job_description: str, top_k: int = 15) -> List[Dict]:
        """
        Find the most relevant resume sections for a job description.
        
        Args:
            job_description: The job posting text to match against
            top_k: Number of top matching chunks to return (default: 15)
        
        Returns:
            List of dictionaries containing:
                - id: Chunk ID
                - text: Chunk text content
                - metadata: Chunk metadata (resume_id, section_type, etc.)
                - distance: Similarity distance (lower is more similar)
                - similarity_score: Normalized similarity score (0-1, higher is more similar)
        
        Raises:
            ValueError: If job_description is empty
        """
        if not job_description or not job_description.strip():
            raise ValueError("Job description cannot be empty")
        
        logger.info(f"Retrieving top {top_k} chunks for job description")
        
        # Generate embedding for the job description
        query_embedding = self.embedding_engine.embed_text(job_description)
        logger.info(f"Generated query embedding with dimension: {len(query_embedding)}")
        
        # Query the vector store
        results = self.vector_store.query(query_embedding, top_k=top_k)
        
        # Add similarity scores (convert distance to similarity)
        # ChromaDB uses L2 distance, so we convert to similarity score
        for result in results:
            if result.get('distance') is not None:
                # Convert L2 distance to similarity score (0-1 range)
                # For normalized embeddings, L2 distance ranges from 0 to 2
                # Similarity = 1 - (distance / 2)
                result['similarity_score'] = max(0.0, 1.0 - (result['distance'] / 2.0))
            else:
                result['similarity_score'] = None
        
        logger.info(f"Retrieved {len(results)} relevant chunks")
        
        return results
