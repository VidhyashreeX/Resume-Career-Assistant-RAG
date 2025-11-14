"""
Vector Store Manager for ChromaDB operations.

This module provides functionality to store and retrieve resume chunks
using ChromaDB as the vector database backend.
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
import uuid
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manages ChromaDB operations for storing and querying resume chunks."""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        Initialize the Vector Store Manager.
        
        Args:
            persist_directory: Directory path for persistent ChromaDB storage
        """
        self.persist_directory = persist_directory
        
        # Initialize ChromaDB client with persistent storage
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            is_persistent=True
        ))
        
        # Get or create the resumes collection
        self.collection = self.client.get_or_create_collection(
            name="resumes",
            metadata={"description": "Resume chunks with embeddings"}
        )
        
        logger.info(f"Initialized VectorStoreManager with persist_directory: {persist_directory}")
    
    def add_resume(self, chunks: List[Dict]) -> None:
        """
        Add resume chunks to the vector store.
        
        Args:
            chunks: List of chunk dictionaries containing:
                - text: The chunk text content
                - embedding: Vector embedding (list of floats)
                - metadata: Dict with resume_id, section_type, chunk_index, source_file
        
        Raises:
            ValueError: If chunks are missing required fields
        """
        if not chunks:
            logger.warning("No chunks provided to add_resume")
            return
        
        # Prepare data for ChromaDB
        ids = []
        embeddings = []
        documents = []
        metadatas = []
        
        for chunk in chunks:
            # Validate chunk structure
            if 'text' not in chunk or 'embedding' not in chunk or 'metadata' not in chunk:
                raise ValueError("Each chunk must contain 'text', 'embedding', and 'metadata'")
            
            # Generate unique ID for each chunk
            chunk_id = str(uuid.uuid4())
            ids.append(chunk_id)
            
            # Extract embedding
            embeddings.append(chunk['embedding'])
            
            # Extract text
            documents.append(chunk['text'])
            
            # Extract metadata
            metadatas.append(chunk['metadata'])
        
        # Add to ChromaDB collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        
        resume_id = chunks[0]['metadata'].get('resume_id', 'unknown')
        logger.info(f"Added {len(chunks)} chunks for resume: {resume_id}")
    
    def query(self, query_embedding: List[float], top_k: int = 10) -> List[Dict]:
        """
        Retrieve most similar chunks from the vector store.
        
        Args:
            query_embedding: Vector embedding of the query (e.g., job description)
            top_k: Number of top results to return
        
        Returns:
            List of dictionaries containing:
                - id: Chunk ID
                - text: Chunk text content
                - metadata: Chunk metadata
                - distance: Similarity distance (lower is more similar)
        """
        if not query_embedding:
            raise ValueError("query_embedding cannot be empty")
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Format results
        formatted_results = []
        
        if results['ids'] and len(results['ids']) > 0:
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    'id': results['ids'][0][i],
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                })
        
        logger.info(f"Retrieved {len(formatted_results)} chunks from query")
        return formatted_results
    
    def list_resumes(self) -> List[str]:
        """
        List all unique resume IDs stored in the vector store.
        
        Returns:
            List of resume IDs
        """
        # Get all items from collection
        all_items = self.collection.get()
        
        # Extract unique resume IDs from metadata
        resume_ids = set()
        if all_items['metadatas']:
            for metadata in all_items['metadatas']:
                if 'resume_id' in metadata:
                    resume_ids.add(metadata['resume_id'])
        
        resume_list = sorted(list(resume_ids))
        logger.info(f"Found {len(resume_list)} unique resumes in vector store")
        return resume_list
    
    def delete_resume(self, resume_id: str) -> None:
        """
        Remove all chunks for a specific resume from the vector store.
        
        Args:
            resume_id: The resume ID to delete
        """
        # Get all items
        all_items = self.collection.get()
        
        # Find IDs matching the resume_id
        ids_to_delete = []
        if all_items['metadatas']:
            for i, metadata in enumerate(all_items['metadatas']):
                if metadata.get('resume_id') == resume_id:
                    ids_to_delete.append(all_items['ids'][i])
        
        # Delete matching chunks
        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)
            logger.info(f"Deleted {len(ids_to_delete)} chunks for resume: {resume_id}")
        else:
            logger.warning(f"No chunks found for resume: {resume_id}")
    
    def get_collection_stats(self) -> Dict:
        """
        Get statistics about the vector store collection.
        
        Returns:
            Dictionary with collection statistics
        """
        all_items = self.collection.get()
        total_chunks = len(all_items['ids']) if all_items['ids'] else 0
        unique_resumes = len(self.list_resumes())
        
        stats = {
            'total_chunks': total_chunks,
            'unique_resumes': unique_resumes,
            'collection_name': self.collection.name
        }
        
        logger.info(f"Collection stats: {stats}")
        return stats
