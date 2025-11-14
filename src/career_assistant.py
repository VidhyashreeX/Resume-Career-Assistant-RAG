"""
Career Assistant - Main application orchestrator.

This module coordinates all components to provide a complete RAG-based
resume analysis system. It handles resume processing and job analysis workflows.
"""

import os
import uuid
from pathlib import Path
from typing import List, Dict, Optional
import logging

from src.pdf_processor import PDFProcessor
from src.text_chunker import TextChunker, Chunk
from src.embedding_engine import EmbeddingEngine
from src.vector_store import VectorStoreManager
from src.retrieval_engine import RetrievalEngine
from src.llm_generator import LLMGenerator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CareerAssistant:
    """
    Main application class that orchestrates the resume career assistant system.
    
    Coordinates PDF processing, text chunking, embedding generation, vector storage,
    retrieval, and LLM-based analysis to provide comprehensive career assistance.
    """
    
    def __init__(
        self,
        chroma_db_path: str = "./chroma_db",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "gemma3:1b",
        ollama_url: str = "http://localhost:11434"
    ):
        """
        Initialize the Career Assistant with all required components.
        
        Args:
            chroma_db_path: Path for ChromaDB persistent storage
            embedding_model: Name of the sentence-transformers model
            llm_model: Name of the Ollama model to use
            ollama_url: Base URL for Ollama API
        """
        logger.info("Initializing Career Assistant...")
        
        # Initialize all components
        self.pdf_processor = PDFProcessor()
        self.text_chunker = TextChunker(chunk_size=350, overlap=50)
        self.embedding_engine = EmbeddingEngine(model_name=embedding_model)
        self.vector_store = VectorStoreManager(persist_directory=chroma_db_path)
        self.retrieval_engine = RetrievalEngine(
            vector_store=self.vector_store,
            embedding_engine=self.embedding_engine
        )
        self.llm_generator = LLMGenerator(
            model_name=llm_model,
            base_url=ollama_url
        )
        
        logger.info("Career Assistant initialized successfully")

    def process_resume_folder(
        self,
        folder_path: str = "./ENGINEERING",
        skip_existing: bool = True
    ) -> Dict[str, any]:
        """
        Process all PDF resumes in a folder and add them to the vector store.
        
        This method performs batch processing of resumes:
        1. Scans the folder for PDF files
        2. Extracts text from each PDF
        3. Chunks the text into semantic sections
        4. Generates embeddings for each chunk
        5. Stores chunks in the vector database
        
        Args:
            folder_path: Path to folder containing PDF resumes
            skip_existing: If True, skip resumes already in the vector store
            
        Returns:
            Dictionary with processing statistics:
                - total_files: Total PDF files found
                - processed: Number of files successfully processed
                - skipped: Number of files skipped
                - failed: Number of files that failed processing
                - errors: List of error messages
        """
        logger.info(f"Starting batch processing of resumes in: {folder_path}")
        
        folder = Path(folder_path)
        if not folder.exists() or not folder.is_dir():
            error_msg = f"Folder not found: {folder_path}"
            logger.error(error_msg)
            return {
                "total_files": 0,
                "processed": 0,
                "skipped": 0,
                "failed": 0,
                "errors": [error_msg]
            }
        
        # Find all PDF files
        pdf_files = list(folder.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        # Get existing resume IDs if skip_existing is True
        existing_resumes = set()
        if skip_existing:
            existing_resumes = set(self.vector_store.list_resumes())
            logger.info(f"Found {len(existing_resumes)} existing resumes in vector store")
        
        # Process each PDF
        stats = {
            "total_files": len(pdf_files),
            "processed": 0,
            "skipped": 0,
            "failed": 0,
            "errors": []
        }
        
        for pdf_file in pdf_files:
            try:
                # Generate resume ID from filename
                resume_id = pdf_file.stem  # Filename without extension
                
                # Skip if already processed
                if skip_existing and resume_id in existing_resumes:
                    logger.info(f"Skipping already processed resume: {resume_id}")
                    stats["skipped"] += 1
                    continue
                
                logger.info(f"Processing resume: {resume_id}")
                
                # Extract text from PDF
                text = self.pdf_processor.extract_text(str(pdf_file))
                
                if not text:
                    error_msg = f"No text extracted from {pdf_file.name}"
                    logger.warning(error_msg)
                    stats["failed"] += 1
                    stats["errors"].append(error_msg)
                    continue
                
                # Chunk the text
                chunks = self.text_chunker.chunk_resume(
                    text=text,
                    resume_id=resume_id,
                    source_file=pdf_file.name
                )
                
                if not chunks:
                    error_msg = f"No chunks created for {pdf_file.name}"
                    logger.warning(error_msg)
                    stats["failed"] += 1
                    stats["errors"].append(error_msg)
                    continue
                
                # Generate embeddings for all chunks
                chunk_texts = [chunk.text for chunk in chunks]
                embeddings = self.embedding_engine.embed_batch(chunk_texts)
                
                # Prepare chunks for storage
                chunks_with_embeddings = []
                for chunk, embedding in zip(chunks, embeddings):
                    chunks_with_embeddings.append({
                        "text": chunk.text,
                        "embedding": embedding,
                        "metadata": chunk.metadata
                    })
                
                # Add to vector store
                self.vector_store.add_resume(chunks_with_embeddings)
                
                stats["processed"] += 1
                logger.info(f"Successfully processed {resume_id} ({len(chunks)} chunks)")
                
            except Exception as e:
                error_msg = f"Error processing {pdf_file.name}: {str(e)}"
                logger.error(error_msg)
                stats["failed"] += 1
                stats["errors"].append(error_msg)
        
        # Log summary
        logger.info(f"Batch processing complete: {stats}")
        return stats

    def analyze_job(
        self,
        job_description: str,
        job_title: str = "",
        top_k: int = 15
    ) -> Dict[str, any]:
        """
        Perform complete analysis of a job description against stored resumes.
        
        This method executes the full RAG pipeline:
        1. Retrieves relevant resume chunks based on job description
        2. Generates job fit score with reasoning
        3. Analyzes matching and missing skills
        4. Provides resume improvement suggestions
        5. Creates a tailored cover letter
        
        Args:
            job_description: The job posting text to analyze
            job_title: Optional job title for personalization
            top_k: Number of relevant chunks to retrieve (default: 15)
            
        Returns:
            Dictionary containing complete analysis:
                - job_fit_score: Score from 0-100
                - fit_reasoning: Explanation of the score
                - matching_qualifications: List of matching skills/experiences
                - missing_qualifications: List of gaps
                - skill_analysis: Detailed skill breakdown
                - resume_improvements: List of actionable suggestions
                - cover_letter: Generated cover letter text
                - retrieved_chunks: Reference chunks used for analysis
                - metadata: Additional information (resume sources, etc.)
        """
        logger.info("Starting job analysis...")
        
        if not job_description or not job_description.strip():
            error_msg = "Job description cannot be empty"
            logger.error(error_msg)
            return {
                "error": error_msg,
                "job_fit_score": 0,
                "fit_reasoning": error_msg
            }
        
        try:
            # Step 1: Retrieve relevant resume chunks
            logger.info(f"Retrieving top {top_k} relevant chunks...")
            retrieved_chunks = self.retrieval_engine.retrieve_relevant_chunks(
                job_description=job_description,
                top_k=top_k
            )
            
            if not retrieved_chunks:
                logger.warning("No relevant chunks found in vector store")
                return {
                    "error": "No resumes found in the database. Please process resumes first.",
                    "job_fit_score": 0,
                    "fit_reasoning": "No resume data available for analysis"
                }
            
            logger.info(f"Retrieved {len(retrieved_chunks)} chunks")
            
            # Step 2: Generate job fit score
            logger.info("Generating job fit score...")
            fit_analysis = self.llm_generator.generate_job_fit_score(
                resume_chunks=retrieved_chunks,
                job_description=job_description
            )
            
            # Step 3: Analyze skills
            logger.info("Analyzing skills...")
            skill_analysis = self.llm_generator.generate_skill_analysis(
                resume_chunks=retrieved_chunks,
                job_description=job_description
            )
            
            # Step 4: Generate resume improvements
            logger.info("Generating resume improvements...")
            improvements = self.llm_generator.generate_resume_improvements(
                resume_chunks=retrieved_chunks,
                job_description=job_description
            )
            
            # Step 5: Generate cover letter
            logger.info("Generating cover letter...")
            cover_letter = self.llm_generator.generate_cover_letter(
                resume_chunks=retrieved_chunks,
                job_description=job_description,
                job_title=job_title
            )
            
            # Compile results
            result = {
                "job_fit_score": fit_analysis.get("score", 0),
                "fit_reasoning": fit_analysis.get("reasoning", ""),
                "matching_qualifications": fit_analysis.get("matching_qualifications", []),
                "missing_qualifications": fit_analysis.get("missing_qualifications", []),
                "skill_analysis": skill_analysis,
                "resume_improvements": improvements,
                "cover_letter": cover_letter,
                "retrieved_chunks": retrieved_chunks,
                "metadata": {
                    "chunks_analyzed": len(retrieved_chunks),
                    "unique_resumes": len(set(
                        chunk.get("metadata", {}).get("resume_id", "unknown")
                        for chunk in retrieved_chunks
                    )),
                    "job_title": job_title
                }
            }
            
            logger.info("Job analysis complete")
            return result
            
        except Exception as e:
            error_msg = f"Error during job analysis: {str(e)}"
            logger.error(error_msg)
            return {
                "error": error_msg,
                "job_fit_score": 0,
                "fit_reasoning": "Analysis failed due to an error"
            }
    
    def get_stats(self) -> Dict[str, any]:
        """
        Get statistics about the current state of the system.
        
        Returns:
            Dictionary with system statistics:
                - vector_store_stats: ChromaDB collection statistics
                - embedding_dimension: Dimension of embedding vectors
        """
        stats = {
            "vector_store_stats": self.vector_store.get_collection_stats(),
            "embedding_dimension": self.embedding_engine.get_embedding_dimension()
        }
        return stats
    
    def check_ollama_connection(self) -> bool:
        """
        Check if Ollama is running and accessible.
        
        Returns:
            True if Ollama is accessible, False otherwise
        """
        return self.llm_generator._check_connection()
