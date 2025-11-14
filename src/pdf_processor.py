"""PDF text extraction module for Resume Career Assistant.

This module handles extracting text content from PDF resume files using pdfplumber.
It provides error handling for corrupted or unreadable files.
"""

import pdfplumber
from pathlib import Path
from typing import Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFProcessor:
    """Handles PDF text extraction operations."""
    
    def extract_text(self, pdf_path: str) -> Optional[str]:
        """Extract all text from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file to process
            
        Returns:
            Extracted text as a string, or None if extraction fails
            
        Raises:
            FileNotFoundError: If the PDF file doesn't exist
        """
        path = Path(pdf_path)
        
        # Check if file exists
        if not path.exists():
            logger.error(f"PDF file not found: {pdf_path}")
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                # Check if PDF has pages
                if len(pdf.pages) == 0:
                    logger.warning(f"PDF file is empty: {pdf_path}")
                    return None
                
                # Extract text from all pages
                text_parts = []
                for page_num, page in enumerate(pdf.pages, start=1):
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                    else:
                        logger.warning(f"No text found on page {page_num} of {pdf_path}")
                
                # Combine all pages with double newline separator
                full_text = "\n\n".join(text_parts)
                
                if not full_text.strip():
                    logger.warning(f"No text content extracted from {pdf_path}")
                    return None
                
                logger.info(f"Successfully extracted {len(full_text)} characters from {pdf_path}")
                return full_text
                
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            return None
    
    def extract_text_with_structure(self, pdf_path: str) -> Optional[dict]:
        """Extract text from PDF preserving page structure.
        
        Args:
            pdf_path: Path to the PDF file to process
            
        Returns:
            Dictionary with page-by-page text and metadata, or None if extraction fails
        """
        path = Path(pdf_path)
        
        if not path.exists():
            logger.error(f"PDF file not found: {pdf_path}")
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                if len(pdf.pages) == 0:
                    logger.warning(f"PDF file is empty: {pdf_path}")
                    return None
                
                result = {
                    "source_file": str(path.name),
                    "total_pages": len(pdf.pages),
                    "pages": []
                }
                
                for page_num, page in enumerate(pdf.pages, start=1):
                    page_text = page.extract_text()
                    result["pages"].append({
                        "page_number": page_num,
                        "text": page_text if page_text else ""
                    })
                
                logger.info(f"Successfully extracted structured text from {pdf_path}")
                return result
                
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            return None
