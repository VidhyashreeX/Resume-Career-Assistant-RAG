"""Text chunking module for Resume Career Assistant.

This module handles splitting resume text into semantic chunks with metadata.
It identifies resume sections and creates chunks of 300-400 tokens with overlap.
"""

import re
import uuid
from typing import List, Optional
from dataclasses import dataclass
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    id: str
    text: str
    metadata: dict
    embedding: Optional[List[float]] = None


class TextChunker:
    """Handles text chunking operations for resumes."""
    
    # Common resume section headers
    SECTION_PATTERNS = [
        r'(?i)^(professional\s+)?summary',
        r'(?i)^(professional\s+)?experience',
        r'(?i)^work\s+experience',
        r'(?i)^employment\s+history',
        r'(?i)^education',
        r'(?i)^academic\s+background',
        r'(?i)^skills',
        r'(?i)^technical\s+skills',
        r'(?i)^core\s+competencies',
        r'(?i)^certifications?',
        r'(?i)^projects?',
        r'(?i)^publications?',
        r'(?i)^awards?',
        r'(?i)^honors?',
        r'(?i)^volunteer',
        r'(?i)^languages?',
        r'(?i)^interests?',
        r'(?i)^references?',
    ]
    
    def __init__(self, chunk_size: int = 350, overlap: int = 50):
        """Initialize the text chunker.
        
        Args:
            chunk_size: Target number of tokens per chunk (default 350)
            overlap: Number of tokens to overlap between chunks (default 50)
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count using simple word-based approximation.
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            Estimated token count (roughly 1.3 tokens per word)
        """
        words = text.split()
        return int(len(words) * 1.3)
    
    def _identify_section_type(self, text: str) -> str:
        """Identify the section type based on content.
        
        Args:
            text: Text to analyze
            
        Returns:
            Section type identifier
        """
        # Check first few lines for section headers
        lines = text.strip().split('\n')[:3]
        header_text = '\n'.join(lines)
        
        for pattern in self.SECTION_PATTERNS:
            if re.search(pattern, header_text):
                # Extract the matched section name
                match = re.search(pattern, header_text)
                if match:
                    section_name = match.group(0).strip().lower()
                    # Normalize section names
                    if 'experience' in section_name or 'employment' in section_name:
                        return 'experience'
                    elif 'education' in section_name or 'academic' in section_name:
                        return 'education'
                    elif 'skill' in section_name or 'competenc' in section_name:
                        return 'skills'
                    elif 'summary' in section_name:
                        return 'summary'
                    elif 'certification' in section_name:
                        return 'certifications'
                    elif 'project' in section_name:
                        return 'projects'
                    elif 'publication' in section_name:
                        return 'publications'
                    elif 'award' in section_name or 'honor' in section_name:
                        return 'awards'
                    elif 'volunteer' in section_name:
                        return 'volunteer'
                    elif 'language' in section_name:
                        return 'languages'
                    else:
                        return section_name.replace(' ', '_')
        
        return 'general'

    def identify_sections(self, text: str) -> List[dict]:
        """Identify resume sections based on common patterns.
        
        Args:
            text: Full resume text
            
        Returns:
            List of sections with start positions and types
        """
        sections = []
        lines = text.split('\n')
        
        current_pos = 0
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Check if line matches a section header pattern
            for pattern in self.SECTION_PATTERNS:
                if re.search(pattern, line_stripped) and len(line_stripped) < 50:
                    # Calculate character position
                    section_start = sum(len(l) + 1 for l in lines[:i])
                    section_type = self._identify_section_type(line_stripped)
                    
                    sections.append({
                        'start': section_start,
                        'line_number': i,
                        'type': section_type,
                        'header': line_stripped
                    })
                    break
        
        # If no sections found, treat entire text as one section
        if not sections:
            sections.append({
                'start': 0,
                'line_number': 0,
                'type': 'general',
                'header': 'Resume Content'
            })
        
        logger.info(f"Identified {len(sections)} sections in resume")
        return sections
    
    def _split_text_by_tokens(self, text: str, max_tokens: int, overlap_tokens: int) -> List[str]:
        """Split text into chunks based on token count.
        
        Args:
            text: Text to split
            max_tokens: Maximum tokens per chunk
            overlap_tokens: Number of tokens to overlap
            
        Returns:
            List of text chunks
        """
        words = text.split()
        chunks = []
        
        # Approximate words per chunk (tokens / 1.3)
        words_per_chunk = int(max_tokens / 1.3)
        overlap_words = int(overlap_tokens / 1.3)
        
        start = 0
        while start < len(words):
            end = start + words_per_chunk
            chunk_words = words[start:end]
            chunks.append(' '.join(chunk_words))
            
            # Move start position with overlap
            start = end - overlap_words
            
            # Prevent infinite loop if overlap is too large
            if start <= 0 and len(chunks) > 0:
                break
        
        return chunks
    
    def chunk_resume(self, text: str, resume_id: str, source_file: str = "") -> List[Chunk]:
        """Split resume text into semantic chunks with metadata.
        
        Args:
            text: Full resume text to chunk
            resume_id: Unique identifier for the resume
            source_file: Original filename
            
        Returns:
            List of Chunk objects with text and metadata
        """
        if not text or not text.strip():
            logger.warning(f"Empty text provided for resume {resume_id}")
            return []
        
        chunks = []
        sections = self.identify_sections(text)
        
        # Process each section
        for i, section in enumerate(sections):
            # Determine section boundaries
            section_start = section['start']
            section_end = sections[i + 1]['start'] if i + 1 < len(sections) else len(text)
            section_text = text[section_start:section_end].strip()
            
            # Skip empty sections
            if not section_text:
                continue
            
            # Estimate tokens in this section
            section_tokens = self._estimate_tokens(section_text)
            
            # If section is small enough, keep as single chunk
            if section_tokens <= self.chunk_size:
                chunk = Chunk(
                    id=str(uuid.uuid4()),
                    text=section_text,
                    metadata={
                        'resume_id': resume_id,
                        'section_type': section['type'],
                        'chunk_index': len(chunks),
                        'source_file': source_file,
                        'section_header': section['header']
                    }
                )
                chunks.append(chunk)
            else:
                # Split large sections into multiple chunks
                section_chunks = self._split_text_by_tokens(
                    section_text, 
                    self.chunk_size, 
                    self.overlap
                )
                
                for j, chunk_text in enumerate(section_chunks):
                    chunk = Chunk(
                        id=str(uuid.uuid4()),
                        text=chunk_text,
                        metadata={
                            'resume_id': resume_id,
                            'section_type': section['type'],
                            'chunk_index': len(chunks),
                            'source_file': source_file,
                            'section_header': section['header'],
                            'sub_chunk': j
                        }
                    )
                    chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} chunks for resume {resume_id}")
        return chunks
