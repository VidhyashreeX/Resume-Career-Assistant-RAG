"""
LLM Generator module for generating career assistance outputs using Ollama.
"""

import requests
import json
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMGenerator:
    """Handles all LLM generation tasks using Ollama with gemma3:1b model."""
    
    def __init__(self, model_name: str = "gemma3:1b", base_url: str = "http://localhost:11434"):
        """
        Initialize the LLM Generator.
        
        Args:
            model_name: Name of the Ollama model to use
            base_url: Base URL for Ollama API
        """
        self.model_name = model_name
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
        self.timeout = 60
        
    def _check_connection(self) -> bool:
        """
        Check if Ollama is running and accessible.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            return False
    
    def _generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 500) -> Optional[str]:
        """
        Generate text using Ollama API.
        
        Args:
            prompt: The prompt to send to the model
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated text or None if generation fails
        """
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            logger.error("Request to Ollama timed out")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse response: {e}")
            return None

    def generate_job_fit_score(self, resume_chunks: List[Dict], job_description: str) -> Dict:
        """
        Calculate job fit score with reasoning.
        
        Args:
            resume_chunks: List of relevant resume chunks with text and metadata
            job_description: The job description to match against
            
        Returns:
            Dictionary containing score, reasoning, and key points
        """
        if not self._check_connection():
            return {
                "score": 0,
                "reasoning": "Error: Cannot connect to Ollama. Please ensure Ollama is running.",
                "matching_qualifications": [],
                "missing_qualifications": []
            }
        
        # Format resume chunks
        resume_text = "\n\n".join([
            f"Section: {chunk.get('metadata', {}).get('section_type', 'Unknown')}\n{chunk.get('text', '')}"
            for chunk in resume_chunks[:10]  # Limit to top 10 chunks
        ])
        
        prompt = f"""You are an expert career advisor. Analyze the following resume sections against the job description and provide:
1. A job fit score (0-100%)
2. Clear reasoning for the score
3. Key matching qualifications (list 3-5 items)
4. Missing qualifications (list 3-5 items)

Resume sections:
{resume_text}

Job description:
{job_description}

Provide your analysis in this exact format:
SCORE: [number between 0-100]
REASONING: [2-3 sentences explaining the score]
MATCHING: [bullet point list of matching qualifications]
MISSING: [bullet point list of missing qualifications]"""

        response = self._generate(prompt, temperature=0.7, max_tokens=500)
        
        if not response:
            return {
                "score": 0,
                "reasoning": "Error: Failed to generate analysis",
                "matching_qualifications": [],
                "missing_qualifications": []
            }
        
        # Parse the response
        return self._parse_job_fit_response(response)
    
    def _parse_job_fit_response(self, response: str) -> Dict:
        """Parse the job fit score response into structured data."""
        result = {
            "score": 0,
            "reasoning": "",
            "matching_qualifications": [],
            "missing_qualifications": []
        }
        
        lines = response.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if line.startswith("SCORE:"):
                try:
                    score_text = line.replace("SCORE:", "").strip().rstrip('%')
                    result["score"] = float(score_text)
                except ValueError:
                    result["score"] = 0
            elif line.startswith("REASONING:"):
                result["reasoning"] = line.replace("REASONING:", "").strip()
                current_section = "reasoning"
            elif line.startswith("MATCHING:"):
                current_section = "matching"
            elif line.startswith("MISSING:"):
                current_section = "missing"
            elif line.startswith("-") or line.startswith("•"):
                item = line.lstrip("-•").strip()
                if current_section == "matching" and item:
                    result["matching_qualifications"].append(item)
                elif current_section == "missing" and item:
                    result["missing_qualifications"].append(item)
            elif current_section == "reasoning" and line and not line.startswith(("MATCHING:", "MISSING:")):
                result["reasoning"] += " " + line
        
        result["reasoning"] = result["reasoning"].strip()
        return result

    def generate_skill_analysis(self, resume_chunks: List[Dict], job_description: str) -> Dict:
        """
        Identify matching and missing skills.
        
        Args:
            resume_chunks: List of relevant resume chunks with text and metadata
            job_description: The job description to analyze
            
        Returns:
            Dictionary containing matching_skills and missing_skills lists
        """
        if not self._check_connection():
            return {
                "matching_skills": [],
                "missing_skills": [],
                "skill_gaps": []
            }
        
        # Format resume chunks
        resume_text = "\n\n".join([
            f"{chunk.get('text', '')}"
            for chunk in resume_chunks[:10]
        ])
        
        prompt = f"""You are an expert career advisor. Analyze the skills in the resume against the job requirements.

Resume content:
{resume_text}

Job description:
{job_description}

Provide a detailed skill analysis in this format:
MATCHING SKILLS: [list skills the candidate has that match the job]
MISSING SKILLS: [list required skills the candidate lacks]
SKILL GAPS: [list areas where the candidate has partial but not complete expertise]

Use bullet points for each list."""

        response = self._generate(prompt, temperature=0.7, max_tokens=500)
        
        if not response:
            return {
                "matching_skills": [],
                "missing_skills": [],
                "skill_gaps": []
            }
        
        return self._parse_skill_analysis_response(response)
    
    def _parse_skill_analysis_response(self, response: str) -> Dict:
        """Parse the skill analysis response into structured data."""
        result = {
            "matching_skills": [],
            "missing_skills": [],
            "skill_gaps": []
        }
        
        lines = response.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if "MATCHING SKILLS:" in line.upper():
                current_section = "matching"
            elif "MISSING SKILLS:" in line.upper():
                current_section = "missing"
            elif "SKILL GAPS:" in line.upper():
                current_section = "gaps"
            elif line.startswith("-") or line.startswith("•") or line.startswith("*"):
                item = line.lstrip("-•*").strip()
                if current_section == "matching" and item:
                    result["matching_skills"].append(item)
                elif current_section == "missing" and item:
                    result["missing_skills"].append(item)
                elif current_section == "gaps" and item:
                    result["skill_gaps"].append(item)
        
        return result

    def generate_resume_improvements(self, resume_chunks: List[Dict], job_description: str) -> List[str]:
        """
        Suggest resume improvements tailored to the job.
        
        Args:
            resume_chunks: List of relevant resume chunks with text and metadata
            job_description: The job description to tailor improvements for
            
        Returns:
            List of actionable improvement suggestions
        """
        if not self._check_connection():
            return ["Error: Cannot connect to Ollama. Please ensure Ollama is running."]
        
        # Format resume chunks
        resume_text = "\n\n".join([
            f"{chunk.get('text', '')}"
            for chunk in resume_chunks[:10]
        ])
        
        prompt = f"""You are an expert resume writer. Review the resume content and suggest specific improvements to better match the job description.

Resume content:
{resume_text}

Job description:
{job_description}

Provide 5-7 specific, actionable resume improvements. Focus on:
- Better phrasing with action verbs and quantifiable results
- Keywords from the job description to incorporate
- Formatting improvements for ATS compatibility
- Specific examples of how to rewrite sections

Format each suggestion as a bullet point starting with a dash (-)."""

        response = self._generate(prompt, temperature=0.7, max_tokens=500)
        
        if not response:
            return ["Error: Failed to generate improvements"]
        
        # Parse bullet points
        improvements = []
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith("-") or line.startswith("•") or line.startswith("*"):
                improvement = line.lstrip("-•*").strip()
                if improvement:
                    improvements.append(improvement)
        
        return improvements if improvements else ["No specific improvements identified"]

    def generate_cover_letter(self, resume_chunks: List[Dict], job_description: str, job_title: str = "") -> str:
        """
        Create a tailored cover letter for the job application.
        
        Args:
            resume_chunks: List of relevant resume chunks with text and metadata
            job_description: The job description
            job_title: Optional job title for personalization
            
        Returns:
            Professional cover letter text (200-300 words)
        """
        if not self._check_connection():
            return "Error: Cannot connect to Ollama. Please ensure Ollama is running."
        
        # Format resume chunks - focus on most relevant experiences
        resume_text = "\n\n".join([
            f"{chunk.get('text', '')}"
            for chunk in resume_chunks[:8]  # Use top 8 most relevant chunks
        ])
        
        job_title_text = f" for the {job_title} position" if job_title else ""
        
        prompt = f"""You are a professional cover letter writer. Write a compelling cover letter{job_title_text} based on the candidate's relevant experiences and skills.

Candidate's relevant experience and skills:
{resume_text}

Job description:
{job_description}

Write a professional cover letter (200-300 words) that:
- Opens with enthusiasm for the position
- Highlights 2-3 most relevant experiences/achievements from the resume
- Explains why the candidate is a great fit
- Closes with a call to action
- Uses a persuasive, human-like tone
- Avoids generic phrases

Write the complete cover letter now:"""

        response = self._generate(prompt, temperature=0.7, max_tokens=400)
        
        if not response:
            return "Error: Failed to generate cover letter"
        
        return response.strip()
