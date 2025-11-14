"""
Gradio UI for Resume Career Assistant.

This module provides a web-based user interface for the Resume Career Assistant
using Gradio. It includes tabs for processing resumes and analyzing job descriptions.
"""

import gradio as gr
import logging
from pathlib import Path
from src.career_assistant import CareerAssistant

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize the Career Assistant
assistant = CareerAssistant()


def process_resumes(folder_path: str = "./ENGINEERING") -> str:
    """
    Process all resumes in the specified folder.
    
    Args:
        folder_path: Path to folder containing PDF resumes
        
    Returns:
        Status message with processing results
    """
    try:
        logger.info(f"Processing resumes from: {folder_path}")
        stats = assistant.process_resume_folder(folder_path=folder_path)
        
        # Format the results
        result = f"""
### Resume Processing Complete

**Summary:**
- Total PDF files found: {stats['total_files']}
- Successfully processed: {stats['processed']}
- Skipped (already in database): {stats['skipped']}
- Failed: {stats['failed']}

"""
        
        if stats['errors']:
            result += "**Errors:**\n"
            for error in stats['errors'][:5]:  # Show first 5 errors
                result += f"- {error}\n"
            if len(stats['errors']) > 5:
                result += f"- ... and {len(stats['errors']) - 5} more errors\n"
        
        if stats['processed'] > 0:
            result += "\n✅ Resumes are now ready for job analysis!"
        
        return result
        
    except Exception as e:
        error_msg = f"Error processing resumes: {str(e)}"
        logger.error(error_msg)
        return f"❌ {error_msg}"


def analyze_job_description(job_description: str, job_title: str = "") -> tuple:
    """
    Analyze a job description against stored resumes.
    
    Args:
        job_description: The job posting text
        job_title: Optional job title
        
    Returns:
        Tuple of (fit_score, skill_analysis, improvements, cover_letter)
    """
    try:
        if not job_description or not job_description.strip():
            error_msg = "⚠️ Please enter a job description"
            return error_msg, "", "", ""
        
        logger.info("Analyzing job description...")
        result = assistant.analyze_job(
            job_description=job_description,
            job_title=job_title
        )
        
        # Check for errors
        if "error" in result:
            error_msg = f"❌ {result['error']}"
            return error_msg, "", "", ""
        
        # Format job fit score
        fit_score_text = f"""
### Job Fit Score: {result['job_fit_score']}%

**Reasoning:**
{result['fit_reasoning']}

**Matching Qualifications:**
"""
        for qual in result.get('matching_qualifications', []):
            fit_score_text += f"- {qual}\n"
        
        fit_score_text += "\n**Missing Qualifications:**\n"
        for qual in result.get('missing_qualifications', []):
            fit_score_text += f"- {qual}\n"
        
        # Format skill analysis
        skill_analysis_text = f"""
### Skill Analysis

**Matching Skills:**
"""
        matching_skills = result.get('skill_analysis', {}).get('matching_skills', [])
        for skill in matching_skills:
            skill_analysis_text += f"- {skill}\n"
        
        skill_analysis_text += "\n**Missing/Weak Skills:**\n"
        missing_skills = result.get('skill_analysis', {}).get('missing_skills', [])
        for skill in missing_skills:
            skill_analysis_text += f"- {skill}\n"
        
        # Format resume improvements
        improvements_text = "### Resume Improvements\n\n"
        for i, improvement in enumerate(result.get('resume_improvements', []), 1):
            improvements_text += f"{i}. {improvement}\n\n"
        
        # Get cover letter
        cover_letter_text = result.get('cover_letter', '')
        
        # Add metadata
        metadata = result.get('metadata', {})
        fit_score_text += f"\n\n---\n*Analysis based on {metadata.get('chunks_analyzed', 0)} relevant sections from {metadata.get('unique_resumes', 0)} resume(s)*"
        
        return fit_score_text, skill_analysis_text, improvements_text, cover_letter_text
        
    except Exception as e:
        error_msg = f"❌ Error during analysis: {str(e)}"
        logger.error(error_msg)
        return error_msg, "", "", ""


def check_system_status() -> str:
    """
    Check the status of the system components.
    
    Returns:
        Status message
    """
    try:
        # Check Ollama connection
        ollama_status = "✅ Connected" if assistant.check_ollama_connection() else "❌ Not connected"
        
        # Get system stats
        stats = assistant.get_stats()
        vector_stats = stats.get('vector_store_stats', {})
        
        status_text = f"""
### System Status

**Ollama LLM:** {ollama_status}
**Embedding Model:** all-MiniLM-L6-v2
**Vector Store:** ChromaDB

**Database Statistics:**
- Total chunks stored: {vector_stats.get('total_chunks', 0)}
- Unique resumes: {vector_stats.get('unique_resumes', 0)}
- Embedding dimension: {stats.get('embedding_dimension', 384)}
"""
        return status_text
        
    except Exception as e:
        return f"❌ Error checking status: {str(e)}"


# Create the Gradio interface
def create_ui():
    """Create and configure the Gradio interface."""
    
    with gr.Blocks(title="Resume Career Assistant", theme=gr.themes.Soft()) as app:
        gr.Markdown(
            """
            # 🎯 Resume Career Assistant
            
            AI-powered resume analysis and career guidance using RAG and local LLM.
            """
        )
        
        with gr.Tabs():
            # Tab 1: Process Resumes
            with gr.Tab("📁 Process Resumes"):
                gr.Markdown(
                    """
                    ### Process Resume PDFs
                    
                    Load PDF resumes from a folder into the vector database.
                    This needs to be done before analyzing job descriptions.
                    """
                )
                
                with gr.Row():
                    folder_input = gr.Textbox(
                        label="Folder Path",
                        value="./ENGINEERING",
                        placeholder="./ENGINEERING"
                    )
                
                process_btn = gr.Button("🚀 Process Resumes", variant="primary")
                process_output = gr.Markdown(label="Status")
                
                process_btn.click(
                    fn=process_resumes,
                    inputs=[folder_input],
                    outputs=[process_output]
                )
            
            # Tab 2: Analyze Job
            with gr.Tab("🔍 Analyze Job"):
                gr.Markdown(
                    """
                    ### Analyze Job Description
                    
                    Enter a job description to get:
                    - Job fit score and reasoning
                    - Skill gap analysis
                    - Resume improvement suggestions
                    - Tailored cover letter
                    """
                )
                
                with gr.Row():
                    with gr.Column():
                        job_title_input = gr.Textbox(
                            label="Job Title (Optional)",
                            placeholder="e.g., Senior Software Engineer"
                        )
                        job_desc_input = gr.Textbox(
                            label="Job Description",
                            placeholder="Paste the job description here...",
                            lines=10
                        )
                        analyze_btn = gr.Button("🎯 Analyze Job", variant="primary")
                
                with gr.Row():
                    with gr.Column():
                        fit_score_output = gr.Markdown(label="Job Fit Score")
                        skill_analysis_output = gr.Markdown(label="Skill Analysis")
                    
                    with gr.Column():
                        improvements_output = gr.Markdown(label="Resume Improvements")
                        cover_letter_output = gr.Textbox(
                            label="Cover Letter",
                            lines=12
                        )
                
                analyze_btn.click(
                    fn=analyze_job_description,
                    inputs=[job_desc_input, job_title_input],
                    outputs=[
                        fit_score_output,
                        skill_analysis_output,
                        improvements_output,
                        cover_letter_output
                    ]
                )
            
            # Tab 3: System Status
            with gr.Tab("⚙️ System Status"):
                gr.Markdown(
                    """
                    ### System Information
                    
                    Check the status of system components and database statistics.
                    """
                )
                
                status_btn = gr.Button("🔄 Refresh Status")
                status_output = gr.Markdown()
                
                status_btn.click(
                    fn=check_system_status,
                    inputs=[],
                    outputs=[status_output]
                )
                
                # Load status on page load
                app.load(fn=check_system_status, outputs=[status_output])
        
        gr.Markdown(
            """
            ---
            **Note:** This application runs entirely locally using Ollama for LLM inference.
            Your resume data never leaves your machine.
            """
        )
    
    return app


if __name__ == "__main__":
    logger.info("Starting Resume Career Assistant UI...")
    
    # Create and launch the UI
    app = create_ui()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False
    )
