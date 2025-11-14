#!/usr/bin/env python3
"""
Main entry point for Resume Career Assistant.

This module provides a command-line interface for the Resume Career Assistant
with options to process resumes or launch the web UI.
"""

import argparse
import sys
import logging
from pathlib import Path

from src.career_assistant import CareerAssistant

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('career_assistant.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def check_ollama_connection(assistant: CareerAssistant) -> bool:
    """
    Check if Ollama is running and accessible.
    
    Args:
        assistant: CareerAssistant instance
        
    Returns:
        True if Ollama is connected, False otherwise
    """
    logger.info("Checking Ollama connection...")
    
    if assistant.check_ollama_connection():
        logger.info("✅ Ollama is running and accessible")
        return True
    else:
        logger.error("❌ Cannot connect to Ollama")
        logger.error("Please ensure Ollama is running:")
        logger.error("  1. Install Ollama from https://ollama.ai")
        logger.error("  2. Start Ollama service")
        logger.error("  3. Pull the model: ollama pull gemma3:1b")
        return False


def process_resumes(assistant: CareerAssistant, folder_path: str) -> int:
    """
    Process all resumes in the specified folder.
    
    Args:
        assistant: CareerAssistant instance
        folder_path: Path to folder containing PDF resumes
        
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    logger.info(f"Processing resumes from: {folder_path}")
    
    # Verify folder exists
    folder = Path(folder_path)
    if not folder.exists():
        logger.error(f"Folder not found: {folder_path}")
        return 1
    
    if not folder.is_dir():
        logger.error(f"Path is not a directory: {folder_path}")
        return 1
    
    # Process resumes
    stats = assistant.process_resume_folder(folder_path=folder_path)
    
    # Display results
    print("\n" + "="*60)
    print("RESUME PROCESSING COMPLETE")
    print("="*60)
    print(f"Total PDF files found:      {stats['total_files']}")
    print(f"Successfully processed:     {stats['processed']}")
    print(f"Skipped (already in DB):    {stats['skipped']}")
    print(f"Failed:                     {stats['failed']}")
    print("="*60)
    
    if stats['errors']:
        print("\nErrors encountered:")
        for error in stats['errors'][:10]:  # Show first 10 errors
            print(f"  - {error}")
        if len(stats['errors']) > 10:
            print(f"  ... and {len(stats['errors']) - 10} more errors")
    
    if stats['processed'] > 0:
        print("\n✅ Resumes are now ready for job analysis!")
        print("   Run with --ui flag to launch the web interface")
    
    return 0 if stats['failed'] == 0 else 1


def launch_ui(assistant: CareerAssistant, host: str, port: int, share: bool) -> int:
    """
    Launch the Gradio web interface.
    
    Args:
        assistant: CareerAssistant instance
        host: Server host address
        port: Server port number
        share: Whether to create a public share link
        
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        # Import Gradio and app module
        import gradio as gr
        from app import create_ui
        
        logger.info("Launching Gradio web interface...")
        
        # Create and launch the UI
        app = create_ui()
        app.launch(
            server_name=host,
            server_port=port,
            share=share
        )
        
        return 0
        
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        logger.error("Please ensure Gradio is installed: pip install gradio")
        return 1
    except Exception as e:
        logger.error(f"Failed to launch UI: {e}")
        return 1


def main():
    """Main entry point for the application."""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Resume Career Assistant - AI-powered resume analysis and career guidance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --process                    # Process resumes from ./ENGINEERING folder
  %(prog)s --process --folder ./resumes # Process resumes from custom folder
  %(prog)s --ui                         # Launch web interface
  %(prog)s --ui --port 8080             # Launch UI on custom port
  %(prog)s --process --ui               # Process resumes then launch UI
        """
    )
    
    parser.add_argument(
        '--process',
        action='store_true',
        help='Process PDF resumes from the specified folder'
    )
    
    parser.add_argument(
        '--folder',
        type=str,
        default='./ENGINEERING',
        help='Path to folder containing PDF resumes (default: ./ENGINEERING)'
    )
    
    parser.add_argument(
        '--ui',
        action='store_true',
        help='Launch the Gradio web interface'
    )
    
    parser.add_argument(
        '--host',
        type=str,
        default='127.0.0.1',
        help='Host address for web interface (default: 127.0.0.1)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=7860,
        help='Port for web interface (default: 7860)'
    )
    
    parser.add_argument(
        '--share',
        action='store_true',
        help='Create a public share link for the web interface'
    )
    
    parser.add_argument(
        '--chroma-db',
        type=str,
        default='./chroma_db',
        help='Path for ChromaDB storage (default: ./chroma_db)'
    )
    
    parser.add_argument(
        '--ollama-url',
        type=str,
        default='http://localhost:11434',
        help='Ollama API URL (default: http://localhost:11434)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='gemma3:1b',
        help='Ollama model name (default: gemma3:1b)'
    )
    
    args = parser.parse_args()
    
    # Check if at least one action is specified
    if not args.process and not args.ui:
        parser.print_help()
        print("\n❌ Error: Please specify at least one action (--process or --ui)")
        return 1
    
    # Display banner
    print("\n" + "="*60)
    print("Resume Career Assistant")
    print("AI-powered resume analysis using RAG and local LLM")
    print("="*60 + "\n")
    
    try:
        # Initialize Career Assistant
        logger.info("Initializing Career Assistant...")
        assistant = CareerAssistant(
            chroma_db_path=args.chroma_db,
            llm_model=args.model,
            ollama_url=args.ollama_url
        )
        
        # Check Ollama connection on startup
        if not check_ollama_connection(assistant):
            logger.error("Cannot proceed without Ollama connection")
            return 1
        
        exit_code = 0
        
        # Process resumes if requested
        if args.process:
            result = process_resumes(assistant, args.folder)
            if result != 0:
                exit_code = result
        
        # Launch UI if requested
        if args.ui:
            result = launch_ui(assistant, args.host, args.port, args.share)
            if result != 0:
                exit_code = result
        
        return exit_code
        
    except KeyboardInterrupt:
        logger.info("\nOperation cancelled by user")
        return 0
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
