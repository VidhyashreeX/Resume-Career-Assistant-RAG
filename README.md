# Resume Career Assistant

A handy tool that helps you analyze your resume, match it with job descriptions, and generate tailored cover letters - all while keeping your data private on your local machine.

##  Quick Start

1. **Setup**
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

2. **Dataset**
- i have downloaded a zip  `ENGINEERING` folder from KAggale 
- If you are unable to access it , create a dummy version

3. **Run the App**
```bash
python main.py
```

##  What It Does

-  Analyzes your resume against job descriptions
-  Shows job fit scores with detailed feedback
-  Identifies skill gaps
- Generates customized cover letters
- use of local LLM - your data never leaves your computer

##  Tech Stack

- **Backend**: Python
- **AI**: Local LLM with Ollama
- **Vector DB**: ChromaDB
- **UI**: Gradio
- **PDF Processing**: pdfplumber

##  Requirements

- Python 3.8+
- 8GB RAM (16GB recommended)
- About 5GB free space

### Project Structure

```
RAG_PROJECT/
│
├── main.py                      # CLI entry point
├── app.py                       # Gradio web UI
├── requirements.txt             # Python dependencies
├── config.yaml                  # Configuration settings
│
├── src/                         # Source code modules
│   ├── __init__.py
│   ├── pdf_processor.py         # PDF text extraction
│   ├── text_chunker.py          # Text chunking logic
│   ├── embedding_engine.py      # Embedding generation
│   ├── vector_store.py          # ChromaDB operations
│   ├── retrieval_engine.py      # Similarity search
│   ├── llm_generator.py         # Ollama LLM interface
│   └── career_assistant.py      # Main orchestrator
│
├── ENGINEERING/                 # Resume PDFs folder
│   ├── resume1.pdf
│   ├── resume2.pdf
│   └── ...
│
├── chroma_db/                   # ChromaDB persistent storage
│   └── [vector database files]
│
├── tests/                       # Unit and integration tests
│   ├── test_pdf_processor.py
│   ├── test_chunker.py
│   └── ...

```
