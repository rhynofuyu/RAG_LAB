# RAG LAB - Modular Architecture

A modular RAG (Retrieval-Augmented Generation) application built with Streamlit.

## Project Structure

```
RAG_LAB/
├── main.py                     # Application entry point
├── app.py                      # Original monolithic file (backup)
├── config/
│   ├── __init__.py
│   └── settings.py             # Configuration settings
├── database/
│   ├── __init__.py
│   └── db_manager.py           # Database operations
├── document_processing/
│   ├── __init__.py
│   └── processor.py            # PDF processing and chunking
├── vector_store/
│   ├── __init__.py
│   └── manager.py              # Vector store and embeddings
├── query_processing/
│   ├── __init__.py
│   └── processor.py            # Query handling and multi-query RAG
├── ui/
│   ├── __init__.py
│   └── components.py           # Streamlit UI components
└── requirements.txt
```

## Usage

To run the application:
```bash
streamlit run main.py
```

## Modules

### Database (`database/`)
- User authentication and registration
- Chat history management
- SQLite database operations

### Document Processing (`document_processing/`)
- PDF text extraction and chunking
- Metadata handling
- Text preprocessing

### Vector Store (`vector_store/`)
- Pinecone vector database management
- Embedding generation
- Retriever setup

### Query Processing (`query_processing/`)
- Multi-query RAG implementation
- Query variation generation
- Document retrieval and scoring

### UI Components (`ui/`)
- Streamlit interface components
- Authentication forms
- Chat interface
- Settings sidebar

### Configuration (`config/`)
- Application settings and constants
- Model configurations
- Default parameters
