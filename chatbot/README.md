 # RAG Chatbot — Groq + FAISS (LangChain)

This repository demonstrates a Retrieval-Augmented Generation (RAG) chatbot built with:
- LangChain (retrieval components)
- FAISS vector store
- SentenceTransformers embeddings (HuggingFace `all-MiniLM-L6-v2`)
- A minimal Groq LLM wrapper (user requested Groq)
- Streamlit web UI

Documents included:
- `data/FAQ.docx`
- `data/Assessment (Data Science).pdf`

## Features
- Document ingestion (docx, pdf, md, txt)
- Chunking and indexing to FAISS
- Retrieval of top-K chunks for a query
- LLM generation using Groq (modular wrapper)
- Streamlit web UI for queries

## Setup

1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate     # macOS/Linux
.venv\Scripts\activate        # Windows

