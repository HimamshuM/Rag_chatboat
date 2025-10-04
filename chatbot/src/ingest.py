 
"""
Ingest documents into FAISS vector store.

- Loads docx and pdf from data/
- Chunks text
- Computes embeddings via sentence-transformers (HF)
- Builds FAISS index saved to INDEX_PATH
"""

import os
import pickle
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()

from config import settings
from utils import ensure_dir, make_text_splitter
from embeddings import HFEmbeddings

def load_documents(data_dir="data"):
    docs = []
    for p in Path(data_dir).iterdir():
        if p.suffix.lower() in [".docx", ".doc"]:
            loader = UnstructuredWordDocumentLoader(str(p))
            docs.extend(loader.load())
        elif p.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(p))
            docs.extend(loader.load())
        elif p.suffix.lower() in [".md", ".txt"]:
            # simple loader for plain text
            with open(p, "r", encoding="utf-8") as f:
                txt = f.read()
            docs.append(Document(page_content=txt, metadata={"source": str(p)}))
    return docs

def main():
    print("Loading documents...")
    docs = load_documents("chatbot\src\data")
    print(f"Loaded {len(docs)} raw documents")

    text_splitter = make_text_splitter(settings.chunk_size, settings.chunk_overlap)
    print("Splitting into chunks...")
    docs_chunks = text_splitter.split_documents(docs)
    print(f"Total chunks: {len(docs_chunks)}")

    # Use sentence-transformers via LangChain HuggingFaceEmbeddings wrapper
    print("Computing embeddings...")
    hf = HuggingFaceEmbeddings(model_name=settings.embedding_model)

    # Build vectorstore
    print("Building FAISS index...")
    faiss_index = FAISS.from_documents(docs_chunks, hf)

    # ensure dir
    ensure_dir(settings.index_path)
    with open(settings.index_path, "wb") as f:
        pickle.dump(faiss_index, f)
    print(f"Saved FAISS index to {settings.index_path}")

if __name__ == "__main__":
    main()
