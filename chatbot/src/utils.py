import os
from langchain.text_splitter import RecursiveCharacterTextSplitter

def make_text_splitter(chunk_size: int = 1000, chunk_overlap: int = 200):
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )

def ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
