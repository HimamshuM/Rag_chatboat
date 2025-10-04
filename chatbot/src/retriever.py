 
"""
Retriever + RAG orchestration.

- Loads FAISS index (created by ingest.py)
- Given a query, retrieves top_k chunks
- Constructs a prompt using prompt template and calls groq_client.generate
"""
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
import pickle
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from config import settings

load_dotenv()

from config import settings
from groq_client import generate
from prompts import make_prompt_for_answer

def load_index(index_path: str = None):
    if index_path is None:
        index_path = settings.index_path
    with open(index_path, "rb") as f:
        faiss_index = pickle.load(f)
    return faiss_index

def answer_query(query: str, top_k: int = 5) -> str:
    faiss_index = load_index()
    hf = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    retriever = faiss_index.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
    docs = retriever.get_relevant_documents(query)

    # Build system/prompt
    prompt = make_prompt_for_answer(query, docs)

    # Call Groq LLM wrapper
    answer = generate(prompt, max_tokens=512, temperature=0.0)
    return answer
