 
import streamlit as st
from src.retriever import answer_query, load_index
from src.config import settings
from dotenv import load_dotenv
from src.config import some_function
from src.retriever import get_response
from src.config import some_function
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import settings

load_dotenv()

st.set_page_config(page_title="RAG Chat (Groq + FAISS)", page_icon="🤖")

st.title("RAG Chatbot — Groq + FAISS (LangChain)")
st.write("Ask questions about the uploaded FAQ and Assessment documents.")

if st.sidebar.button("Reload index / Build if missing"):
    st.sidebar.write("If you haven't run ingest.py, run it now (or this will attempt to load index).")

query = st.text_input("Your question", placeholder="e.g., How do I cancel TontonUp subscription?")

top_k = st.slider("Number of retrieved chunks (k)", min_value=1, max_value=10, value=4)

if st.button("Ask"):
    if not query.strip():
        st.warning("Please type a question.")
    else:
        with st.spinner("Retrieving and generating answer..."):
            try:
                answer = answer_query(query, top_k=top_k)
                st.markdown("### Answer")
                st.write(answer)
            except Exception as e:
                st.error(f"Error: {e}")
                st.info("Make sure you ran `python src/ingest.py` to create the FAISS index and set API keys in .env")
