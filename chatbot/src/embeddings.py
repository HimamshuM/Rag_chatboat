 
from sentence_transformers import SentenceTransformer
import numpy as np

class HFEmbeddings:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        vec = self.model.encode([text], convert_to_numpy=True)
        return vec[0].tolist()
