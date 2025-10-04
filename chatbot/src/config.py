from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    groq_api_key: str | None = None
    openai_api_key: str | None = None
    index_path: str = "./indexes/faiss_index.pkl"
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 1000
    chunk_overlap: int = 200

    class Config:
        env_file = ".env"

settings = Settings()
