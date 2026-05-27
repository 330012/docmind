"""
Central configuration for DocMind.

Loads environment variables from .env and exposes them as typed settings.
All other modules should import from here — never read env vars directly.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # ─── LLM Provider ───
    llm_provider: str = "groq"
    llm_model: str = "llama-3.1-8b-instant"
    groq_api_key: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # ─── LangFuse ───
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # ─── Embeddings ───
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ─── Chunking ───
    chunk_size: int = 512
    chunk_overlap: int = 50

    # ─── Vector Store ───
    vector_store_path: Path = Path("./data/faiss_index")

    # ─── App ───
    log_level: str = "INFO"
    max_agent_iterations: int = 2

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


# Singleton instance — import this everywhere
settings = Settings()
