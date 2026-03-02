"""Application settings loaded from environment / .env file."""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # General
    APP_ENV: str = "local"

    # Groq
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"

    # Vector store
    VECTOR_STORE: str = "faiss"
    FAISS_DIR: str = str(_PROJECT_ROOT / "data" / "faiss_index")

    # Chunking
    CHUNK_TOKENS: int = 500
    CHUNK_OVERLAP: int = 80
    TOP_K: int = 5
    MIN_SCORE: float = 0.35

    # Guardrails
    ENABLE_GROUNDING_CHECK: bool = True
    ENABLE_FALLBACK: bool = True

    # Derived paths
    @property
    def knowledge_base_dir(self) -> Path:
        return _PROJECT_ROOT / "app" / "knowledge_base"

    @property
    def faiss_dir_path(self) -> Path:
        return Path(self.FAISS_DIR)


settings = Settings()
