# shared/config.py
# Central configuration with typed settings and validation.
# All secrets come from environment variables with sensible defaults.

import logging
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator

logger = logging.getLogger(__name__)


class AppSettings(BaseSettings):
    """Typed configuration with environment variable validation."""

    # Application environment
    app_env: str = Field(default="development", description="dev|prod|test")

    # LLM (Orchestrator)
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL for Ollama service"
    )
    ollama_model: str = Field(
        default="deepseek-r1:7b",
        description="Model name for Ollama"
    )
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key (optional)"
    )

    # Vector Store (Search Engine)
    chroma_host: str = Field(default="localhost", description="Chroma server host")
    chroma_port: int = Field(default=8000, ge=1, le=65535, description="Chroma server port")
    chroma_collection: str = Field(default="crop_traits", description="Default collection name")
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence transformer model"
    )

    # Research APIs
    semantic_scholar_key: Optional[str] = Field(
        default=None,
        description="Semantic Scholar API key (optional)"
    )
    arxiv_max_results: int = Field(default=5, ge=1, le=100, description="Max ArXiv results per query")

    # Pipeline
    max_retries: int = Field(default=3, ge=1, le=10, description="Max retry attempts")
    search_top_k: int = Field(default=10, ge=1, le=100, description="Top-K for similarity search")
    min_confidence: float = Field(default=0.6, ge=0.0, le=1.0, description="Minimum confidence threshold")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @field_validator("app_env")
    @classmethod
    def validate_env(cls, v):
        if v not in ("development", "prod", "test"):
            raise ValueError("app_env must be one of: development, prod, test")
        return v


# Singleton configuration instance
_settings: Optional[AppSettings] = None


def get_settings() -> AppSettings:
    """Get or create the global settings object."""
    global _settings
    if _settings is None:
        _settings = AppSettings()
        logger.info(f"Settings loaded for environment: {_settings.app_env}")
    return _settings


# Convenience exports (backwards compatible)
settings = get_settings()
OLLAMA_BASE_URL = settings.ollama_base_url
OLLAMA_MODEL = settings.ollama_model
OPENAI_API_KEY = settings.openai_api_key
CHROMA_HOST = settings.chroma_host
CHROMA_PORT = settings.chroma_port
CHROMA_COLLECTION = settings.chroma_collection
EMBEDDING_MODEL = settings.embedding_model
SEMANTIC_SCHOLAR_KEY = settings.semantic_scholar_key
ARXIV_MAX_RESULTS = settings.arxiv_max_results
MAX_RETRIES = settings.max_retries
SEARCH_TOP_K = settings.search_top_k
MIN_CONFIDENCE = settings.min_confidence
