# shared/config.py
# Central configuration. All secrets come from environment variables.

import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM (Orchestrator) ────────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "deepseek-r1:7b")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")

# ── Vector Store (Search Engine) ─────────────────────────────────────────────
CHROMA_HOST       = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT       = int(os.getenv("CHROMA_PORT", "8000"))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "crop_traits")
EMBEDDING_MODEL   = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ── Research APIs ─────────────────────────────────────────────────────────────
SEMANTIC_SCHOLAR_KEY = os.getenv("SEMANTIC_SCHOLAR_KEY", "")
ARXIV_MAX_RESULTS    = int(os.getenv("ARXIV_MAX_RESULTS", "5"))

# ── Pipeline ──────────────────────────────────────────────────────────────────
MAX_RETRIES     = int(os.getenv("MAX_RETRIES", "3"))
SEARCH_TOP_K    = int(os.getenv("SEARCH_TOP_K", "10"))
MIN_CONFIDENCE  = float(os.getenv("MIN_CONFIDENCE", "0.6"))
