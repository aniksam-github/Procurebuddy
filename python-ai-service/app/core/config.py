"""Environment configuration and application settings.

Loads environment variables from .env files and provides a typed Settings class.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger("procurebuddy-ai")

# ── Load environment ────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # python-ai-service/
ROOT_DIR = BASE_DIR.parent  # project root

load_dotenv(BASE_DIR / ".env", override=True)
load_dotenv(ROOT_DIR / ".env", override=True)

# ── Mask and validate API key ───────────────────────────────────────────────
_raw_key = os.getenv("GROQ_API_KEY") or ""
print("ENV BASE:", BASE_DIR)
print("ENV ROOT:", ROOT_DIR)
print("GROQ_API_KEY:", f"****{_raw_key[-4:]}" if len(_raw_key) > 4 else "<NOT SET>")


def reload_runtime_env() -> None:
    """Reload .env files so runtime secret changes can be picked up safely."""

    load_dotenv(BASE_DIR / ".env", override=True)
    load_dotenv(ROOT_DIR / ".env", override=True)


def resolve_data_dir() -> Path:
    """Discover the knowledge-base data directory."""
    configured = os.getenv("PROCUREBUDDY_DATA_DIR")
    if configured:
        return Path(configured).expanduser()

    service_dir = Path(__file__).resolve().parent.parent.parent
    repo_dir = service_dir.parent
    candidates = [
        repo_dir / "data",
        Path.cwd() / "data",
    ]

    for candidate in candidates:
        if candidate.exists():
            logger.info("Using discovered knowledge base directory: %s", candidate)
            return candidate

    fallback = Path("/home/ec2-user/procurebuddy-data")
    logger.info("Using fallback knowledge base directory: %s", fallback)
    return fallback


class Settings:
    """Typed application settings, loaded from environment variables."""

    def __init__(self) -> None:
        self.data_dir = resolve_data_dir()
        self.index_dir = Path(os.getenv("PROCUREBUDDY_INDEX_DIR", str(self.data_dir / ".index")))
        self.embedding_model = os.getenv("PROCUREBUDDY_EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
        self.chunk_size = int(os.getenv("PROCUREBUDDY_CHUNK_SIZE", "1000"))
        self.chunk_overlap = int(os.getenv("PROCUREBUDDY_CHUNK_OVERLAP", "200"))
        self.top_k = min(8, max(3, int(os.getenv("PROCUREBUDDY_TOP_K", "5"))))
        self.min_score = float(os.getenv("PROCUREBUDDY_MIN_SCORE", "0.10"))
        self.answer_cache_size = max(100, int(os.getenv("PROCUREBUDDY_ANSWER_CACHE_SIZE", "500")))
        self.answer_cache_ttl_seconds = max(60, int(os.getenv("PROCUREBUDDY_ANSWER_CACHE_TTL_SECONDS", "3600")))
        self.llm_api_key = os.getenv("GROQ_API_KEY")
        if not self.llm_api_key:
            raise RuntimeError("GROQ_API_KEY is not loaded. Check .env file path.")
        configured_model = (os.getenv("GROQ_MODEL") or "llama-3.1-8b-instant").strip()
        if configured_model in ("llama3-8b-8192", "llama-3.3-70b-versatile"):
            configured_model = "llama-3.1-8b-instant"
        self.llm_model = configured_model

    def refresh_llm_settings(self) -> None:
        """Refresh LLM-related settings from the latest environment values."""

        reload_runtime_env()
        self.llm_api_key = os.getenv("GROQ_API_KEY")
        if not self.llm_api_key:
            raise RuntimeError("GROQ_API_KEY is not loaded. Check .env file path.")
        configured_model = (os.getenv("GROQ_MODEL") or "llama-3.1-8b-instant").strip()
        if configured_model in ("llama3-8b-8192", "llama-3.3-70b-versatile"):
            configured_model = "llama-3.1-8b-instant"
        self.llm_model = configured_model


# ── Singleton ───────────────────────────────────────────────────────────────
settings = Settings()
os.makedirs(settings.index_dir, exist_ok=True)
