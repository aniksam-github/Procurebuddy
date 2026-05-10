"""Environment configuration and application settings.

Loads environment variables from .env files and provides a typed Settings class.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger("procurebuddy-ai")

DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"
DISABLED_GROQ_MODELS = {
    "gpt-oss-120b",
    "openai/gpt-oss-120b",
    "gpt-oss-20b",
    "openai/gpt-oss-20b",
    "llama-3.3-70b-versatile",
    "llama3-8b-8192",
}

# ── Load environment ────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # python-ai-service/
ROOT_DIR = BASE_DIR.parent  # project root

load_dotenv(BASE_DIR / ".env", override=True)
load_dotenv(ROOT_DIR / ".env", override=True)

# ── Mask and validate API key ───────────────────────────────────────────────
_keys = [os.getenv(f"GROQ_API_KEY_{i}") for i in range(1, 20) if os.getenv(f"GROQ_API_KEY_{i}")]
if os.getenv("GROQ_API_KEY") and os.getenv("GROQ_API_KEY") not in _keys:
    _keys.insert(0, os.getenv("GROQ_API_KEY"))

print("ENV BASE:", BASE_DIR)
print("ENV ROOT:", ROOT_DIR)
print("GROQ_API_KEYS:", f"{len(_keys)} keys loaded" if _keys else "<NOT SET>")


def reload_runtime_env() -> None:
    """Reload .env files so runtime secret changes can be picked up safely."""

    load_dotenv(BASE_DIR / ".env", override=True)
    load_dotenv(ROOT_DIR / ".env", override=True)


def normalize_groq_model(configured_model: str | None) -> str:
    """Map disabled or risky Groq model selections to the stable default."""

    model_name = (configured_model or DEFAULT_GROQ_MODEL).strip()
    if model_name in DISABLED_GROQ_MODELS:
        logger.warning(
            "Configured Groq model '%s' is disabled; using '%s' instead",
            model_name,
            DEFAULT_GROQ_MODEL,
        )
        return DEFAULT_GROQ_MODEL
    return model_name


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
        self.llm_api_keys: list[str] = []
        main_key = os.getenv("GROQ_API_KEY")
        if main_key:
            self.llm_api_keys.append(main_key)
            
        for i in range(1, 20):
            key = os.getenv(f"GROQ_API_KEY_{i}")
            if key and key not in self.llm_api_keys:
                self.llm_api_keys.append(key)

        if not self.llm_api_keys:
            logger.warning("No GROQ_API_KEY or GROQ_API_KEY_1..19 found. LLM features will fall back to rule-based responses.")
            self.llm_api_key = ""
        else:
            self.llm_api_key = self.llm_api_keys[0]
        self.llm_model = normalize_groq_model(os.getenv("GROQ_MODEL"))

    def refresh_llm_settings(self) -> None:
        """Refresh LLM-related settings from the latest environment values."""

        reload_runtime_env()
        self.llm_api_keys = []
        main_key = os.getenv("GROQ_API_KEY")
        if main_key:
            self.llm_api_keys.append(main_key)
            
        for i in range(1, 20):
            key = os.getenv(f"GROQ_API_KEY_{i}")
            if key and key not in self.llm_api_keys:
                self.llm_api_keys.append(key)

        if not self.llm_api_keys:
            logger.warning("No GROQ_API_KEY or GROQ_API_KEY_1..19 found after refresh. LLM features will stay in rule-based mode.")
            self.llm_api_key = ""
        else:
            self.llm_api_key = self.llm_api_keys[0]
        self.llm_model = normalize_groq_model(os.getenv("GROQ_MODEL"))


# ── Singleton ───────────────────────────────────────────────────────────────
settings = Settings()
os.makedirs(settings.index_dir, exist_ok=True)
