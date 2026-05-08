"""ProcureBuddy AI Service — Application Entry Point.

Creates the FastAPI app instance, registers routes, and runs
the knowledge base initialization on startup.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI

from app.api.chat_router import router
from app.core.config import settings
from app.services.knowledge_base import knowledge_base

logger = logging.getLogger("procurebuddy-ai")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ── FastAPI App ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="ProcureBuddy AI Service",
    description="Expert RAG pipeline for CSIR procurement policy queries",
    version="2.0.0",
)

# Register all routes at the root level (no prefix)
# This preserves backward compatibility: /chat, /search, /reload, /health
app.include_router(router)


@app.on_event("startup")
def startup() -> None:
    """Initialize the knowledge base and log configuration."""
    logger.info("Initializing knowledge base from %s", settings.data_dir)
    logger.info(
        "Config: model=%s top_k=%d min_score=%.2f cache_size=%d cache_ttl=%ds",
        settings.llm_model,
        settings.top_k,
        settings.min_score,
        settings.answer_cache_size,
        settings.answer_cache_ttl_seconds,
    )
    logger.info("LLM ENABLED: %s", bool(settings.llm_api_key))
    knowledge_base.initialize()
    logger.info("✅ ProcureBuddy AI Service v2.0 — modular architecture ready")
