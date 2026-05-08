"""ProcureBuddy AI Service — Backward-Compatible Entry Point.

This thin wrapper re-exports the FastAPI `app` from the modular package
so that existing `uvicorn main:app` commands still work without changes.

The original monolith has been preserved as `main_legacy.py` for reference.
"""

from app.main import app  # noqa: F401

__all__ = ["app"]
