"""High-level response generation service.

This service is the safe entry point for batch workloads. It always returns a
valid answer payload and never lets LLM transport/rate-limit failures bubble
up to the API client.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from app.core.orchestrator import run_planned_orchestration
from app.core.rag_engine import build_no_match_response, normalize_response_hashes
from app.services.knowledge_base import SearchMatch

logger = logging.getLogger("procurebuddy-ai")

VALID_DECISIONS = ("APPROVE", "REJECT", "MODIFY", "VERIFY")


def extract_decision(text: str) -> str | None:
    """Return a valid FINAL DECISION value if one is already present."""

    match = re.search(r"FINAL DECISION:\s*(.+)", text, flags=re.IGNORECASE)
    if not match:
        return None
    decision_text = match.group(1).strip().upper()
    for decision in VALID_DECISIONS:
        if decision in decision_text:
            return decision
    return None


def infer_decision_from_answer(text: str) -> str:
    """Infer a safe audit decision from the current answer text."""

    lowered = text.lower()
    if "non-compliant" in lowered or "cannot proceed" in lowered or "must be rejected" in lowered:
        return "REJECT"
    if "conditional" in lowered or "subject to" in lowered or "revise" in lowered or "modify" in lowered:
        return "MODIFY"
    if "verify" in lowered or "check" in lowered or "confirm" in lowered or "examine" in lowered:
        return "VERIFY"
    if "compliant" in lowered:
        return "APPROVE"
    return "VERIFY"


def inject_decision(answer: str) -> tuple[str, bool]:
    """Append FINAL DECISION when the answer does not already include one."""

    cleaned = str(answer or "").strip()
    if not cleaned:
        return cleaned, False
    existing = extract_decision(cleaned)
    if existing:
        return cleaned, False
    inferred = infer_decision_from_answer(cleaned)
    return f"{cleaned}\nFINAL DECISION: {inferred}", True


def generate_response(
    query: str,
    user: str = "anonymous",
    bypass_cache: bool = False,
    blocked_chunk_ids: list[int] | None = None,
    blocked_response_hashes: list[str] | set[str] | None = None,
) -> dict[str, Any]:
    """Generate a chat answer and always return a valid payload.

    Response format:
        {
            "answer": "...",
            "generation_mode": "llm" | "rule_based",
        }
    """

    try:
        from app.core.engine_v2 import run_v2_flow
        graph_state = run_v2_flow(query)

        raw_answer = str(graph_state.get("generation", "")).strip() or build_no_match_response(query)
        logger.info("decision_injector query='%s' before=%s", query[:120], raw_answer[-120:] if raw_answer else "<empty>")
        answer, decision_injected = inject_decision(raw_answer)
        logger.info("decision_injector query='%s' injected=%s after=%s", query[:120], decision_injected, answer[-120:] if answer else "<empty>")
        matches = graph_state.get("documents", [])
        metadata = dict(graph_state.get("metadata", {}))
        generation_mode = str(metadata.get("generation_mode", "rule_based")).strip().lower()
        if generation_mode not in {"llm", "llm-retry", "rule_based"}:
            generation_mode = "rule_based"

        metadata["generation_mode"] = generation_mode
        metadata["decision_injected"] = decision_injected
        logger.info("generation_mode=%s query='%s'", generation_mode, query[:120])

        return {
            "answer": answer,
            "generation_mode": generation_mode,
            "documents": matches,
            "metadata": metadata,
        }
    except Exception:
        logger.exception("Response service failed; returning safe rule-based fallback")
        raw_fallback = build_no_match_response(query)
        logger.info("decision_injector_fallback query='%s' before=%s", query[:120], raw_fallback[-120:] if raw_fallback else "<empty>")
        fallback, decision_injected = inject_decision(raw_fallback)
        logger.info("decision_injector_fallback query='%s' injected=%s after=%s", query[:120], decision_injected, fallback[-120:] if fallback else "<empty>")
        logger.info("generation_mode=rule_based query='%s'", query[:120])
        return {
            "answer": fallback,
            "generation_mode": "rule_based",
            "documents": [],
            "metadata": {
                "generation_mode": "rule_based",
                "service_fallback": True,
                "decision_injected": decision_injected,
            },
        }
