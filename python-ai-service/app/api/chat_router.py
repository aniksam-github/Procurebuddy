"""FastAPI route handlers for /chat, /search, /reload, and /health.

All endpoints maintain the same paths as the original monolith so the
Spring Boot PythonAiService.java requires zero changes.
"""

from __future__ import annotations

import logging
from threading import RLock
from typing import Any

from cachetools import TTLCache
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from app.core.config import settings
from app.core.rag_engine import build_cache_key, build_no_match_response, response_hash, retrieve_candidates
from app.services.knowledge_base import SearchMatch, SearchResponse, knowledge_base
from app.services.response_service import generate_response, inject_decision
from app.utils.processors import detect_intent, extract_username

logger = logging.getLogger("procurebuddy-ai")

router = APIRouter()

answer_cache: TTLCache = TTLCache(
    maxsize=settings.answer_cache_size,
    ttl=settings.answer_cache_ttl_seconds,
)
answer_cache_lock = RLock()


class ChatRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    message: str = Field(min_length=1)
    user: str = Field(min_length=1)
    display_name: str | None = Field(None, alias="displayName")
    username: str | None = None
    email: str | None = None
    bypass_cache: bool = Field(False, alias="bypass_cache")
    blocked_chunk_ids: list[int] = Field(default_factory=list, alias="blocked_chunk_ids")
    blocked_response_hashes: list[str] = Field(default_factory=list, alias="blocked_response_hashes")


class ChatResponse(BaseModel):
    answer: str
    generation_mode: str
    response: str
    response_id: str
    response_hash: str
    source_chunk_ids: list[int] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)


def build_chat_response(
    response_text: str,
    matches: list[SearchMatch],
    generation_mode: str = "rule_based",
    metadata: dict[str, Any] | None = None,
) -> ChatResponse:
    """Build a backward-compatible chat payload with explicit generation mode."""

    answer = response_text.strip() or build_no_match_response("the request")
    answer_hash = response_hash(answer)
    payload_metadata = dict(metadata or {})
    payload_metadata["generation_mode"] = generation_mode

    return ChatResponse(
        answer=answer,
        generation_mode=generation_mode,
        response=answer,
        response_id=f"resp_{answer_hash[:16]}",
        response_hash=answer_hash,
        source_chunk_ids=[match.chunk_id for match in matches[:settings.top_k]],
        metadata=payload_metadata,
    )


@router.get("/health")
def health() -> dict[str, Any]:
    return knowledge_base.status()


@router.post("/reload")
def reload_index() -> dict[str, Any]:
    result = knowledge_base.reload()
    with answer_cache_lock:
        answer_cache.clear()
    logger.info("Cleared answer cache after knowledge base reload")
    return result


@router.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    matches, _ = retrieve_candidates(request.query, blocked_chunk_ids=[], top_k=settings.top_k)
    return SearchResponse(matches=matches, count=len(matches))


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    message = request.message.strip()
    user = request.user.strip()
    if not message:
        raise HTTPException(status_code=400, detail="message is required")
    if not user:
        raise HTTPException(status_code=400, detail="user is required")

    logger.info("Chat request user='%s' message='%s'", user, message[:120])

    try:
        intent = detect_intent(message)
        if intent == "GREETING":
            resolved_name = extract_username({
                "displayName": request.display_name,
                "username": request.username,
                "email": request.email or user,
            })
            greeting_text = (
                f"Namaste {resolved_name}! \U0001f60a\n"
                "How can I help you with procurement today?"
            )
            logger.info("Greeting detected for user='%s' resolved_name='%s' - skipping RAG pipeline", user, resolved_name)
            return build_chat_response(
                greeting_text,
                [],
                generation_mode="rule_based",
                metadata={"generation_mode": "rule_based", "shortcut": "greeting"},
            )

        result = generate_response(
            query=message,
            user=user,
            bypass_cache=request.bypass_cache,
            blocked_chunk_ids=request.blocked_chunk_ids,
            blocked_response_hashes=request.blocked_response_hashes,
        )
        matches = result.get("documents", [])
        payload = build_chat_response(
            str(result.get("answer", "")),
            matches,
            generation_mode=str(result.get("generation_mode", "rule_based")),
            metadata=result.get("metadata", {}),
        )
        with answer_cache_lock:
            answer_cache[build_cache_key(message)] = payload.model_dump()
        return payload

    except HTTPException:
        raise
    except Exception:
        logger.exception("Unhandled error in /chat for user='%s' message='%s'", user, message[:120])
        error_response = (
            "I encountered an internal processing error while generating your answer. "
            "Please try again in a moment, or rephrase your question."
        )
        error_response, _ = inject_decision(error_response)
        return build_chat_response(
            error_response,
            [],
            generation_mode="rule_based",
            metadata={"generation_mode": "rule_based", "router_fallback": True},
        )
