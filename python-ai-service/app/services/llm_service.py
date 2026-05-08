"""Groq LLM client with cascading model fallback and rate-limit handling.

When a model hits a rate limit, the service automatically switches to the
next model in the chain instead of returning None. Only returns None when
ALL models are exhausted, allowing higher layers to fall back to grounded
rule-based responses.

Model cascade order:
  1. llama-3.3-70b-versatile  (primary – best quality)
  2. gpt-oss-120b             (fallback 1)
  3. gpt-oss-20b              (fallback 2)
  4. llama-3.1-8b-instant     (fallback 3 – fastest, lightest)
"""

from __future__ import annotations

import logging
import os
import time

from groq import APIConnectionError, APITimeoutError, BadRequestError, Groq, RateLimitError

from app.core.config import settings
from app.core.constants import SYSTEM_PROMPT

logger = logging.getLogger("procurebuddy-ai")

_client: Groq | None = None
_client_api_key = ""

# ── Model cascade ──────────────────────────────────────────────────────
# Order matters: first model is tried first; on rate-limit exhaustion the
# next model in the list is attempted automatically.
MODELS: tuple[str, ...] = (
    "llama-3.3-70b-versatile",
    "gpt-oss-120b",
    "gpt-oss-20b",
    "llama-3.1-8b-instant",
)

# ── Tuning knobs ───────────────────────────────────────────────────────
RATE_LIMIT_MAX_RETRIES = int(os.getenv("PROCUREBUDDY_LLM_RATE_LIMIT_RETRIES", "2"))
RATE_LIMIT_BACKOFF_BASE_SECONDS = float(os.getenv("PROCUREBUDDY_LLM_RATE_LIMIT_BACKOFF_BASE_SECONDS", "8"))
RATE_LIMIT_BACKOFF_CAP_SECONDS = float(os.getenv("PROCUREBUDDY_LLM_RATE_LIMIT_BACKOFF_CAP_SECONDS", "20"))
LLM_TEMPERATURE = float(os.getenv("PROCUREBUDDY_LLM_TEMPERATURE", "0.1"))
LLM_MAX_TOKENS = int(os.getenv("PROCUREBUDDY_LLM_MAX_TOKENS", "1800"))
LLM_TIMEOUT_SECONDS = float(os.getenv("PROCUREBUDDY_LLM_TIMEOUT_SECONDS", "60"))
_rate_limit_cooldown_until = 0.0


def _candidate_models() -> list[str]:
    """Build the ordered model list: configured model first, then MODELS cascade."""

    configured = (settings.llm_model or "").strip()
    candidates: list[str] = []

    # If a model is explicitly configured, put it at the front.
    if configured and configured not in MODELS:
        candidates.append(configured)

    # Add all cascade models in order, avoiding duplicates.
    for model in MODELS:
        if model not in candidates:
            candidates.append(model)

    return candidates


def _get_client() -> Groq:
    """Return a Groq client bound to the latest configured API key."""

    global _client, _client_api_key

    settings.refresh_llm_settings()
    current_api_key = settings.llm_api_key
    if _client is None or _client_api_key != current_api_key:
        _client = Groq(api_key=current_api_key)
        _client_api_key = current_api_key
        logger.info("Rebound Groq client using the latest configured API key")
    return _client


def _rate_limit_backoff_seconds(attempt: int) -> float:
    """Return a bounded exponential-style backoff duration."""

    return min(RATE_LIMIT_BACKOFF_CAP_SECONDS, RATE_LIMIT_BACKOFF_BASE_SECONDS + (attempt * 4.0))


def _wait_for_shared_cooldown() -> None:
    """Honor any recent upstream cooldown discovered by another request."""

    remaining = _rate_limit_cooldown_until - time.monotonic()
    if remaining > 0:
        logger.warning("Honoring shared Groq cooldown for %.1fs before next request", remaining)
        time.sleep(remaining)


def generate_llm_response(
    prompt: str,
    system_prompt: str = SYSTEM_PROMPT,
) -> str | None:
    """Call the Groq LLM with cascading model fallback.

    Tries each model in the cascade order. For each model, retries up to
    RATE_LIMIT_MAX_RETRIES times on rate-limit errors with exponential
    backoff. If all retries for a model are exhausted, moves to the next
    model. Returns None only when ALL models have been exhausted.
    """

    global _rate_limit_cooldown_until

    candidates = _candidate_models()
    total_models = len(candidates)

    try:
        last_bad_request: BadRequestError | None = None

        for model_idx, model_name in enumerate(candidates, 1):
            logger.info(
                "Trying LLM model='%s' (%d/%d) prompt_length=%s",
                model_name, model_idx, total_models, len(prompt),
            )

            for attempt in range(RATE_LIMIT_MAX_RETRIES + 1):
                try:
                    _wait_for_shared_cooldown()
                    response = _get_client().chat.completions.create(
                        model=model_name,
                        temperature=LLM_TEMPERATURE,
                        max_tokens=LLM_MAX_TOKENS,
                        timeout=LLM_TIMEOUT_SECONDS,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt},
                        ],
                    )
                    content = (response.choices[0].message.content or "").strip()
                    if not content:
                        logger.info("LLM returned empty response from model='%s'", model_name)
                        return None
                    logger.info(
                        "LLM success model='%s' content_length=%s",
                        model_name, len(content),
                    )
                    return content

                except RateLimitError as exc:
                    backoff_seconds = _rate_limit_backoff_seconds(attempt)
                    _rate_limit_cooldown_until = time.monotonic() + backoff_seconds
                    logger.warning(
                        "Rate limit hit model='%s' attempt=%d/%d; backing off %.1fs",
                        model_name,
                        attempt + 1,
                        RATE_LIMIT_MAX_RETRIES + 1,
                        backoff_seconds,
                    )
                    if attempt >= RATE_LIMIT_MAX_RETRIES:
                        # All retries exhausted for this model → switch to next
                        remaining_models = total_models - model_idx
                        if remaining_models > 0:
                            logger.warning(
                                "Rate limit persisted for model='%s'; "
                                "switching to next model (%d remaining)",
                                model_name, remaining_models,
                            )
                        else:
                            logger.error(
                                "Rate limit persisted for model='%s' and no more "
                                "fallback models available. Returning None.",
                                model_name,
                            )
                        break  # break retry loop → continue to next model
                    time.sleep(backoff_seconds)

                except BadRequestError as exc:
                    last_bad_request = exc
                    error_payload = getattr(exc, "body", {}) or {}
                    error_code = str((error_payload.get("error") or {}).get("code", "")).strip().lower()
                    error_message = str((error_payload.get("error") or {}).get("message", "")).strip().lower()
                    if error_code == "model_decommissioned" or "decommissioned" in error_message:
                        logger.warning(
                            "Model '%s' decommissioned; switching to next model",
                            model_name,
                        )
                        break  # try next model
                    if "model_not_found" in error_code or "does not exist" in error_message:
                        logger.warning(
                            "Model '%s' not found; switching to next model",
                            model_name,
                        )
                        break  # try next model
                    raise

                except (APITimeoutError, APIConnectionError) as exc:
                    # Network/timeout errors: try next model instead of giving up
                    logger.warning(
                        "Model '%s' network error (%s): %s; switching to next model",
                        model_name, type(exc).__name__, exc,
                    )
                    break  # try next model

        # All models exhausted
        if last_bad_request is not None:
            raise last_bad_request
        logger.error("All %d models exhausted. Returning None for rule-based fallback.", total_models)
        return None

    except Exception:
        logger.exception("LLM generation failed; returning None for fallback")
        return None
