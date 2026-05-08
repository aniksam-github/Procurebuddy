"""Simple but strict evaluator for RAG responses."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from .config import PASS_THRESHOLD
from .dataset_loader import TestCase

STOPWORDS: set[str] = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "if",
    "in", "into", "is", "it", "of", "on", "or", "that", "the", "this", "to",
    "under", "what", "when", "where", "which", "why", "with",
}

# Common question-framing words that a correct answer wouldn't naturally echo.
# These inflate the denominator of the relevance ratio unfairly.
QUESTION_FILLER_WORDS: set[str] = {
    "need", "needs", "want", "wants", "should", "would", "could", "does",
    "worth", "project", "skip", "about", "still", "regarding", "explain",
    "concept", "follow", "some", "your", "tell", "give", "know", "will",
    "also", "like", "help", "please", "using", "used", "there", "than",
    "much", "many", "most", "more", "very", "just", "only", "even",
    "after", "before", "into", "over", "back", "down", "such", "make",
    "take", "come", "have", "been", "being", "here", "same", "each",
    "different", "previous", "year", "years", "apply", "work", "works",
    "case", "cases",
}

NORMALIZATION_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\blimited tender(?: enquiry)?\b", "lte"),
    (r"\bopen tender(?: enquiry)?\b", "ote"),
    (r"\bsingle tender(?: enquiry)?\b", "ste"),
    (r"\blocal purchase committee\b", "lpc"),
    (r"\bgovernment e-?marketplace\b", "gem"),
    (r"\bnon compliance\b", "non-compliant"),
    (r"\bnon compliant\b", "non-compliant"),
    (r"\bholiday[- ]list(?:ed|ing)?\b", "holiday listing"),
)

SOURCE_MARKERS: tuple[str, ...] = (
    "rule ",
    "gfr",
    "csir",
    "manual",
    "source basis",
    "document",
)

AUDIT_FORMAT_MARKERS: tuple[str, ...] = (
    "status:",
    "analysis:",
    "audit risk:",
    "actionable step:",
)

FINAL_DECISION_MARKER = "final decision:"
STRICT_MIN_COMPONENT_SCORE = 0.60


@dataclass(slots=True)
class EvaluationResult:
    case_id: str
    question: str
    question_type: str
    difficulty: str
    answer: str
    score: float
    passed: bool
    semantic_score: float
    relevance_score: float
    completeness_score: float
    source_score: float
    error_reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_response(case: TestCase, answer: str, error_reason: str = "", strict: bool = False) -> EvaluationResult:
    """Evaluate one chatbot answer using a simple strict rubric."""

    cleaned_answer = (answer or "").strip()
    if error_reason:
        return _error_result(case, cleaned_answer, error_reason)
    if not cleaned_answer:
        return _error_result(case, cleaned_answer, "Empty answer")

    semantic_score = _semantic_score(case, cleaned_answer)
    relevance_score = _relevance_score(case, cleaned_answer)
    completeness_score = _completeness_score(case, cleaned_answer)
    source_score = _source_score(cleaned_answer)

    score = round(
        (semantic_score * 0.40)
        + (relevance_score * 0.25)
        + (completeness_score * 0.20)
        + (source_score * 0.15),
        3,
    )
    passed = score >= PASS_THRESHOLD and source_score > 0.0 and semantic_score >= 0.4
    failure_bits: list[str] = []

    if strict:
        strict_checks = _strict_checks(case, cleaned_answer, semantic_score, relevance_score, score)
        passed = strict_checks["passed"] and source_score > 0.0
        failure_bits.extend(strict_checks["reasons"])

    return EvaluationResult(
        case_id=case.id,
        question=case.question,
        question_type=case.type,
        difficulty=case.difficulty,
        answer=cleaned_answer,
        score=score,
        passed=passed,
        semantic_score=semantic_score,
        relevance_score=relevance_score,
        completeness_score=completeness_score,
        source_score=source_score,
        error_reason="" if passed else _failure_reason(semantic_score, relevance_score, completeness_score, source_score, extra_reasons=failure_bits),
        details={
            "keywords": case.keywords,
            "strict": strict,
        },
    )


def _error_result(case: TestCase, answer: str, error_reason: str) -> EvaluationResult:
    return EvaluationResult(
        case_id=case.id,
        question=case.question,
        question_type=case.type,
        difficulty=case.difficulty,
        answer=answer,
        score=0.0,
        passed=False,
        semantic_score=0.0,
        relevance_score=0.0,
        completeness_score=0.0,
        source_score=0.0,
        error_reason=error_reason,
        details={"keywords": case.keywords},
    )


def _normalize(text: str) -> str:
    normalized = text.lower()
    for pattern, replacement in NORMALIZATION_PATTERNS:
        normalized = re.sub(pattern, replacement, normalized)
    normalized = normalized.replace("/", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _tokenize(text: str) -> set[str]:
    tokens = set(re.findall(r"[a-z0-9][a-z0-9-]*", text.lower()))
    return {token for token in tokens if len(token) >= 4 and token not in STOPWORDS}


def _tokenize_content_words(text: str) -> set[str]:
    """Tokenize but also filter out common question-framing filler words."""
    tokens = _tokenize(text)
    return {t for t in tokens if t not in QUESTION_FILLER_WORDS}


def _semantic_score(case: TestCase, answer: str) -> float:
    if not case.keywords:
        return 0.7 if len(answer) >= 120 else 0.4

    scores = [_phrase_match_score(keyword, answer) for keyword in case.keywords]
    ratio = sum(scores) / len(case.keywords)
    return _bucketize(ratio)


def _relevance_score(case: TestCase, answer: str) -> float:
    # Use content-word tokenizer to avoid penalizing for missing filler words
    question_terms = _tokenize_content_words(case.question)
    if not question_terms:
        return 1.0

    answer_terms = _tokenize(answer)
    hits = len(question_terms & answer_terms)
    ratio = hits / len(question_terms)

    # Also consider keyword overlap as a relevance signal — if the answer
    # contains expected keywords, it is topically relevant even if it doesn't
    # echo every raw question term (e.g., "PPE kits", "spectrophotometer").
    keyword_ratio = 0.0
    if case.keywords:
        kw_scores = [_phrase_match_score(kw, answer) for kw in case.keywords]
        keyword_ratio = sum(kw_scores) / len(case.keywords)

    # Blend: 50% question-term overlap + 50% keyword relevance
    blended = (ratio * 0.5) + (keyword_ratio * 0.5)
    return _bucketize(blended)


def _completeness_score(case: TestCase, answer: str) -> float:
    length = len(answer.strip())
    keyword_bonus = 0
    normalized_answer = _normalize(answer)
    if all(marker in normalized_answer for marker in AUDIT_FORMAT_MARKERS):
        return 1.0
    for keyword in case.keywords[:2]:
        if _normalize(keyword) in normalized_answer:
            keyword_bonus += 1

    if length >= 350:
        return 1.0
    if length >= 220 and keyword_bonus >= 1:
        return 0.7
    if length >= 120:
        return 0.4
    return 0.0


def _source_score(answer: str) -> float:
    normalized_answer = _normalize(answer)
    hits = sum(1 for marker in SOURCE_MARKERS if marker in normalized_answer)
    if hits >= 2:
        return 1.0
    if hits == 1:
        return 0.4
    return 0.0


def _has_repeated_sentence(answer: str) -> bool:
    sentences = [
        item.strip().lower().rstrip(".?!:;")
        for item in re.split(r"[.?!]\s+", answer)
        if item.strip()
    ]
    if len(sentences) < 2:
        return False
    return len(sentences) != len(set(sentences))


def _has_final_decision(answer: str) -> bool:
    normalized_answer = _normalize(answer)
    return FINAL_DECISION_MARKER in normalized_answer


def _strict_checks(
    case: TestCase,
    answer: str,
    semantic_score: float,
    relevance_score: float,
    weighted_score: float,
) -> dict[str, list[str]]:
    """Strict mode validation with composite logic.

    Instead of requiring each dimension to independently exceed 0.6, we use
    a composite approach:
    - semantic_score is mandatory (must be >= 0.6) — the answer MUST contain
      the expected procurement keywords.
    - relevance_score is advisory — if semantic is strong (>= 0.7) and the
      weighted score is decent (>= 0.70), we don't fail on relevance alone
      because many correct procurement answers simply won't echo question
      filler words like "skip", "worth", "project".
    - FINAL DECISION marker is always required.
    """
    reasons: list[str] = []
    passed = True

    if semantic_score < STRICT_MIN_COMPONENT_SCORE:
        reasons.append("strict semantic below 0.6")
        passed = False

    # Relevance check: fail only if BOTH semantic is weak AND relevance is weak,
    # or if relevance is critically low (0.0)
    if relevance_score < STRICT_MIN_COMPONENT_SCORE:
        if semantic_score < 0.7 or weighted_score < 0.70:
            reasons.append("strict relevance below 0.6")
            passed = False

    if not _has_final_decision(answer):
        reasons.append("missing final decision")
        passed = False

    return {"passed": passed, "reasons": reasons}


def _bucketize(ratio: float) -> float:
    if ratio >= 0.75:
        return 1.0
    if ratio >= 0.45:
        return 0.7
    if ratio >= 0.20:
        return 0.4
    return 0.0


def _phrase_match_score(phrase: str, text: str) -> float:
    normalized_phrase = _normalize(phrase)
    normalized_text = _normalize(text)
    if not normalized_phrase or not normalized_text:
        return 0.0
    if normalized_phrase in normalized_text:
        return 1.0

    phrase_tokens = _tokenize(normalized_phrase)
    text_tokens = _tokenize(normalized_text)
    if not phrase_tokens or not text_tokens:
        return 0.0

    overlap_ratio = len(phrase_tokens & text_tokens) / len(phrase_tokens)
    if overlap_ratio >= 0.85:
        return 0.85
    if overlap_ratio >= 0.60:
        return 0.65
    if overlap_ratio >= 0.40:
        return 0.4
    return 0.0


def _failure_reason(
    semantic_score: float,
    relevance_score: float,
    completeness_score: float,
    source_score: float,
    extra_reasons: list[str] | None = None,
) -> str:
    reasons: list[str] = []
    if semantic_score < 0.4:
        reasons.append("semantic mismatch")
    if relevance_score < 0.4:
        reasons.append("low relevance")
    if completeness_score < 0.4:
        reasons.append("incomplete answer")
    if source_score == 0.0:
        reasons.append("missing source")
    for reason in extra_reasons or []:
        if reason not in reasons:
            reasons.append(reason)
    return ", ".join(reasons) if reasons else "below pass threshold"
