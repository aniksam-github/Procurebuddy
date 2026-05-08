"""Intent detection, amount extraction, keyword expansion, and query processing.

Handles the classification of user queries into GREETING/WORKFLOW/PROCESS/
ANALYTICAL/GENERAL and provides query-enhancement utilities for the RAG pipeline.
"""

from __future__ import annotations

import re
from typing import Any

from app.core.constants import (
    gfr_keywords_for_amount,
    gfr_slab_for_amount,
    GREETING_WORDS,
    PROCUREMENT_SIGNALS,
)
from app.utils.text_cleaner import clean_text

_ANALYTICAL_TERMS = (
    "compare",
    "comparison",
    "difference",
    "different",
    "versus",
    "vs",
    "table",
    "matrix",
    "why",
    "distinguish",
    "distinction",
    "contrast",
    "conflict",
    "rather than",
    "not just",
    "conceptual",
    "foundational",
    "benchmarking",
    "justification",
)

_ANALYTICAL_METHOD_ALIASES: dict[str, tuple[str, ...]] = {
    "LTE": ("lte", "limited tender", "limited tender enquiry"),
    "STE": ("ste", "single tender", "single tender enquiry", "single source", "proprietary"),
    "OTE": ("ote", "open tender", "open tender enquiry"),
    "LPC": ("lpc", "local purchase committee", "local purchase"),
    "DIRECT PURCHASE": ("direct purchase", "without quotation", "direct procurement"),
}

_WORKFLOW_SYNONYMS = (
    "SOP",
    "standard operating procedure",
    "steps",
    "approval flow",
    "approval chain",
    "sequence",
    "indenting",
    "T&PC",
    "TSC",
    "technical specification committee",
    "purchase committee",
)

_THRESHOLD_SYNONYMS = (
    "value band",
    "slab",
    "limit",
    "procurement method",
    "rule 154",
    "rule 155",
    "rule 161",
    "rule 162",
    "rule 166",
)

_PROCESS_KEYWORDS = (
    "process",
    "procedure",
    "steps",
    "workflow",
    "scrutiny",
    "evaluation",
    "recording",
    "approval",
    "committee",
    "bid opening",
    "responsiveness",
    "rejection",
    "documentation",
)

_ANALYTICAL_BROAD_TERMS = (
    "definition",
    "applicability",
    "threshold",
    "comparison",
    "approval steps",
)

_SCENARIO_TERMS = (
    "scenario",
    "suppose",
    "assume",
    "if ",
    "when ",
    "a note says",
    "buyer still wants",
    "lab proposes",
    "in this case",
    "what should",
    "must be answered first",
)


def _scenario_focus_terms(query: str) -> tuple[str, ...]:
    """Return retrieval hints for recurring scenario and edge-case families."""

    cleaned = clean_text(query).lower()
    terms: list[str] = []

    if any(token in cleaned for token in ("holiday", "debar", "blacklist")):
        terms.extend(
            [
                "holiday listing vendor verification bid eligibility before award",
                "debarment status bidder participation award restriction",
            ]
        )
    if any(token in cleaned for token in ("single remaining responsive bid", "single responsive bid", "one-bid", "single bid", "only bidder", "only one bidder")):
        terms.extend(
            [
                "single responsive bid technical rejection price reasonableness speaking order",
                "single offer after technical evaluation acceptance justification",
            ]
        )
    if any(token in cleaned for token in ("pac", "single-source", "single source", "proprietary", "confidentiality")):
        terms.extend(
            [
                "PAC proprietary justification confidentiality standardisation single tender",
                "single source procurement proprietary certificate approved exception",
            ]
        )
    if any(token in cleaned for token in ("make in india", "local supplier", "local content", "purchase preference")):
        terms.extend(
            [
                "Make in India local supplier preference bid evaluation declaration local content",
                "class i class ii supplier purchase preference verification",
            ]
        )
    if any(token in cleaned for token in ("approver", "head of office", "director", "competent authority", "technical recommender")):
        terms.extend(
            [
                "approval authority oversight sanction responsibility committee recommendation",
                "head of office director approver accountability procurement decision",
            ]
        )
    if "unauthorized channel" in cleaned or any(token in cleaned for token in ("authorized channel", "authorised channel", "authorized source", "authorised source", "oem", "reseller")):
        terms.extend(
            [
                "authorized source OEM reseller legitimacy warranty support admissibility",
                "source legitimacy channel authorization supplier eligibility",
            ]
        )
    if "gem" in cleaned:
        terms.extend(
            [
                "GeM platform suitability exact functional requirement departure justification",
                "GeM authorized reseller OEM listing exact specification",
            ]
        )
    if any(token in cleaned for token in ("gst", "tender fee", "document fee", "registration fee")):
        terms.extend(
            [
                "GST treatment tender document fee vendor registration fee accounting clarification",
                "tax treatment procurement fees disclosure tender conditions",
            ]
        )
    if any(token in cleaned for token in ("value for money", "value-for-money", "reasonableness", "restrictive specification")):
        terms.extend(
            [
                "value for money audit competition price reasonableness specification neutrality",
                "comparative justification market reasonableness audit defensibility",
            ]
        )

    deduped: list[str] = []
    seen: set[str] = set()
    for term in terms:
        if term not in seen:
            seen.add(term)
            deduped.append(term)
    return tuple(deduped)

def detect_intent(message: str) -> str:
    """Classify the user message into GREETING, SCENARIO, WORKFLOW, ANALYTICAL, PROCESS, or GENERAL."""
    lowered = message.strip().lower()
    cleaned = clean_text(message).lower()

    if any(re.search(rf"\b{re.escape(word.lower())}\b", lowered) for word in GREETING_WORDS):
        has_procurement_signal = any(sig in lowered for sig in PROCUREMENT_SIGNALS)
        word_count = len(re.findall(r"[a-z0-9]+", cleaned))
        if not has_procurement_signal and word_count <= 6:
            return "GREETING"

    if any(term in cleaned for term in _ANALYTICAL_TERMS):
        return "ANALYTICAL"

    # ── Scenario Detection (BEFORE amount-based PROCESS) ──
    # Scenario questions contain an amount but ask about a specific situation,
    # edge case, or hypothetical. They MUST go to LLM, not the rule-based template.
    scenario_signals = (
        "urgently", "urgent", "emergency", "what if", "what should",
        "suppose", "assume", "scenario", "claims", "proprietary",
        "sole", "only one", "single bid", "no bid", "no vendor",
        "not available on gem", "not on gem", "skip gem", "gem down",
        "split into", "split a", "club them",
        "l1 is not", "l1 not meeting", "l1 did not", "award to l2",
        "debarred", "blacklist", "without approval", "without pac",
        "expired", "previous year", "crosses financial",
        "can we", "can i", "is it allowed", "is this valid",
        "is this allowed", "is this permissible",
        "make in india", "local supplier", "local content",
        "how to handle", "how to verify",
        "not meeting", "does not meet", "did not meet",
        "received only", "only 1 quotation", "only 2 quotation",
        "exceeds", "exceed the", "all bids exceed",
        "foreign vendor", "foreign manufacturer", "import",
        "rate contract", "repeat", "extend the bid",
        "unregistered vendor", "non-competent",
        "retired scientist", "same person",
        "delivery.*delayed", "delayed beyond",
    )
    if any(term in cleaned for term in scenario_signals):
        return "SCENARIO"

    workflow_terms = (
        "workflow",
        "sop",
        "approval flow",
        "approval chain",
        "step by step",
        "step-by-step",
        "sequence of",
        "how to procure",
        "procurement process",
        "how to purchase",
        "purchase process",
        "how to tender",
        "tendering process",
        "indenting officer",
        "t&pc",
        "tsc",
    )
    if any(term in cleaned for term in workflow_terms):
        return "WORKFLOW"

    if extract_amount_lakhs(message) is not None:
        return "PROCESS"
    process_terms = ("process", "procedure", "steps", "how", "route", "approval")
    if any(term in cleaned for term in process_terms):
        return "PROCESS"

    return "GENERAL"


def is_threshold_route_query(message: str) -> bool:
    """Return True when the query is mainly about threshold bands or route mapping."""
    cleaned = clean_text(message).lower()
    if extract_amount_lakhs(message) is not None:
        return True
    return any(term in cleaned for term in _THRESHOLD_SYNONYMS)


def looks_like_scenario_query(message: str) -> bool:
    """Return True when the user frames the question as a practical scenario or case."""
    cleaned = f" {clean_text(message).lower()} "
    return any(term in cleaned for term in _SCENARIO_TERMS)


def get_analytical_method_variants(method: str) -> tuple[str, ...]:
    """Return known synonyms for a procurement method label."""
    return _ANALYTICAL_METHOD_ALIASES.get(method.upper(), (method.lower(),))


def extract_analytical_terms(query: str) -> list[str]:
    """Extract canonical procurement methods mentioned in a comparison query."""
    cleaned = clean_text(query).lower()
    found: list[str] = []
    for canonical, variants in _ANALYTICAL_METHOD_ALIASES.items():
        for variant in variants:
            normalized_variant = variant.lower().strip()
            if len(normalized_variant) <= 3 and normalized_variant.isalpha():
                if re.search(rf"\b{re.escape(normalized_variant)}\b", cleaned):
                    found.append(canonical)
                    break
                continue
            if normalized_variant in cleaned:
                found.append(canonical)
                break
    return found


def expand_query_keywords(query: str, intent: str) -> str:
    """Expand the query with synonyms based on detected intent."""
    cleaned_query = clean_text(query).lower()
    amount = extract_amount_lakhs(query)
    amount_keywords = " ".join(amount_to_context_keywords(amount, query))
    explicit_rule_terms = " ".join(_extract_explicit_rule_terms(query))
    scenario_focus = " ".join(_scenario_focus_terms(query))
    slab = gfr_slab_for_amount(amount, query)
    slab_keywords = ""
    if slab:
        slab_keywords = " ".join(
            [
                str(slab["method"]),
                str(slab["rule"]),
                str(slab["value_band"]),
                "GFR 2025 truth table",
                "CSIR Manual 2019 process",
                "Make in India SnT special provisions exemptions",
            ]
        )
    route_sensitive = any(
        token in cleaned_query
        for token in (
            "threshold",
            "value band",
            "slab",
            "route",
            "rule 149",
            "rule 154",
            "rule 155",
            "rule 161",
            "rule 162",
            "rule 166",
            "gem",
            "tender",
            "lpc",
            "lte",
            "ote",
            "ste",
        )
    )
    route_slab_keywords = slab_keywords if route_sensitive else ""
    if intent == "ANALYTICAL":
        parts = [query, explicit_rule_terms, " ".join(_ANALYTICAL_BROAD_TERMS), scenario_focus, route_slab_keywords]
        for method in extract_analytical_terms(query):
            parts.append(" ".join(get_analytical_method_variants(method)))
        return " ".join(part for part in parts if part).strip()
    if intent == "SCENARIO":
        return " ".join(
            part
            for part in (query, explicit_rule_terms, scenario_focus, amount_keywords, route_slab_keywords)
            if part
        ).strip()
    if intent == "WORKFLOW":
        synonyms = " ".join(_WORKFLOW_SYNONYMS)
        return " ".join(
            part for part in (query, explicit_rule_terms, synonyms, scenario_focus, amount_keywords, route_slab_keywords) if part
        ).strip()
    if intent == "PROCESS":
        if route_sensitive or amount is not None:
            synonyms = " ".join(_THRESHOLD_SYNONYMS)
        else:
            synonyms = " ".join(_PROCESS_KEYWORDS)
        return " ".join(
            part for part in (query, explicit_rule_terms, synonyms, scenario_focus, amount_keywords, route_slab_keywords) if part
        ).strip()
    return " ".join(part for part in (query, explicit_rule_terms, scenario_focus, amount_keywords, route_slab_keywords) if part).strip()


def _extract_explicit_rule_terms(query: str) -> tuple[str, ...]:
    """Return exact rule-oriented expansion terms for rule-number queries."""
    normalized = clean_text(query)
    matches = re.findall(r"\brule\s+(\d{3})\b", normalized, flags=re.IGNORECASE)
    if not matches:
        return ()

    expanded: list[str] = []
    for rule_number in matches:
        expanded.extend(
            [
                f"Rule {rule_number}",
                f"GFR Rule {rule_number}",
                f"rule {rule_number} procurement",
            ]
        )
        if rule_number == "149":
            expanded.append("GeM mandatory procurement")
        elif rule_number == "154":
            expanded.append("direct purchase market reasonableness")
        elif rule_number == "155":
            expanded.append("local purchase committee three quotations")
        elif rule_number == "161":
            expanded.append("open tender wide publicity")
        elif rule_number == "162":
            expanded.append("limited tender enquiry three firms")
        elif rule_number == "166":
            expanded.append("single tender proprietary PAC certificate")
    return tuple(dict.fromkeys(expanded))


def extract_amount_lakhs(message: str) -> float | None:
    """Extract monetary amount in lakhs from the message.

    Priority order:
    1. Explicit crore unit
    2. Explicit lakh/lac/lakhs unit  (NOT bare 'l' — avoids false matches)
    3. Explicit thousand/k unit
    4. Bare Rs./INR/rupee prefix without unit  → kept as rupees
    5. Comma-formatted rupee amounts (e.g. 50,00,000) when accompanied by currency intent words
    6. Compact 5-8 digit number  ONLY if lakh/crore context word nearby
    7. Shorthand tokens Nl / Ncr
    """
    cleaned = clean_text(message).lower()
    # Keep a version with commas for comma-form numeric detection.
    cleaned_with_commas = cleaned
    normalized = cleaned.replace(",", "")

    # 1. Crore — explicit unit
    crore_match = re.search(
        r"(?:rs\.?\s*|inr\s*|₹\s*)?([\d]+(?:\.\d+)?)\s*(?:crore|crores|cr)\b",
        normalized,
    )
    if crore_match:
        return float(crore_match.group(1)) * 100.0

    # 2. Lakh — explicit keyword only (NO bare 'l' — prevents "Rs 56 l" false match)
    lakh_match = re.search(
        r"(?:rs\.?\s*|inr\s*|₹\s*)?([\d]+(?:\.\d+)?)\s*(?:lakh|lakhs|lac|lacs)\b",
        normalized,
    )
    if lakh_match:
        return float(lakh_match.group(1))

    # 3. Thousand / k
    thousand_match = re.search(
        r"(?:rs\.?\s*|inr\s*|₹\s*)?([\d]+(?:\.\d+)?)\s*(?:thousand|k)\b",
        normalized,
    )
    if thousand_match:
        return float(thousand_match.group(1)) / 100.0

    # 4. Bare currency prefix without explicit unit — keep as rupees
    #    "Rs. 56" = 56 rupees = 0.00056 lakh (NOT 56 lakh)
    rupee_match = re.search(r"(?:rs\.?|inr|₹)\s*([\d]+(?:\.\d+)?)\b", normalized)
    if rupee_match:
        val = float(rupee_match.group(1))
        return val / 100000.0

    # 5. Comma-formatted rupee amount without explicit lakh/crore unit.
    #    Example: "procurement worth Rs 50,00,000" OR "worth 50,00,000" (no Rs/INR).
    comma_rupee_int = re.search(
        r"\b(?:worth|costing|amount(?:ing)? to|value(?:d)? at|purchase of|procurement of|estimated at|for)\s+(\d{1,3}(?:,\d{2}){2,3})\b",
        cleaned_with_commas,
    )
    if comma_rupee_int:
        numeric = _parse_numeric_token(comma_rupee_int.group(1))
        if numeric is not None:
            return numeric / 100000.0

    # 5b. Standalone comma-number preceded by a rupee intent token (looser)
    comma_rupee_loose = re.search(
        r"(?:worth|costing|purchase of|procurement of|estimated|value(?:d)? at)\s+(\d{1,3}(?:,\d{2}){2,3})",
        cleaned_with_commas,
    )
    if comma_rupee_loose:
        numeric = _parse_numeric_token(comma_rupee_loose.group(1))
        if numeric is not None:
            return numeric / 100000.0

    # 6. Compact 5-8 digit bare number ONLY if lakh/crore context word nearby
    lakh_context = any(
        kw in normalized
        for kw in ("lakh", "lac", "crore", "threshold", "limit", "value band")
    )
    compact_match = re.search(r"\b(\d{5,8})\b", normalized)
    if compact_match and lakh_context:
        return float(compact_match.group(1)) / 100000.0

    # 7. Shorthand: Nl = N lakhs, Ncr = N*100 lakhs
    shorthand_match = re.search(r"\b([\d]+(?:\.[\d]+)?)(l|cr)\b", normalized)
    if shorthand_match:
        val = float(shorthand_match.group(1))
        unit = shorthand_match.group(2)
        return val * 100.0 if unit == "cr" else val

    return None


def amount_to_context_keywords(amount_lakhs: float | None, query: str | None = None) -> tuple[str, ...]:
    """Return threshold-bracket keywords for a detected amount in lakhs."""
    return gfr_keywords_for_amount(amount_lakhs, query)


def format_lakh_amount(amount_lakhs: float) -> str:
    """Format a lakh amount into readable text."""
    if amount_lakhs < 1:
        return f"Rs. {int(round(amount_lakhs * 100000)):,}"
    if float(amount_lakhs).is_integer():
        return f"Rs. {int(amount_lakhs)} lakhs"
    return f"Rs. {amount_lakhs:g} lakhs"


def _parse_numeric_token(value: str) -> float | None:
    cleaned = value.replace(",", "").strip()
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def extract_amount_rupees(message: str) -> int | None:
    """Extract the first procurement amount from free text as integer rupees."""
    normalized = clean_text(message).lower()
    normalized = normalized.replace("₹", " rs ")
    normalized = re.sub(r"\brupees?\b", " rs ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    unit_patterns: tuple[tuple[re.Pattern[str], int], ...] = (
        (
            re.compile(r"(?:\brs\.?\s*)?(\d[\d,]*(?:\.\d+)?)\s*(?:crore|crores|cr)\b"),
            10_000_000,
        ),
        (
            re.compile(r"(?:\brs\.?\s*)?(\d[\d,]*(?:\.\d+)?)\s*(?:lakh|lakhs|lac|lacs)\b"),
            100_000,
        ),
        (
            re.compile(r"(?:\brs\.?\s*)?(\d[\d,]*(?:\.\d+)?)\s*(?:thousand|k)\b"),
            1_000,
        ),
    )
    for pattern, multiplier in unit_patterns:
        match = pattern.search(normalized)
        if not match:
            continue
        numeric = _parse_numeric_token(match.group(1))
        if numeric is not None:
            return int(round(numeric * multiplier))

    shorthand_match = re.search(r"\b(\d+(?:\.\d+)?)(l|lk|cr)\b", normalized)
    if shorthand_match:
        numeric = _parse_numeric_token(shorthand_match.group(1))
        if numeric is not None:
            multiplier = 10_000_000 if shorthand_match.group(2) == "cr" else 100_000
            return int(round(numeric * multiplier))

    currency_match = re.search(r"\brs\.?\s*(\d[\d,]*(?:\.\d+)?)\b", normalized)
    if currency_match:
        numeric = _parse_numeric_token(currency_match.group(1))
        if numeric is not None:
            return int(round(numeric))

    context_match = re.search(
        r"\b(?:worth|costing|amount(?:ing)? to|value(?:d)? at|purchase of|procurement of)\s+(\d[\d,]*(?:\.\d+)?)\b",
        normalized,
    )
    if context_match:
        numeric = _parse_numeric_token(context_match.group(1))
        if numeric is not None and numeric >= 1_000:
            return int(round(numeric))

    return None


def extract_amount_lakhs(message: str) -> float | None:
    """Extract monetary amount in lakhs from the message using strict rupee normalization."""
    amount_rupees = extract_amount_rupees(message)
    if amount_rupees is None:
        return None
    return amount_rupees / 100000.0


def score_amount_relevance(candidate: str, amount_lakhs: float | None) -> int:
    """Score how relevant a candidate sentence is to the queried amount."""
    if amount_lakhs is None:
        return 0
    lowered = candidate.lower()
    score = 0
    if "lakh" in lowered or "rs." in lowered or "₹" in candidate:
        score += 2
    if amount_lakhs <= 50 and any(term in lowered for term in ("50 lakhs or less", "up to rs. 50 lakhs", "up to 50 lakhs", "only local suppliers")):
        score += 3
    if amount_lakhs <= 50 and any(term in lowered for term in ("more than 50 lakhs", "above rs. 50 lakhs")):
        score -= 1
    slab = gfr_slab_for_amount(amount_lakhs)
    if slab and slab.get("key") not in {"DIRECT PURCHASE", "LPC"} and (
        "2,50,000" in candidate or "2.5 lakhs" in lowered or "local purchase committee" in lowered
    ):
        score -= 2
    return score


def score_process_relevance(query: str, candidate: str) -> int:
    """Score how relevant a candidate is to a process/procedure query."""
    query_lower = query.lower()
    candidate_lower = candidate.lower()
    score = 0
    if "process" in query_lower and any(term in candidate_lower for term in ("procedure", "procurement", "committee", "eligible", "award", "l1")):
        score += 2
    if "table" in query_lower and "|" in candidate:
        score += 1
    return score


def score_analytical_relevance(query: str, candidate: str) -> int:
    """Score how useful a candidate is for comparison-style questions."""
    query_lower = query.lower()
    candidate_lower = candidate.lower()
    if not any(term in query_lower for term in _ANALYTICAL_TERMS):
        return 0

    score = 0
    methods = extract_analytical_terms(query)
    for method in methods:
        if any(alias in candidate_lower for alias in get_analytical_method_variants(method)):
            score += 2
    if any(marker in candidate_lower for marker in ("is used for", "is meant for", "applies to", "applies when", "means", "refers to", "shall be used")):
        score += 3
    if any(marker in candidate_lower for marker in ("definition", "applicability", "threshold", "approval", "quotation")):
        score += 2
    if any(marker in candidate_lower for marker in ("single offer", "rule 21 of dfpr", "court", "litigation", "legal case")):
        score -= 2
    return score


def extract_username(data: Any) -> str:
    """Extract display name from user metadata."""
    if not isinstance(data, dict):
        return "User"
    if data.get("displayName"):
        return str(data["displayName"]).strip()
    if data.get("username"):
        return str(data["username"]).strip()
    if data.get("email"):
        return str(data["email"]).split("@")[0].capitalize()
    return "User"


def describe_user_request(message: str) -> str:
    """Create a shortened description of the user's question."""
    cleaned = clean_text(message).strip().rstrip("?").rstrip(".")
    lowered = cleaned.lower()
    prefixes = [
        "what is ",
        "what are ",
        "show me ",
        "show ",
        "tell me ",
        "explain ",
    ]
    for prefix in prefixes:
        if lowered.startswith(prefix):
            cleaned = cleaned[len(prefix):]
            break
    cleaned = cleaned[:1].lower() + cleaned[1:] if cleaned else "the request"
    return cleaned


def tokenize(value: str) -> set[str]:
    """Tokenize text into a set of lowercase terms (min 2 chars)."""
    normalized = re.sub(r"[^a-zA-Z0-9\s]", " ", value.lower())
    return {token for token in normalized.split() if len(token) >= 2}


def split_sentences(content: str) -> list[str]:
    """Split text into sentences."""
    parts = re.split(r"(?<=[.!?])\s+|\n+", content)
    return [part.strip() for part in parts if part.strip()]


def semantic_dedup_key(value: str) -> str:
    """Generate a deduplication key for semantically similar sentences."""
    lowered = clean_text(value).lower()
    if "local purchase committee" in lowered or "2,50,000" in lowered or "2.5 lakhs" in lowered:
        return "lpc-threshold"
    if "only local suppliers" in lowered and "50 lakhs" in lowered:
        return "local-supplier-under-50"
    if "above rs. 50 lakh" in lowered or "more than 50 lakhs" in lowered or "next purchase-preference rules" in lowered:
        return "above-50-rule"
    if "l1" in lowered and "50%" in lowered:
        return "l1-supplier-split"
    if "limited tender" in lowered or "lte" in lowered:
        return "lte-" + lowered
    if "single tender" in lowered or "ste" in lowered or "proprietary" in lowered:
        return "ste-" + lowered
    return lowered
