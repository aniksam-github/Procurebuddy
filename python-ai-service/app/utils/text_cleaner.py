"""Advanced text sanitization for the RAG pipeline.

Handles mojibake removal, PDF artifact cleaning, TOC detection,
scientist-list filtering, and currency normalization.
"""

from __future__ import annotations

import re

from app.core.constants import (
    MOJIBAKE_REPLACEMENTS,
    PROCUREMENT_ACTION_VERBS,
)


# ── Core cleaner ────────────────────────────────────────────────────────────

def clean_text(value: str) -> str:
    """Remove mojibake, normalize whitespace, fix broken hyphenation."""
    cleaned = value or ""
    for source, target in MOJIBAKE_REPLACEMENTS.items():
        cleaned = cleaned.replace(source, target)
    cleaned = cleaned.replace("\x00", " ")
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"-\s*\n\s*", "", cleaned)        # broken hyphenation
    cleaned = re.sub(r"\n+", " ", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    return cleaned.strip()


# ── PDF artifact removal ────────────────────────────────────────────────────

_PDF_ARTIFACT_RE = re.compile(
    r"S~2-\$|\\S~2-\$|[^\x20-\x7E\u00A0-\uFFFF]",
)


def remove_pdf_artifacts(value: str) -> str:
    """Strip broken PDF encoding characters like S~2-$, stray control chars."""
    cleaned = _PDF_ARTIFACT_RE.sub(" ", value)
    return re.sub(r" {2,}", " ", cleaned).strip()


# ── Currency normalization ──────────────────────────────────────────────────

def normalize_currency(value: str) -> str:
    """Standardize all currency symbols to 'Rs.'."""
    cleaned = value
    cleaned = cleaned.replace("₹", "Rs. ")
    cleaned = re.sub(r"\bINR\s*", "Rs. ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\brupees?\s+", "Rs. ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"Rs\.\s+", "Rs. ", cleaned)  # normalize spacing
    return cleaned


# ── Scientist / name-list detection ─────────────────────────────────────────

_NAME_PATTERN = re.compile(
    r"\b(?:Dr|Prof|Shri|Smt|Mr|Ms|Mrs)\.\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b"
)

_METHOD_DEFINITION_RE = re.compile(
    r"\b(?:limited tender(?: enquiry)?|single tender(?: enquiry)?|open tender(?: enquiry)?|local purchase committee|lte|ste|ote|lpc)\b.*\b(?:is|means|refers to|applies to|applies when|shall be used|is used for|is meant for)\b",
    re.IGNORECASE,
)

_LEGAL_NOISE_RE = re.compile(
    r"\b(?:rule\s*21\s*of\s*dfpr|dfpr|single offer|sole offer|sole bid|court|writ petition|legal case|litigation)\b",
    re.IGNORECASE,
)


def contains_scientist_list(content: str) -> bool:
    """Return True if chunk looks like a list of scientists/fellows/members
    rather than procurement content (5+ proper names, no action verbs)."""
    names = _NAME_PATTERN.findall(content)
    if len(names) < 5:
        return False
    lowered = content.lower()
    has_action_verb = any(verb in lowered for verb in PROCUREMENT_ACTION_VERBS)
    return not has_action_verb


def has_definition_style(content: str) -> bool:
    """Return True when a chunk defines or explains the core procurement method."""
    return bool(_METHOD_DEFINITION_RE.search(clean_text(content)))


def legalistic_noise_penalty(content: str) -> float:
    """Penalty for legal-case/exception-heavy chunks with weak method explanation."""
    cleaned = clean_text(content)
    if not cleaned:
        return 0.0
    if _LEGAL_NOISE_RE.search(cleaned) and not has_definition_style(cleaned):
        return -0.10
    return 0.0


def action_sentence_density(content: str) -> float:
    """Estimate how much of the chunk describes actions or procedure."""
    cleaned = clean_text(content)
    if not cleaned:
        return 0.0
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", cleaned) if part.strip()]
    if not sentences:
        return 0.0
    action_hits = 0
    for sentence in sentences:
        lowered = sentence.lower()
        if any(verb in lowered for verb in PROCUREMENT_ACTION_VERBS):
            action_hits += 1
    return action_hits / len(sentences)


def numeric_reference_count(content: str) -> int:
    """Count section-like or page-like numeric references inside a chunk."""
    cleaned = clean_text(content)
    return len(re.findall(r"\b\d+(?:\.\d+)*\b", cleaned))


def audit_chunk_quality(content: str) -> dict[str, float | str | bool | int]:
    """Tag a chunk as PROCEDURAL or REFERENCE_ONLY and detect noisy/index-like text."""
    cleaned = clean_text(content)
    letters = sum(1 for char in cleaned if char.isalpha())
    noisy_chars = sum(1 for char in cleaned if char.isdigit() or (not char.isalnum() and not char.isspace()))
    alpha_ratio = letters / max(1, letters + noisy_chars)
    action_density = action_sentence_density(cleaned)
    number_refs = numeric_reference_count(cleaned)
    tag = "PROCEDURAL" if action_density >= 0.25 else "REFERENCE_ONLY"
    discard = (
        looks_like_table_of_contents(cleaned)
        or contains_scientist_list(cleaned)
        or alpha_ratio < 0.55
        or (number_refs >= 8 and action_density == 0.0)
    )
    return {
        "tag": tag,
        "action_sentence_density": round(action_density, 4),
        "numeric_reference_count": number_refs,
        "alpha_ratio": round(alpha_ratio, 4),
        "discard": discard,
    }


# ── Table of Contents detection ─────────────────────────────────────────────

def looks_like_table_of_contents(value: str) -> bool:
    """Detect chunks that are TOC / index pages rather than content."""
    cleaned = clean_text(value).lower()
    if not cleaned:
        return False
    if "table of contents" in cleaned or "title page no." in cleaned:
        return True
    if "chapter-" in cleaned or "chapter –" in cleaned or "chapter -" in cleaned:
        section_hits = len(re.findall(r"\b\d+\.\d+\b", cleaned))
        if section_hits >= 3:
            return True
    dense_numbering = len(re.findall(r"\b\d+\.\d+\b", cleaned))
    page_number_hits = len(re.findall(r"\b\d{1,3}\b", cleaned))
    if dense_numbering >= 5 and page_number_hits >= 8:
        return True
    toc_phrases = (
        "constitution of technical & purchase committee",
        "functions of technical & purchase committee",
        "procurement planning",
        "supplier relationship management",
        "grades of debarment of suppliers",
    )
    if sum(1 for phrase in toc_phrases if phrase in cleaned) >= 2:
        return True
    return False


# ── Text quality checks ────────────────────────────────────────────────────

def is_clean_chunk(content: str) -> bool:
    """Return True only if the chunk has enough readable, domain-relevant text."""
    if not content or len(content.strip()) < 50:
        return False
    alpha = sum(1 for c in content if c.isalpha())
    if alpha < max(20, len(content) * 0.25):
        return False
    audit = audit_chunk_quality(content)
    if audit["discard"]:
        return False
    return True


# ── Sentence-level utilities ────────────────────────────────────────────────

def strip_leading_noise(value: str) -> str:
    """Remove chapter/annexure headers, dense numbering, and stray punctuation."""
    cleaned = value
    cleaned = re.sub(r"^(?:chapter|annexure|appendix|table|form)\s*[-:0-9a-zA-Z.() ]+\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^(?:\d+(?:\.\d+){0,4}\s*)+", "", cleaned)
    cleaned = re.sub(r"^(?:[A-Z]{1,4}\s+){2,}", "", cleaned)
    cleaned = re.sub(r"^\W+", "", cleaned)
    return cleaned.strip()


def ensure_sentence_punctuation(value: str) -> str:
    """Ensure the text ends with proper sentence punctuation."""
    cleaned = value.strip().rstrip(",;:-")
    if not cleaned:
        return ""
    if cleaned[-1] not in ".!?":
        cleaned = f"{cleaned}."
    return cleaned


def is_useful_sentence(value: str) -> bool:
    """Check if a sentence has enough substance to include in output."""
    cleaned = value.strip()
    if len(cleaned.split()) < 7:
        return False
    alpha_chars = sum(1 for char in cleaned if char.isalpha())
    digit_chars = sum(1 for char in cleaned if char.isdigit())
    if alpha_chars < max(12, digit_chars * 2):
        return False
    if cleaned.lower().startswith(("document:", "summary:", "source:")):
        return False
    if cleaned.lower().startswith("in procurement of goods, services or works in respect of which"):
        return False
    if looks_like_table_of_contents(cleaned):
        return False
    if cleaned.lower().endswith(("as the.", "as the case.", "for the remaining.", "up to rs.", "where the.")):
        return False
    return True


def smooth_summary_text(value: str, max_length: int = 220) -> str:
    """Clean and truncate text to a readable summary length."""
    cleaned = clean_text(value)
    cleaned = strip_leading_noise(cleaned)
    cleaned = cleaned.replace("₹", "Rs. ")
    cleaned = re.sub(r"\b\d+\s*$", "", cleaned).strip()
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    if len(cleaned) <= max_length:
        return ensure_sentence_punctuation(cleaned)
    truncated = cleaned[:max_length].rsplit(" ", 1)[0].rstrip(",;:-")
    return ensure_sentence_punctuation(truncated)


def rewrite_procurement_sentence(value: str) -> str:
    """Rewrite known procurement patterns into clean, readable sentences."""
    cleaned = clean_text(value)
    lowered = cleaned.lower()

    if "certified that we, members of the purchase committee" in lowered:
        return "The purchase committee must record a joint certificate that the recommended goods meet the required specification and quality and that the proposed rate is reasonable."
    if "i am personally satisfied" in lowered and "reasonable price" in lowered:
        return "The approving officer should record satisfaction that the goods meet the required specification and quality and have been obtained from a reliable source at a reasonable price."
    if "copies of the bidding document" in lowered and "registered suppliers" in lowered:
        return "Under limited tender, the bid documents are issued directly to a limited set of suitable or registered suppliers instead of being opened to the public."
    if "an item is said to be not available in gem" in lowered:
        return "An item should be treated as unavailable on GeM only when the required specification or delivery requirement cannot be met there."
    if "only when the item is not available on gem" in lowered and "open market" in lowered:
        return "Open-market procurement should be considered only after recording that the required item is not available on GeM."
    if "procurement from a single source may be resorted to" in lowered:
        return "Single-source procurement is allowed only in limited situations such as proprietary supply, standardisation, or genuine emergency, and the justification must be recorded."
    if "wide publicity" in lowered and ("cpp portal" in lowered or "portal" in lowered):
        return "Wide publicity means the tender is publicly advertised through the approved portal so all eligible bidders have a fair opportunity to participate."
    if "local purchase committee" in lowered and (
        ("25,000" in cleaned and ("2,50,000" in cleaned or "two lakh fifty thousand" in lowered))
        or ("50,000" in cleaned and ("5,00,000" in cleaned or "five lakh" in lowered))
    ):
        return (
            "Local Purchase Committee procurement is meant for smaller purchases above Rs. 50,000 and up to Rs. 5 lakhs under the GFR 2025 truth table, "
            "including minor fabrication, repairs, and small works or services."
        )
    if "estimated value of procurement is" in lowered and "50 lakhs or less" in lowered and "only local suppliers shall be eligible" in lowered:
        return "For categories where the nodal ministry has confirmed enough local capacity and competition, procurements up to Rs. 50 lakhs are limited to local suppliers."
    if "more than" in lowered and "50 lakhs" in lowered and ("sub-paragraph" in lowered or "sub- paragraph" in lowered):
        return "For procurements above Rs. 50 lakhs, the manual moves to the next purchase-preference rules instead of the up-to-Rs. 50 lakh condition."
    if "sufficient local capacity and local competition" in lowered:
        return "This condition applies only when the nodal ministry has confirmed enough local capacity and competition in the relevant category."
    if "lowest bid will be termed as l1" in lowered and "50% of the order quantity" in lowered:
        return "If the lowest eligible bid is not from a local supplier in a divisible procurement, part of the quantity may still be offered to an eligible local supplier that matches the L1 price."
    if "if l1 is from a local supplier" in lowered and "full quantity will be awarded to l1" in lowered:
        return "When the lowest eligible bidder is a local supplier, the full quantity can be awarded to that supplier."
    if "lowest bidder among the local suppliers" in lowered and "remaining 50% quantity" in lowered:
        return "For divisible procurement, an eligible local supplier may be invited to match the L1 price for the remaining quantity when purchase-preference conditions are met."
    if "sufficient local capacity and local competition" in lowered and ("50.00 lakhs" in cleaned or "50 lakhs" in lowered):
        return "The local-supplier preference up to Rs. 50 lakhs applies only where the nodal ministry has confirmed adequate local capacity and competition."
    if "this mode of procurement is used for procurements valued above" in lowered and "goods not available on gem" in lowered:
        return "This small-purchase route is intended for low-value procurements not available on GeM, not for higher-value procurement cases."
    return cleaned
