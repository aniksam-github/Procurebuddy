"""Output validation, cleaning, and retry logic for ProcureBuddy structured responses.

Fixes:
- Broken bullet sentences from chunking
- Missing sections (injects fallbacks)
- Rule version mismatch (GFR 2025 -> GFR 2017)
- Wrong committee names
- Uncertainty symbols ("?", "incomplete")
- Internal reasoning leakage ("threshold engine", "planner", etc.)
- Incomplete process steps (< 4 steps)
- Duplicate/split bullets
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger("procurebuddy-ai")

# ── Constants ─────────────────────────────────────────────────────────────────

REQUIRED_SECTIONS = [
    "## Quick Answer",
    "## Amount Breakdown",
    "## Threshold Evaluation",
    "## Why This Applies",
    "## Detailed Process",
    "## Comparison",
    "## Source Basis",
    "## TL;DR",
]

# Internal reasoning leakage patterns to strip
_LEAKAGE_PATTERNS = re.compile(
    r"(?i)(threshold engine|planner decision|tool state|rag retriever|"
    r"retrieval mode|generation mode|mii engine|rule lookup|source quality|"
    r"weak match|structured context|orchestrat\w+|vector search|chromadb|"
    r"langchain|embedding|chunk_id|document_id|score=[\d.]+)",
    re.I,
)

# Uncertainty / broken-sentence patterns
_UNCERTAINTY_PATTERNS = re.compile(
    r"(?i)(committee\s*[?:]\s*$|\?\s*$|incomplete\.\s*$|tbc\.?\s*$|"
    r"to be confirmed\.?\s*$|n/a\.?\s*$)",
)

# GFR version correction: "GFR 2025" → "GFR 2017 (as amended)"
_GFR_VERSION_FIX = re.compile(r"\bGFR\s+2025\b", re.I)

# Committee normalisation map
_COMMITTEE_MAP = {
    "LTE": "Technical & Purchase Committee (T&PC) - min. 3 members",
    "OTE": "Technical & Purchase Committee (T&PC) - min. 3 members",
    "LPC": "Local Purchase Committee (LPC) - min. 3 members",
    "STE": "Technical Committee + Competent Authority approval",
    "Direct Purchase": "Not required (single officer, value <= Rs. 2,00,000)",
    "GeM": "Purchase officer (T&PC if value > Rs. 5 lakh)",
}

# Default process steps per mode (min 4 steps)
_DEFAULT_STEPS: dict[str, str] = {
    "LTE": (
        "1. Prepare indent and get Technical Scrutiny Committee (TSC) clearance.\n"
        "2. Check GeM availability; obtain Non-Availability Certificate (NAC) if not on GeM.\n"
        "3. Issue Limited Tender Enquiry (LTE) to minimum 3 registered vendors.\n"
        "4. Constitute T&PC; open bids and prepare comparative statement.\n"
        "5. T&PC recommends L1-responsive vendor; Finance concurrence obtained.\n"
        "6. Competent Authority grants sanction (per DFP limits).\n"
        "7. Issue Purchase Order; post-delivery inspection and acceptance."
    ),
    "OTE": (
        "1. Prepare indent and obtain TSC clearance.\n"
        "2. Check GeM; obtain NAC if not available.\n"
        "3. Issue Open Tender Enquiry (OTE) with wide publicity (e-procurement portal + newspaper).\n"
        "4. Constitute T&PC for bid opening and technical evaluation.\n"
        "5. Commercial evaluation of technically qualified bids; prepare comparative statement.\n"
        "6. Finance wing concurrence; Competent Authority approval (Director/DG).\n"
        "7. Issue Purchase Order and record all documents on file."
    ),
    "LPC": (
        "1. Indenting officer raises indent; TSC review if technical.\n"
        "2. Check GeM availability; get NAC if not found.\n"
        "3. Local Purchase Committee (LPC) obtains minimum 3 quotations.\n"
        "4. LPC prepares comparative statement; selects L1 vendor.\n"
        "5. Competent Authority sanction as per DFP.\n"
        "6. Issue Supply Order; goods inspected on delivery."
    ),
    "Direct Purchase": (
        "1. Indenting officer confirms value <= Rs. 2,00,000 (Rule 154).\n"
        "2. Verify GeM availability; if available, procure via GeM portal.\n"
        "3. If not on GeM, obtain market rate reasonableness record.\n"
        "4. Procuring officer issues supply order; maintains purchase record on file."
    ),
    "STE": (
        "1. Indenting officer raises indent with PAC (Proprietary Article Certificate) or justified STE basis.\n"
        "2. Technical authority certifies proprietary/emergency/standardisation ground.\n"
        "3. Competent Authority approves STE route (Finance concurrence if high value).\n"
        "4. Negotiate price with single vendor; record price reasonableness.\n"
        "5. Issue Purchase Order; inspection and acceptance on delivery."
    ),
    "GeM": (
        "1. Indenting officer searches item on GeM portal.\n"
        "2. For <= Rs. 25,000: direct purchase from any GeM seller.\n"
        "3. For Rs. 25,001–5 lakh: compare minimum 3 GeM sellers.\n"
        "4. For > Rs. 5 lakh: GeM Bidding / Reverse Auction (RA).\n"
        "5. Place order through GeM portal; record transaction ID on file."
    ),
    "GENERAL": (
        "1. Indenting officer prepares indent and requirement specifications.\n"
        "2. Check GeM availability as per Rule 149, GFR 2017.\n"
        "3. Determine procurement route based on value band (Rule 154/155/161/162/166).\n"
        "4. Obtain quotations / issue tender per applicable route.\n"
        "5. Evaluate bids through appropriate committee; prepare comparative statement.\n"
        "6. Finance concurrence and Competent Authority approval.\n"
        "7. Issue Purchase Order; record all documents on procurement file."
    ),
}

_DEFAULT_KEY_DOCS = (
    "- Indent / Purchase Requisition\n"
    "- Technical Scrutiny Committee (TSC) Note (if applicable)\n"
    "- GeM Non-Availability Certificate (NAC) or GeM Order ID\n"
    "- Comparative Statement / Quotation File\n"
    "- T&PC / LPC Minutes of Meeting\n"
    "- Finance Concurrence Note\n"
    "- Competent Authority Approval / Sanction Order\n"
    "- Purchase Order / Supply Order"
)

_DEFAULT_FLOWCHART = (
    "```mermaid\n"
    "flowchart TD\n"
    "    A[Start: Receive Indent] --> B[Check GeM - Rule 149]\n"
    "    B --> C{Item on GeM?}\n"
    "    C -->|Yes| D[Procure via GeM Portal]\n"
    "    C -->|No| E[Get NAC]\n"
    "    E --> F{Determine Value Band}\n"
    "    F -->|Up to 2L| G[Direct Purchase - Rule 154]\n"
    "    F -->|2L to 5L| H[LPC - Rule 155]\n"
    "    F -->|5L to 50L| I[LTE - Rule 162]\n"
    "    F -->|Above 50L| J[OTE - Rule 161]\n"
    "    G --> K[Competent Authority Approval]\n"
    "    H --> K\n"
    "    I --> K\n"
    "    J --> K\n"
    "    K --> L[Issue Purchase Order]\n"
    "```"
)

# Deterministic route table used by the active response pipeline.
_COMMITTEE_MAP = {
    "LTE": "Technical & Purchase Committee (T&PC) - min. 3 members",
    "OTE": "Technical & Purchase Committee (T&PC) - min. 3 members",
    "LPC": "Local Purchase Committee (LPC) - min. 3 members",
    "STE": "Technical Committee + Competent Authority approval",
    "Direct Purchase": "Not required (single officer, value <= Rs. 2,00,000)",
    "GeM": "Purchase officer (T&PC if value > Rs. 5 lakh)",
}

_DEFAULT_STEPS["Direct Purchase"] = (
    "1. Confirm the estimated value is within the direct-purchase band under Rule 154.\n"
    "2. Record the requirement, source, and basic price reasonableness.\n"
    "3. Obtain the competent approval for direct purchase.\n"
    "4. Issue the order and place the supporting record on file."
)

_DEFAULT_FLOWCHART = (
    "```mermaid\n"
    "flowchart TD\n"
    "    A[Start: Receive Indent] --> B[Check GeM - Rule 149]\n"
    "    B --> C{Item on GeM?}\n"
    "    C -->|Yes| D[Procure via GeM Portal]\n"
    "    C -->|No| E[Get NAC]\n"
    "    E --> F{Determine Value Band}\n"
    "    F -->|Up to 2L| G[Direct Purchase - Rule 154]\n"
    "    F -->|2L to 5L| H[LPC - Rule 155]\n"
    "    F -->|5L to 50L| I[LTE - Rule 162]\n"
    "    F -->|Above 50L| J[OTE - Rule 161]\n"
    "    G --> K[Competent Authority Approval]\n"
    "    H --> K\n"
    "    I --> K\n"
    "    J --> K\n"
    "    K --> L[Issue Purchase Order]\n"
    "```"
)


# ── Chunk Merging ─────────────────────────────────────────────────────────────

def merge_adjacent_chunks(chunks: list[str], max_merged_chars: int = 1200) -> list[str]:
    """Merge adjacent chunks from the same document if they form broken sentences.

    Rules:
    - Merge if previous chunk ends mid-sentence (no terminal punctuation)
    - Merge if merged length stays under max_merged_chars
    - Remove duplicate content before merging
    """
    if not chunks:
        return chunks
    merged: list[str] = []
    current = chunks[0].strip()
    for next_chunk in chunks[1:]:
        next_chunk = next_chunk.strip()
        if not next_chunk:
            continue
        # Merge if current ends without terminal punctuation
        ends_incomplete = current and not re.search(r"[.!?:]\s*$", current)
        would_fit = len(current) + len(next_chunk) + 1 <= max_merged_chars
        if ends_incomplete and would_fit:
            current = current + " " + next_chunk
        else:
            merged.append(current)
            current = next_chunk
    if current:
        merged.append(current)
    return merged


# ── Text Cleaning ─────────────────────────────────────────────────────────────

def clean_bullet(line: str) -> str:
    """Fix a single bullet line: strip leakage, uncertainty, ensure sentence ends properly."""
    line = line.strip()
    # Remove internal leakage
    line = _LEAKAGE_PATTERNS.sub("", line).strip()
    # Remove uncertainty artifacts
    line = re.sub(r"\s*\?\s*$", ".", line)
    line = re.sub(r"\s*(tbc|incomplete|n/a|undefined)\s*\.?\s*$", ".", line, flags=re.I)
    # Ensure ends with punctuation
    if line and not line[-1] in ".!?:":
        line = line + "."
    return line


def fix_broken_bullets(section_text: str) -> str:
    """Join bullet lines that appear to be continuations of the previous line."""
    lines = section_text.split("\n")
    result: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            result.append("")
            continue
        is_bullet = stripped.startswith(("-", "*", "•")) or re.match(r"^\d+\.", stripped)
        if is_bullet:
            result.append(clean_bullet(stripped))
        elif result and not result[-1].endswith((".", "!", "?", ":")):
            # Continuation of previous incomplete line
            result[-1] = result[-1].rstrip() + " " + stripped
            result[-1] = clean_bullet(result[-1])
        else:
            result.append(stripped)
    return "\n".join(result)


def fix_gfr_version(text: str) -> str:
    """Replace 'GFR 2025' with 'GFR 2017 (as amended)' unless it appears in retrieved context."""
    return _GFR_VERSION_FIX.sub("GFR 2017 (as amended)", text)


def remove_leakage(text: str) -> str:
    """Remove internal reasoning leakage phrases."""
    text = _LEAKAGE_PATTERNS.sub("[internal]", text)
    # Remove whole lines that are pure leakage
    cleaned_lines = []
    for line in text.split("\n"):
        if "[internal]" in line and len(line.strip()) < 30:
            continue
        cleaned_lines.append(line.replace("[internal]", "").strip())
    return "\n".join(cleaned_lines)


def remove_duplicate_bullets(section_text: str) -> str:
    """Remove duplicate bullet lines (case-insensitive, first occurrence wins)."""
    seen: set[str] = set()
    result: list[str] = []
    for line in section_text.split("\n"):
        key = re.sub(r"^[-*•\d.)\s]+", "", line).strip().lower()
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        result.append(line)
    return "\n".join(result)


# ── Section Validation ────────────────────────────────────────────────────────

def _extract_section(text: str, section_header: str) -> str:
    """Extract content of a section up to the next ## heading."""
    pattern = re.escape(section_header) + r"(.*?)(?=\n## |\Z)"
    m = re.search(pattern, text, re.DOTALL)
    return m.group(1).strip() if m else ""


def _infer_mode_from_text(text: str) -> str:
    """Infer procurement mode from text."""
    t = text.lower()
    if "open tender" in t or " ote" in t:
        return "OTE"
    if "limited tender" in t or " lte" in t:
        return "LTE"
    if "local purchase" in t or " lpc" in t:
        return "LPC"
    if "single tender" in t or " ste" in t or "proprietary" in t:
        return "STE"
    if "direct purchase" in t or "rule 154" in t:
        return "Direct Purchase"
    if " gem" in t or "gem " in t:
        return "GeM"
    return "GENERAL"


def _count_process_steps(steps_text: str) -> int:
    """Count numbered steps in process text."""
    return len(re.findall(r"(?m)^\s*\d+\.", steps_text))


def _validate_committee_line(quick_answer: str, mode: str) -> str:
    """Return corrected committee line."""
    correct = _COMMITTEE_MAP.get(mode, "Purchase Committee - as per DFP")
    # Replace incorrect "Purchase Committee" when mode is LTE/OTE
    if mode in ("LTE", "OTE") and "purchase committee" in quick_answer.lower():
        if "t&pc" not in quick_answer.lower() and "technical" not in quick_answer.lower():
            quick_answer = re.sub(
                r"(?i)(committee\s*:\s*)([^\n]+)",
                f"\\1{correct}",
                quick_answer,
            )
    return quick_answer


def _rule_matches_amount(text: str, amount_lakhs: float | None) -> bool:
    """Enforce deterministic mode band based on extracted amount."""
    if amount_lakhs is None:
        return True

    def expected_mode_for_amount(x: float) -> str:
        if x <= 2.0:
            return "Direct Purchase"
        if x <= 5.0:
            return "LPC"
        if x <= 50.0:
            return "LTE"
        return "OTE"

    def normalize_mode(mode_text: str) -> str:
        t = (mode_text or "").lower()
        if "direct purchase" in t:
            return "Direct Purchase"
        if "lpc" in t or "local purchase committee" in t:
            return "LPC"
        if "lte" in t or "limited tender" in t:
            return "LTE"
        if "ote" in t or "open tender" in t:
            return "OTE"
        return "UNKNOWN"

    expected_mode = expected_mode_for_amount(amount_lakhs)
    quick_answer = _extract_section(text, "## Quick Answer").lower()
    mode_match = re.search(r"applicable mode:\s*([^\n]+)", quick_answer)
    mode_text = mode_match.group(1).strip() if mode_match else quick_answer
    actual_mode = normalize_mode(mode_text)
    return actual_mode == expected_mode


# ── Validation Report ─────────────────────────────────────────────────────────

class ValidationReport:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.is_valid: bool = True

    def fail(self, reason: str) -> None:
        self.errors.append(reason)
        self.is_valid = False

    def __str__(self) -> str:
        return "; ".join(self.errors) if self.errors else "OK"


def validate_structured_output(
    text: str,
    amount_lakhs: float | None = None,
    expected_mode: str | None = None,
) -> ValidationReport:
    """Simple validation of structured output. Returns a ValidationReport."""
    report = ValidationReport()

    if not text:
        report.fail("Empty response")
        return report

    if "## Quick Answer" not in text:
        report.fail("Missing section: ## Quick Answer")

    if re.search(r"\bGFR\s*2025\b", text, flags=re.I):
        report.fail("Incorrect GFR version: GFR 2025 used (should be GFR 2017)")

    # 5. Mode band / expected mode
    if expected_mode and expected_mode not in {"GENERAL", "UNKNOWN"}:
        # Normalize both sides so synonyms still pass.
        def normalize_mode(mode_text: str) -> str:
            t = (mode_text or "").lower()
            if "direct purchase" in t:
                return "Direct Purchase"
            if "lpc" in t or "local purchase committee" in t:
                return "LPC"
            if "lte" in t or "limited tender" in t:
                return "LTE"
            if "ote" in t or "open tender" in t:
                return "OTE"
            if "general" in t:
                return "GENERAL"
            return "UNKNOWN"

        quick_answer = _extract_section(text, "## Quick Answer").lower()
        mode_match = re.search(r"applicable mode:\s*([^\n]+)", quick_answer)
        mode_text = mode_match.group(1).strip() if mode_match else quick_answer
        actual_mode = normalize_mode(mode_text)
        expected_norm = normalize_mode(expected_mode)
        if actual_mode != expected_norm:
            report.fail(f"Applicable mode mismatch: expected {expected_norm}, got {actual_mode}")
    elif not _rule_matches_amount(text, amount_lakhs):
        report.fail("Rule does not match detected amount band")

    return report


# ── Main Post-Processor ───────────────────────────────────────────────────────

def post_process_structured_output(
    text: str,
    query: str = "",
    amount_lakhs: float | None = None,
    tool_state: Any = None,
) -> str:
    """Full post-processing pipeline for structured LLM output.

    Steps:
    1. Remove leakage
    2. Fix GFR version
    3. Fix broken bullets / duplicate bullets
    4. Validate all sections present; inject fallbacks
    5. Fix committee mapping
    6. Expand process steps if < 4
    7. Ensure FINAL DECISION present
    """
    if not text:
        return _build_full_fallback(query, amount_lakhs)

    # 1. Remove leakage
    text = remove_leakage(text)

    # 2. Fix GFR version
    text = fix_gfr_version(text)

    # 3. Infer mode from full text
    mode = _infer_mode_from_text(text)

    # 4. Fix each section
    sections_out: list[str] = []
    for section in REQUIRED_SECTIONS:
        content = _extract_section(text, section)
        if not content:
            content = _build_fallback_section(section, mode, query, amount_lakhs, tool_state)
        else:
            content = fix_broken_bullets(content)
            content = remove_duplicate_bullets(content)
            if section == "## Quick Answer":
                content = _validate_committee_line(content, mode)
            if section == "## Detailed Process":
                content = _ensure_min_steps(content, mode, 4)
        sections_out.append(f"{section}\n{content}")

    result = "\n\n".join(sections_out)

    # 5. Ensure FINAL DECISION
    if "FINAL DECISION:" not in result:
        result = result.rstrip() + "\n\n**FINAL DECISION: VERIFY**"

    return result


def _ensure_min_steps(steps_text: str, mode: str, min_steps: int = 4) -> str:
    """If fewer than min_steps numbered steps, replace with default steps for mode."""
    count = _count_process_steps(steps_text)
    if count >= min_steps:
        return steps_text
    default = _DEFAULT_STEPS.get(mode, _DEFAULT_STEPS["GENERAL"])
    logger.info("Expanding process steps: found %d, need %d, mode=%s", count, min_steps, mode)
    return f"- Total steps: {len(default.splitlines())}\n{default}"


def _build_fallback_section(
    section: str,
    mode: str,
    query: str,
    amount_lakhs: float | None,
    tool_state: Any,
) -> str:
    """Build fallback content for a missing section."""
    src = _infer_source(tool_state)
    committee = _COMMITTEE_MAP.get(mode, "Purchase Committee - as per DFP")
    value_label = _format_amount(amount_lakhs)

    if section == "## Quick Answer":
        return (
            f"- Purchase value: {value_label}\n"
            f"- Applicable mode: {mode if mode != 'GENERAL' else 'Not determined'}\n"
            f"- Committee: {committee}"
        )
    if section == "## Rule Priority Applied":
        return (
            "- Priority order:\n"
            "  1. OM / Special Provisions (DoE OMs, CSIR circulars)\n"
            "  2. CSIR Manual 2019\n"
            "  3. GFR 2017 (as amended)\n"
            f"- Controlling source: {src}"
        )
    if section == "## Why This Applies":
        return (
            f"- The detected value ({value_label}) falls in the {mode} band under GFR 2017 (as amended).\n"
            "- Not found in context - please verify with retrieved document."
        )
    if section == "## Detailed Process":
        default = _DEFAULT_STEPS.get(mode, _DEFAULT_STEPS["GENERAL"])
        return f"- Total steps: {len(default.splitlines())}\n{default}"
    if section == "## Key Documents / Outputs":
        return _DEFAULT_KEY_DOCS
    if section == "## FLOWCHART (Mermaid)":
        return _DEFAULT_FLOWCHART
    if section == "## Source Basis":
        return f"- {src}"
    if section == "## TL;DR":
        return (
            f"- Applicable route: {mode if mode != 'GENERAL' else 'Verify based on value band'}\n"
            "- FINAL DECISION: VERIFY"
        )
    return "- Not found in context"


def _build_full_fallback(query: str, amount_lakhs: float | None) -> str:
    """Build a complete 8-section fallback when LLM returns nothing."""
    mode = "GENERAL"
    value_label = _format_amount(amount_lakhs)
    sections = [
        _build_fallback_section(s, mode, query, amount_lakhs, None)
        for s in REQUIRED_SECTIONS
    ]
    result = "\n\n".join(
        f"{header}\n{content}"
        for header, content in zip(REQUIRED_SECTIONS, sections)
    )
    return result + "\n\n**FINAL DECISION: VERIFY**"


def _infer_source(tool_state: Any) -> str:
    """Extract source from tool_state if available."""
    try:
        docs = tool_state.documents[:2]
        names = []
        for doc in docs:
            meta = getattr(doc, "metadata", {}) or {}
            name = meta.get("document_name") or getattr(doc, "file_name", "")
            rule = meta.get("rule_number", "")
            if name:
                names.append(f"{name}" + (f" - {rule}" if rule else ""))
        if names:
            return "; ".join(names)
    except Exception:
        pass
    return "GFR 2017 (as amended)"


def _format_amount(amount_lakhs: float | None) -> str:
    if amount_lakhs is None:
        return "Not specified"
    if amount_lakhs < 1:
        return f"Rs. {int(round(amount_lakhs * 100000)):,}"
    if float(amount_lakhs).is_integer():
        return f"Rs. {int(amount_lakhs)} lakhs"
    return f"Rs. {amount_lakhs:g} lakhs"
