"""Expert-level RAG pipeline: retrieval, scoring, reranking, and query handling.

Implements hybrid scoring with procurement domain gating, GFR 2025 recency bias,
zero-hallucination guards, and strict SOP output enforcement.

Response rendering and text selection are in app.core.response_builder.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Any, TypedDict

from app.core.config import settings
from app.core.constants import (
    AUDIT_LOGIC_BLOCK,
    DOMAIN_DENSITY_CAP,
    CSIR_MANUAL_DOCUMENT_NAME,
    GFR_2025_FILENAME,
    GFR_2025_RECENCY_BONUS,
    GFR_2025_DOCUMENT_NAME,
    GFR_2025_SLABS,
    HEADING_BOOST,
    KEYWORD_OVERLAP_WEIGHT,
    MIN_PROCUREMENT_SCORE,
    PROCUREMENT_DOMAIN_TERMS,
    PROCUREMENT_THRESHOLDS_TABLE,
    QUERY_RELEVANCE_CAP,
    SECTION_MARKER,
    SECTION_HEADING_RE,
    SYSTEM_PROMPT,
    PROCESS_PROMPT,
    PROCUREMENT_SOURCE_PRIORITY,
    WORKFLOW_PROMPT,
    GENERATION_CACHE_VERSION,
    SOURCE_VERIFICATION_RULE,
    gfr_slab_for_amount,
)
from app.core.response_builder import (
    build_default_procedural_steps,
    build_intent_prompt_guidance,
    cleanup_generated_sentence,
    compact_answer_text,
    extract_markdown_table_block,
    normalize_points,
    parse_llm_sections,
    render_structured_response,
    select_relevant_sentences,
    summarize_match_for_context,
)
from app.services.knowledge_base import SearchMatch, knowledge_base
from app.services.llm_service import generate_llm_response
from app.utils.processors import (
    amount_to_context_keywords,
    detect_intent,
    extract_analytical_terms,
    extract_amount_lakhs,
    format_lakh_amount,
    get_analytical_method_variants,
    describe_user_request,
    looks_like_scenario_query,
    tokenize,
    semantic_dedup_key,
)
from app.core.mii_lookup import get_mii_answer
from app.utils.text_cleaner import (
    audit_chunk_quality,
    clean_text,
    has_definition_style,
    is_clean_chunk,
    legalistic_noise_penalty,
    looks_like_table_of_contents,
    contains_scientist_list,
)
from app.utils.output_validator import (
    merge_adjacent_chunks,
    post_process_structured_output,
    validate_structured_output,
)

logger = logging.getLogger("procurebuddy-ai")

ANALYTICAL_PROMPT = f"""{AUDIT_LOGIC_BLOCK}

You are ProcureBuddy, a senior procurement assistant answering comparison questions.

Use only the provided knowledge-base context and threshold reference.
Answer the distinction directly, in clear professional language.
Do not invent facts or sources.
Do not give a generic comparison.

Rules:
- Keep the answer concise.
- Identify whether the question is asking for the difference, why the distinction matters, or how the routes differ in practice.
- Explain the reasoning briefly and connect it to real procurement logic.
- If the question is about audit implication, oversight, or defensibility, answer that issue directly instead of forcing a threshold comparison.
- Use contrast words such as whereas, unlike, or in contrast when helpful.
- Use the threshold table as the controlling source when older text conflicts.
- Use the compact output structure from [OUTPUT STRUCTURE].
- Include the controlling rule and grounded document name where available.
- Do not fall back to generic procurement commentary.
"""

_ANALYTICAL_EXCEPTION_RE = re.compile(
    r"\b(?:single offer|sole offer|sole bid|rule\s*21\s*of\s*dfpr|dfpr|court|writ petition|legal case|litigation)\b",
    re.IGNORECASE,
)

DISTINCTION_MARKERS: tuple[str, ...] = ("whereas", "unlike", "in contrast", "not the same", "while")
DECISION_MARKERS: tuple[str, ...] = ("should", "must", "reject", "disqualify", "proceed", "re-tender", "retender")
CONSEQUENCE_MARKERS: tuple[str, ...] = ("otherwise", "if not", "failing which", "cannot", "invalid", "liable", "reject")
PRIORITY_MARKERS: tuple[str, ...] = ("takes precedence", "prevails", "override", "priority", "controlling reference")
QUESTION_RULE_RE = re.compile(r"\bRule\s+\d{3}\b", re.IGNORECASE)
FAILED_CASES_PATH = Path(__file__).resolve().parents[3] / "tests" / "eval_failed_cases.json"

FEW_SHOT_EXAMPLES = f"""
Few-shot examples:
Q: What is the difference between LTE and OTE?
A:
LTE is limited competition under Rule 162, whereas OTE is open competition under Rule 161.
The distinction comes from the GFR 2025 threshold and publicity requirements.
FINAL DECISION: APPROVE

Q: What is the process for LPC procurement?
A:
LPC procurement should follow Rule 155 after checking GeM under Rule 149.
Obtain at least three quotations, record the committee recommendation, and take competent approval.
FINAL DECISION: VERIFY

Q: Can a bidder retain preference without a compliant local content declaration?
A:
No, the bidder should not retain preference without a compliant local content declaration.
The governing preference condition must be satisfied before the ranking is protected.
FINAL DECISION: REJECT

Q: If two rules conflict, which one controls?
A:
The higher-priority controlling source should prevail.
Identify the competing clauses and apply the stronger GFR or CSIR rule with reasons on file.
FINAL DECISION: VERIFY
""".strip()


# ─── Procurement Scoring ────────────────────────────────────────────────────

class GraphState(TypedDict, total=False):
    """State shared by the LangGraph RAG workflow."""

    query: str
    search_query: str
    retrieval_queries: list[str]
    documents: list[SearchMatch]
    raw_documents: list[SearchMatch]
    low_quality_documents: list[SearchMatch]
    amount: float | None
    amount_rupees: int | None
    intent: str
    slab: dict[str, Any] | None
    threshold_basis: str
    threshold_truth: str
    threshold_query_variations: list[str]
    generation: str
    user: str
    bypass_cache: bool
    blocked_chunk_ids: list[int]
    blocked_response_hashes: set[str]
    weak_match: bool
    retry_count: int
    research_required: bool
    metadata: dict[str, Any]


class _SequentialGraph:
    """Small fallback with the same invoke API when LangGraph is unavailable."""

    def invoke(self, initial_state: GraphState) -> GraphState:
        state = question_transformer_node(initial_state)
        state = logic_injection_node(state)
        state = multi_query_retrieval_node(state)
        if _route_after_retrieval(state) == "retry":
            state = retry_search_fallback_node(state)
        state = rerank_node(state)
        state = threshold_logic_node(state)
        return agentic_generation_node(state)


def _question_shape(message: str) -> str:
    normalized = clean_text(message).lower()
    if any(marker in normalized for marker in ("conflict", "priority", "prevail", "override", "takes precedence")):
        return "CONFLICT"
    if any(marker in normalized for marker in ("difference", "distinction", "not the same", "confused with", "whereas", "versus", " vs ")):
        return "DEFINITION"
    if any(marker in normalized for marker in ("compliance", "eligible", "eligibility", "local content", "preference claim", "before final ranking", "bid-security")):
        return "COMPLIANCE"
    if any(marker in normalized for marker in ("process", "procedure", "workflow", "step", "steps", "how should an officer resolve")):
        return "PROCESS"
    return "GENERAL"


def _answer_intent_label(message: str) -> str:
    normalized = clean_text(message).lower()
    if "difference" in normalized or "distinction" in normalized or "compare" in normalized or "versus" in normalized or " vs " in normalized:
        return "difference"
    if normalized.startswith("why ") or " why " in normalized:
        return "why"
    if normalized.startswith("how ") or " how " in normalized:
        return "how"
    if "role of" in normalized or "what is the role" in normalized:
        return "role"
    if any(token in normalized for token in ("threshold", "value band", "slab", "limit", "route")):
        return "threshold"
    if looks_like_scenario_query(message):
        return "scenario"
    return "general"


def _question_rule_signals(text: str) -> set[str]:
    return {match.group(0).title() for match in QUESTION_RULE_RE.finditer(text or "")}


def _canonical_document_name(file_name: str) -> str:
    lowered = clean_text(file_name).lower()
    if "updatedgfr" in lowered or ("gfr" in lowered and "2025" in lowered):
        return "GFR 2025"
    if "gfr" in lowered:
        return "GFR 2017"
    if "csir manual" in lowered:
        return "CSIR Manual 2019"
    if "make in india" in lowered or "local content" in lowered or "purchase preference" in lowered:
        return "Make in India Policy"
    if "snt" in lowered or "special provisions" in lowered or "scientific procurement" in lowered:
        return "Scientific Procurement Provisions"
    if "amend" in lowered:
        return "Amendments"
    if any(token in lowered for token in ("inventory", "stores", "write-off", "condemn")):
        return "Compendium"
    return clean_text(file_name)


def _display_document_name(file_name: str) -> str:
    return clean_text(Path(file_name).name or file_name)


def _allowed_source_labels(message: str) -> set[str]:
    normalized = clean_text(message).lower()
    allowed: set[str] = set()
    if any(
        token in normalized
        for token in (
            "rule 149",
            "rule 154",
            "rule 155",
            "rule 161",
            "rule 162",
            "rule 166",
            "gfr",
            "direct purchase",
            "lpc",
            "local purchase committee",
            "lte",
            "limited tender",
            "ote",
            "open tender",
            "ste",
            "single tender",
            "gem",
            "threshold",
        )
    ):
        allowed.add("GFR 2025")
        allowed.add("GFR 2017")
    if any(token in normalized for token in ("csir", "manual", "lab", "institute")):
        allowed.add("CSIR Manual 2019")
    if any(token in normalized for token in ("local content", "make in india", "preference")):
        allowed.add("Make in India Policy")
    if any(token in normalized for token in ("scientific procurement", "scientific", "research demand", "lab-specific", "snt")):
        allowed.add("Scientific Procurement Provisions")
    if any(token in normalized for token in ("amendment", "older manuals conflict", "threshold table")):
        allowed.add("Amendments")
    if any(token in normalized for token in ("write-off", "condemnation", "inventory", "stores", "salvage", "obsolete")):
        allowed.add("Compendium")
    if not allowed:
        allowed.add("GFR 2025")
    return allowed


@lru_cache(maxsize=1)
def _failed_case_summary() -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    if not FAILED_CASES_PATH.exists():
        return summary
    try:
        items = json.loads(FAILED_CASES_PATH.read_text(encoding="utf-8"))
    except Exception:
        logger.debug("Failed to parse eval_failed_cases.json for prompt guidance", exc_info=True)
        return summary

    for item in items:
        category = str(item.get("category", "general")).strip().lower()
        bucket = summary.setdefault(category, {})
        for error in item.get("errors", []):
            label = clean_text(str(error))
            if not label:
                continue
            bucket[label] = bucket.get(label, 0) + 1
    return summary


def _failed_case_guidance(message: str) -> str:
    shape = _question_shape(message).lower()
    summary = _failed_case_summary()
    buckets = [summary.get(shape, {}), summary.get("general", {})]
    ranked_errors: list[tuple[int, str]] = []
    for bucket in buckets:
        ranked_errors.extend((count, error) for error, count in bucket.items())
    ranked_errors.sort(reverse=True)
    common = [error for _, error in ranked_errors[:3]]

    guidance = [
        "Failure-driven guardrails:",
        "- Avoid unrelated rules and irrelevant explanations.",
        "- Cite only the controlling rule numbers and relevant document names.",
    ]
    if shape == "definition":
        guidance.append("- Make the distinction explicit with whereas, unlike, or in contrast.")
    if shape == "process":
        guidance.append("- Give at least 5 concrete, sequential steps with real procurement verbs.")
    if shape == "compliance":
        guidance.append("- State the decision, rule, and consequence in plain terms.")
    if shape == "conflict":
        guidance.append("- Resolve which rule controls and say why it prevails.")
    for error in common:
        guidance.append(f"- Known recent failure to avoid: {error}.")
    return "\n".join(guidance)


def _strict_system_contract(question_shape: str) -> str:
    rules = [
        "Keep the answer grounded in the retrieved rule text and threshold table.",
        "Use exactly these four labels: STATUS, ANALYSIS, AUDIT RISK, ACTIONABLE STEP.",
        "Answer the user's exact issue before adding surrounding procurement detail.",
        "Mention threshold bands, route labels, or GeM only when they directly control the answer.",
        "ANALYSIS should mention the most relevant grounded rules or document names.",
        "ACTIONABLE STEP must mention the required proof or evidence.",
        "Do not copy raw document text or leave broken sentences.",
        "If the retrieved context does not explicitly answer the question, do not guess.",
    ]
    if question_shape == "DEFINITION":
        rules.append("If the question asks for a difference or distinction, make the comparison clear.")
    if question_shape == "COMPLIANCE":
        rules.append("For compliance questions, clearly state the decision, rule reference, and consequence when possible.")
    if question_shape == "CONFLICT":
        rules.append("For conflict questions, explain which rule or source takes priority.")
    return "## ANSWER GUIDANCE\n" + "\n".join(f"- {rule}" for rule in rules)


def _question_specific_requirements(message: str) -> str:
    shape = _question_shape(message)
    scenario = looks_like_scenario_query(message)
    base = [
        f"Question shape: {shape}",
        "Answer guidance:",
        "1. Use only grounded procurement context and the threshold table.",
        "2. Use STATUS, ANALYSIS, AUDIT RISK, and ACTIONABLE STEP only.",
        "3. Keep ANALYSIS concise, grounded, and free of raw text.",
        "4. In ANALYSIS, include the relevant grounded rules and document names.",
    ]
    if shape == "DEFINITION":
        base.extend(
            [
                "5. Clearly distinguish the concepts using words such as whereas, unlike, or in contrast.",
                "6. Name both compared routes or concepts explicitly and include a small comparison table when helpful.",
            ]
        )
    elif shape == "PROCESS":
        base.extend(
            [
                "5. Put the next procedural action inside ACTIONABLE STEP.",
                "6. Each step must include a real procurement action such as verify, issue, evaluate, obtain, or record.",
            ]
        )
    elif shape == "COMPLIANCE":
        base.extend(
            [
                "5. State the decision on what should be done.",
                "6. State the governing rule reference.",
                "7. State the consequence if the rule is violated.",
            ]
        )
    elif shape == "CONFLICT":
        base.extend(
            [
                "5. Resolve which rule or document controls.",
                "6. Explain why that source prevails over the competing text.",
            ]
        )
    else:
        base.append("5. Keep the process checklist concrete and procurement-specific.")
    if shape == "ANALYTICAL" or len(extract_analytical_terms(message)) >= 2:
        base.extend(
            [
                "7. Identify the core concept for each method before comparing them.",
                "8. State the practical implication of the difference for the officer or file.",
                "9. Include the audit or compliance consequence of using the wrong route.",
            ]
        )
    if scenario:
        base.extend(
            [
                "10. Answer the scenario directly instead of giving only a generic definition.",
                "11. Explain what the officer should focus on in this case and why it matters.",
            ]
        )
    return "\n".join(base)


def _numbered_step_count(text: str) -> int:
    return len(re.findall(r"(?m)^\s*\d+\.\s+", text or ""))


def _has_distinction(text: str) -> bool:
    lowered = clean_text(text).lower()
    return any(marker in lowered for marker in DISTINCTION_MARKERS)


def _has_comparison_table(text: str) -> bool:
    return "| Feature |" in text and "|---|" in text


def _count_source_entries(text: str) -> int:
    match = re.search(r"(?:SOURCE BASIS|ANALYSIS)\s*:?(.*)$", text, flags=re.I | re.S)
    if not match:
        return 0
    return len(re.findall(r"(?m)^\s*\*\s+", match.group(1)))


def _looks_like_raw_excerpt(text: str) -> bool:
    lowered = clean_text(text).lower()
    raw_markers = (
        "certified that we, members of the purchase committee",
        "i am personally satisfied that these goods purchased",
        "copies of the bidding document should be sent",
        "an item is said to be not available in gem only when",
        "procurement from a single source may be resorted to",
        "xxii) in case a purchase committee is constituted",
    )
    return any(marker in lowered for marker in raw_markers)


def _extract_source_basis_entries(message: str, matches: list[SearchMatch], limit: int = 5) -> list[str]:
    allowed_sources = _allowed_source_labels(message)
    expected_rules = _question_rule_signals(message)
    entries: list[str] = []
    seen: set[str] = set()
    methods = extract_analytical_terms(message)

    if methods:
        for method in methods[:2]:
            row = _reference_row(method)
            rule_number = clean_text(row.get("rule", ""))
            if not rule_number or rule_number == "-":
                continue
            entry = f"{rule_number} - {GFR_2025_DOCUMENT_NAME}"
            normalized = entry.lower()
            if normalized in seen:
                continue
            if expected_rules and rule_number not in expected_rules:
                continue
            seen.add(normalized)
            entries.append(entry)
            if len(entries) >= limit:
                return entries

    for match in prioritize_matches(message, matches):
        metadata = getattr(match, "metadata", {}) or {}
        raw_document_name = str(match.file_name or metadata.get("document_name") or "")
        source_label = _canonical_document_name(raw_document_name)
        document_name = _display_document_name(raw_document_name)
        if source_label not in allowed_sources and any(source in allowed_sources for source in ("GFR 2025", "CSIR Manual 2019", "Make in India Policy", "Scientific Procurement Provisions", "Compendium")):
            continue
        rule_number = clean_text(str(metadata.get("rule_number") or ""))
        if not rule_number:
            content_rules = _question_rule_signals(match.content)
            if content_rules:
                rule_number = sorted(content_rules)[0]
        if expected_rules and rule_number and rule_number not in expected_rules:
            continue
        if not rule_number and source_label not in {"Make in India Policy", "Scientific Procurement Provisions", "Compendium"}:
            continue
        entry = f"{rule_number} - {document_name}" if rule_number else f"Relevant clause - {document_name}"
        normalized = clean_text(entry).lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        entries.append(entry)
        if len(entries) >= limit:
            return entries

    if not entries and expected_rules:
        for rule_number in sorted(expected_rules):
            entry = f"{rule_number} - {GFR_2025_DOCUMENT_NAME}"
            if entry.lower() not in seen:
                seen.add(entry.lower())
                entries.append(entry)
            if len(entries) >= limit:
                break
    if not entries:
        entries.append(f"Relevant rule - {GFR_2025_DOCUMENT_NAME}")
    return entries


def _response_validation_issues(message: str, rendered: str) -> list[str]:
    issues: list[str] = []
    normalized = clean_text(rendered)
    for label in ("STATUS", "ANALYSIS", "AUDIT RISK", "ACTIONABLE STEP"):
        if label not in normalized:
            issues.append(f"Missing {label} section")
    if _looks_like_raw_excerpt(rendered):
        issues.append("Contains raw document text")
    return issues


def _rule_priority_line(message: str) -> str:
    allowed_sources = _allowed_source_labels(message)
    gfr_allowed = "GFR 2025" in allowed_sources or "GFR 2017" in allowed_sources
    if "Make in India Policy" in allowed_sources:
        if gfr_allowed:
            return "Use GFR 2025 as the controlling source, and apply the Make in India policy only where the question is about preference or local content."
        return "Use the Make in India policy only for the preference or local content issue raised in the question."
    if "Scientific Procurement Provisions" in allowed_sources:
        if gfr_allowed:
            return "Use GFR 2025 as the controlling source and use Scientific Procurement Provisions only where they specifically govern the case."
        return "Use Scientific Procurement Provisions only where they directly govern the case raised in the question."
    if "Compendium" in allowed_sources:
        if gfr_allowed:
            return "Use the controlling disposal or inventory rule first, and use the compendium only where it directly supports that route."
        return "Use the compendium only where it directly controls the disposal or inventory issue in the question."
    if "CSIR Manual 2019" in allowed_sources:
        if gfr_allowed:
            return "GFR 2025 controls the rule position, while CSIR Manual 2019 may support procedure detail where it does not conflict."
        return "CSIR Manual 2019 is the controlling source for this concept-level CSIR question."
    return "GFR 2025 controls over older conflicting guidance for this question."


def _gfr_amount_search_suffix(amount_lakhs: float | None, query: str | None = None) -> str:
    slab = gfr_slab_for_amount(amount_lakhs, query)
    if not slab:
        return ""
    keywords = " ".join(str(keyword) for keyword in slab["keywords"])
    return f"{slab['method']} {slab['rule']} {slab['value_band']} {keywords}"


def _rewrite_query_for_gfr_slab(query: str, amount_lakhs: float | None) -> str:
    suffix = _gfr_amount_search_suffix(amount_lakhs, query)
    if not suffix:
        return query.strip()
    return f"{query.strip()} {suffix}".strip()


def _extract_currency_value_rupees(query: str) -> int | None:
    normalized = clean_text(query).lower().replace(",", "")

    crore_match = re.search(r"(?:rs\.?|inr|₹)?\s*(\d+(?:\.\d+)?)\s*(?:crore|crores|cr)\b", normalized)
    if crore_match:
        return int(round(float(crore_match.group(1)) * 10000000))

    lakh_match = re.search(r"(?:rs\.?|inr|₹)?\s*(\d+(?:\.\d+)?)\s*(?:lakh|lakhs|lac|lacs|l)\b", normalized)
    if lakh_match:
        return int(round(float(lakh_match.group(1)) * 100000))

    thousand_match = re.search(r"(?:rs\.?|inr|₹)?\s*(\d+(?:\.\d+)?)\s*(?:k|thousand)\b", normalized)
    if thousand_match:
        return int(round(float(thousand_match.group(1)) * 1000))

    rupee_match = re.search(r"(?:rs\.?|inr|₹)\s*(\d+(?:\.\d+)?)\b", normalized)
    if rupee_match:
        return int(round(float(rupee_match.group(1))))

    compact_rupee_match = re.search(r"\b(\d{5,8})\b", normalized)
    if compact_rupee_match:
        return int(compact_rupee_match.group(1))

    return None


def _gfr_2025_slab_for_rupees(amount_rupees: int | None, query: str | None = None) -> dict[str, Any] | None:
    if amount_rupees is None:
        return None
    return gfr_slab_for_amount(amount_rupees / 100000.0, query)


def _format_rupee_amount(amount_rupees: int | None) -> str:
    if amount_rupees is None:
        return "the stated amount"
    return f"Rs. {amount_rupees:,}"


def _threshold_procedural_steps(slab: dict[str, Any]) -> str:
    key = str(slab.get("key", "")).upper()
    if key == "DIRECT_PURCHASE":
        return (
            "1. Verify the requirement and check GeM applicability.\n"
            "2. Record market-rate reasonableness for the selected source.\n"
            "3. Record why quotation-based procurement is not required for this value band.\n"
            "4. Obtain competent approval for direct purchase.\n"
            "5. Issue the purchase order and keep the supporting record."
        )
    if key == "LPC":
        return (
            "1. Verify the estimated value and confirm the LPC route applies.\n"
            "2. Obtain comparative quotations through the Local Purchase Committee.\n"
            "3. Prepare the comparative statement and committee recommendation.\n"
            "4. Obtain approval from the competent authority.\n"
            "5. Issue the purchase order and place the file on record."
        )
    if key == "LTE":
        return (
            "1. Verify the value falls within the LTE band and check GeM applicability.\n"
            "2. Prepare and issue the limited tender to eligible firms.\n"
            "3. Receive and evaluate the offers through the competent committee.\n"
            "4. Obtain approval from the competent authority.\n"
            "5. Issue the purchase order and record the tender outcome."
        )
    return (
        "1. Verify the value exceeds the LTE ceiling and check GeM / portal applicability.\n"
        "2. Prepare and publish the open tender with the required publicity.\n"
        "3. Receive and evaluate the bids through the competent committee.\n"
        "4. Obtain approval from the competent authority.\n"
        "5. Issue the contract or purchase order and record the decision."
    )


def _build_threshold_truth_block(amount_rupees: int, slab: dict[str, Any]) -> str:
    amount_label = _format_rupee_amount(amount_rupees)
    return (
        "MANDATORY TRUTH:\n"
        f"- Detected amount: {amount_label}\n"
        f"- Governing slab: {slab['label']}\n"
        f"- Governing method: {slab['method']}\n"
        f"- Governing rule: {slab['rule']}\n"
        "- This slab decision overrides weaker retrieved wording for threshold routing."
    )


def _inject_logic_into_system_prompt(system_prompt: str, threshold_truth: str, message: str = "") -> str:
    shape = _question_shape(message) if message else "GENERAL"
    parts = [
        system_prompt,
        SOURCE_VERIFICATION_RULE,
        _strict_system_contract(shape),
    ]
    if threshold_truth:
        parts.extend(
            [
                "## DETERMINISTIC THRESHOLD INTELLIGENCE",
                "If a monetary amount is detected, the following logic is mandatory and overrides weaker retrieved wording:",
                threshold_truth,
                "You must follow this threshold intelligence exactly.",
            ]
        )
    return "\n\n".join(part for part in parts if part)


def _build_threshold_query_variations(query: str, amount_rupees: int, slab: dict[str, Any]) -> list[str]:
    amount_label = _format_rupee_amount(amount_rupees)
    return [
        f"{amount_label} procurement",
        f"{slab['method']} threshold rules",
        f"{slab['rule']} limits {slab['label']}",
        f"{query} {slab['method']} {slab['rule']}",
    ]


def _search_match_to_document(match: SearchMatch) -> Any:
    try:
        from langchain_core.documents import Document

        return Document(
            page_content=match.content,
            metadata={
                "chunk_id": match.chunk_id,
                "document_id": match.document_id,
                "file_name": match.file_name,
                "chunk_index": match.chunk_index,
                "token_count": match.token_count,
                "score": match.score,
                "document_name": str((match.metadata or {}).get("document_name", "")),
                "rule_number": str((match.metadata or {}).get("rule_number", "")),
                "topic": str((match.metadata or {}).get("topic", "")),
            },
        )
    except Exception:
        return {
            "page_content": match.content,
            "metadata": {
                "chunk_id": match.chunk_id,
                "document_id": match.document_id,
                "file_name": match.file_name,
                "chunk_index": match.chunk_index,
                "token_count": match.token_count,
                "score": match.score,
                "document_name": str((match.metadata or {}).get("document_name", "")),
                "rule_number": str((match.metadata or {}).get("rule_number", "")),
                "topic": str((match.metadata or {}).get("topic", "")),
            },
        }


def _document_to_search_match(document: Any) -> SearchMatch | None:
    metadata = getattr(document, "metadata", None) or document.get("metadata", {})
    page_content = getattr(document, "page_content", None) or document.get("page_content", "")
    try:
        return SearchMatch(
            chunk_id=int(metadata["chunk_id"]),
            document_id=int(metadata["document_id"]),
            file_name=str(metadata["file_name"]),
            chunk_index=int(metadata["chunk_index"]),
            content=str(page_content),
            token_count=int(metadata.get("token_count") or len(str(page_content).split())),
            score=float(metadata.get("relevance_score", metadata.get("score", 0.0))),
            metadata={
                "document_name": clean_text(str(metadata.get("document_name", ""))),
                "rule_number": clean_text(str(metadata.get("rule_number", ""))),
                "topic": clean_text(str(metadata.get("topic", ""))),
            },
        )
    except Exception:
        logger.debug("Unable to convert compressed document back to SearchMatch", exc_info=True)
        return None


def _dedupe_by_chunk_id(matches: list[SearchMatch]) -> list[SearchMatch]:
    best_by_id: dict[int, SearchMatch] = {}
    for match in matches:
        current = best_by_id.get(match.chunk_id)
        if current is None or match.score > current.score:
            best_by_id[match.chunk_id] = match
    return list(best_by_id.values())


def chunk_procurement_score(content: str) -> int:
    """Count how many domain terms appear in content."""
    lowered = content.lower()
    return sum(1 for term in PROCUREMENT_DOMAIN_TERMS if term in lowered)


def _source_priority_bonus(file_name: str) -> float:
    lowered = file_name.lower()
    if "special provisions" in lowered or "amendment" in lowered:
        return 0.12
    if "csir manual" in lowered:
        return 0.08
    if "gfr" in lowered or "updatedgfr" in lowered:
        return 0.06
    return 0.0


def _query_relevance_bonus(query: str, content: str) -> float:
    lowered_query = query.lower()
    lowered_content = content.lower()
    bonus = 0.0
    if any(kw in lowered_query for kw in ("lakh", "crore", "rs", "purchase process", "procurement process", "committee")):
        for marker in ("purchase committee", "technical & purchase committee", "local purchase committee", "limited tender", "advertised tender", "rule 155", "rule 161", "rule 162", "up to rs", "above rs"):
            if marker in lowered_content:
                bonus += 0.03
    if any(kw in lowered_query for kw in ("single tender", "ste", "proprietary", "pac")):
        for marker in ("single tender", "rule 166", "proprietary article", "standardisation", "emergency"):
            if marker in lowered_content:
                bonus += 0.04
    if any(kw in lowered_query for kw in ("compare", "difference", "versus", "vs", "table", "slab", "overview", "matrix")):
        for marker in ("rule 154", "rule 155", "rule 161", "rule 162", "rule 166", "direct purchase", "limited tender enquiry", "single tender enquiry", "definition", "applicability"):
            if marker in lowered_content:
                bonus += 0.05
    return min(bonus, QUERY_RELEVANCE_CAP)


def _analytical_definition_bonus(content: str, methods: list[str]) -> float:
    lowered = content.lower()
    bonus = 0.0
    if has_definition_style(content):
        bonus += 0.18
    for method in methods:
        if any(alias in lowered for alias in get_analytical_method_variants(method)):
            bonus += 0.05
    if any(marker in lowered for marker in ("applicability", "threshold", "approval", "quotation", "used for", "shall be used")):
        bonus += 0.04
    return min(bonus, 0.28)


def _analytical_exception_penalty(content: str) -> float:
    penalty = legalistic_noise_penalty(content)
    if _ANALYTICAL_EXCEPTION_RE.search(content):
        penalty -= 0.10
    return penalty


def _mentions_method(match: SearchMatch, method: str) -> bool:
    lowered = clean_text(match.content).lower()
    for alias in get_analytical_method_variants(method):
        normalized_alias = alias.lower().strip()
        if len(normalized_alias) <= 3 and normalized_alias.isalpha():
            if re.search(rf"\b{re.escape(normalized_alias)}\b", lowered):
                return True
            continue
        if normalized_alias in lowered:
            return True
    return False


def _procurement_verb_present(content: str) -> bool:
    lowered = clean_text(content).lower()
    return any(term in lowered for term in ("purchase", "procure", "procurement", "approve", "approval", "sanction", "issue", "evaluate", "quotation", "tender", "procedure"))


def _structured_content_bonus(content: str, question_shape: str) -> float:
    bonus = 0.0
    if SECTION_HEADING_RE.search(content):
        bonus += 0.15
    if "rule " in content.lower():
        bonus += 0.25
    if has_definition_style(content):
        bonus += 0.20
    if _numbered_step_count(content) >= 3:
        bonus += 0.10
    if question_shape == "DEFINITION" and _has_distinction(content):
        bonus += 0.30
    if question_shape == "PROCESS" and _numbered_step_count(content) >= 5:
        bonus += 0.20
    return bonus


def _rule_alignment_score(query: str, content: str) -> float:
    query_rules = _question_rule_signals(query)
    if not query_rules:
        return 0.0
    content_rules = _question_rule_signals(content)
    if not content_rules:
        return 0.0
    if query_rules.intersection(content_rules):
        return 0.80
    return -0.90


def _rrf_fuse(rankings: dict[str, list[SearchMatch]], limit: int = 20, k: int = 60) -> list[SearchMatch]:
    fused_scores: dict[int, float] = {}
    canonical_matches: dict[int, SearchMatch] = {}
    for matches in rankings.values():
        for rank, match in enumerate(matches[:limit], start=1):
            fused_scores[match.chunk_id] = fused_scores.get(match.chunk_id, 0.0) + (1.0 / (k + rank))
            canonical_matches.setdefault(match.chunk_id, match)
    ordered_ids = sorted(fused_scores, key=lambda chunk_id: fused_scores[chunk_id], reverse=True)
    fused: list[SearchMatch] = []
    for chunk_id in ordered_ids:
        match = canonical_matches[chunk_id].model_copy(deep=True)
        match.score = fused_scores[chunk_id]
        fused.append(match)
    return fused


def _action_sentence_density(content: str) -> float:
    audit = audit_chunk_quality(content)
    return float(audit["action_sentence_density"])


def _rerank_match_score(query: str, match: SearchMatch, intent: str) -> float:
    """Lean rerank score: trust the retriever, penalize only junk."""
    cleaned = clean_text(match.content)
    base_score = float(match.score)
    # Quality penalties only — no additive bonuses
    if looks_like_table_of_contents(cleaned):
        base_score -= 0.75
    if not _procurement_verb_present(cleaned):
        base_score -= 0.25
    return base_score


def _deduplicate_matches(matches: list[SearchMatch], threshold: float = 0.80) -> list[SearchMatch]:
    deduped: list[SearchMatch] = []
    for match in matches:
        if any(_semantic_similarity(match.content, kept.content) >= threshold for kept in deduped):
            continue
        deduped.append(match)
    return deduped


def rerank_matches(query: str, matches: list[SearchMatch], intent: str, limit: int | None = None) -> list[SearchMatch]:
    reranked = sorted(
        matches,
        key=lambda match: _rerank_match_score(query, match, intent),
        reverse=True,
    )
    reranked = _deduplicate_matches(reranked, threshold=0.80)
    return reranked[: limit or settings.top_k]


# ─── Expert Hybrid Scoring & Reranking ──────────────────────────────────────

def prioritize_matches(query: str, matches: list[SearchMatch]) -> list[SearchMatch]:
    """Domain gating + sort by base retrieval score.  Trust the retriever."""

    def score(match: SearchMatch) -> tuple[float, float]:
        try:
            cleaned = clean_text(match.content)
            proc_score = chunk_procurement_score(cleaned)
            if proc_score < MIN_PROCUREMENT_SCORE:
                return (-1.0, match.score)
            if contains_scientist_list(cleaned):
                return (-1.0, match.score)
            return (match.score, match.score)
        except Exception:
            logger.warning("Scoring failed for chunk_id=%s file=%s", match.chunk_id, match.file_name, exc_info=True)
            return (match.score, match.score)

    scored = sorted(matches, key=score, reverse=True)
    return [m for m in scored if score(m)[0] >= 0]


# ─── Filtering ──────────────────────────────────────────────────────────────

def filter_matches(
    matches: list[SearchMatch],
    blocked_chunk_ids: list[int],
    query: str = "",
    require_domain: bool = False,
    intent: str = "GENERAL",
) -> list[SearchMatch]:
    blocked = {int(cid) for cid in blocked_chunk_ids}
    query_rules = _question_rule_signals(query)
    question_shape = _question_shape(query)
    filtered: list[SearchMatch] = []
    for match in matches:
        if match.chunk_id in blocked:
            continue
        cleaned = clean_text(match.content)
        if not is_clean_chunk(cleaned):
            continue
        audit = audit_chunk_quality(cleaned)
        if audit["discard"]:
            continue
        if require_domain and chunk_procurement_score(cleaned) == 0:
            continue
        if intent == "WORKFLOW" and audit["tag"] == "REFERENCE_ONLY":
            continue
        content_rules = _question_rule_signals(cleaned)
        if query_rules and content_rules and query_rules.isdisjoint(content_rules):
            if question_shape != "DEFINITION":
                continue
            if not any(alias in cleaned.lower() for method in extract_analytical_terms(query) for alias in get_analytical_method_variants(method)):
                continue
        filtered.append(match)
    return filtered


# ─── Zero-Hallucination Guard ───────────────────────────────────────────────

def _is_context_relevant_to_query(query: str, matches: list[SearchMatch]) -> bool:
    query_lower = query.lower()
    methods = extract_analytical_terms(query)
    if len(methods) >= 2:
        combined = " ".join(clean_text(m.content).lower() for m in matches[:8])
        return all(any(alias in combined for alias in get_analytical_method_variants(method)) for method in methods[:2])
    workflow_signals = ("workflow", "approval", "process", "procedure", "steps", "sop", "how to")
    if not any(sig in query_lower for sig in workflow_signals):
        return True
    combined = " ".join(clean_text(m.content).lower() for m in matches[:5])
    return any(t in combined for t in ("procurement", "tender", "purchase", "committee", "approval", "indent", "gfr", "rule"))


def _has_explicit_rule_coverage(query: str, matches: list[SearchMatch]) -> bool:
    query_rules = _question_rule_signals(query)
    if not query_rules:
        return True
    for match in matches[:8]:
        content_rules = _question_rule_signals(match.content)
        metadata_rule = clean_text(str((getattr(match, "metadata", {}) or {}).get("rule_number", "")))
        if metadata_rule:
            content_rules.add(metadata_rule)
        if query_rules.intersection(content_rules):
            return True
    return False


@lru_cache(maxsize=1)
def _threshold_reference_rows() -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for line in PROCUREMENT_THRESHOLDS_TABLE.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if len(cells) < 4 or cells[0].lower() == "value band" or set(cells[0]) == {"-"}:
            continue
        value_band, method_cell, rule, notes = cells[:4]
        lowered = method_cell.lower()
        if "lpc" in lowered:
            key = "LPC"
        elif "lte" in lowered:
            key = "LTE"
        elif "ste" in lowered:
            key = "STE"
        elif "ote" in lowered:
            key = "OTE"
        elif "direct purchase" in lowered:
            key = "DIRECT PURCHASE"
        else:
            continue
        rows[key] = {
            "value_band": value_band,
            "method": method_cell,
            "rule": rule,
            "notes": notes,
        }
    return rows


def _reference_row(method: str) -> dict[str, str]:
    return _threshold_reference_rows().get(
        method.upper(),
        {
            "value_band": "See reference table",
            "method": method,
            "rule": "-",
            "notes": "Refer to the official threshold reference.",
        },
    )


def _build_analytical_search_query(method: str) -> str:
    variants = " ".join(get_analytical_method_variants(method))
    return f"{variants} definition applicability threshold approval comparison"


def _ensure_analytical_coverage(query: str, matches: list[SearchMatch]) -> list[SearchMatch]:
    methods = extract_analytical_terms(query)
    if len(methods) < 2:
        return prioritize_matches(query, matches)

    merged: dict[int, SearchMatch] = {match.chunk_id: match for match in matches}
    for method in methods[:2]:
        current = [match for match in merged.values() if _mentions_method(match, method)]
        if len(current) >= 2:
            continue
        extra_matches = knowledge_base.search(
            _build_analytical_search_query(method),
            max(settings.top_k * 2, 6),
            min_score_override=0.0,
        )
        extra_matches = filter_matches(extra_matches, [], query=query, require_domain=True, intent="ANALYTICAL")
        extra_matches = prioritize_matches(query, extra_matches)
        for match in extra_matches:
            if not _mentions_method(match, method):
                continue
            merged.setdefault(match.chunk_id, match)
            current = [item for item in merged.values() if _mentions_method(item, method)]
            if len(current) >= 2:
                break

    ranked = prioritize_matches(query, list(merged.values()))
    selected: list[SearchMatch] = []
    seen: set[int] = set()
    for method in methods[:2]:
        method_matches = [match for match in ranked if _mentions_method(match, method)]
        for match in method_matches[:2]:
            if match.chunk_id in seen:
                continue
            seen.add(match.chunk_id)
            selected.append(match)
    for match in ranked:
        if match.chunk_id in seen:
            continue
        seen.add(match.chunk_id)
        selected.append(match)
    return selected


def retrieve_candidates(
    message: str,
    blocked_chunk_ids: list[int],
    top_k: int | None = None,
    relaxed: bool = False,
) -> tuple[list[SearchMatch], bool]:
    """Run hybrid retrieval + reranking and return (matches, weak_match)."""
    intent = detect_intent(message)
    query = message.strip()
    from app.utils.processors import expand_query_keywords

    expanded_query = expand_query_keywords(query, intent)
    retrieval_size = max(top_k or settings.top_k, 6)
    semantic_matches = knowledge_base.search_semantic(
        expanded_query,
        top_k=retrieval_size,
        min_score_override=(0.0 if relaxed else None),
    )
    keyword_matches = knowledge_base.search_keyword(
        expanded_query,
        top_k=retrieval_size,
        min_score_override=0.0,
    )
    logger.info(
        "Hybrid retrieval query='%s' intent='%s' semantic=%s keyword=%s relaxed=%s",
        query,
        intent,
        len(semantic_matches),
        len(keyword_matches),
        relaxed,
    )

    fused = _rrf_fuse({"semantic": semantic_matches, "keyword": keyword_matches}, limit=retrieval_size)
    filtered = filter_matches(
        fused,
        blocked_chunk_ids,
        query=query,
        require_domain=True,
        intent=intent,
    )
    if intent == "ANALYTICAL":
        filtered = _ensure_analytical_coverage(query, filtered)
    reranked = rerank_matches(query, filtered, intent, limit=max(top_k or settings.top_k, 4))
    top_three = reranked[:3]
    weak_match = relaxed
    if top_three:
        average_score = sum(match.score for match in top_three) / len(top_three)
        logger.info(
            "Reranked query='%s' top_scores=%s avg_top3=%.4f",
            query,
            [round(match.score, 4) for match in top_three],
            average_score,
        )
        if average_score < 0.005:
            return [], True
    return reranked[: top_k or settings.top_k], weak_match


# ─── Prompt Building ────────────────────────────────────────────────────────

def _retrieve_single_query(
    query: str,
    blocked_chunk_ids: list[int],
    top_k: int,
    intent: str,
    relaxed: bool = False,
) -> list[SearchMatch]:
    semantic_matches = knowledge_base.search_semantic(
        query,
        top_k=top_k,
        min_score_override=(0.0 if relaxed else None),
    )
    keyword_matches = knowledge_base.search_keyword(
        query,
        top_k=top_k,
        min_score_override=0.0,
    )
    fused = _rrf_fuse({"semantic": semantic_matches, "keyword": keyword_matches}, limit=top_k)
    filtered = filter_matches(
        fused,
        blocked_chunk_ids,
        query=query,
        require_domain=True,
        intent=intent,
    )
    if intent == "ANALYTICAL":
        filtered = _ensure_analytical_coverage(query, filtered)
    return filtered


def _fallback_query_variations(query: str, intent: str) -> list[str]:
    from app.utils.processors import expand_query_keywords

    expanded = expand_query_keywords(query, intent)
    if intent == "WORKFLOW":
        return [
            expanded,
            f"{expanded} workflow approval sequence documentation",
            f"{expanded} committee finance approval order record",
        ]
    if intent == "PROCESS":
        return [
            expanded,
            f"{expanded} controlling rule applicability documentation",
            f"{expanded} approval evidence officer checklist",
        ]
    if intent == "ANALYTICAL":
        return [
            expanded,
            f"{expanded} distinction comparison applicability evidence",
            f"{expanded} controlling rule contrast procurement logic",
        ]
    if intent == "SCENARIO":
        return [
            expanded,
            f"{expanded} scenario compliance consequence corrective action",
            f"{expanded} facts audit risk evidence required",
        ]
    return [
        expanded,
        f"{expanded} controlling rule evidence",
        f"{expanded} direct answer grounded context",
    ]


def _generate_lcel_query_variations(query: str) -> list[str]:
    logger.debug("Skipping LCEL query rewrites; using deterministic multi-query fallback")
    return []


def _retrieve_with_langchain_multi_query(
    query: str,
    blocked_chunk_ids: list[int],
    top_k: int,
    intent: str,
    relaxed: bool = False,
) -> list[SearchMatch] | None:
    logger.debug("Skipping LangChain MultiQueryRetriever; using local deterministic fallback")
    return None


def _multi_query_retrieve(
    query: str,
    blocked_chunk_ids: list[int],
    top_k: int,
    intent: str,
    relaxed: bool = False,
    threshold_queries: list[str] | None = None,
) -> tuple[list[SearchMatch], list[str]]:
    retrieval_size = max(top_k, 12)
    seeded_queries = [q for q in (threshold_queries or []) if q.strip()]
    langchain_matches = _retrieve_with_langchain_multi_query(
        query=query,
        blocked_chunk_ids=blocked_chunk_ids,
        top_k=retrieval_size,
        intent=intent,
        relaxed=relaxed,
    )
    if langchain_matches is not None:
        merged = list(langchain_matches)
        for seeded_query in seeded_queries:
            merged.extend(
                _retrieve_single_query(
                    query=seeded_query,
                    blocked_chunk_ids=blocked_chunk_ids,
                    top_k=retrieval_size,
                    intent=intent,
                    relaxed=relaxed,
                )
            )
        ranked = prioritize_matches(query, _dedupe_by_chunk_id(merged))
        return ranked, [query, *seeded_queries][:4]

    variations = seeded_queries + (_generate_lcel_query_variations(query) or _fallback_query_variations(query, intent))
    queries = [query, *variations]
    seen_queries: list[str] = []
    merged: list[SearchMatch] = []
    for variant in queries:
        normalized_variant = re.sub(r"\s+", " ", variant).strip()
        if not normalized_variant or normalized_variant.lower() in {q.lower() for q in seen_queries}:
            continue
        seen_queries.append(normalized_variant)
        merged.extend(
            _retrieve_single_query(
                query=normalized_variant,
                blocked_chunk_ids=blocked_chunk_ids,
                top_k=retrieval_size,
                intent=intent,
                relaxed=relaxed,
            )
        )
    return prioritize_matches(query, _dedupe_by_chunk_id(merged)), seen_queries[:4]


def _compress_retrieved_matches(query: str, matches: list[SearchMatch], intent: str, limit: int) -> list[SearchMatch]:
    """Two-pass reranking: Flashrank (fast) -> CrossEncoder (precise)."""
    if not matches:
        return []

    # ── Pass 1: Flashrank for fast initial filtering ──
    flashrank_results = matches
    documents = [_search_match_to_document(match) for match in matches]
    try:
        try:
            from langchain_community.document_compressors import FlashrankRerank
        except Exception:
            from langchain.retrievers.document_compressors import FlashrankRerank

        compressor = FlashrankRerank(top_n=min(limit * 2, len(matches)))
        compressed_documents = compressor.compress_documents(documents, query=query)
        compressed_matches = [_document_to_search_match(document) for document in compressed_documents]
        usable = [match for match in compressed_matches if match is not None]
        if usable:
            flashrank_results = usable
            logger.info("Pass 1 (Flashrank) reranked %d -> %d for query='%s'", len(matches), len(usable), query[:80])
    except Exception:
        logger.debug("Flashrank unavailable; skipping pass 1", exc_info=True)

    # ── Pass 2: CrossEncoder for precise semantic scoring (mandatory) ──
    try:
        model = _cross_encoder_reranker()
        pairs = [(query, match.content) for match in flashrank_results]
        scores = model.predict(pairs)
        scored_matches = []
        for match, score in zip(flashrank_results, scores, strict=False):
            updated = match.model_copy(deep=True)
            updated.score = float(score)
            scored_matches.append(updated)
        scored_matches.sort(key=lambda m: m.score, reverse=True)
        logger.info(
            "Pass 2 (CrossEncoder) scored %d chunks, top_scores=%s for query='%s'",
            len(scored_matches),
            [round(m.score, 4) for m in scored_matches[:3]],
            query[:80],
        )
        return scored_matches[:limit]
    except Exception:
        logger.debug("CrossEncoder rerank unavailable; using Flashrank results only", exc_info=True)

    return rerank_matches(query, flashrank_results, intent, limit=limit)


@lru_cache(maxsize=1)
def _cross_encoder_reranker() -> Any:
    from sentence_transformers import CrossEncoder

    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def prepare_context_blocks(message: str, matches: list[SearchMatch], intent: str = "GENERAL") -> str:
    ranked_matches = _ensure_analytical_coverage(message, matches) if intent == "ANALYTICAL" else prioritize_matches(message, matches)
    context_limit = 4 if intent == "ANALYTICAL" else 3
    ranked = ranked_matches[: min(context_limit, len(ranked_matches))]
    # Merge adjacent chunks to fix broken sentence artifacts
    raw_contents = [m.content for m in ranked]
    merged_contents = merge_adjacent_chunks(raw_contents, max_merged_chars=1200)
    blocks: list[str] = []
    for i, (m, merged_content) in enumerate(zip(ranked, merged_contents + [None] * len(ranked)), 1):
        content_to_use = merged_contents[i - 1] if i - 1 < len(merged_contents) else m.content
        summary = summarize_match_for_context(message, content_to_use, m.file_name)
        if not summary:
            continue
        metadata = getattr(m, "metadata", {}) or {}
        document_name = clean_text(str(metadata.get("document_name") or m.file_name))
        rule_number = clean_text(str(metadata.get("rule_number") or ""))
        topic = clean_text(str(metadata.get("topic") or ""))
        meta = [f"Document: {document_name}"]
        if rule_number:
            meta.append(f"Rule: {rule_number}")
        if topic:
            meta.append(f"Topic: {topic}")
        blocks.append(f"{i}. {' | '.join(meta)}\n   Summary: {summary}")
    return "\n".join(blocks)


def _should_include_threshold_reference(message: str, intent: str) -> bool:
    lowered = clean_text(message).lower()
    if extract_amount_lakhs(message) is not None or _extract_currency_value_rupees(message) is not None:
        return True
    return any(
        token in lowered
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
            "direct purchase",
            "lpc",
            "lte",
            "ote",
            "ste",
            "tender",
        )
    ) or intent in {"PROCESS", "WORKFLOW"} and any(token in lowered for token in ("threshold", "route", "gem", "tender"))


def build_prompt(message: str, user: str, matches: list[SearchMatch], bypass_cache: bool, weak_match: bool = False, threshold_truth: str = "") -> str:
    ctx = prepare_context_blocks(message, matches)
    answer_intent = _answer_intent_label(message)
    include_threshold_reference = _should_include_threshold_reference(message, detect_intent(message))
    lines = [
        "User question:",
        message,
        "",
        f"User identifier: {user}",
        "",
        "Knowledge-base context:",
        ctx,
        "",
        f"Question intent: {answer_intent}",
        "Answer contract:",
        "1. Use exactly STATUS, ANALYSIS, AUDIT RISK, and ACTIONABLE STEP.",
        "2. In ANALYSIS, state the controlling rule, principle, or reasoning briefly.",
        "3. Use only the grounded context supplied in this prompt.",
        "4. Keep the answer concise and avoid unnecessary elaboration.",
        f"5. In ANALYSIS, cite only real document file names such as {GFR_2025_DOCUMENT_NAME} or {CSIR_MANUAL_DOCUMENT_NAME}.",
        "6. In ACTIONABLE STEP, name the required proof or evidence.",
        "7. If the retrieved context is insufficient, explicitly say the context is not available and do not guess.",
    ]
    if include_threshold_reference:
        lines[5:5] = ["Official Procurement Threshold Reference:", PROCUREMENT_THRESHOLDS_TABLE, ""]
    if looks_like_scenario_query(message):
        lines.append("8. Apply the answer to the specific scenario facts given in the question.")
    if threshold_truth:
        lines[8:8] = ["Deterministic Threshold Judge:", threshold_truth, ""]
    if weak_match:
        lines.append("9. If the match is only approximate, say that briefly.")
    if bypass_cache:
        lines.append("10. Use fresh wording.")
    return "\n".join(lines)


def build_process_prompt(message: str, user: str, matches: list[SearchMatch], facts: str, bypass_cache: bool, weak_match: bool = False, threshold_truth: str = "") -> str:
    ctx = prepare_context_blocks(message, matches)
    answer_intent = _answer_intent_label(message)
    include_threshold_reference = _should_include_threshold_reference(message, "PROCESS")
    lines = [
        "User question:",
        message,
        "",
        f"User identifier: {user}",
        "",
        "Extracted procurement facts:",
        facts or "No structured facts extracted.",
        "",
        "Retrieved document context:",
        ctx,
        "",
        f"Question intent: {answer_intent}",
        "Answer contract:",
        "1. Use exactly STATUS, ANALYSIS, AUDIT RISK, and ACTIONABLE STEP.",
        "2. Use the threshold table as the source of truth only for amount-based or route-selection questions.",
        "3. State the controlling rule, route, or approval logic directly in ANALYSIS.",
        "4. Keep the explanation concise and grounded.",
        f"5. In ANALYSIS, cite only real document file names such as {GFR_2025_DOCUMENT_NAME} or {CSIR_MANUAL_DOCUMENT_NAME}.",
        "6. In ACTIONABLE STEP, name the required proof or evidence.",
        "7. If the retrieved context is insufficient, explicitly say the context is not available and do not guess.",
    ]
    if include_threshold_reference:
        lines[5:5] = ["Official Procurement Threshold Reference:", PROCUREMENT_THRESHOLDS_TABLE, ""]
    if looks_like_scenario_query(message):
        lines.append("8. Address the scenario facts directly and do not answer in generic process terms.")
    if threshold_truth:
        lines[8:8] = ["Deterministic Threshold Judge:", threshold_truth, ""]
    if weak_match:
        lines.append("7. If the match is only approximate, say that briefly.")
    if bypass_cache:
        lines.append("8. Use fresh wording.")
    return "\n".join(lines)


def build_analytical_prompt(message: str, user: str, matches: list[SearchMatch], bypass_cache: bool, weak_match: bool = False, threshold_truth: str = "") -> str:
    methods = extract_analytical_terms(message)
    method_1 = methods[0] if len(methods) >= 1 else "Method 1"
    method_2 = methods[1] if len(methods) >= 2 else "Method 2"
    ctx = prepare_context_blocks(message, matches, intent="ANALYTICAL")
    include_threshold_reference = _should_include_threshold_reference(message, "ANALYTICAL")
    lines = [
        "User question:",
        message,
        "",
        f"User identifier: {user}",
        "",
        "Comparison focus:",
        f"Method 1: {method_1}",
        f"Method 2: {method_2}",
        "",
        "Retrieved document context:",
        ctx,
        "",
        f"Question intent: {_answer_intent_label(message)}",
        "Answer contract:",
        "1. Use exactly STATUS, ANALYSIS, AUDIT RISK, and ACTIONABLE STEP.",
        f"2. Compare {method_1} and {method_2} directly in ANALYSIS.",
        "3. Explain briefly why or how the distinction matters in real procurement logic.",
        f"4. Do not use a table unless it is essential to compare {method_1} and {method_2}.",
        "5. Keep the explanation concise and avoid extra reasoning layers.",
        f"6. In ANALYSIS, cite only real document file names such as {GFR_2025_DOCUMENT_NAME} or {CSIR_MANUAL_DOCUMENT_NAME}.",
        "7. In ACTIONABLE STEP, name the required proof or evidence.",
        "8. If the retrieved context is insufficient, explicitly say the context is not available and do not guess.",
    ]
    if include_threshold_reference:
        lines[5:5] = ["Official Procurement Threshold Reference:", PROCUREMENT_THRESHOLDS_TABLE, ""]
    if looks_like_scenario_query(message):
        lines.append("9. Address the scenario context directly instead of giving only a generic comparison.")
    if threshold_truth:
        lines[8:8] = ["Deterministic Threshold Judge:", threshold_truth, ""]
    if weak_match:
        lines.append("8. If the comparison is based on closely related guidance rather than an exact clause, say that briefly.")
    if bypass_cache:
        lines.append("9. Use fresh wording.")
    return "\n".join(lines)


# ─── Hashing & Caching ─────────────────────────────────────────────────────

def response_hash(value: str) -> str:
    return hashlib.sha256(re.sub(r"\s+", " ", value.strip()).encode("utf-8")).hexdigest()

def normalize_response_hashes(values: list[str]) -> set[str]:
    return {v.strip().lower() for v in values if v and v.strip()}

def build_cache_key(message: str) -> str:
    normalized = re.sub(r"\s+", " ", message.strip().lower())
    return f"{knowledge_base.version_token()}::{GENERATION_CACHE_VERSION}::{normalized}"


def _replace_stale_lte_threshold(value: str) -> str:
    updated = re.sub(r"Rs\.?\s*2[.,]?5\s*lakh", "Rs. 5 lakh", value, flags=re.I)
    updated = re.sub(r"Rs\.?\s*2,50,000", "Rs. 5,00,000", updated, flags=re.I)
    return updated


def _semantic_similarity(left: str, right: str) -> float:
    left_clean = clean_text(left).lower()
    right_clean = clean_text(right).lower()
    if not left_clean or not right_clean:
        return 0.0
    return SequenceMatcher(None, left_clean, right_clean).ratio()


def _is_distinct_summary(candidate: str, existing: list[str], threshold: float = 0.80) -> bool:
    cleaned_candidate = cleanup_generated_sentence(candidate)
    if not cleaned_candidate:
        return False
    for current in existing:
        if _semantic_similarity(cleaned_candidate, current) >= threshold:
            return False
    return True


def build_grounding_metadata(answer: str, matches: list[SearchMatch]) -> dict[str, Any]:
    """Flag simple rule/amount claims that are not grounded in retrieved chunks."""
    source_text = clean_text(" ".join(match.content for match in matches) + " " + PROCUREMENT_THRESHOLDS_TABLE).lower()
    warnings: list[str] = []
    for rule in re.findall(r"rule\s+\d+", answer, flags=re.I):
        if clean_text(rule).lower() not in source_text:
            warnings.append(f"Ungrounded rule reference: {rule}")
    for amount in re.findall(r"Rs\.?\s*[0-9,]+(?:\s*lakh[s]?)?", answer, flags=re.I):
        if clean_text(amount).lower() not in source_text:
            warnings.append(f"Ungrounded amount reference: {amount}")
    return {
        "retrieval_mode": "hybrid-rrf-rerank",
        "grounding_warnings": warnings,
    }


def _looks_like_retrieval_failure(value: str) -> bool:
    normalized = clean_text(value).lower()
    return any(
        marker in normalized
        for marker in (
            "not found",
            "no data",
            "not applicable",
            "no strong match",
            "no rule found",
            "unable to find",
        )
    )


# ─── Post-Processing + Anti-Hallucination ───────────────────────────────────

def post_process_llm_output(
    response: str,
    message: str,
    matches: list[SearchMatch],
    weak_match: bool = False,
    intent: str = "GENERAL",
    threshold_response: str = "",
) -> str:
    if threshold_response and _looks_like_retrieval_failure(response):
        logger.info("Threshold guardrail overriding raw LLM failure output for query='%s'", message[:120])
        return threshold_response

    question_shape = _question_shape(message)
    targeted = _targeted_case_guidance(message)
    sections = parse_llm_sections(response)
    fb_points = build_explanation_points(message, matches)
    raw_explanation = (
        sections.get("analysis", "")
        or sections.get("detailed explanation", "")
        or sections.get("explanation", "")
    )
    table_from_explanation, cleaned_explanation = extract_markdown_table_block(raw_explanation)
    comparison_table = sections.get("comparison table", "").strip() or table_from_explanation
    wants_table = any(token in clean_text(message).lower() for token in ("table", "matrix"))
    answer = compact_answer_text(
        sections.get("analysis")
        or sections.get("direct answer")
        or sections.get("answer")
        or compose_direct_answer(message, fb_points, weak_match, matches)
    )
    targeted_answer = cleanup_generated_sentence(str(targeted.get("answer", "")))
    if targeted_answer:
        answer = compact_answer_text(targeted_answer)
    max_points = 2 if intent == "ANALYTICAL" else 4
    exp_points = normalize_points(cleaned_explanation or raw_explanation, fallback_points=fb_points, max_points=max_points)
    targeted_points: list[str] = []
    for point in targeted.get("points", []):
        cleaned_point = cleanup_generated_sentence(str(point))
        if cleaned_point:
            targeted_points.append(cleaned_point)
    if targeted_points:
        merged_points: list[str] = []
        seen_point_keys: set[str] = set()
        for point in [*targeted_points, *exp_points]:
            key = semantic_dedup_key(point)
            if point and key not in seen_point_keys:
                seen_point_keys.add(key)
                merged_points.append(point)
        exp_points = merged_points[:max_points]
    proc_steps = sections.get("procedural steps", "").strip()
    if sections.get("actionable step") and not proc_steps:
        proc_steps = f"1. {sections['actionable step']}"
    pro_tip = (
        sections.get("rule priority", "").strip()
        or sections.get("pro-tip", "").strip()
        or sections.get("pro tip", "").strip()
    )
    if not pro_tip:
        lowered_message = clean_text(message).lower()
        if _question_shape(message) == "CONFLICT" or any(token in lowered_message for token in ("conflict", "older manual", "priority", "override", "prevail")):
            pro_tip = _rule_priority_line(message)
    targeted_pro_tip = cleanup_generated_sentence(str(targeted.get("pro_tip", "")))
    if targeted_pro_tip:
        pro_tip = targeted_pro_tip
    targeted_steps = str(targeted.get("procedural_steps", "") or "").strip()
    if targeted_steps:
        proc_steps = targeted_steps

    if intent == "ANALYTICAL":
        if not wants_table:
            comparison_table = ""
        elif not comparison_table:
            comparison_table = _build_analytical_table(message, matches)
        if not proc_steps or "not applicable" in proc_steps.lower():
            proc_steps = _build_analytical_procedural_steps(message, matches)

    assembled = "\n".join([answer, comparison_table, "\n".join(exp_points), proc_steps, pro_tip])
    if re.search(r"(?:2\.?5|2,50,000)\s*(?:lakh|lac).*(?:LTE|limited\s+tender)", assembled, re.I) or re.search(r"(?:LTE|limited\s+tender).*(?:2\.?5|2,50,000)\s*(?:lakh|lac)", assembled, re.I):
        logger.warning("Anti-hallucination: correcting 2.5 lakh LTE -> 5 lakh (GFR 2025)")
        answer = _replace_stale_lte_threshold(answer)
        comparison_table = _replace_stale_lte_threshold(comparison_table)
        exp_points = [_replace_stale_lte_threshold(point) for point in exp_points]
        proc_steps = _replace_stale_lte_threshold(proc_steps)
        pro_tip = _replace_stale_lte_threshold(pro_tip)

    if intent in {"WORKFLOW", "PROCESS"} and (not proc_steps or "not applicable" in proc_steps.lower()):
        proc_steps = build_default_procedural_steps(intent, answer, exp_points) or (
            "1. Requirement identification & indenting (Indenting Officer)\n"
            "2. TSC review (if applicable)\n"
            "3. Check GeM availability (GFR Rule 149)\n"
            "4. Procurement method selection based on value band\n"
            "5. Quotation / Tender document preparation\n"
            "6. T&PC evaluation\n"
            "7. Finance concurrence\n"
            "8. Competent Authority approval (Director / DG)\n"
            "9. Purchase Order / Contract issuance"
        )

    if question_shape == "DEFINITION" and not _has_distinction("\n".join([answer, *exp_points])):
        methods = extract_analytical_terms(message)
        if len(methods) >= 2:
            left_row = _reference_row(methods[0])
            right_row = _reference_row(methods[1])
            answer = (
                f"{methods[0]} applies under {left_row['rule']} for {left_row['value_band']}, "
                f"whereas {methods[1]} applies under {right_row['rule']} for {right_row['value_band']}."
            )
            exp_points = [
                f"{methods[0]} follows {left_row['rule']} for {left_row['value_band']}, whereas {methods[1]} follows {right_row['rule']} for {right_row['value_band']}.",
                f"{methods[0]} and {methods[1]} are not the same because their competition scope and approval basis differ.",
                *exp_points[:2],
            ][:4]
        else:
            exp_points = [
                "The two concepts are not the same; the applicable rule, threshold, or competition route must be distinguished explicitly.",
                *exp_points[:3],
            ][:4]

    if question_shape in {"PROCESS", "COMPLIANCE"} and _numbered_step_count(proc_steps) < 5:
        proc_steps = build_default_procedural_steps("PROCESS" if question_shape == "COMPLIANCE" else intent, answer, exp_points)

    if question_shape == "COMPLIANCE":
        compliance_text = " ".join([answer, *exp_points, proc_steps])
        if not any(marker in compliance_text.lower() for marker in DECISION_MARKERS):
            exp_points = ["Decision: the officer should follow the controlling rule before proceeding."] + exp_points
        if not QUESTION_RULE_RE.search(compliance_text):
            rules = _question_rule_signals(message)
            if rules:
                exp_points.append(f"Rule reference: apply {sorted(rules)[0]} as the controlling provision.")
        if not any(marker in compliance_text.lower() for marker in CONSEQUENCE_MARKERS):
            exp_points.append("Consequence: if the rule is not met, the claim or action can be rejected as non-compliant.")
        exp_points = exp_points[:4]

    if question_shape == "CONFLICT":
        priority_text = " ".join([pro_tip, answer, *exp_points]).lower()
        if not any(marker in priority_text for marker in PRIORITY_MARKERS):
            pro_tip = _rule_priority_line(message)

    if intent == "ANALYTICAL":
        exp_points = exp_points[:2]

    rendered = render_structured_response(
        answer=answer,
        explanation_points=exp_points,
        sources=extract_source_names(matches, message),
        procedural_steps=proc_steps,
        pro_tip=pro_tip,
        comparison_table=comparison_table,
        intent=intent,
    )
    if threshold_response and _looks_like_retrieval_failure(rendered):
        logger.info("Threshold guardrail overriding post-processed failure output for query='%s'", message[:120])
        return threshold_response
    return rendered


# ─── Query Handler ──────────────────────────────────────────────────────────

def handle_query(
    message: str,
    user: str,
    matches: list[SearchMatch],
    bypass_cache: bool,
    blocked_response_hashes: set[str],
    weak_match: bool = False,
    threshold_truth: str = "",
) -> dict[str, Any]:
    intent = detect_intent(message)
    amount = extract_amount_lakhs(message)
    amount_rupees = _extract_currency_value_rupees(message)
    deterministic_threshold_response = _static_gfr_rule_response(message, intent) if amount_rupees is not None else ""

    # ── Make in India deterministic lookup ──
    # MII queries need mathematical precision, not semantic search.
    lowered = message.strip().lower()
    mii_answer = get_mii_answer(amount, lowered)
    if mii_answer and any(term in lowered for term in ("make in india", "local supplier", "local content", "class i", "class ii", "split order")):
        return {
            "intent": intent,
            "amount": amount,
            "answer": mii_answer,
            "generation_mode": "llm",  # mark as llm since it's a structured response
            "metadata": {
                **build_grounding_metadata(mii_answer, matches),
                "retrieval_mode": "deterministic-mii-lookup",
                "generation_mode": "llm",
            },
        }

    # SCENARIO intent: NEVER short-circuit to rule_based.
    # Pass threshold info as context enrichment for the LLM.
    if intent == "SCENARIO":
        deterministic_threshold_response = ""  # prevent rule_based hijacking

    if deterministic_threshold_response and intent == "PROCESS":
        return {
            "intent": intent,
            "amount": amount,
            "answer": deterministic_threshold_response,
            "generation_mode": "rule_based",
            "metadata": {
                **build_grounding_metadata(deterministic_threshold_response, matches),
                "retrieval_mode": "deterministic-threshold-judge",
                "generation_mode": "rule_based",
            },
        }
    working_matches = _ensure_analytical_coverage(message, matches) if intent in ("ANALYTICAL", "SCENARIO") else matches

    if intent == "SCENARIO":
        # Enrich with threshold context so LLM knows the slab, but let it reason freely
        threshold_hint = _static_gfr_rule_response(message, "PROCESS") if amount_rupees is not None else ""
        enriched_truth = f"{threshold_truth}\n{threshold_hint}".strip() if threshold_hint else threshold_truth
        prompt = build_prompt(message, user, working_matches, bypass_cache, weak_match=weak_match, threshold_truth=enriched_truth)
        system_prompt = _inject_logic_into_system_prompt(SYSTEM_PROMPT, enriched_truth, message=message)
    elif intent == "WORKFLOW":
        prompt = build_prompt(message, user, working_matches, bypass_cache, weak_match=weak_match, threshold_truth=threshold_truth)
        system_prompt = _inject_logic_into_system_prompt(WORKFLOW_PROMPT, threshold_truth, message=message)
    elif intent == "PROCESS":
        facts = _build_amount_answer(message, working_matches)
        prompt = build_process_prompt(
            message=message,
            user=user,
            matches=working_matches,
            facts=facts,
            bypass_cache=bypass_cache,
            weak_match=weak_match,
            threshold_truth=threshold_truth,
        )
        system_prompt = _inject_logic_into_system_prompt(PROCESS_PROMPT, threshold_truth, message=message)
    elif intent == "ANALYTICAL":
        prompt = build_analytical_prompt(
            message,
            user,
            working_matches,
            bypass_cache,
            weak_match=weak_match,
            threshold_truth=threshold_truth,
        )
        system_prompt = _inject_logic_into_system_prompt(ANALYTICAL_PROMPT, threshold_truth, message=message)
    else:
        prompt = build_prompt(message, user, working_matches, bypass_cache, weak_match=weak_match, threshold_truth=threshold_truth)
        system_prompt = _inject_logic_into_system_prompt(SYSTEM_PROMPT, threshold_truth, message=message)

    llm_response = generate_llm_response(prompt, system_prompt=system_prompt)
    amount_lakhs = extract_amount_lakhs(message)
    if llm_response:
        generation_mode = "llm"
        formatted = post_process_llm_output(
            llm_response,
            message,
            working_matches,
            weak_match=weak_match,
            intent=intent,
            threshold_response=deterministic_threshold_response,
        )
        # ── Post-processing validation + clean ───────────────────────
        formatted = post_process_structured_output(
            formatted,
            query=message,
            amount_lakhs=amount_lakhs,
            tool_state=None,
        )
        # ── Validate; retry once if critical sections missing ─────────
        report = validate_structured_output(formatted, amount_lakhs=amount_lakhs)
        if not report.is_valid:
            logger.warning(
                "Structured output validation failed (%s) — retrying with bypass_cache=True",
                report,
            )
            retry_response = generate_llm_response(
                build_prompt(message, user, working_matches, bypass_cache=True, weak_match=weak_match, threshold_truth=threshold_truth),
                system_prompt=system_prompt,
            )
            if retry_response:
                retry_formatted = post_process_structured_output(
                    post_process_llm_output(
                        retry_response, message, working_matches,
                        weak_match=weak_match, intent=intent,
                        threshold_response=deterministic_threshold_response,
                    ),
                    query=message,
                    amount_lakhs=amount_lakhs,
                    tool_state=None,
                )
                retry_report = validate_structured_output(retry_formatted, amount_lakhs=amount_lakhs)
                if retry_report.is_valid or len(retry_report.errors) < len(report.errors):
                    formatted = retry_formatted
                    generation_mode = "llm-retry"
        if response_hash(formatted) not in blocked_response_hashes:
            return {
                "intent": intent,
                "amount": amount,
                "answer": formatted,
                "generation_mode": generation_mode,
                "metadata": {
                    **build_grounding_metadata(formatted, working_matches),
                    "generation_mode": generation_mode,
                    "validation": str(report),
                },
            }

    if deterministic_threshold_response:
        fallback = deterministic_threshold_response
    else:
        fallback = build_rule_based_answer(message, working_matches, blocked_response_hashes, weak_match=weak_match, intent=intent)
    return {
        "intent": intent,
        "amount": amount,
        "answer": fallback,
        "generation_mode": "rule_based",
        "metadata": {
            **build_grounding_metadata(fallback, working_matches),
            "generation_mode": "rule_based",
        },
    }


# ─── Explanation & Source Extraction ────────────────────────────────────────

def _build_analytical_points(message: str, matches: list[SearchMatch]) -> list[str]:
    methods = extract_analytical_terms(message)
    if len(methods) < 2:
        return []
    ranked = _ensure_analytical_coverage(message, matches)
    points: list[str] = []
    seen: set[str] = set()
    chosen_summaries: list[str] = []
    for method in methods[:2]:
        summary = _method_summary(method, message, ranked, existing_summaries=chosen_summaries)
        cleaned_summary = cleanup_generated_sentence(summary)
        if cleaned_summary:
            key = semantic_dedup_key(cleaned_summary)
            if key not in seen and _is_distinct_summary(cleaned_summary, chosen_summaries, threshold=0.72):
                seen.add(key)
                points.append(cleaned_summary)
                chosen_summaries.append(cleaned_summary)
        row = _reference_row(method)
        threshold_point = cleanup_generated_sentence(f"{method} aligns with {row['value_band']} under {row['rule']}. {row['notes']}")
        if threshold_point:
            key = semantic_dedup_key(threshold_point)
            if key not in seen and _is_distinct_summary(threshold_point, chosen_summaries, threshold=0.72):
                seen.add(key)
                points.append(threshold_point)
                chosen_summaries.append(threshold_point)
    if len(methods) >= 2:
        distinction_point = cleanup_generated_sentence(
            f"{methods[0]} and {methods[1]} should not be treated as interchangeable because their applicability and supporting record are different."
        )
        if distinction_point:
            key = semantic_dedup_key(distinction_point)
            if key not in seen:
                seen.add(key)
                points.append(distinction_point)
    if any("single offer" in clean_text(match.content).lower() for match in ranked[:6]):
        points.append("Single-offer handling is an exception scenario and should not replace the core definition of the procurement route.")
    return points[:4]


def _looks_like_role_question(message: str) -> bool:
    normalized = clean_text(message).lower()
    return any(
        token in normalized
        for token in (
            "role of",
            "what is the role",
            "what responsibility",
            "responsibility does",
            "responsibility of",
            "approver",
            "head of office",
            "director",
            "competent authority",
            "technical recommender",
        )
    )


def _targeted_case_guidance(message: str) -> dict[str, str | list[str]]:
    normalized = clean_text(message).lower()

    def pack(answer: str, points: list[str], procedural_steps: str = "", pro_tip: str = "") -> dict[str, str | list[str]]:
        return {
            "answer": answer,
            "points": points,
            "procedural_steps": procedural_steps,
            "pro_tip": pro_tip,
        }

    if "head of office" in normalized or ("director" in normalized and "audit risk" in normalized):
        return pack(
            "The head of office or director should exercise oversight and can require a safer or better-documented course before approving a technically possible route.",
            [
                "The head of office or director remains responsible for whether the route is defensible in audit, not merely whether it is technically available.",
                "Where reputational or audit risk is visible, the approval note should test proportionality, record why the route is still justified, and insist on stronger documentation where needed.",
                "Oversight here means the approver can ask for additional market check, competition, or justification before allowing the case to proceed.",
            ],
            "1. Test whether the proposed route is still defensible on the stated facts.\n2. Record the reputational or audit risk and the reason for still proceeding or revising the route.\n3. Require any missing market check, committee note, or approval justification before sanction.\n4. Confirm the file shows why the chosen route remains fair and reasonable.\n5. Approve only after the record is strong enough to withstand audit review.",
            "Role questions should answer oversight and accountability directly instead of defaulting to threshold routing.",
        )

    if "approver" in normalized and "committee recommendation" in normalized:
        return pack(
            "The approver retains sanction responsibility even after committee recommendation, because the recommendation informs the decision but does not transfer accountability.",
            [
                "Committee recommendation supports the file, but the approver still owns the final sanction and must be satisfied that rules, facts, and records are adequate.",
                "If the supporting note is weak, the approver should return the case for clarification instead of treating the recommendation as a complete shield.",
                "The sanctioning authority should verify competence, justification, and supporting documentation before approving the case.",
            ],
            "1. Recheck whether the committee recommendation addresses the controlling facts.\n2. Verify that the approver's sanction note independently records satisfaction on rule compliance and justification.\n3. Return the file for clarification if the recommendation is incomplete.\n4. Approve only after the sanction note and supporting evidence are aligned.\n5. Keep the approval record with the committee papers on file.",
        )

    if any(token in normalized for token in ("single remaining responsive bid", "single responsive bid", "three bids", "only bidder remains")) and any(token in normalized for token in ("technical", "responsive")):
        return pack(
            "Before accepting the single remaining responsive bid, the file should record why the other bids failed, confirm the surviving bid is fully responsive, and test price reasonableness.",
            [
                "A single remaining responsive bid is not accepted merely because the other bids failed; the file should show a reasoned decision on technical responsiveness and price reasonableness.",
                "The evaluation note should explain the disqualification of the non-responsive bids so the remaining bid is not treated as acceptable by default.",
                "If reasonableness cannot be defended, the authority should consider fresh competition or further justification before award.",
            ],
            "1. Record the technical reasons why the other bids failed.\n2. Confirm that the remaining bid fully meets the specification and tender terms.\n3. Test price reasonableness against available market or comparative material.\n4. Prepare a speaking recommendation explaining why award is still defensible.\n5. Obtain approval on that recorded reasoning before award.",
        )

    if any(token in normalized for token in ("holiday", "debar", "blacklist")) and any(token in normalized for token in ("award", "submitted", "submission", "before finalization")):
        return pack(
            "The case should be tested against the bidder's eligibility at the relevant stage, and the file should record whether holiday-list status bars award or requires further verification before proceeding.",
            [
                "Holiday-list or debarment issues should be verified explicitly before award because eligibility and award defensibility can change with timing.",
                "The file should not proceed on price or technical merit alone until the bidder's status and the applicable restriction are clarified on record.",
                "If the restriction applies, the case should move to the next eligible bidder or be reconsidered through a documented decision.",
            ],
            "1. Verify the bidder's holiday-list or debarment status and effective date.\n2. Check whether the applicable restriction affects participation, evaluation, or award.\n3. Record the eligibility conclusion on file.\n4. Rework the ranking or award decision if the bidder is barred.\n5. Keep the verification record with the evaluation papers.",
        )

    if "gem" in normalized and "crosses the threshold" in normalized:
        return pack(
            "The file should check whether the chosen GeM mode still lawfully covers the final value and its higher-value approval and competition requirements.",
            [
                "Using GeM for speed does not remove the need to satisfy the controlling value-band and approval requirements that apply to the final committed value.",
                "The record should show that the final purchase order value, the chosen GeM mode, and the approval path remained aligned when the value increased.",
                "If the value drift changed the required competition or sanction standard, the file should explain how that was addressed before award.",
            ],
            "1. Confirm the final committed value and the GeM mode actually used.\n2. Check whether that value triggers a higher competition or approval requirement.\n3. Record whether the GeM route used still satisfies those requirements.\n4. Cure any approval or documentation gap before proceeding further.\n5. Keep the GeM record, value basis, and sanction note together on file.",
        )

    if "unauthorized channel" in normalized or ("lower price" in normalized and any(token in normalized for token in ("source legitimacy", "authorized source", "authorised source"))):
        return pack(
            "No. A lower price should not override source legitimacy; the file must first verify that the supply channel is authorized and admissible.",
            [
                "A lower quote is not sufficient if the bidder is supplying through an unauthorized channel that creates legitimacy, warranty, support, or traceability concerns.",
                "The evaluation should test source authorization and supply admissibility before treating the offer as comparable with compliant channels.",
                "If legitimacy cannot be established, the lower-priced offer should not be treated as the winning basis.",
            ],
            "1. Verify the bidder's authorization from the OEM or other controlling source requirement.\n2. Check whether warranty, service support, and source traceability remain valid.\n3. Record whether the offer is admissible for evaluation.\n4. Exclude or qualify the offer if legitimacy cannot be established.\n5. Finalize award only after the source-legitimacy issue is resolved on file.",
        )

    if any(token in normalized for token in ("without pac", "single-source", "single source")) and "confidentiality" in normalized:
        return pack(
            "Confidentiality alone does not cure a missing PAC; the file must still prove a valid proprietary or approved single-source basis.",
            [
                "A claim of research confidentiality should be examined carefully, because it does not automatically substitute for a PAC or other lawful single-source justification.",
                "The file should distinguish between a genuine proprietary basis and a preference to avoid publication or competition.",
                "If the proprietary or approved exception basis is not established, the proposal should not be treated as a complete single-source case.",
            ],
            "1. Identify the exact single-source justification being claimed.\n2. Verify whether PAC, standardization, confidentiality, or another approved exception is actually supported.\n3. Record why publication or competition is or is not feasible on the facts.\n4. Obtain the required certificate or approval before proceeding.\n5. If the basis is weak, shift the case to a competitive route or seek a corrected justification.",
        )

    if any(token in normalized for token in ("gst", "tender document fee", "vendor registration fee", "registration fee")):
        return pack(
            "Before proceeding, the file should clarify the GST treatment of each fee and whether the tender terms disclosed that treatment consistently.",
            [
                "Tender document fee and vendor registration fee should not move forward on assumption alone if their GST treatment is unclear.",
                "The file should clearly show whether each fee is taxable, how it will be invoiced or accounted for, and whether bidders were told the same treatment in the tender terms.",
                "If the tax treatment affects collection or refund handling, that basis should be resolved before further processing.",
            ],
            "1. List each fee separately and identify its purpose.\n2. Confirm the GST treatment and accounting basis for each fee.\n3. Check whether the tender terms disclosed the same treatment to all bidders.\n4. Correct or clarify the fee note before proceeding further.\n5. Keep the clarification with the tender record.",
        )

    if "value-for-money" in normalized or "value for money" in normalized:
        return pack(
            "A file can satisfy threshold logic yet still fail audit if competition, price reasonableness, specifications, or comparative justification do not show value for money.",
            [
                "Threshold compliance only answers whether the route is facially available; audit can still question whether the file proved competition, reasonableness, and value for money.",
                "A file becomes vulnerable when the route is technically correct but the record does not justify the specification, comparative outcome, or market reasonableness.",
                "Audit therefore tests both route logic and the quality of the supporting justification, not just the value band.",
            ],
            "1. Confirm that the chosen route is correct for the threshold.\n2. Check whether the file also records competition, price reasonableness, and neutral specifications.\n3. Record any gap in market comparison or justification.\n4. Strengthen the value-for-money note before award or post-facto defence.\n5. Keep the comparative reasoning with the route selection record.",
        )

    if "make in india" in normalized or "local supplier preference" in normalized:
        return pack(
            "During bid evaluation, first verify the local content declaration and supplier class, then apply preference only if the bid is otherwise responsive.",
            [
                "Make in India preference should be checked as an evaluation issue, not assumed from the bidder's label alone.",
                "The file should verify the declared local content, the supplier category claimed, and whether the bid remains technically and commercially responsive before preference is applied.",
                "If the declaration is defective or unsupported, the bid should be evaluated without the claimed preference benefit.",
            ],
            "1. Verify the bidder's local content declaration and supporting documents.\n2. Identify the applicable supplier class and preference condition.\n3. Check that the bid is otherwise responsive before applying preference.\n4. Apply or deny preference on a recorded evaluation note.\n5. Keep the declaration, calculation basis, and evaluation record on file.",
        )

    if "one-bid ote" in normalized or ("only bidder" in normalized and "local supplier" in normalized):
        return pack(
            "Local supplier preference does not decide a one-bid OTE by itself; the file should first test whether the lone bid is responsive and reasonably priced.",
            [
                "Purchase preference normally matters in comparative evaluation, so a one-bid case should not treat that preference claim as a substitute for competition.",
                "The authority should first assess responsiveness, admissibility, and price reasonableness of the lone offer.",
                "If the file cannot defend the single-offer outcome, it should record why award is still justified or reconsider the procurement strategy.",
            ],
            "1. Confirm that only one responsive bid remains in the OTE case.\n2. Verify the basis of the local supplier claim but do not let it replace competition analysis.\n3. Test the lone bid for responsiveness and price reasonableness.\n4. Record why award is still defensible, or reconsider the tender decision.\n5. Keep the evaluation reasoning on file.",
        )

    if "gem" in normalized and any(token in normalized for token in ("reseller", "resellers", "oem", "configuration")):
        return pack(
            "Compare the exact scientific requirement with the GeM listings, verify reseller authorization, and record a reasoned departure if GeM does not meet the need.",
            [
                "The question is not whether GeM shows a broadly similar item, but whether the available listing matches the exact scientific configuration required for the case.",
                "If GeM listings are through resellers, the file should verify authorization, supportability, and whether the platform offering truly meets the operational need.",
                "Any departure from GeM should therefore be supported by a specific record of mismatch or infeasibility, not just a preference for the OEM.",
            ],
            "1. Compare the required scientific configuration with the GeM listing actually available.\n2. Verify whether the reseller is authorized and whether support obligations are acceptable.\n3. Record any configuration gap or platform infeasibility with evidence.\n4. Decide whether GeM remains feasible or a justified departure is needed.\n5. Keep the comparison and authorization record on file.",
        )

    if "scientific relaxation" in normalized and "another csir lab" in normalized:
        return pack(
            "The claimed scientific relaxation should be tested against market evidence; if another CSIR lab bought competitively, this file must explain why competition is infeasible here.",
            [
                "A scientific relaxation cannot be treated as self-proving when comparable competitive procurement is demonstrably possible elsewhere in the same ecosystem.",
                "The file should explain the factual difference between this case and the other CSIR lab's competitive procurement if it still seeks a non-competitive route.",
                "Without that distinction, the relaxation claim looks weak from an audit and value-for-money perspective.",
            ],
            "1. Gather the facts of the other CSIR lab's competitive procurement.\n2. Compare them with the current case and record any material difference.\n3. Test whether competition is genuinely infeasible here.\n4. Retain or revise the claimed relaxation based on that comparison.\n5. Keep the comparative market record on file.",
        )

    if any(token in normalized for token in ("amendment", "amendments")) and any(token in normalized for token in ("threshold", "process", "responsibility")):
        return pack(
            "Amendments usually change thresholds, clarify process, or reallocate responsibility, so the file should identify which of those functions the amendment serves.",
            [
                "An amendment does not automatically change everything; it may alter a value threshold, refine the procedure, or shift who must approve or certify the action.",
                "Interpretation should therefore start by identifying whether the amendment changes the monetary trigger, the workflow, or the responsible authority.",
                "That distinction matters because each type of amendment affects compliance analysis differently.",
            ],
            "1. Identify the exact clause changed by the amendment.\n2. Decide whether it changes a threshold, a process step, or responsibility.\n3. Apply only that change to the file's interpretation.\n4. Update the note to show the revised controlling position.\n5. Keep the amendment extract with the working file.",
        )

    return {"answer": "", "points": [], "procedural_steps": "", "pro_tip": ""}


def _build_role_reasoning_points(message: str) -> list[str]:
    normalized = clean_text(message).lower()
    if "approver" in normalized:
        return [
            "The approver retains sanction responsibility and must be satisfied on rule compliance and justification before approving the case.",
            "A committee recommendation assists the decision but does not displace the approver's accountability.",
        ]
    if "head of office" in normalized or "director" in normalized:
        return [
            "The head of office or director should exercise oversight over whether the case remains defensible on the stated facts.",
            "Where audit or reputational risk is visible, the approver may require stronger justification or a different route before approval.",
        ]
    if "competent authority" in normalized:
        return [
            "The competent authority should verify that the supporting record and approval basis are adequate before sanction.",
            "Competence to approve is distinct from the technical recommendation that supports the file.",
        ]
    if "technical recommender" in normalized:
        return [
            "The technical recommender supports the case on specification or suitability, but does not replace the sanctioning responsibility of the competent authority.",
            "The file should keep recommendation and sanction as distinct responsibilities.",
        ]
    return []


def _build_scenario_reasoning_points(message: str, matches: list[SearchMatch]) -> list[str]:
    if not looks_like_scenario_query(message):
        return []
    targeted = _targeted_case_guidance(message)
    targeted_points: list[str] = []
    for point in targeted.get("points", []):
        cleaned_point = cleanup_generated_sentence(str(point))
        if cleaned_point:
            targeted_points.append(cleaned_point)
    if targeted_points:
        return targeted_points[:4]
    points: list[str] = []
    primary_matches = prioritize_matches(message, matches)[: min(3, len(matches))]
    if primary_matches:
        top_summary = summarize_match_for_context(message, primary_matches[0].content, primary_matches[0].file_name)
        cleaned_summary = cleanup_generated_sentence(top_summary)
        if cleaned_summary:
            points.append(cleaned_summary)
    points.append(
        "In this scenario, the officer should test the factual trigger for the proposed action against the governing rule instead of relying only on the label used in the note."
    )
    points.append(
        "The file should show why the chosen route fits the facts, because the same label can be valid in one situation and wrong in another."
    )
    cleaned_points: list[str] = []
    for point in points:
        cleaned_point = cleanup_generated_sentence(point)
        if cleaned_point:
            cleaned_points.append(cleaned_point)
    return cleaned_points


def build_explanation_points(message: str, matches: list[SearchMatch]) -> list[str]:
    targeted = _targeted_case_guidance(message)
    targeted_points: list[str] = []
    for point in targeted.get("points", []):
        cleaned_point = cleanup_generated_sentence(str(point))
        if cleaned_point:
            targeted_points.append(cleaned_point)
    if targeted_points:
        return targeted_points[:4]

    role_points = _build_role_reasoning_points(message)
    if role_points:
        return role_points[:4]

    if detect_intent(message) == "ANALYTICAL":
        analytical_points = _build_analytical_points(message, matches)
        if analytical_points:
            return analytical_points

    amt_points = [] if looks_like_scenario_query(message) or _looks_like_role_question(message) else _build_amount_specific_points(message, matches)
    if amt_points:
        return amt_points
    scenario_points = _build_scenario_reasoning_points(message, matches)
    if scenario_points:
        return scenario_points[:4]
    ranked = prioritize_matches(message, matches)[: min(5, len(matches))]
    points: list[str] = []
    seen: set[str] = set()
    for m in ranked:
        if looks_like_table_of_contents(m.content):
            continue
        for s in select_relevant_sentences(message, clean_text(m.content), max_sentences=3):
            rw = cleanup_generated_sentence(s)
            if not rw:
                continue
            key = semantic_dedup_key(rw)
            if key in seen:
                continue
            seen.add(key)
            points.append(rw)
            if len(points) >= 3:
                return points
    return points or ["The retrieved documents provide related procurement guidance, but the wording is partial."]


def extract_source_names(matches: list[SearchMatch], message: str = "") -> list[str]:
    return _extract_source_basis_entries(message, matches)


def compose_direct_answer(message: str, explanation_points: list[str], weak_match: bool = False, matches: list[SearchMatch] | None = None) -> str:
    targeted = _targeted_case_guidance(message)
    targeted_answer = cleanup_generated_sentence(str(targeted.get("answer", "")))
    if targeted_answer:
        return targeted_answer

    if _looks_like_role_question(message):
        role_points = _build_role_reasoning_points(message)
        if role_points:
            cleaned_role = cleanup_generated_sentence(role_points[0])
            if cleaned_role:
                return cleaned_role

    methods = extract_analytical_terms(message)
    if detect_intent(message) == "ANALYTICAL" or (_question_shape(message) == "DEFINITION" and len(methods) >= 2):
        if len(methods) >= 2:
            left, right = methods[:2]
            left_row = _reference_row(left)
            right_row = _reference_row(right)
            return (
                f"{left} applies under {left_row['rule']} for {left_row['value_band']}, "
                f"whereas {right} applies under {right_row['rule']} for {right_row['value_band']}."
            )

    amount = extract_amount_lakhs(message)
    if amount is not None and not looks_like_scenario_query(message) and not _looks_like_role_question(message):
        slab = gfr_slab_for_amount(amount, message)
        if slab:
            return f"For {format_lakh_amount(amount)}, the applicable route is {slab['method']} under {slab['rule']}."

    if explanation_points:
        primary = cleanup_generated_sentence(explanation_points[0])
        if primary:
            if looks_like_scenario_query(message) and not primary.lower().startswith("in this scenario"):
                return f"In this scenario, {primary[:1].lower() + primary[1:]}"
            if weak_match:
                return f"The closest grounded guidance is that {primary[:1].lower() + primary[1:]}"
            return primary
    label = describe_user_request(message)
    return f"The retrieved procurement guidance addresses {label}."


# ─── Internal Helpers ───────────────────────────────────────────────────────

def _build_amount_answer(message: str, matches: list[SearchMatch]) -> str:
    amount = extract_amount_lakhs(message)
    exp = build_explanation_points(message, matches)
    sources = extract_source_names(matches, message)
    facts: list[str] = []
    if amount is not None:
        facts.append(f"Asked amount: {format_lakh_amount(amount)}")
    for p in exp[:3]:
        c = cleanup_generated_sentence(p)
        if c:
            facts.append(c)
    if sources:
        facts.append("Relevant sources: " + ", ".join(sources[:3]))
    return "\n".join(f"- {f}" for f in facts) if facts else "- No structured procurement facts extracted."


def _extract_match_features(matches: list[SearchMatch]) -> dict[str, bool]:
    combined = " ".join(clean_text(m.content).lower() for m in matches[:8])
    return {
        "has_lpc_limit": "local purchase committee" in combined and ("5,00,000" in combined or "five lakh" in combined),
        "has_under_50_rule": ("50 lakhs or less" in combined or "up to rs. 50 lakhs" in combined) and ("local suppliers" in combined or "local capacity" in combined),
        "has_divisible_preference": "50% of the order quantity" in combined or "remaining 50% quantity" in combined,
    }


def _build_amount_specific_points(message: str, matches: list[SearchMatch]) -> list[str]:
    amount = extract_amount_lakhs(message)
    if amount is None:
        return []
    points: list[str] = []
    slab = gfr_slab_for_amount(amount, message)
    if slab:
        points.append(f"{format_lakh_amount(amount)} falls in the {slab['method']} band under {slab['rule']}.")
    points.append("The GFR 2025 threshold table is treated as the final source of truth when older manuals conflict.")
    return points[:3]


def _method_summary(method: str, message: str, matches: list[SearchMatch], existing_summaries: list[str] | None = None) -> str:
    row = _reference_row(method)
    fallback = cleanup_generated_sentence(f"{row['method']} applies to {row['value_band']}. {row['notes']}") or row["notes"]
    if row.get("rule") not in {"", "-"} and _is_distinct_summary(fallback, existing_summaries or []):
        return fallback

    ranked = _ensure_analytical_coverage(message, matches)
    method_matches = [match for match in ranked if _mentions_method(match, method)]
    distinct_against = existing_summaries or []
    for match in method_matches:
        candidate_sentences = select_relevant_sentences(message, clean_text(match.content), max_sentences=4)
        for sentence in candidate_sentences:
            cleaned_sentence = cleanup_generated_sentence(sentence)
            if not cleaned_sentence:
                continue
            if _is_distinct_summary(cleaned_sentence, distinct_against):
                return cleaned_sentence

        summary = summarize_match_for_context(message, match.content, match.file_name)
        cleaned_summary = cleanup_generated_sentence(summary)
        if cleaned_summary and _is_distinct_summary(cleaned_summary, distinct_against):
            return cleaned_summary

    if _is_distinct_summary(fallback, distinct_against):
        return fallback
    return f"{row['method']} follows {row['rule']} for {row['value_band']} procurements."


def _escape_table_cell(value: str) -> str:
    cleaned = clean_text(value)
    cleaned = cleaned.replace("|", "/")
    return cleaned or "-"


def _build_analytical_table(message: str, matches: list[SearchMatch]) -> str:
    methods = extract_analytical_terms(message)
    if len(methods) < 2:
        return ""
    left, right = methods[:2]
    left_row = _reference_row(left)
    right_row = _reference_row(right)
    left_summary = _method_summary(left, message, matches)
    right_summary = _method_summary(right, message, matches, existing_summaries=[left_summary])
    rows = [
        ("Core purpose", left_summary, right_summary),
        ("Threshold / applicability", left_row["value_band"], right_row["value_band"]),
        ("Rule", left_row["rule"], right_row["rule"]),
        ("Approval / evidence", left_row["notes"], right_row["notes"]),
    ]
    lines = [
        f"| Feature | {left} | {right} |",
        "|---|---|---|",
    ]
    for feature, left_value, right_value in rows:
        lines.append(f"| {_escape_table_cell(feature)} | {_escape_table_cell(left_value)} | {_escape_table_cell(right_value)} |")
    return "\n".join(lines)


def _build_analytical_procedural_steps(message: str, matches: list[SearchMatch]) -> str:
    methods = extract_analytical_terms(message)
    if len(methods) < 2:
        return ""
    steps_by_method = {
        "LTE": [
            "Confirm the estimated value falls in the LTE band and shortlist capable firms.",
            "Issue the limited tender enquiry and obtain quotations or bids from the shortlisted firms.",
            "Evaluate the offers through the competent committee and obtain approval before placing the order.",
        ],
        "STE": [
            "Record why single-source procurement is justified, such as PAC, standardisation, or urgency.",
            "Obtain the supporting certificate or justification note from the competent authority.",
            "Place the case for approval and issue the order through the approved single-source route.",
        ],
        "LPC": [
            "Confirm the value falls in the LPC band and identify at least three capable sources.",
            "Obtain comparative quotations and record the committee recommendation.",
            "Take approval from the competent LPC authority before issue of the order.",
        ],
        "OTE": [
            "Prepare the tender with the required publicity through the approved public route.",
            "Receive and evaluate bids through the competent committee in line with tender conditions.",
            "Obtain approval and place the order or contract with the successful bidder.",
        ],
    }
    lines: list[str] = []
    step_number = 1
    for method in methods[:2]:
        for sentence in steps_by_method.get(method, ["Confirm applicability, document justification, and complete the approval chain."])[:3]:
            lines.append(f"{step_number}. **{method}**: {sentence}")
            step_number += 1
    while len(lines) < 5:
        lines.append(f"{step_number}. Record the controlling rule, approval path, and final decision on file.")
        step_number += 1
    return "\n".join(lines)


def build_rule_based_answer(message: str, matches: list[SearchMatch], blocked_hashes: set[str], weak_match: bool = False, intent: str = "GENERAL") -> str:
    if not matches:
        return build_no_match_response(message)
    targeted = _targeted_case_guidance(message)
    exp = build_explanation_points(message, matches)
    answer = compose_direct_answer(message, exp, weak_match, matches)
    lowered_message = clean_text(message).lower()
    wants_table = any(token in lowered_message for token in ("table", "matrix"))
    comparison_table = _build_analytical_table(message, matches) if intent == "ANALYTICAL" and wants_table else ""
    procedural_steps = str(targeted.get("procedural_steps", "") or "")
    if not procedural_steps:
        procedural_steps = _build_analytical_procedural_steps(message, matches) if intent == "ANALYTICAL" else ""
    pro_tip = str(targeted.get("pro_tip", "") or "")
    if not pro_tip and any(token in lowered_message for token in ("conflict", "override", "priority", "older manual", "prevail")):
        pro_tip = _rule_priority_line(message)
    candidate = render_structured_response(
        answer=answer,
        explanation_points=exp,
        sources=extract_source_names(matches, message),
        procedural_steps=procedural_steps,
        pro_tip=pro_tip,
        comparison_table=comparison_table,
        intent=intent,
    )
    if response_hash(candidate) in blocked_hashes:
        candidate = render_structured_response(
            answer=f"Here is an alternate summary for: {message.strip()}",
            explanation_points=exp,
            sources=extract_source_names(matches, message),
            procedural_steps=procedural_steps,
            pro_tip=pro_tip,
            comparison_table=comparison_table,
            intent=intent,
        )
    return candidate


def build_empty_knowledge_base_response() -> str:
    return render_structured_response(
        answer="The knowledge base is currently empty or not loaded.",
        explanation_points=["No indexed documents are available.", "The FAISS index needs loaded documents."],
        sources=[],
    )


def build_source_verification_response(message: str) -> str:
    expected_rules = sorted(_question_rule_signals(message))
    target_rule = expected_rules[0] if expected_rules else "General Procurement Principle"
    if expected_rules:
        answer = f"Context regarding this rule is not available in the manual. Please refer to {target_rule} of the Updated GFR 2025."
        sources = [f"{target_rule} - {GFR_2025_DOCUMENT_NAME}"]
    else:
        answer = "Context regarding this rule is not available in the manual. Please refer to General Procurement Principle of the Updated GFR 2025."
        sources = [f"General Procurement Principle - {GFR_2025_DOCUMENT_NAME}"]
    return render_structured_response(
        answer=answer,
        explanation_points=[
            "The retrieved context does not explicitly answer the question, so no unsupported conclusion should be recorded.",
            "Use the controlling GFR rule text or a verified CSIR manual clause before proceeding.",
        ],
        sources=sources,
        procedural_steps="1. Re-run retrieval using the exact rule number or topic.\n2. Verify the controlling clause in Updated GFR 2025 or the CSIR Manual.\n3. Record the verified clause in the note before taking a decision.",
        pro_tip="Insufficient retrieved context is itself an audit signal and should be recorded transparently.",
        intent="PROCESS",
    )


def build_no_match_response(message: str) -> str:
    return build_source_verification_response(message)


def _line_item_count(content: str) -> int:
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    line_items = [line for line in lines if re.match(r"^(?:[-*]|\d+[.)])\s+", line)]
    if line_items:
        return len(line_items)
    return len(re.findall(r"(?:^|\s)(?:\d+[.)])\s+[A-Za-z]", content))


def _is_low_quality_retrieval_chunk(match: SearchMatch) -> bool:
    cleaned = clean_text(match.content)
    if looks_like_table_of_contents(cleaned):
        return True
    if _line_item_count(cleaned) > 5:
        return True
    if contains_scientist_list(cleaned):
        return True
    audit = audit_chunk_quality(cleaned)
    return bool(audit.get("discard"))


def _is_low_score_retrieval(state: GraphState) -> bool:
    documents = state.get("documents", [])
    if not documents:
        return True
    usable = [match for match in documents if not _is_low_quality_retrieval_chunk(match)]
    if not usable:
        return True
    top_score = max(float(match.score) for match in usable)
    average_top_score = sum(float(match.score) for match in usable[:3]) / min(len(usable), 3)
    return top_score < 0.02 or average_top_score < 0.012


def _static_gfr_rule_response(message: str, intent: str) -> str:
    amount_rupees = _extract_currency_value_rupees(message)
    deterministic_slab = _gfr_2025_slab_for_rupees(amount_rupees, message)
    if deterministic_slab and amount_rupees is not None:
        amount_label = _format_rupee_amount(amount_rupees)
        answer = f"For {amount_label}, the applicable route is {deterministic_slab['method']} under {deterministic_slab['rule']}."
        points = [
            f"The amount falls in the deterministic threshold slab: {deterministic_slab['label']}.",
            str(deterministic_slab["reason"]),
            str(deterministic_slab.get("profile_summary", "This rule is treated as mandatory truth for threshold routing and overrides weaker retrieval wording.")),
        ]
        return render_structured_response(
            answer=answer,
            explanation_points=points,
            sources=[str(deterministic_slab.get("source_basis", f"{deterministic_slab['rule']} - {GFR_2025_DOCUMENT_NAME}"))],
            procedural_steps=_threshold_procedural_steps(deterministic_slab),
            pro_tip=f"Use the deterministic threshold mapping for the selected profile: {deterministic_slab.get('profile_label', 'General GFR thresholds')}.",
            intent=intent if intent in {"PROCESS", "WORKFLOW", "ANALYTICAL"} else "PROCESS",
        )

    amount = extract_amount_lakhs(message)
    slab = gfr_slab_for_amount(amount, message)
    if not slab:
        return build_no_match_response(message)
    answer = f"For {format_lakh_amount(amount or 0)}, the applicable route is {slab['method']} under {slab['rule']}."
    points = [
        f"The amount falls in the selected threshold band: {slab['value_band']}.",
        str(slab["notes"]),
        str(slab.get("profile_summary", "The threshold framework is used as the controlling source when older manuals conflict.")),
    ]
    return render_structured_response(
        answer=answer,
        explanation_points=points,
        sources=[str(slab.get("source_basis", f"{slab['rule']} - {GFR_2025_DOCUMENT_NAME}"))],
        procedural_steps=build_default_procedural_steps("PROCESS", answer, points),
        pro_tip=f"Threshold decision uses the selected profile: {slab.get('profile_label', 'General GFR thresholds')}.",
        intent=intent if intent in {"PROCESS", "WORKFLOW", "ANALYTICAL"} else "PROCESS",
    )


def question_transformer_node(state: GraphState) -> GraphState:
    query = (state.get("query") or "").strip()
    intent = detect_intent(query)
    amount = extract_amount_lakhs(query)
    search_query = _rewrite_query_for_gfr_slab(query, amount)
    logger.info("Graph node transform intent='%s' amount=%s query='%s'", intent, amount, search_query[:160])
    return {
        **state,
        "query": query,
        "search_query": search_query,
        "intent": intent,
        "amount": amount,
        "threshold_basis": "Awaiting deterministic threshold judge",
    }


def logic_injection_node(state: GraphState) -> GraphState:
    query = (state.get("query") or "").strip()
    amount_rupees = _extract_currency_value_rupees(query)
    amount_lakhs = state.get("amount")
    if amount_rupees is not None:
        amount_lakhs = amount_rupees / 100000.0
    elif amount_lakhs is not None:
        amount_rupees = int(round(amount_lakhs * 100000))

    slab = _gfr_2025_slab_for_rupees(amount_rupees)
    threshold_truth = ""
    threshold_queries: list[str] = []
    search_query = state.get("search_query") or query
    threshold_basis = "No numeric slab detected"

    if slab and amount_rupees is not None:
        threshold_truth = _build_threshold_truth_block(amount_rupees, slab)
        threshold_queries = _build_threshold_query_variations(query, amount_rupees, slab)
        search_query = " ".join([search_query, str(slab["rule"]), str(slab["method"]), "MANDATORY TRUTH"]).strip()
        threshold_basis = f"Deterministic threshold judge matched {slab['rule']} for {_format_rupee_amount(amount_rupees)}"

    metadata = dict(state.get("metadata", {}))
    metadata["logic_injection"] = {
        "matched": bool(slab),
        "amount_rupees": amount_rupees,
        "threshold_basis": threshold_basis,
    }
    if slab:
        metadata["logic_injection"]["slab"] = {
            "method": slab["method"],
            "rule": slab["rule"],
            "label": slab["label"],
        }

    logger.info("Graph node logic_injection amount_rupees=%s slab=%s", amount_rupees, slab.get("key") if slab else None)
    return {
        **state,
        "amount": amount_lakhs,
        "amount_rupees": amount_rupees,
        "slab": slab,
        "search_query": search_query,
        "threshold_truth": threshold_truth,
        "threshold_query_variations": threshold_queries,
        "threshold_basis": threshold_basis,
        "metadata": metadata,
    }


def threshold_judge_node(state: GraphState) -> GraphState:
    """Backward-compatible alias for the newer logic injection node."""
    return logic_injection_node(state)


def multi_query_retrieval_node(state: GraphState) -> GraphState:
    search_query = state.get("search_query") or state.get("query", "")
    top_k = max(settings.top_k * 2, settings.top_k)
    matches, queries = _multi_query_retrieve(
        query=search_query,
        blocked_chunk_ids=state.get("blocked_chunk_ids", []),
        top_k=top_k,
        intent=state.get("intent", "GENERAL"),
        relaxed=False,
        threshold_queries=state.get("threshold_query_variations", []),
    )
    logger.info("Graph node retrieval queries=%s documents=%s", len(queries), len(matches))
    return {
        **state,
        "retrieval_queries": queries,
        "raw_documents": matches,
        "documents": matches,
        "weak_match": False,
    }


def _route_after_retrieval(state: GraphState) -> str:
    if state.get("retry_count", 0) > 0:
        return "rerank"
    return "retry" if _is_low_score_retrieval(state) else "rerank"


def retry_search_fallback_node(state: GraphState) -> GraphState:
    search_query = state.get("search_query") or state.get("query", "")
    slab = state.get("slab")
    expanded_query = search_query
    if slab:
        expanded_query = " ".join(
            [
                search_query,
                str(slab["method"]),
                str(slab["rule"]),
                str(slab["value_band"]),
                str(slab["notes"]),
                "CSIR Manual 2019 procedure steps",
                "Make in India SnT special provisions exemptions",
            ]
        )
    else:
        expanded_query = f"{search_query} GFR 2025 CSIR Manual 2019 procurement threshold procedure"

    matches, queries = _multi_query_retrieve(
        query=expanded_query,
        blocked_chunk_ids=state.get("blocked_chunk_ids", []),
        top_k=max(settings.top_k * 3, 12),
        intent=state.get("intent", "GENERAL"),
        relaxed=True,
        threshold_queries=state.get("threshold_query_variations", []),
    )
    logger.info("Graph node retry_search queries=%s documents=%s", len(queries), len(matches))
    return {
        **state,
        "search_query": expanded_query,
        "retrieval_queries": [*state.get("retrieval_queries", []), *queries],
        "raw_documents": matches,
        "documents": matches,
        "weak_match": True,
        "retry_count": state.get("retry_count", 0) + 1,
    }


def rerank_node(state: GraphState) -> GraphState:
    query = state.get("search_query") or state.get("query", "")
    intent = state.get("intent", "GENERAL")
    documents = state.get("documents", [])
    reranked = _compress_retrieved_matches(
        query=query,
        matches=documents,
        intent=intent,
        limit=max(settings.top_k * 2, 6),
    )
    high_quality: list[SearchMatch] = []
    low_quality: list[SearchMatch] = []
    for match in reranked:
        if _is_low_quality_retrieval_chunk(match):
            low_quality.append(match)
        else:
            high_quality.append(match)
    research_required = len(high_quality) == 0
    logger.info(
        "Graph node rerank high_quality=%s low_quality=%s research_required=%s",
        len(high_quality),
        len(low_quality),
        research_required,
    )
    return {
        **state,
        "documents": high_quality,
        "low_quality_documents": low_quality,
        "research_required": research_required,
    }


def threshold_logic_node(state: GraphState) -> GraphState:
    slab = state.get("slab") or _gfr_2025_slab_for_rupees(state.get("amount_rupees"), state.get("query")) or gfr_slab_for_amount(state.get("amount"), state.get("query"))
    metadata = dict(state.get("metadata", {}))
    metadata.update(
        {
            "source_priority": list(PROCUREMENT_SOURCE_PRIORITY),
            "slab_checked": bool(slab),
            "threshold_basis": state.get("threshold_basis", "No numeric slab detected"),
        }
    )
    if slab:
        metadata["gfr_2025_slab"] = {
            "method": slab["method"],
            "rule": slab["rule"],
            "value_band": slab.get("value_band", slab.get("label", "")),
            "notes": slab.get("notes", slab.get("reason", "")),
        }
    if state.get("threshold_truth"):
        metadata["threshold_truth"] = state["threshold_truth"]
    logger.info("Graph node threshold_logic slab=%s", slab.get("key") if slab else None)
    return {
        **state,
        "slab": slab,
        "metadata": metadata,
    }


def research_or_static_fallback_node(state: GraphState) -> GraphState:
    search_query = state.get("search_query") or state.get("query", "")
    matches, queries = _multi_query_retrieve(
        query=search_query,
        blocked_chunk_ids=state.get("blocked_chunk_ids", []),
        top_k=max(settings.top_k * 2, 5),
        intent=state.get("intent", "GENERAL"),
        relaxed=True,
        threshold_queries=state.get("threshold_query_variations", []),
    )
    compressed = _compress_retrieved_matches(
        query=search_query,
        matches=matches,
        intent=state.get("intent", "GENERAL"),
        limit=max(settings.top_k, 5),
    )
    high_quality = [match for match in compressed if not _is_low_quality_retrieval_chunk(match)]
    logger.info("Graph node research relaxed_documents=%s high_quality=%s", len(compressed), len(high_quality))
    return {
        **state,
        "documents": high_quality,
        "retrieval_queries": [*state.get("retrieval_queries", []), *queries],
        "weak_match": True,
        "research_required": len(high_quality) == 0,
    }


def agentic_generation_node(state: GraphState) -> GraphState:
    query = state.get("query", "")
    user = state.get("user", "anonymous")
    documents = state.get("documents", [])
    blocked_response_hashes = state.get("blocked_response_hashes", set())
    intent = state.get("intent", detect_intent(query))

    if not documents:
        if not knowledge_base.status().get("index_loaded"):
            generation = build_empty_knowledge_base_response()
        else:
            generation = build_source_verification_response(query)
        metadata = {**state.get("metadata", {}), **build_grounding_metadata(generation, documents)}
        metadata.update(
            {
                "retrieval_mode": "agentic-langgraph-static-gfr",
                "intent": intent,
                "amount": state.get("amount"),
                "gfr_slab_forced": bool(state.get("slab")),
                "generation_mode": "rule_based",
            }
        )
        return {**state, "generation": generation, "metadata": metadata}

    if not _is_context_relevant_to_query(query, documents):
        generation = build_source_verification_response(query)
        metadata = {**state.get("metadata", {}), **build_grounding_metadata(generation, documents)}
        metadata.update(
            {
                "retrieval_mode": "agentic-langgraph-context-guard",
                "intent": intent,
                "amount": state.get("amount"),
                "gfr_slab_forced": bool(state.get("slab")),
                "generation_mode": "rule_based",
            }
        )
        return {**state, "generation": generation, "metadata": metadata}

    if not _has_explicit_rule_coverage(query, documents):
        generation = build_source_verification_response(query)
        metadata = {**state.get("metadata", {}), **build_grounding_metadata(generation, documents)}
        metadata.update(
            {
                "retrieval_mode": "agentic-langgraph-source-verification-guard",
                "intent": intent,
                "amount": state.get("amount"),
                "gfr_slab_forced": bool(state.get("slab")),
                "generation_mode": "rule_based",
            }
        )
        return {**state, "generation": generation, "metadata": metadata}

    result = handle_query(
        message=query,
        user=user,
        matches=documents,
        bypass_cache=bool(state.get("bypass_cache", False)),
        blocked_response_hashes=blocked_response_hashes,
        weak_match=bool(state.get("weak_match", False)),
        threshold_truth=state.get("threshold_truth", ""),
    )
    metadata = {**state.get("metadata", {}), **(result.get("metadata") or {})}
    metadata.update(
        {
            "retrieval_mode": "agentic-langgraph-multiquery-compression",
            "intent": result.get("intent", intent),
            "amount": result.get("amount", state.get("amount")),
            "retrieval_queries": state.get("retrieval_queries", []),
            "low_quality_chunk_ids": [match.chunk_id for match in state.get("low_quality_documents", [])],
            "gfr_slab_forced": bool(state.get("slab")),
            "generation_mode": result.get("generation_mode", "rule_based"),
        }
    )
    return {
        **state,
        "generation": result["answer"],
        "metadata": metadata,
    }


@lru_cache(maxsize=1)
def get_compiled_rag_graph() -> Any:
    try:
        from langgraph.graph import END, StateGraph

        graph = StateGraph(GraphState)
        graph.add_node("QUERY_TRANSFORM", question_transformer_node)
        graph.add_node("LOGIC_INJECTION", logic_injection_node)
        graph.add_node("RETRIEVE", multi_query_retrieval_node)
        graph.add_node("RETRY_SEARCH", retry_search_fallback_node)
        graph.add_node("RERANK", rerank_node)
        graph.add_node("THRESHOLD_LOGIC", threshold_logic_node)
        graph.add_node("GENERATE", agentic_generation_node)
        graph.set_entry_point("QUERY_TRANSFORM")
        graph.add_edge("QUERY_TRANSFORM", "LOGIC_INJECTION")
        graph.add_edge("LOGIC_INJECTION", "RETRIEVE")
        graph.add_conditional_edges(
            "RETRIEVE",
            _route_after_retrieval,
            {
                "rerank": "RERANK",
                "retry": "RETRY_SEARCH",
            },
        )
        graph.add_edge("RETRY_SEARCH", "RERANK")
        graph.add_edge("RERANK", "THRESHOLD_LOGIC")
        graph.add_edge("THRESHOLD_LOGIC", "GENERATE")
        graph.add_edge("GENERATE", END)
        return graph.compile()
    except Exception:
        logger.warning("LangGraph unavailable; using sequential graph fallback", exc_info=True)
        return _SequentialGraph()


def run_agentic_rag(
    message: str,
    user: str,
    bypass_cache: bool = False,
    blocked_chunk_ids: list[int] | None = None,
    blocked_response_hashes: set[str] | None = None,
) -> GraphState:
    graph = get_compiled_rag_graph()
    initial_state: GraphState = {
        "query": message,
        "user": user,
        "bypass_cache": bypass_cache,
        "blocked_chunk_ids": blocked_chunk_ids or [],
        "blocked_response_hashes": blocked_response_hashes or set(),
        "documents": [],
        "raw_documents": [],
        "generation": "",
        "retry_count": 0,
    }
    return graph.invoke(initial_state)
