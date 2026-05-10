import concurrent.futures
import logging
import re
from typing import Any

from app.services.knowledge_base import knowledge_base
from app.services.llm_service import generate_llm_response
from app.utils.output_validator import merge_adjacent_chunks, validate_structured_output

logger = logging.getLogger("procurebuddy-ai")

_METHOD_ALIASES: dict[str, tuple[str, ...]] = {
    "STE": ("single tender enquiry", "single tender", "ste", "single source", "proprietary"),
    "LTE": ("limited tender enquiry", "limited tender", "lte"),
    "OTE": ("open tender enquiry", "open tender", "ote"),
    "LPC": ("local purchase committee", "local purchase", "lpc"),
    "Direct Purchase": ("direct purchase", "direct procurement"),
}

_METHOD_DETAILS: dict[str, dict[str, str]] = {
    "STE": {
        "label": "Single Tender Enquiry (STE)",
        "purpose": "Exceptional route used where the case is justified from the start on single-source grounds such as proprietary supply, standardisation, or urgency with recorded reasons.",
        "competition": "No open competition; enquiry is addressed to one identified source.",
        "trigger": "Use only when the file records a valid exception such as PAC / proprietary justification, compatibility need, standardisation, or another specifically defensible reason.",
        "approval": "Technical justification plus competent authority approval; PAC or equivalent record is normally expected.",
        "documents": "PAC / proprietary certificate, exception justification note, approval note, and price reasonableness record.",
        "source": "Rule 166 / single tender exception principles in GFR 2017 (as amended) and CSIR procurement guidance.",
    },
    "LTE": {
        "label": "Limited Tender Enquiry (LTE)",
        "purpose": "Competitive route where a limited number of suitable vendors are invited to quote.",
        "competition": "Competition is retained, but only among the shortlisted or known capable vendors.",
        "trigger": "Used when the value band and procurement profile permit LTE and the case does not require open publicity.",
        "approval": "Tender processing through the competent purchase committee with comparative evaluation and approval on file.",
        "documents": "Tender enquiry, bidder list, quotations / bids, comparative statement, committee recommendation, and approval note.",
        "source": "Rule 162 / limited tender route principles in GFR 2017 (as amended) and CSIR procurement guidance.",
    },
    "OTE": {
        "label": "Open Tender Enquiry (OTE)",
        "purpose": "Competitive route with wide publicity so all eligible suppliers can participate.",
        "competition": "Open competition with broad market access and formal tender evaluation.",
        "trigger": "Used when the value band or procurement conditions require wide publicity rather than limited invitation.",
        "approval": "Tender approval, publication, committee evaluation, and competent authority approval are required.",
        "documents": "Tender notice, published bid documents, technical and commercial evaluation, and approval note.",
        "source": "Rule 161 / open tender route principles in GFR 2017 (as amended).",
    },
    "LPC": {
        "label": "Local Purchase Committee (LPC)",
        "purpose": "Committee-based quotation route for lower-value procurement bands.",
        "competition": "Competition through collection and comparison of quotations, typically through the LPC.",
        "trigger": "Used in the LPC value band where committee quotation comparison is required instead of direct purchase.",
        "approval": "Local Purchase Committee recommendation with competent authority approval.",
        "documents": "Quotations, comparative statement, LPC proceedings, and approval note.",
        "source": "Rule 155 / LPC route principles in GFR 2017 (as amended).",
    },
    "Direct Purchase": {
        "label": "Direct Purchase",
        "purpose": "Lowest-value route where direct procurement is allowed subject to reasonableness and record requirements.",
        "competition": "No formal quotation competition is mandatory at the lowest band, but market reasonableness must be recorded.",
        "trigger": "Used only within the direct-purchase value limit.",
        "approval": "Competent officer approval with record of source and price reasonableness.",
        "documents": "Indent, price reasonableness basis, approval record, and supply order.",
        "source": "Rule 154 / direct purchase principles in GFR 2017 (as amended).",
    },
}


def _format_rupees(amount_rupees: float | None) -> str:
    if amount_rupees is None:
        return "Not specified"
    if float(amount_rupees).is_integer():
        return f"Rs. {int(amount_rupees):,}"
    return f"Rs. {amount_rupees:,.2f}"


def _committee_for_mode(mode: str) -> str:
    mapping = {
        "Direct Purchase": "Not required (single officer, value <= Rs. 2,00,000)",
        "LPC": "Local Purchase Committee (LPC) - min. 3 members",
        "LTE": "Technical & Purchase Committee (T&PC) - min. 3 members",
        "OTE": "Technical & Purchase Committee (T&PC) - min. 3 members",
    }
    return mapping.get(mode, "Purchase Committee - as per DFP")


def normalize_amount(text: str) -> float | None:
    t = text.lower().replace(",", "").replace("â‚¹", "").replace("rs", "").strip()
    t = re.sub(r"\binr\b", "", t).strip()

    if "crore" in t:
        m = re.search(r"(\d+(?:\.\d+)?)\s*crore", t)
        if m:
            return float(m.group(1)) * 10000000.0

    if "lakh" in t:
        m = re.search(r"(\d+(?:\.\d+)?)\s*lakh", t)
        if m:
            return float(m.group(1)) * 100000.0

    num = "".join(c for c in t if c.isdigit() or c == ".")
    num = num.strip(".")
    if num.count(".") > 1:
        parts = num.split(".")
        num = "".join(parts[:-1]) + "." + parts[-1]
    return float(num) if num else None


def get_procurement_mode(amount: float | None) -> str:
    if amount is None:
        return "UNKNOWN"
    if amount <= 200000:
        return "Direct Purchase"
    if amount <= 500000:
        return "LPC"
    if amount <= 5000000:
        return "LTE"
    return "OTE"


def rule_engine_from_amount(amount_rupees: float | None) -> dict[str, Any]:
    if amount_rupees is None:
        return {
            "mode": "UNKNOWN",
            "rule_number": None,
            "band_label": "Not determined in query",
        }
    mode = get_procurement_mode(amount_rupees)
    if mode == "Direct Purchase":
        return {"mode": mode, "rule_number": "154", "band_label": "<= Rs. 2,00,000"}
    if mode == "LPC":
        return {"mode": mode, "rule_number": "155", "band_label": "> Rs. 2,00,000 and <= Rs. 5,00,000"}
    if mode == "LTE":
        return {"mode": mode, "rule_number": "162", "band_label": "> Rs. 5,00,000 and <= Rs. 50,00,000"}
    return {"mode": mode, "rule_number": "161", "band_label": "> Rs. 50,00,000"}


def detect_concept_query(query: str) -> str | None:
    q = query.lower()
    has_open_tender = ("open tender enquiry" in q) or ("ote" in q)
    has_limited_tender = ("limited tender enquiry" in q) or ("limited tender" in q) or ("lte" in q)
    has_local_purchase = ("local purchase committee" in q) or ("lpc" in q)
    has_direct_purchase = ("direct purchase" in q) or ("direct procurement" in q)

    if has_open_tender and ("mandatory" in q or "when" in q):
        return "OTE_THRESHOLD"

    if has_limited_tender and ("limit" in q or "upper" in q):
        return "LTE_LIMIT"

    if has_local_purchase and ("limit" in q or "upper" in q):
        return "LPC_LIMIT"

    if has_direct_purchase and ("limit" in q or "upper" in q or "threshold" in q):
        return "DIRECT_LIMIT"

    wants_any_band = any(k in q for k in ("direct purchase", "local purchase committee", "limited tender", "open tender"))
    if ("slab" in q) and not wants_any_band:
        return "ALL_THRESHOLDS"
    return None


def handle_concept(concept: str) -> dict[str, str]:
    if concept == "OTE_THRESHOLD":
        return {
            "mode": "OTE",
            "threshold_eval": "Open Tender Enquiry (OTE) is used for procurements above Rs. 50 lakh.",
        }
    if concept == "LTE_LIMIT":
        return {
            "mode": "LTE",
            "threshold_eval": "Limited Tender Enquiry (LTE) applies up to Rs. 50 lakh.",
        }
    if concept == "LPC_LIMIT":
        return {
            "mode": "LPC",
            "threshold_eval": "Local Purchase Committee (LPC) applies up to Rs. 5 lakh.",
        }
    if concept == "DIRECT_LIMIT":
        return {
            "mode": "Direct Purchase",
            "threshold_eval": "Direct Purchase applies up to Rs. 2 lakh.",
        }
    return {
        "mode": "Direct Purchase",
        "threshold_eval": (
            "Procurement Thresholds:\n"
            "- Direct Purchase: up to Rs. 2 lakh\n"
            "- LPC: up to Rs. 5 lakh\n"
            "- LTE: up to Rs. 50 lakh\n"
            "- OTE: above Rs. 50 lakh"
        ),
    }


def _is_comparison_query(query: str) -> bool:
    q = query.lower()
    return "compare" in q or " vs " in q or " versus " in q


def compare_amounts(a_text: str, b_text: str) -> dict[str, Any]:
    a = normalize_amount(a_text)
    b = normalize_amount(b_text)
    mode_a = get_procurement_mode(a)
    mode_b = get_procurement_mode(b)
    return {
        "A_value": a,
        "B_value": b,
        "A_mode": mode_a,
        "B_mode": mode_b,
    }


def _extract_two_amount_parts(query: str) -> tuple[str, str] | None:
    normalized = query.lower().replace(" versus ", " vs ")
    if " vs " in normalized:
        parts = re.split(r"\bvs\b", normalized, maxsplit=1)
        if len(parts) == 2:
            return parts[0], parts[1]
    if "compare" in normalized and "and" in normalized:
        after_compare = normalized.split("compare", 1)[1]
        parts = after_compare.split("and", 1)
        if len(parts) == 2:
            return parts[0], parts[1]
    return None


def _extract_method_mentions(query: str) -> list[str]:
    lowered = query.lower()
    found: list[str] = []
    for canonical, aliases in _METHOD_ALIASES.items():
        for alias in aliases:
            if alias in lowered:
                found.append(canonical)
                break
    return found


def _is_conceptual_comparison_query(query: str) -> bool:
    return _is_comparison_query(query) and normalize_amount(query) is None and len(_extract_method_mentions(query)) >= 2


def _is_workflow_query(query: str) -> bool:
    lowered = query.lower()
    return any(
        term in lowered
        for term in (
            "workflow",
            "approval workflow",
            "approval process",
            "committee approval",
            "process",
            "procedure",
            "steps",
            "how does",
            "how do",
            "summarize",
            "summary",
            "simple language",
        )
    )


def _context_points(context: str, limit: int = 4) -> list[str]:
    snippets: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", context):
        cleaned = re.sub(r"\s+", " ", sentence).strip()
        if len(cleaned) < 40:
            continue
        snippets.append(cleaned)
        if len(snippets) >= limit:
            break
    return snippets


def _clean_context_text(text: str) -> str:
    text = text.replace("\u2028", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\bGFR\s*2025\b", "GFR 2017 (as amended)", text, flags=re.IGNORECASE)
    return text


def _merge_context(chunks: list[str]) -> str:
    chunks = [_clean_context_text(c) for c in chunks if c and len(c.strip()) >= 40]
    if not chunks:
        return ""
    merged = merge_adjacent_chunks(chunks, max_merged_chars=1200)
    merged = [c for c in merged if c.strip()]
    return "\n".join(merged)[:2500]


def _retrieve_context(query: str, top_k: int = 3) -> tuple[str, list[str], list[Any]]:
    matches = knowledge_base.search(query, top_k=top_k)
    file_names = []
    for match in matches:
        try:
            if match.file_name and match.file_name not in file_names:
                file_names.append(match.file_name)
        except Exception:
            continue
    context = _merge_context([match.content for match in matches])
    return context, file_names, matches


def _build_method_comparison_block(methods: list[str]) -> str:
    if len(methods) < 2:
        return "- Not applicable."

    left = _METHOD_DETAILS.get(methods[0], {})
    right = _METHOD_DETAILS.get(methods[1], {})
    left_label = left.get("label", methods[0])
    right_label = right.get("label", methods[1])

    rows = [
        ("Primary use", left.get("purpose", "Not found in context."), right.get("purpose", "Not found in context.")),
        ("Competition style", left.get("competition", "Not found in context."), right.get("competition", "Not found in context.")),
        ("When used", left.get("trigger", "Not found in context."), right.get("trigger", "Not found in context.")),
        ("Approval / justification", left.get("approval", "Not found in context."), right.get("approval", "Not found in context.")),
        ("Typical records", left.get("documents", "Not found in context."), right.get("documents", "Not found in context.")),
    ]

    lines = [
        f"| Aspect | {left_label} | {right_label} |",
        "| --- | --- | --- |",
    ]
    lines.extend(f"| {aspect} | {left_value} | {right_value} |" for aspect, left_value, right_value in rows)
    return "\n".join(lines)


def _build_structured_response(
    amount_rupees: float | None,
    mode_info: dict[str, Any],
    threshold_eval: str,
    comparison_block: str,
    retrieved_files: list[str] | None = None,
) -> str:
    mode = mode_info.get("mode", "UNKNOWN")
    rule_number = mode_info.get("rule_number", "")
    band_label = mode_info.get("band_label", "Not determined")
    value_label = _format_rupees(amount_rupees)
    committee = _committee_for_mode(mode)
    source_files = retrieved_files or []
    files_line = (
        f"- Retrieved documents (for explanation only): {', '.join(source_files)}"
        if source_files
        else "- Retrieved documents (for explanation only): Not found / not used"
    )

    why_bullets = [
        f"- The parsed value is {value_label}.",
        f"- This falls in band: {band_label}.",
        f"- Therefore deterministic mode is: {mode}.",
        "- Rule engine is final authority; no LLM mode decision is used.",
    ]
    detailed_steps = [
        "1. Receive user query.",
        "2. Detect concept intent (if present).",
        "3. Parse amount deterministically.",
        "4. Apply threshold rule engine to get mode.",
        "5. Build structured response and validate output.",
    ]
    rule_line = f"- Rule applied: {rule_number}" if rule_number else "- Rule applied: Not explicit"

    return "\n".join(
        [
            "## Quick Answer",
            f"- Applicable mode: {mode}",
            f"- Committee: {committee}",
            "",
            "## Amount Breakdown",
            f"- Parsed amount: {value_label}",
            f"- Threshold band: {band_label}",
            "",
            "## Threshold Evaluation",
            threshold_eval,
            rule_line,
            "",
            "## Why This Applies",
            *why_bullets,
            "",
            "## Detailed Process",
            *detailed_steps,
            "",
            "## Comparison",
            comparison_block or "- Not applicable.",
            "",
            "## Source Basis",
            "- GFR 2017 (as amended)",
            files_line,
            "",
            "## TL;DR",
            f"- {mode} applies based on threshold rules.",
            "- FINAL DECISION: VERIFY",
        ]
    )


def _build_conceptual_comparison_response(methods: list[str], retrieved_files: list[str] | None = None) -> str:
    left = _METHOD_DETAILS.get(methods[0], {})
    right = _METHOD_DETAILS.get(methods[1], {})
    left_label = left.get("label", methods[0])
    right_label = right.get("label", methods[1])
    files_line = (
        f"- Retrieved documents (for explanation only): {', '.join(retrieved_files)}"
        if retrieved_files
        else "- Retrieved documents (for explanation only): Not found / not used"
    )

    return "\n".join(
        [
            "## Quick Answer",
            f"- Query type: Conceptual comparison between {left_label} and {right_label}",
            "- Applicable mode: Not a single amount-based route decision",
            "- Committee: Depends on which route is actually chosen in the case file",
            "",
            "## Amount Breakdown",
            "- Parsed amount: Not specified in the query",
            "- Threshold band: Not applicable for a pure route-to-route comparison",
            "",
            "## Threshold Evaluation",
            f"- {left_label} and {right_label} should be compared on legal basis, competition model, and approval requirements rather than by assigning a purchase value.",
            "- Rule applied: Conceptual comparison only",
            "",
            "## Why This Applies",
            f"- {left_label} is an exception-based route, while {right_label} is a competition-based route.",
            f"- {left_label} needs special justification on file; {right_label} normally relies on comparative bidding among a limited set of vendors.",
            "- The deciding factor is not a default zero amount, but whether the facts justify an exception or require competitive enquiry.",
            "",
            "## Detailed Process",
            "1. Identify whether the user is asking for a concept comparison or a live procurement route decision.",
            f"2. Check the legal trigger for {left_label} and the normal applicability conditions for {right_label}.",
            "3. Compare the level of competition, justification burden, and committee scrutiny.",
            "4. Record the required documents and approval path before choosing either route in a real case.",
            "",
            "## Comparison",
            _build_method_comparison_block(methods),
            "",
            "## Source Basis",
            f"- {left.get('source', 'GFR 2017 (as amended)')}",
            f"- {right.get('source', 'GFR 2017 (as amended)')}",
            files_line,
            "",
            "## TL;DR",
            f"- {left_label} is an exception route; {right_label} is a competitive route.",
            "- FINAL DECISION: VERIFY",
        ]
    )


def _build_generic_context_response(
    query: str,
    context: str,
    retrieved_files: list[str] | None = None,
    workflow_mode: bool = False,
) -> str:
    points = _context_points(context, limit=4)
    files_line = (
        f"- Retrieved documents (for explanation only): {', '.join(retrieved_files)}"
        if retrieved_files
        else "- Retrieved documents (for explanation only): Not found / not used"
    )
    quick_answer = (
        "In simple terms, the file usually moves from technical checking to committee scrutiny, then finance / competent authority approval, and finally issue of the purchase order."
        if workflow_mode
        else "This is a procedural / conceptual procurement query, so the answer should come from the approval flow and governing records rather than an amount band."
    )
    why_lines = points[:3] or ["- Not found in retrieved context."]
    process_lines = (
        [
            "1. Indenting side prepares the requirement and technical basis.",
            "2. The technical / purchase committee checks suitability and comparative position.",
            "3. Finance or competent authority reviews the recommendation where required.",
            "4. After approval, the order is issued and the record is kept on file.",
        ]
        if workflow_mode
        else [
            "1. Identify the controlling procurement question and record the facts.",
            "2. Check the relevant committee, approval, and supporting documents.",
            "3. Apply the governing rule from the retrieved procurement context.",
            "4. Record the final recommendation and approval on file.",
        ]
    )

    return "\n".join(
        [
            "## Quick Answer",
            f"- {quick_answer}",
            "",
            "## Amount Breakdown",
            "- Parsed amount: Not specified in the query",
            "- Threshold band: Not required for this workflow / conceptual answer",
            "",
            "## Threshold Evaluation",
            "- This query should not be resolved by assigning an UNKNOWN procurement mode.",
            "- Rule applied: Context-driven workflow / process summary",
            "",
            "## Why This Applies",
            *[f"- {line}" for line in why_lines],
            "",
            "## Detailed Process",
            *process_lines,
            "",
            "## Comparison",
            "- Not applicable.",
            "",
            "## Source Basis",
            files_line,
            "- GFR 2017 (as amended)",
            "",
            "## TL;DR",
            "- This answer is based on the approval flow and retrieved procurement context, not on an amount threshold.",
            "- FINAL DECISION: VERIFY",
        ]
    )


def _amount_prompt(
    query: str,
    mode: str,
    numeric_value: str,
    committee_details: str,
    range_bracket: str,
    lower_mode: str,
    higher_mode: str,
    context: str,
) -> str:
    return f"""Role: You are the ProcureBuddy Core Engine. Your only job is to format procurement data into a professional, audit-ready report.

STRICT DATA CONSTRAINTS:
- Injected Mode: The procurement mode {mode} and amount {numeric_value} have been pre-validated by our deterministic engine. You MUST NOT change them.
- No Hallucinations: Do not mention GFR 2025 or any non-existent rules. Stick to GFR 2017/CSIR 2019 guidelines provided in context.
- No Omissions: Every single one of the 9 sections below must be present.

Context: {context[:1500]}

REQUIRED OUTPUT FORMAT:

## Quick Answer
Purchase value: â‚¹{numeric_value}
Applicable mode: {mode}
Committee: {committee_details}

## Amount Breakdown
Input amount: â‚¹{query}
Normalized amount: â‚¹{numeric_value}
Comparison:
â‰¤ â‚¹2,00,000 â†’ Direct Purchase
â‰¤ â‚¹5,00,000 â†’ LPC (Local Purchase Committee)
â‰¤ â‚¹50,00,000 â†’ LTE (Limited Tender Enquiry)
> â‚¹50,00,000 â†’ OTE (Open Tender Enquiry)

## Threshold Evaluation
Given amount falls under: {range_bracket}
Therefore applicable mode: {mode}
Boundary check: Checked and validated
Rule applied correctly: Yes

## Why This Applies
- [Explain why {mode} is chosen for â‚¹{numeric_value}]
- [Mention transparency and competition levels]
- [Cite relevant financial threshold from GFR]
- [Explain why a higher/lower mode was rejected]

## Detailed Process
1. Identify procurement requirement.
2. Estimate cost and confirm threshold band.
3. Select {mode} based on threshold.
4. [Add specific step for {mode}]
5. Obtain final approval and issue order.

## Key Documents / Outputs
- Purchase Requisition & Cost Estimate
- Tender/Quotation Documents
- Committee Minutes (if applicable)
- Comparative Statement & Approval Note

## Comparison
Amount | Mode | Reason
--- | --- | ---
< â‚¹{numeric_value} | {lower_mode} | Below threshold
â‚¹{numeric_value} | {mode} | Current Selection
> â‚¹{numeric_value} | {higher_mode} | Exceeds threshold

## Source Basis
- GFR 2017 (as amended)
- CSIR 2019 Procurement Guidelines

## TL;DR
Mode: {mode}
Reason: Based on strict threshold validation.
Status: 100% GFR Compliant.
FINAL DECISION: VERIFY
"""


def _comparison_prompt(query: str, methods: list[str], context: str) -> str:
    left = _METHOD_DETAILS.get(methods[0], {})
    right = _METHOD_DETAILS.get(methods[1], {})
    left_label = left.get("label", methods[0])
    right_label = right.get("label", methods[1])
    return f"""Role: You are the ProcureBuddy Core Engine. Format this as a conceptual procurement comparison, not as a threshold or amount decision.

STRICT DATA CONSTRAINTS:
- No amount was provided. Do not invent a purchase value, normalized amount, or threshold band.
- The user wants a table comparing {left_label} and {right_label}.
- Use GFR 2017 (as amended) / CSIR procurement language only. Do not mention GFR 2025.
- Keep all required section headings exactly as given below.

Context: {context[:1500]}

REQUIRED OUTPUT FORMAT:

## Quick Answer
Query type: Conceptual comparison between {left_label} and {right_label}
Applicable mode: Not a single amount-based route decision
Committee: Depends on the route actually chosen

## Amount Breakdown
Parsed amount: Not specified in the query
Threshold band: Not applicable for a pure route comparison

## Threshold Evaluation
State that this is a route-to-route comparison, not an amount-band decision.
Rule applied: Conceptual comparison only

## Why This Applies
- Explain the core difference between {left_label} and {right_label}
- Explain competition model
- Explain approval / justification difference

## Detailed Process
1. Identify whether the case really qualifies for {left_label} or should follow {right_label}.
2. Check the controlling rule and facts.
3. Record justification, competition basis, and approvals.
4. Finalize the route only after the file supports it.

## Comparison
Return a markdown table with columns: Aspect | {left_label} | {right_label}
Include rows for Primary use, Competition style, When used, Approval / justification, Typical records.

## Source Basis
- GFR 2017 (as amended)
- CSIR 2019 Procurement Guidelines

## TL;DR
Summarize the difference in 1-2 lines.
FINAL DECISION: VERIFY
"""


def _generic_prompt(query: str, context: str, workflow_mode: bool) -> str:
    if workflow_mode:
        return f"""Role: You are the ProcureBuddy Core Engine. Answer this procurement workflow question in simple language using only the retrieved context.

STRICT DATA CONSTRAINTS:
- No amount was provided. Do not invent a purchase value or threshold band.
- This is a workflow / approval-chain question, not a threshold-routing question.
- Use GFR 2017 (as amended) / CSIR procurement language only.
- Keep all section headings exactly as given below.

Context: {context[:1500]}
Question: {query}

REQUIRED OUTPUT FORMAT:

## Quick Answer
Give a 1-2 line plain-language summary of the committee approval workflow.

## Amount Breakdown
Say that no amount is specified and threshold routing is not required for this answer.

## Threshold Evaluation
State that this is a workflow summary, not an amount-band decision.

## Why This Applies
- Explain the role of the technical / purchase side
- Explain where approval or concurrence enters
- Keep the language simple

## Detailed Process
1. Start from requirement / indent.
2. Move through technical / committee scrutiny.
3. Mention approval / concurrence.
4. End with order / record.

## Comparison
Write: Not applicable.

## Source Basis
- Cite the retrieved document names only.

## TL;DR
Summarize the flow in one line.
FINAL DECISION: VERIFY
"""

    return f"""Role: You are the ProcureBuddy Core Engine. Answer this procurement concept question from the retrieved context without forcing it into a threshold template.

STRICT DATA CONSTRAINTS:
- No amount was provided. Do not invent a purchase value or threshold band.
- Use GFR 2017 (as amended) / CSIR procurement language only.
- Keep all section headings exactly as given below.

Context: {context[:1500]}
Question: {query}

REQUIRED OUTPUT FORMAT:

## Quick Answer
Answer the question directly in 1-2 lines.

## Amount Breakdown
Say that no amount is specified and an amount threshold is not required for this answer.

## Threshold Evaluation
State that this is a conceptual / procedural answer, not an amount-band decision.

## Why This Applies
- Explain the main grounded points from context

## Detailed Process
1. Give the practical officer checklist only if relevant.
2. Keep it concise and grounded.

## Comparison
Write: Not applicable, unless the question itself asks for a comparison.

## Source Basis
- Cite the retrieved document names only.

## TL;DR
Summarize in one line.
FINAL DECISION: VERIFY
"""


def run_v2_flow(query: str) -> dict[str, Any]:
    concept = detect_concept_query(query)
    conceptual_methods = _extract_method_mentions(query)
    is_conceptual_comparison = _is_conceptual_comparison_query(query)
    is_workflow = _is_workflow_query(query)
    retrieved_files: list[str] = []
    retrieved_docs: list[Any] = []
    comparison_block = "- Not applicable."
    generic_non_amount = False

    if is_conceptual_comparison:
        amount_rupees = None
        mode_info = {"mode": "GENERAL", "rule_number": "", "band_label": "Conceptual comparison"}
        threshold_eval = (
            "This query compares procurement routes conceptually. "
            "Do not force an amount band or treat the question as a threshold lookup."
        )
        comparison_block = _build_method_comparison_block(conceptual_methods[:2])
    elif concept:
        concept_data = handle_concept(concept)
        amount_rupees = None
        mode_info = {"mode": concept_data["mode"], "rule_number": "", "band_label": "Concept query"}
        threshold_eval = concept_data["threshold_eval"]
    else:
        amount_rupees = normalize_amount(query)
        if amount_rupees is None:
            generic_non_amount = True
            mode_info = {"mode": "GENERAL", "rule_number": "", "band_label": "Not applicable"}
            threshold_eval = (
                "No procurement amount is present in the query, so the answer should come from workflow / policy context rather than threshold routing."
            )
        else:
            mode_info = rule_engine_from_amount(amount_rupees)
            threshold_eval = (
                f"{mode_info['mode']} applies for band: {mode_info['band_label']}."
                if mode_info["mode"] != "UNKNOWN"
                else "Could not determine mode."
            )

    mode = mode_info["mode"]
    context = ""

    def fetch_rag() -> tuple[str, list[str], list[Any]]:
        return _retrieve_context(query, top_k=2)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        rag_future = executor.submit(fetch_rag)

        if _is_comparison_query(query) and not is_conceptual_comparison:
            pair = _extract_two_amount_parts(query)
            if pair:
                cmp_data = compare_amounts(pair[0], pair[1])
                comparison_block = "\n".join(
                    [
                        f"- A value: {_format_rupees(cmp_data['A_value'])}",
                        f"- B value: {_format_rupees(cmp_data['B_value'])}",
                        f"- A mode: {cmp_data['A_mode']}",
                        f"- B mode: {cmp_data['B_mode']}",
                        "- Difference: Higher amount maps to higher scrutiny.",
                    ]
                )

        try:
            context_str, retrieved_files, retrieved_docs = rag_future.result(timeout=3.0)
            if context_str:
                threshold_eval += (
                    " Retrieved context used to support the conceptual distinction."
                    if is_conceptual_comparison
                    else " Retrieved context used only for explanation."
                )
                context = context_str
        except concurrent.futures.TimeoutError:
            logger.warning("RAG retrieval timed out, continuing without context.")

    if is_conceptual_comparison:
        rule_based_rendered = _build_conceptual_comparison_response(
            conceptual_methods[:2],
            retrieved_files=retrieved_files,
        )
    elif generic_non_amount:
        rule_based_rendered = _build_generic_context_response(
            query=query,
            context=context,
            retrieved_files=retrieved_files,
            workflow_mode=is_workflow,
        )
    else:
        rule_based_rendered = _build_structured_response(
            amount_rupees=amount_rupees,
            mode_info=mode_info,
            threshold_eval=threshold_eval,
            comparison_block=comparison_block,
            retrieved_files=retrieved_files,
        )

    numeric_value = f"{amount_rupees:,.2f}" if amount_rupees is not None else "Not specified"
    committee_details = _committee_for_mode(mode)
    range_bracket = mode_info.get("band_label", "Concept query")

    if amount_rupees is not None:
        if amount_rupees <= 200000:
            lower_mode, higher_mode = "N/A", "LPC"
        elif amount_rupees <= 500000:
            lower_mode, higher_mode = "Direct Purchase", "LTE"
        elif amount_rupees <= 5000000:
            lower_mode, higher_mode = "LPC", "OTE"
        else:
            lower_mode, higher_mode = "LTE", "N/A"
    else:
        lower_mode, higher_mode = "N/A", "N/A"

    prompt = (
        _comparison_prompt(query, conceptual_methods[:2], context)
        if is_conceptual_comparison
        else _generic_prompt(query, context, is_workflow)
        if generic_non_amount
        else _amount_prompt(
            query=query,
            mode=mode,
            numeric_value=numeric_value,
            committee_details=committee_details,
            range_bracket=range_bracket,
            lower_mode=lower_mode,
            higher_mode=higher_mode,
            context=context,
        )
    )

    llm_response = None
    if not is_conceptual_comparison:
        llm_exec: concurrent.futures.ThreadPoolExecutor | None = None
        future: concurrent.futures.Future[str | None] | None = None
        try:
            llm_exec = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            future = llm_exec.submit(generate_llm_response, prompt)
            llm_response = future.result(timeout=15.0)
        except concurrent.futures.TimeoutError:
            logger.error("LLM timed out after 15 seconds. Falling back to deterministic response.")
            llm_response = None
            if future is not None:
                future.cancel()
        finally:
            if llm_exec is not None:
                llm_exec.shutdown(wait=False, cancel_futures=True)

    if not llm_response or "## Quick Answer" not in llm_response:
        rendered = rule_based_rendered
        generation_mode = "rule_based"
    else:
        rendered = llm_response
        generation_mode = "llm"

    amount_lakhs = None if amount_rupees is None else amount_rupees / 100000.0
    expected_mode = None if (is_conceptual_comparison or generic_non_amount) else mode_info["mode"]
    report = validate_structured_output(rendered, amount_lakhs=amount_lakhs, expected_mode=expected_mode)

    if not report.is_valid:
        logger.warning("Validation failed; using deterministic failsafe.")
        rendered = rule_based_rendered
        generation_mode = "rule_based"

    return {
        "intent": "ANALYTICAL" if is_conceptual_comparison else "WORKFLOW" if is_workflow else "GENERAL" if generic_non_amount else "PROCESS",
        "amount": amount_rupees,
        "generation": rendered,
        "generation_mode": generation_mode,
        "documents": retrieved_docs,
        "metadata": {
            "mode": mode,
            "validation": str(report),
            "retrieved_files": retrieved_files,
            "conceptual_comparison": is_conceptual_comparison,
            "generic_non_amount": generic_non_amount,
        },
    }
