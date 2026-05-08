"""Response rendering, section parsing, and text selection helpers."""

from __future__ import annotations

import re

from app.core.constants import GFR_2025_DOCUMENT_NAME, SECTION_MARKER
from app.utils.processors import (
    extract_amount_lakhs,
    format_lakh_amount,
    describe_user_request,
    score_amount_relevance,
    score_process_relevance,
    score_analytical_relevance,
    tokenize,
    split_sentences,
    semantic_dedup_key,
)
from app.utils.text_cleaner import (
    clean_text,
    audit_chunk_quality,
    looks_like_table_of_contents,
    smooth_summary_text,
    strip_leading_noise,
    ensure_sentence_punctuation,
    is_useful_sentence,
    rewrite_procurement_sentence,
    legalistic_noise_penalty,
)


def build_intent_prompt_guidance(intent: str) -> str:
    """Return dynamic prompt guidance snippets for the current intent."""
    if intent == "WORKFLOW":
        return "Give only the essential workflow sequence grounded in the retrieved procurement text."
    if intent == "PROCESS":
        return "State the applicable rule or threshold clearly, then give only the necessary grounded steps."
    if intent == "ANALYTICAL":
        return "Answer the distinction directly from the retrieved rule text. Avoid separate concept, implication, or audit breakdowns."
    return "Use only the grounded procurement context and ignore decorative lists or annexure-style references."


def split_primary_and_explanation(text: str) -> tuple[str, str]:
    """Split text into a direct first sentence and a preserved explanation tail."""
    cleaned = clean_text(text)
    if not cleaned:
        return "", ""
    protected = cleaned.replace("Rs.", "Rs<dot>")
    sentences = [s.replace("Rs<dot>", "Rs.") for s in split_sentences(protected)]
    if not sentences:
        return smooth_summary_text(cleaned, max_length=420), ""
    primary = smooth_summary_text(sentences[0], max_length=320)
    explanation = smooth_summary_text(" ".join(sentences[1:5]), max_length=560)
    return primary, explanation


def render_compact_response(direct_answer: str, explanation: str, final_decision: str) -> str:
    """Render the end-user answer format while preserving source lines."""
    direct_body, direct_source = _split_source_line(direct_answer)
    explanation_body, explanation_source = _split_source_line(explanation)
    source_line = explanation_source or direct_source

    cleaned_direct = compact_answer_text(direct_body, max_sentences=3, max_length=420) or cleanup_generated_sentence(direct_body)
    cleaned_explanation = smooth_summary_text(clean_text(explanation_body), max_length=560)
    lines: list[str] = []
    if cleaned_direct:
        lines.append(cleaned_direct)
    if cleaned_explanation and cleaned_explanation.lower() != cleaned_direct.lower():
        lines.extend(["", cleaned_explanation])
    if source_line:
        lines.extend(["", source_line])
    lines.extend(["", f"FINAL DECISION: {final_decision}"])
    return "\n".join(lines).strip()


def _split_source_line(value: str) -> tuple[str, str]:
    """Separate any trailing Source: citation from body text."""
    cleaned = clean_text(value).strip()
    if not cleaned:
        return "", ""
    match = re.search(r"(?:^|\s)(Source:\s*.+)$", cleaned, flags=re.IGNORECASE)
    if not match:
        return cleaned, ""
    body = cleaned[:match.start()].strip().rstrip(".")
    source = match.group(1).strip()
    if not source.endswith("."):
        source += "."
    return body, source


def infer_procurement_method(answer: str, explanation_points: list[str]) -> str:
    """Infer the procurement route from answer text."""
    combined = clean_text(" ".join([answer, *explanation_points])).lower()
    if "single tender" in combined or " ste" in f" {combined}" or "proprietary" in combined:
        return "STE"
    if "limited tender" in combined or " lte" in f" {combined}":
        return "LTE"
    if "local purchase committee" in combined or " lpc" in f" {combined}":
        return "LPC"
    if "open tender" in combined or " ote" in f" {combined}":
        return "OTE"
    if "direct purchase" in combined:
        return "DIRECT PURCHASE"
    return ""


def build_default_procedural_steps(intent: str, answer: str, explanation_points: list[str]) -> str:
    """Build safe fallback steps when the LLM omits them."""
    method = infer_procurement_method(answer, explanation_points)
    if intent == "WORKFLOW":
        return (
            "1. Identify the requirement and prepare the indent.\n"
            "2. Obtain technical scrutiny through the competent technical committee, where applicable.\n"
            "3. Check GeM availability and confirm the applicable procurement route.\n"
            "4. Prepare quotations or tender documents and place the case before the competent purchase committee.\n"
            "5. Complete evaluation and obtain finance concurrence, where required.\n"
            "6. Obtain competent approval and issue the order."
        )
    if intent == "PROCESS":
        if method == "LTE":
            return (
                "1. Confirm the value falls in the LTE band under Rule 162 and check GeM applicability.\n"
                "2. Shortlist capable firms and issue the limited tender enquiry.\n"
                "3. Receive and compare the quotations through the competent purchase committee.\n"
                "4. Record the recommendation and obtain approval from the competent authority.\n"
                "5. Issue the purchase order and place the tender record on file."
            )
        if method == "LPC":
            return (
                "1. Confirm the value falls in the LPC band under Rule 155 and check GeM applicability.\n"
                "2. Obtain at least three quotations through the Local Purchase Committee.\n"
                "3. Prepare the comparative statement and record the committee recommendation.\n"
                "4. Obtain approval from the competent authority.\n"
                "5. Issue the purchase order and retain the committee record."
            )
        if method == "STE":
            return (
                "1. Record the single-source justification, such as PAC, standardisation, or urgency.\n"
                "2. Obtain the required certificate or approval note from the competent authority.\n"
                "3. Check GeM or competing-source feasibility and record why it does not apply.\n"
                "4. Process the case through finance or purchase scrutiny, as applicable.\n"
                "5. Approve and place the order through the single-tender route."
            )
        if method == "OTE":
            return (
                "1. Confirm the value falls above the LTE band and check GeM or CPP Portal applicability.\n"
                "2. Prepare the tender with wide publicity and publish it through the approved channel.\n"
                "3. Receive and evaluate bids through the competent committee.\n"
                "4. Record the recommendation and obtain approval from the competent authority.\n"
                "5. Issue the contract or purchase order and preserve the tender record."
            )
        if method == "DIRECT PURCHASE":
            return (
                "1. Confirm the value is within the direct purchase limit and check GeM availability.\n"
                "2. Record the market-rate justification for the identified source.\n"
                "3. Confirm why quotation-based procurement is not required for this case.\n"
                "4. Obtain the competent approval for the direct purchase.\n"
                "5. Issue the purchase order and maintain the supporting record."
            )
    if intent == "ANALYTICAL":
        return (
            "1. Confirm the applicable route and threshold for each method.\n"
            "2. Record the approval and justification required for each route.\n"
            "3. Follow the relevant quotation or tender issue, evaluation, and approval steps."
        )
    return ""


def parse_llm_sections(response: str) -> dict[str, str]:
    """Parse a structured LLM response into named sections."""
    normalized = clean_text(response)
    heading_pattern = rf"(?:^|\n)?\s*(?:#+\s*)?(?:{re.escape(SECTION_MARKER)}|[\u25b9\U0001F539])?"
    section_names = (
        "Status"
        "|Analysis"
        "|Audit Risk"
        "|Actionable Step"
        "|Required Evidence"
        "|Evidence"
        "|Verdict"
        "|Action"
        "|Risk"
        "|Required Proof"
        "|Proof"
        "|"
        "Quick Answer|Direct Answer|Answer"
        "|Rule Priority"
        "|Comparison Table|Markdown Table"
        "|Why This Applies"
        "|Detailed Explanation|Explanation"
        "|Process|Procedural Steps"
        "|Source Basis"
        "|Supporting Context"
        "|Sources|Source Citation"
        "|Pro-Tip|Pro Tip|Caution"
        "|Confidence"
    )
    pattern = re.compile(
        heading_pattern + r"\s*(" + section_names + r")\s*:?\s*",
        re.IGNORECASE,
    )
    matches = list(pattern.finditer(normalized))
    if not matches:
        return {}

    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        key = match.group(1).strip().lower()
        if key == "verdict":
            key = "status"
        elif key == "risk":
            key = "audit risk"
        elif key == "action":
            key = "actionable step"
        elif key in ("required proof", "proof"):
            key = "evidence"
        if key == "answer":
            key = "direct answer"
        elif key == "quick answer":
            key = "direct answer"
        elif key == "explanation":
            key = "detailed explanation"
        elif key == "why this applies":
            key = "detailed explanation"
        elif key == "process":
            key = "procedural steps"
        elif key == "source basis":
            key = "sources"
        elif key in ("source citation", "source"):
            key = "sources"
        elif key == "caution":
            key = "pro-tip"
        elif key == "markdown table":
            key = "comparison table"
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(normalized)
        sections[key] = normalized[start:end].strip()
    return sections


def extract_markdown_table_block(raw_value: str) -> tuple[str, str]:
    """Extract a markdown table block from a string and return (table, remainder)."""
    lines = raw_value.splitlines()
    table_lines: list[str] = []
    remainder: list[str] = []
    in_table = False

    for line in lines:
        stripped = line.strip()
        is_table_line = stripped.startswith("|") and stripped.count("|") >= 2
        if is_table_line:
            table_lines.append(stripped)
            in_table = True
            continue
        if in_table and not stripped:
            continue
        if in_table:
            remainder.append(line)
        else:
            remainder.append(line)

    table = "\n".join(table_lines).strip()
    rest = "\n".join(remainder).strip()
    return table, rest


def render_structured_response(
    answer: str,
    explanation_points: list[str],
    sources: list[str],
    procedural_steps: str = "",
    pro_tip: str = "",
    comparison_table: str = "",
    intent: str = "GENERAL",
) -> str:
    """Assemble a compact grounded response string."""
    cleaned_sources = _clean_source_entries(sources)
    status = _infer_audit_status(answer, explanation_points, pro_tip)
    evidence = _infer_required_evidence(answer, explanation_points, procedural_steps)
    action = _build_actionable_step(answer, explanation_points, procedural_steps, evidence)

    explanation_lines: list[str] = []
    explanation_seen: set[str] = set()
    priority_line = cleanup_generated_sentence(pro_tip) if pro_tip else ""
    if priority_line:
        explanation_lines.append(priority_line)
        explanation_seen.add(semantic_dedup_key(priority_line))
    resolved_steps = procedural_steps or build_default_procedural_steps(intent, answer, explanation_points)
    if "not applicable" in resolved_steps.lower() and intent in {"PROCESS", "WORKFLOW", "ANALYTICAL"}:
        resolved_steps = build_default_procedural_steps(intent, answer, explanation_points)
    for point in explanation_points[:3]:
        cleaned_point = cleanup_generated_sentence(point)
        point_key = semantic_dedup_key(cleaned_point) if cleaned_point else ""
        if cleaned_point and point_key not in explanation_seen:
            explanation_lines.append(cleaned_point)
            explanation_seen.add(point_key)
    if intent in {"PROCESS", "WORKFLOW"}:
        if not resolved_steps:
            resolved_steps = (
                "1. Identify the applicable procurement rule and supporting facts.\n"
                "2. Check the controlling source and record the justification.\n"
                "3. Complete the required scrutiny or committee review.\n"
                "4. Obtain approval from the competent authority.\n"
                "5. Record the decision and issue the next action or order."
            )
        compact_steps = _compress_steps_for_explanation(resolved_steps)
        if compact_steps:
            explanation_lines.append(compact_steps)

    source_line = _build_source_line(cleaned_sources)
    if source_line:
        explanation_lines.append(source_line)

    direct_answer, answer_tail = split_primary_and_explanation(answer)
    extra_explanation = " ".join(explanation_lines[:2]).strip()
    explanation = " ".join(part for part in (answer_tail, extra_explanation) if part).strip()
    explanation = smooth_summary_text(explanation, max_length=260)
    final_decision = _infer_final_decision(status, answer, explanation, action)
    return render_compact_response(direct_answer or answer, explanation, final_decision)


def _clean_source_entries(sources: list[str]) -> list[str]:
    seen_sources: set[str] = set()
    cleaned_sources: list[str] = []
    for source in sources or [f"Relevant rule - {GFR_2025_DOCUMENT_NAME}"]:
        normalized = clean_text(source).strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen_sources:
            continue
        seen_sources.add(key)
        cleaned_sources.append(normalized)
    return cleaned_sources or [f"Relevant rule - {GFR_2025_DOCUMENT_NAME}"]


def _infer_audit_status(answer: str, explanation_points: list[str], pro_tip: str = "") -> str:
    combined = clean_text(" ".join([answer, pro_tip, *explanation_points])).lower()
    if any(marker in combined for marker in ("non-compliant", "not compliant", "cannot", "must not", "should not", "invalid", "violation", "reject")):
        return "NON-COMPLIANT"
    if any(marker in combined for marker in ("if ", "only if", "subject to", "provided", "requires", "depending", "conditional", "closest grounded")):
        return "CONDITIONAL"
    return "COMPLIANT"


def _infer_audit_risk(status: str, answer: str, explanation_points: list[str], pro_tip: str = "") -> str:
    combined = clean_text(" ".join([answer, pro_tip, *explanation_points])).lower()
    if status == "NON-COMPLIANT" or any(marker in combined for marker in ("shortcut", "urgency", "skipped", "without pac", "gem-not-feasible", "high audit risk")):
        return "High"
    if status == "CONDITIONAL" or any(marker in combined for marker in ("pac", "proprietary", "single tender", "ste", "single source", "limited tender")):
        return "Medium"
    return "Low"


def _infer_required_evidence(answer: str, explanation_points: list[str], procedural_steps: str = "") -> str:
    combined = clean_text(" ".join([answer, procedural_steps, *explanation_points])).lower()
    evidence: list[str] = []
    if "gem" in combined:
        evidence.append("GeM availability or GeM-not-feasible screenshots")
    if any(marker in combined for marker in ("pac", "proprietary", "single tender", "single source", "ste")):
        evidence.append("PAC Certificate or proprietary justification")
    if any(marker in combined for marker in ("quotation", "lpc", "local purchase committee")):
        evidence.append("comparative quotations and LPC record")
    if any(marker in combined for marker in ("tender", "ote", "lte", "bid")):
        evidence.append("tender notice, bid evaluation, and approval note")
    if any(marker in combined for marker in ("approval", "competent authority", "dfp", "delegation")):
        evidence.append("competent authority approval under DFP")
    if not evidence:
        evidence.append("rule extract, justification note, and competent approval record")
    return "; ".join(dict.fromkeys(evidence))


def _build_audit_analysis(answer: str, explanation_points: list[str], pro_tip: str, sources: list[str]) -> str:
    candidates: list[str] = []
    direct = compact_answer_text(answer)
    if direct:
        candidates.append(direct)
    if pro_tip:
        cleaned_tip = cleanup_generated_sentence(pro_tip)
        if cleaned_tip:
            candidates.append(cleaned_tip)
    for point in explanation_points[:3]:
        cleaned_point = cleanup_generated_sentence(point)
        if cleaned_point:
            candidates.append(cleaned_point)
    if not candidates:
        candidates.append("The answer is based on the retrieved procurement rule context and threshold table.")
    source_text = "; ".join(sources[:2])
    summary_parts = candidates[:2]
    if source_text:
        summary_parts.append(f"Source basis: {source_text}.")
    elif len(candidates) > 2:
        summary_parts.append(candidates[2])
    return " ".join(summary_parts[:3])


def _build_actionable_step(answer: str, explanation_points: list[str], procedural_steps: str, evidence: str) -> str:
    first_step = ""
    for line in (procedural_steps or "").splitlines():
        match = re.match(r"\s*\d+[.)]\s*(.+)", line)
        if match:
            first_step = cleanup_generated_sentence(match.group(1))
            break
    if not first_step:
        method = infer_procurement_method(answer, explanation_points)
        if method:
            first_step = f"Proceed through the {method} route only after recording the controlling rule and approvals."
        else:
            first_step = "Record the controlling rule, justification, and approval before proceeding."
    return f"{first_step} Required proof: {evidence}."


def _compress_steps_for_explanation(procedural_steps: str) -> str:
    """Compress numbered steps into one short explanation line."""
    steps: list[str] = []
    for line in (procedural_steps or "").splitlines():
        match = re.match(r"\s*\d+[.)]\s*(.+)", line)
        if not match:
            continue
        cleaned = cleanup_generated_sentence(match.group(1))
        if cleaned:
            steps.append(cleaned.rstrip("."))
        if len(steps) >= 3:
            break
    if not steps:
        return ""
    return "Steps: " + "; ".join(steps) + "."


def _build_source_line(sources: list[str]) -> str:
    """Return a short source line that preserves evaluator source markers."""
    if not sources:
        return ""
    return f"Source: {'; '.join(sources[:2])}."


def _infer_final_decision(status: str, answer: str, explanation: str, action: str) -> str:
    """Infer a final decision locally for compact rendered responses."""
    normalized_text = clean_text(" ".join([answer, explanation, action])).lower()
    if any(term in normalized_text for term in ("reject", "cannot proceed", "cannot be accepted", "not compliant")):
        return "REJECT"
    if any(term in normalized_text for term in ("verify", "check", "confirm", "examine", "insufficient context")):
        return "VERIFY"
    if any(term in normalized_text for term in ("revise", "modify", "re-tender", "retender", "correct")):
        return "MODIFY"
    if status == "NON-COMPLIANT":
        return "REJECT"
    if status == "COMPLIANT":
        return "APPROVE"
    return "VERIFY"


def cleanup_generated_sentence(value: str) -> str:
    """Clean, rewrite, and validate a generated sentence."""
    cleaned = rewrite_procurement_sentence(value)
    cleaned = smooth_summary_text(cleaned, max_length=180)
    cleaned = strip_leading_noise(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = cleaned.replace("..", ".")
    cleaned = re.sub(r"\b(?:where|that|which|subject to|provided that)\s+\.$", ".", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(?:as the case may be|etc|and so on)\s*\.$", ".", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'^[\'"`“”]+|[\'"`“”]+$', "", cleaned).strip()
    if not cleaned:
        return ""
    if not is_useful_sentence(cleaned):
        return ""
    return ensure_sentence_punctuation(cleaned)


def compact_answer_text(value: str, max_sentences: int = 3, max_length: int = 420) -> str:
    """Compact an answer while keeping enough detail for grounded responses."""
    cleaned = clean_text(value)
    protected = cleaned.replace("Rs.", "Rs<dot>")
    sentences = [s.replace("Rs<dot>", "Rs.") for s in split_sentences(protected)]
    if not sentences:
        return smooth_summary_text(cleaned, max_length=max_length)
    answer = " ".join(sentences[: min(max_sentences, len(sentences))])
    return smooth_summary_text(answer, max_length=max_length)


def normalize_points(raw_value: str, fallback_points: list[str], max_points: int = 4) -> list[str]:
    """Parse bullet points from raw LLM output, deduplicate, and fallback."""
    points: list[str] = []
    cleaned = clean_text(raw_value)
    if cleaned:
        for item in split_candidate_points(raw_value):
            normalized = cleanup_generated_sentence(item)
            if normalized:
                points.append(normalized)
    deduped: list[str] = []
    seen: set[str] = set()
    for point in points:
        if point.count("|") >= 2:
            continue
        normalized = semantic_dedup_key(point)
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(smooth_summary_text(point, max_length=180))
    if not deduped:
        deduped = [cleanup_generated_sentence(p) for p in fallback_points]
        deduped = [p for p in deduped if p]
    return [p for p in deduped[:max_points] if p]


def split_candidate_points(raw_value: str) -> list[str]:
    """Split raw LLM output into candidate bullet points."""
    if "* " in raw_value and "\n" not in raw_value:
        return [s.strip() for s in raw_value.split("* ") if s.strip()]
    parts: list[str] = []
    for line in raw_value.splitlines():
        if not line.strip():
            continue
        if line.strip().startswith("|"):
            continue
        normalized_line = re.sub(r"^[\-\*\d.)\s]+", "", line.strip())
        if normalized_line:
            parts.append(normalized_line)
    if parts:
        return parts
    return [s for s in re.split(r"(?<=[.!?])\s+", raw_value) if s.strip()]


def summarize_match_for_context(query: str, match_content: str, match_file_name: str) -> str:
    """Create a concise summary of a match for LLM context."""
    cleaned = clean_text(match_content)
    if looks_like_table_of_contents(cleaned):
        return ""
    selected = select_relevant_sentences(query, cleaned, max_sentences=2)
    if not selected:
        return cleanup_generated_sentence(rewrite_procurement_sentence(cleaned))
    return cleanup_generated_sentence(" ".join(selected))


def select_relevant_sentences(query: str, content: str, max_sentences: int = 2) -> list[str]:
    """Select the most relevant sentences from content for the query."""
    if looks_like_table_of_contents(content):
        return []
    candidates = [c for c in extract_readable_units(content) if not looks_like_table_of_contents(c)]
    if not candidates:
        return []
    query_terms = tokenize(query)
    amount_lakhs = extract_amount_lakhs(query)
    ranked: list[tuple[float, int, str]] = []
    for candidate in candidates:
        audit = audit_chunk_quality(candidate)
        if audit["discard"]:
            continue
        sentence_terms = tokenize(candidate)
        overlap = len(query_terms.intersection(sentence_terms))
        amount_bonus = score_amount_relevance(candidate, amount_lakhs)
        process_bonus = score_process_relevance(query, candidate)
        analytical_bonus = score_analytical_relevance(query, candidate)
        noise_penalty = legalistic_noise_penalty(candidate)
        alpha_chars = sum(1 for char in candidate if char.isalpha())
        digit_chars = sum(1 for char in candidate if char.isdigit())
        readability = alpha_chars - digit_chars
        ranked.append((overlap + amount_bonus + process_bonus + analytical_bonus + noise_penalty, readability, candidate))
    ranked.sort(reverse=True, key=lambda item: (item[0], item[1], len(item[2])))
    selected: list[str] = []
    seen: set[str] = set()
    for _, _, sentence in ranked:
        cleaned_sentence = cleanup_generated_sentence(sentence)
        normalized = semantic_dedup_key(cleaned_sentence)
        if normalized in seen:
            continue
        seen.add(normalized)
        if cleaned_sentence:
            selected.append(cleaned_sentence)
        if len(selected) >= max_sentences:
            break
    return [item for item in selected if item]


def extract_readable_units(content: str) -> list[str]:
    """Split content into readable sentence-like units."""
    cleaned = clean_text(content)
    units = re.split(r"(?<=[.!?;:])\s+|,\s+(?=(?:and|but|if|when|where|while|provided|however)\b)", cleaned)
    extracted: list[str] = []
    for unit in units:
        normalized = rewrite_procurement_sentence(unit)
        normalized = strip_leading_noise(normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        if not normalized or len(normalized.split()) < 7:
            continue
        if not any(char.isalpha() for char in normalized):
            continue
        cleaned_unit = cleanup_generated_sentence(normalized)
        if cleaned_unit:
            extracted.append(cleaned_unit)
    return extracted
