import os
import re
import time
import logging
from pathlib import Path
from types import SimpleNamespace

from dotenv import load_dotenv
from groq import Groq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from pypdf import PdfReader

from ingest import create_vector_db

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
DATA_DIR = PROJECT_ROOT / "data"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

logger = logging.getLogger("procurebuddy-legacy-ai")

load_dotenv(dotenv_path=ENV_FILE)

MODELS = (
    "llama-3.1-8b-instant",
    "gemma2-9b-it",
)
TRANSLATION_MODEL = os.getenv("GROQ_TRANSLATION_MODEL", MODELS[0]).strip() or MODELS[0]
ROTATION_SLEEP_SECONDS = float(os.getenv("GROQ_ROTATION_SLEEP_SECONDS", "1"))
NO_CONTEXT_MESSAGE = "This information is not found in the provided rules."
MAX_DOCS = 8
MAX_CHARS_PER_DOC = 1400

PROCESS_PROMPT = """You are ProcureBuddy, a procurement assistant for CBRI and CSIR.

Answer only from the retrieved context. Do not use outside knowledge.
Do not reveal chain-of-thought or internal reasoning.

Priority order when sources conflict:
1. Latest office memorandum or amendment
2. CSIR Manual on Procurement of Goods 2019
3. GFR 2017

If the answer is not supported by the supplied context, reply exactly:
"This information is not found in the provided rules."

For amount-based procurement questions:
- Identify the exact amount from the question.
- Use only the threshold or procedure text that is explicitly present in the retrieved context.
- Do not infer a slab from Make in India, local supplier preference, or unrelated tender clauses.
- Mention the procurement mode only if the applicable threshold or procedure is explicitly visible in context.
- When explicit lower and upper threshold rules are present in the context, compare the amount numerically and pick the matching slab.
- If one mode is outside the applicable range, say that it does not apply instead of leaving the classification ambiguous.
- Do not apply Special Provisions meant only for scientific equipment / consumables / research purpose unless the user question actually falls in that scope.
- Keep the amount numerically consistent throughout the answer. Do not restate Rs 10,00,001 as above Rs 1 crore, or similar arithmetic mistakes.
- State whether a committee is required only if explicitly supported by context.
- Give a practical step-by-step process only if the process is explicitly supported by context.
- Do not convert headings, table of contents entries, or implied logic into procedural steps.
- Do not use words like "implied", "similar language", or "must therefore" unless the supporting sentence is explicit in the retrieved context.
- End with a short TL;DR.
- Add a short `Source Basis` section with only the relevant source numbers.

Use simple Hinglish, clear headings, and audit-friendly language.

Mandatory output format:
## Quick Answer
- Purchase value:
- Applicable mode:
- Committee:

## Rule Priority Applied
- Explicitly state that priority order is:
  1. OM / Special Provisions
  2. CSIR Manual 2019
  3. GFR 2017
- Mention which source level actually controlled the answer.

## Why This Applies
- 3 to 6 detailed bullets.

## Detailed Process
1. Step 1 title: explanation
2. Step 2 title: explanation
3. Continue with as many steps as the context supports.
- Mention total number of steps in one bullet before the list.

## Key Documents / Outputs
- Detailed bullet list.

## FLOWCHART (Mermaid)
```mermaid
flowchart TD
    A[Start] --> B[Next Step]
```

## Source Basis
- Mention document name and page number for each key point.
- Keep OM / Special Provisions first, then CSIR Manual 2019, then GFR 2017.

## TL;DR
- One or two bullets only.

Formatting rules:
- Do not use tables unless the user explicitly asks for a table.
- Give detailed answers because users are scientists and expect depth.
- If some section is not supported by context, omit that specific section instead of guessing.
"""

POLICY_PROMPT = """You are ProcureBuddy, a procurement policy assistant for CBRI and CSIR.

Answer only from the retrieved context. Do not use outside knowledge.
Do not reveal chain-of-thought or internal reasoning.

Explain the rule, when it applies, and any conditions.
If the answer is not supported by the supplied context, reply exactly:
"This information is not found in the provided rules."

Use simple Hinglish, clear headings, and audit-friendly language.

Mandatory output format:
## Quick Answer
- 2 to 4 bullets.

## Rule Priority Applied
- State OM / Special Provisions > CSIR Manual 2019 > GFR 2017.
- Mention which document level controlled the answer.

## Detailed Explanation
- 4 to 8 bullets.

## Conditions / Exceptions
- Include only if supported by context.

## Source Basis
- Mention document name and page number for each key point.
- Keep OM / Special Provisions first, then CSIR Manual 2019, then GFR 2017.

## TL;DR
- One or two bullets only.

Formatting rules:
- Do not use tables unless the user explicitly asks for a table.
- Give detailed but readable answers.
- If the answer is not fully supported by context, reply exactly:
"This information is not found in the provided rules."
"""

TABLE_PROMPT = """You are ProcureBuddy, a procurement assistant for CBRI and CSIR.

Generate a Markdown table only from the retrieved context.
- Do not use outside knowledge.
- Do not infer thresholds not visible in context.
- If the retrieved context does not support a complete table, reply exactly:
"This information is not found in the provided rules."
- Return only the table or the fallback sentence.
"""

_embeddings = None
_vector_db = None
_clients_by_key = {}
_active_key_index = 0


def _load_api_keys():
    keys = []
    for index in range(1, 9):
        value = (os.getenv(f"GROQ_API_KEY_{index}") or "").strip()
        if value and value not in keys:
            keys.append(value)

    single_key = (os.getenv("GROQ_API_KEY") or "").strip()
    if single_key and single_key not in keys:
        keys.append(single_key)

    if not keys:
        raise RuntimeError("No Groq API key configured. Set GROQ_API_KEY or GROQ_API_KEY_1..8.")
    return keys


API_KEYS = _load_api_keys()


def extract_amount(text: str):
    normalized = text.lower().replace(",", "").strip()

    crore_match = re.search(r"(\d+(?:\.\d+)?)\s*(crore|crores|cr)\b", normalized)
    if crore_match:
        return int(float(crore_match.group(1)) * 10000000)

    lakh_match = re.search(r"(\d+(?:\.\d+)?)\s*(lakh|lakhs|lac|lacs)\b", normalized)
    if lakh_match:
        return int(float(lakh_match.group(1)) * 100000)

    patterns = [
        r"₹\s*(\d+)",
        r"rs\.?\s*(\d+)",
        r"inr\s*(\d+)",
        r"worth\s*(\d+)",
        r"amount\s*(\d+)",
        r"\b(\d{5,})\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match:
            return int(match.group(1))

    return None


def detect_intent(text: str):
    lowered = text.lower()
    if extract_amount(text) is not None:
        return "PROCESS"
    if any(
        keyword in lowered
        for keyword in ["procurement process", "purchase process", "how to purchase", "ka process", "kya process", "tender process", "committee"]
    ):
        return "PROCESS"
    if any(keyword in lowered for keyword in ["table", "slab", "show table", "overview", "matrix"]):
        return "TABLE"
    if any(
        keyword in lowered
        for keyword in [
            "approval",
            "amendment",
            "policy",
            "rule",
            "om",
            "office memorandum",
            "conflict",
            "priority",
            "single tender",
            "proprietary",
            "publication",
            "pac",
            "ste",
        ]
    ):
        return "POLICY"
    return "HELP"


def _query_flags(user_text: str):
    lowered = user_text.lower()
    return {
        "local_preference": any(
            keyword in lowered
            for keyword in ["make in india", "local supplier", "local content", "preference policy"]
        ),
        "single_tender": any(keyword in lowered for keyword in ["single tender", "ste", "proprietary", "pac"]),
        "amendment_priority": any(
            keyword in lowered
            for keyword in ["latest amendment", "amendment", "supersede", "override", "priority", "conflict"]
        ),
        "amount_process": extract_amount(user_text) is not None
        or any(
            keyword in lowered
            for keyword in ["procurement process", "purchase process", "tender process", "committee"]
        ),
        "table": any(keyword in lowered for keyword in ["table", "slab", "show table", "overview", "matrix"]),
        "scientific_item": any(
            keyword in lowered
            for keyword in ["equipment", "instrument", "consumable", "research", "scientific"]
        ),
    }


def _format_chat_history(chat_history: list[dict] | None):
    if not chat_history:
        return "No prior conversation."

    lines = []
    for message in chat_history[-6:]:
        role = (message.get("role") or "user").capitalize()
        content = (message.get("content") or "").strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "No prior conversation."


def _extract_message_content(response):
    message = response.choices[0].message
    content = message.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "\n".join(part for part in parts if part).strip()
    return ""


def _get_client_for_key(api_key: str):
    client = _clients_by_key.get(api_key)
    if client is None:
        client = Groq(api_key=api_key)
        _clients_by_key[api_key] = client
    return client


def _is_rate_limit_error(error_text: str):
    return "rate_limit_exceeded" in error_text or "rate limit" in error_text


def _is_minute_limit_error(error_text: str):
    return "tokens per minute" in error_text or "requests per minute" in error_text


def _ordered_models(preferred_models: list[str] | tuple[str, ...] | None = None):
    candidates = []
    for model_name in preferred_models or ():
        cleaned = (model_name or "").strip()
        if cleaned and cleaned not in candidates:
            candidates.append(cleaned)
    for model_name in MODELS:
        if model_name not in candidates:
            candidates.append(model_name)
    return candidates


def _chat_completion(messages: list[dict], *, temperature: float = 0.0, preferred_models=None):
    global _active_key_index

    model_candidates = _ordered_models(preferred_models)
    last_exception = None
    start_index = _active_key_index

    for key_attempt in range(len(API_KEYS)):
        key_index = (start_index + key_attempt) % len(API_KEYS)
        current_key = API_KEYS[key_index]
        client = _get_client_for_key(current_key)

        for model_name in model_candidates:
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    timeout=20.0,
                )
                _active_key_index = key_index
                return response
            except Exception as exc:
                last_exception = exc
                error_text = str(exc).lower()

                if _is_rate_limit_error(error_text):
                    if _is_minute_limit_error(error_text):
                        logger.warning("Groq minute limit hit for model '%s'; trying next model on same key", model_name)
                        continue

                    logger.warning(
                        "Groq daily/key limit hit for key %s; switching to next key",
                        current_key[:10],
                    )
                    break

                logger.warning("Groq call failed for model '%s': %s", model_name, exc)
                continue

        _active_key_index = (key_index + 1) % len(API_KEYS)
        time.sleep(ROTATION_SLEEP_SECONDS)

    if last_exception is not None:
        raise last_exception
    raise RuntimeError("All Groq keys and models were exhausted.")


def _vector_db_ready():
    return CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir())


def _get_resources():
    global _embeddings, _vector_db

    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    if _vector_db is None:
        if not _vector_db_ready():
            create_vector_db()
        _vector_db = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=_embeddings,
        )

    return _vector_db.as_retriever(search_kwargs={"k": 6})


def translate_query(user_text: str):
    messages = [
        {
            "role": "system",
            "content": (
                "Translate Hindi or Hinglish procurement questions into concise professional "
                "English for semantic retrieval. Keep procurement terms intact. "
                "Return only the translated query."
            ),
        },
        {"role": "user", "content": user_text},
    ]
    try:
        response = _chat_completion(
            messages,
            temperature=0.0,
            preferred_models=[TRANSLATION_MODEL],
        )
        translated = _extract_message_content(response)
        return translated or user_text
    except Exception:
        return user_text


def _build_search_query(user_text: str):
    flags = _query_flags(user_text)
    hints = []
    if flags["amount_process"]:
        hints.append("procurement mode threshold committee csir manual gfr")
    if flags["scientific_item"]:
        hints.append("scientific equipment consumables research purpose special provisions")
    if flags["single_tender"]:
        hints.append("single tender enquiry proprietary article certificate rule 166 nomination basis")
    if flags["amendment_priority"]:
        hints.append("latest amendment special provisions override discrepancy csir manual gfr")
    if flags["table"]:
        hints.append("threshold table procurement mode committee")
        hints.append("SnT special provisions 3 rule 154 155 161 162")
    return f"{user_text} {' '.join(hints)}".strip()


def _merge_docs(doc_groups: list[list]):
    merged = []
    seen = set()
    for docs in doc_groups:
        for doc in docs:
            key = (
                doc.metadata.get("source", ""),
                doc.metadata.get("page", ""),
                doc.page_content[:200],
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(doc)
    return merged


def _score_doc(user_text: str, doc):
    flags = _query_flags(user_text)
    source_name = Path(doc.metadata.get("source", "")).name.lower()
    text = doc.page_content.lower()
    score = 0

    special_scope_query = flags["scientific_item"] or flags["amendment_priority"] or "special provision" in user_text.lower()

    if "special provisions" in source_name and special_scope_query:
        score += 6
    elif "special provisions" in source_name:
        score -= 4
    if "amendment" in source_name or "amendments" in source_name:
        score += 5
    if "csir manual" in source_name:
        score += 4
    if "gfr" in source_name:
        score += 3

    if flags["amount_process"]:
        for marker in [
            "purchase committee",
            "technical & purchase committee",
            "technical and purchase committee",
            "local purchase committee",
            "limited tender",
            "advertised tender",
            "rule 155",
            "rule 161",
            "rule 162",
            "up to rs.",
            "above rs.",
        ]:
            if marker in text:
                score += 2

    if flags["table"]:
        for marker in [
            "rule 154",
            "rule 155",
            "rule 161",
            "rule 162",
            "direct purchase",
            "local purchase committee",
            "purchase committee",
            "limited tender enquiry",
            "advertised tender enquiry",
        ]:
            if marker in text:
                score += 3

    if flags["single_tender"]:
        for marker in ["single tender", "rule 166", "proprietary article", "standardisation", "emergency"]:
            if marker in text:
                score += 3

    if flags["amendment_priority"]:
        for marker in ["discrepancy", "prevail", "specific relaxations", "special provisions", "amended limit"]:
            if marker in text:
                score += 3

    return score


def _retrieve_docs(retriever, user_text: str):
    translated_query = translate_query(user_text)
    base_query = _build_search_query(translated_query)
    flags = _query_flags(user_text)
    queries = [base_query]

    if flags["amount_process"]:
        queries.append(
            f"{translated_query} rule 155 rule 161 rule 162 purchase committee technical purchase committee csir manual gfr"
        )
        queries.append(
            f"{translated_query} csir manual purchase committee technical and purchase committee local purchase committee"
        )
        if flags["scientific_item"]:
            queries.append(
                f"{translated_query} scientific equipment consumables research purpose special provisions amended limit"
            )

    if flags["single_tender"]:
        queries.append(f"{translated_query} rule 166 single tender enquiry proprietary article certificate")

    if flags["amendment_priority"]:
        queries.append(f"{translated_query} discrepancy prevail special relaxations ministry of finance")

    if flags["table"]:
        queries.append(
            "SnT special provisions 3 procurement threshold table rule 154 rule 155 rule 161 rule 162 special provisions purchase committee local purchase committee technical purchase committee"
        )
        queries.append(
            "csir manual chapter 4 purchase committee technical purchase committee local purchase committee thresholds"
        )

    doc_groups = [retriever.invoke(query) for query in queries]
    docs = _filter_docs(user_text, _merge_docs(doc_groups))
    docs = sorted(docs, key=lambda doc: _score_doc(user_text, doc), reverse=True)
    return docs[:MAX_DOCS]


def _doc_is_applicable(user_text: str, doc):
    flags = _query_flags(user_text)
    lowered_query = user_text.lower()
    source_name = Path(doc.metadata.get("source", "")).name.lower()
    text = doc.page_content.lower()

    if "special provisions" in source_name:
        scientific_scope = any(
            marker in lowered_query
            for marker in ["scientific", "equipment", "consumable", "research", "research purpose", "special provision"]
        )
        if not scientific_scope and not flags["amendment_priority"]:
            return False

    if "scientific equipment" in text and "research purpose only" in text:
        scientific_scope = any(
            marker in lowered_query
            for marker in ["scientific", "equipment", "consumable", "research", "research purpose"]
        )
        if not scientific_scope and not flags["amendment_priority"]:
            return False

    return True


def _prepare_answer_docs(user_text: str, retriever, *, include_threshold_docs: bool = False):
    doc_groups = [_retrieve_docs(retriever, user_text)]
    if include_threshold_docs:
        doc_groups.append(_load_threshold_docs_from_files())

    docs = _merge_docs(doc_groups)
    docs = [doc for doc in docs if _doc_is_applicable(user_text, doc)]
    docs = _filter_docs(user_text, docs)
    docs = sorted(docs, key=lambda doc: _score_doc(user_text, doc), reverse=True)
    return docs[:MAX_DOCS]


def _filter_docs(user_text: str, docs: list):
    flags = _query_flags(user_text)
    filtered = []

    for doc in docs:
        source_name = Path(doc.metadata.get("source", "")).name.lower()
        text = doc.page_content.lower()

        if not flags["local_preference"]:
            if any(
                marker in text or marker in source_name
                for marker in ["make in india", "local supplier", "local content", "purchase preference"]
            ):
                continue

        if flags["amount_process"]:
            if any(
                marker in text
                for marker in [
                    "model tender",
                    "invitation for bids",
                    "bid validity",
                    "performance security",
                    "table of contents",
                ]
            ):
                continue

        filtered.append(doc)

    return filtered or docs


def _build_help_answer():
    return (
        "Namaste. Main procurement-related questions me help kar sakta hoon.\n\n"
        "Aap in tarah ke questions pooch sakte hain:\n"
        "- amount-based process, for example `8 lakh ka purchase process kya hai`\n"
        "- policy question, for example `single tender kab allowed hai`\n"
        "- table request, for example `show procurement table`\n\n"
        "Aap chahein to apna question simple Hinglish me seedha likh dijiye."
    )


def _detect_response_language(user_text: str):
    if re.search(r"[\u0900-\u097F]", user_text):
        return "Hindi"

    hinglish_markers = [
        "kya", "ka", "ki", "ke", "hai", "hoga", "agar", "kaise",
        "karna", "karni", "hindi", "hinglish", "samjhao", "batao",
        "procurement", "purchase", "tender", "committee",
    ]
    lowered = f" {user_text.lower()} "
    score = sum(1 for marker in hinglish_markers if f" {marker} " in lowered)
    if score >= 2:
        return "Hinglish"
    return "English"


def _build_amount_instruction(user_text: str):
    amount = extract_amount(user_text)
    if amount is None:
        return ""
    return (
        f"Detected amount exactly: Rs {amount} (Indian format: Rs {_format_rupees(amount)}).\n"
        "- You must compare this amount numerically against threshold values mentioned in the retrieved context.\n"
        "- Do not apply an `above threshold` rule when the amount is lower than that threshold.\n"
        "- Do not apply an `up to threshold` rule when the amount is higher than that threshold.\n"
        "- Before finalizing, verify that every amount comparison in the answer is arithmetically consistent with this exact amount.\n"
    )


def _parse_amount_value(raw_value: str, unit_hint: str = ""):
    cleaned = raw_value.replace(",", "").strip()
    try:
        value = float(cleaned)
    except ValueError:
        return None

    unit = unit_hint.lower()
    if "crore" in unit or "cr" == unit:
        return int(value * 10000000)
    if "lakh" in unit or "lac" in unit:
        return int(value * 100000)
    return int(value)


def _extract_amounts_from_text(text: str):
    values = []

    for match in re.finditer(r"₹\s*([\d,]+(?:\.\d+)?)\s*(lakh|lakhs|crore|crores)?", text, re.I):
        amount = _parse_amount_value(match.group(1), match.group(2) or "")
        if amount is not None:
            values.append(amount)

    for match in re.finditer(r"Rs\.?\s*([\d,]+(?:\.\d+)?)\s*(lakh|lakhs|crore|crores)?", text, re.I):
        amount = _parse_amount_value(match.group(1), match.group(2) or "")
        if amount is not None:
            values.append(amount)

    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*(lakh|lakhs|crore|crores)\b", text, re.I):
        amount = _parse_amount_value(match.group(1), match.group(2))
        if amount is not None:
            values.append(amount)

    return values


def _extract_rule_window(text: str, rule_number: str, stop_markers: list[str]):
    start = re.search(rf"\b{rule_number}\b", text)
    if not start:
        return ""
    start_index = start.start()
    end_index = len(text)
    for marker in stop_markers:
        marker_match = re.search(rf"\b{marker}\b", text[start_index + 1 :])
        if marker_match:
            end_index = min(end_index, start_index + 1 + marker_match.start())
    return text[start_index:end_index]


def _source_label(doc):
    return f"{Path(doc.metadata.get('source', 'Unknown')).name} (Page {doc.metadata.get('page', 'N/A')})"


def _extract_procurement_facts(docs: list):
    facts = {
        "direct_purchase_max": None,
        "lpc_max": None,
        "pc_max": None,
        "tpc_above": None,
        "lte_max": None,
        "advertised_above": None,
        "sources": {},
    }

    for doc in docs:
        source_name = Path(doc.metadata.get("source", "")).name.lower()
        text = " ".join(doc.page_content.split())

        if "special provisions" in source_name:
            rule_154_match = re.search(
                r"154.*?Rs\.?\s*([\d,]+(?:\.\d+)?)\s*(lakh|lakhs|crore|crores)?.*?Rs\.?\s*([\d,]+(?:\.\d+)?)\s*(lakh|lakhs|crore|crores)?",
                text,
                re.I,
            )
            rule_155_match = re.search(
                r"155.*?upto\s*Rs\.?\s*([\d,]+(?:\.\d+)?)\s*(lakh|lakhs|crore|crores)?.*?upto\s*Rs\.?\s*([\d,]+(?:\.\d+)?)\s*(lakh|lakhs|crore|crores)?",
                text,
                re.I,
            )
            rule_162_match = re.search(
                r"162.*?U\s*pto\s*Rs\.?\s*([\d,]+(?:\.\d+)?)\s*(lakh|lakhs|crore|crores)?.*?U\s*pto\s*Rs\.?\s*([\d,]+(?:\.\d+)?)\s*(lakh|lakhs|crore|crores)?",
                text,
                re.I,
            )
            rule_161_match = re.search(
                r"161.*?Above\s*Rs\.?\s*([\d,]+(?:\.\d+)?)\s*(lakh|lakhs|crore|crores)?.*?Above\s*Rs\.?\s*([\d,]+(?:\.\d+)?)\s*(lakh|lakhs|crore|crores)?",
                text,
                re.I,
            )

            if rule_154_match:
                amount = _parse_amount_value(rule_154_match.group(3), rule_154_match.group(4) or "")
                if amount is not None:
                    facts["direct_purchase_max"] = amount
                    facts["sources"]["direct_purchase_max"] = _source_label(doc)

            if rule_155_match:
                amount = _parse_amount_value(rule_155_match.group(3), rule_155_match.group(4) or "")
                if amount is not None:
                    facts["pc_max"] = amount
                    facts["sources"]["pc_max"] = _source_label(doc)

            if rule_162_match:
                amount = _parse_amount_value(rule_162_match.group(3), rule_162_match.group(4) or "")
                if amount is not None:
                    facts["lte_max"] = amount
                    facts["sources"]["lte_max"] = _source_label(doc)

            if rule_161_match:
                amount = _parse_amount_value(rule_161_match.group(3), rule_161_match.group(4) or "")
                if amount is not None:
                    facts["advertised_above"] = amount
                    facts["sources"]["advertised_above"] = _source_label(doc)

        if "csir manual" in source_name:
            pc_match = re.search(r"PC will consider procurement of all goods up to\s*₹?\s*([\d,]+(?:\.\d+)?)\s*(lakh|lakhs|crore|crores)?", text, re.I)
            if pc_match:
                amount = _parse_amount_value(pc_match.group(1), pc_match.group(2) or "")
                if amount is not None:
                    facts["pc_max"] = amount
                    facts["sources"]["pc_max"] = _source_label(doc)

            tpc_match = re.search(r"T&PC will consider procurement of all goods above\s*₹?\s*([\d,]+(?:\.\d+)?)\s*(lakh|lakhs|crore|crores)?", text, re.I)
            if tpc_match:
                amount = _parse_amount_value(tpc_match.group(1), tpc_match.group(2) or "")
                if amount is not None:
                    facts["tpc_above"] = amount
                    facts["sources"]["tpc_above"] = _source_label(doc)

            lpc_match = re.search(
                r"valued above\s*₹?\s*([\d,]+(?:\.\d+)?)\s*(lakh|lakhs|crore|crores|thousand)? .*? up to\s*₹?\s*([\d,]+(?:\.\d+)?)\s*(lakh|lakhs|crore|crores|thousand)?",
                text,
                re.I,
            )
            if "local purchase committee" in text.lower() and lpc_match:
                upper = _parse_amount_value(lpc_match.group(3), lpc_match.group(4) or "")
                if upper is not None:
                    facts["lpc_max"] = upper
                    facts["sources"]["lpc_max"] = _source_label(doc)

    return facts


def _extract_procurement_facts_v2(docs: list):
    facts = {
        "direct_purchase_max": None,
        "lpc_max": None,
        "pc_max": None,
        "tpc_above": None,
        "lte_max": None,
        "advertised_above": None,
        "sources": {},
    }

    for doc in docs:
        source_name = Path(doc.metadata.get("source", "")).name.lower()
        text = " ".join(doc.page_content.split())

        if "special provisions" in source_name:
            rule_154_match = re.search(
                r"154.*?Rs\.?\s*([\d,]+(?:\.\d+)?)\s*(lakh|lakhs|crore|crores)?.*?Rs\.?\s*([\d,]+(?:\.\d+)?)\s*(lakh|lakhs|crore|crores)?",
                text,
                re.I,
            )
            rule_155_match = re.search(
                r"155.*?upto\s*Rs\.?\s*([\d,]+(?:\.\d+)?)\s*(lakh|lakhs|crore|crores)?.*?upto\s*Rs\.?\s*([\d,]+(?:\.\d+)?)\s*(lakh|lakhs|crore|crores)?",
                text,
                re.I,
            )
            rule_162_match = re.search(
                r"162.*?u\s*pto\s*Rs\.?\s*([\d,]+(?:\.\d+)?)\s*(lakh|lakhs|crore|crores)?.*?u\s*pto\s*Rs\.?\s*([\d,]+(?:\.\d+)?)\s*(lakh|lakhs|crore|crores)?",
                text,
                re.I,
            )
            rule_161_match = re.search(
                r"161.*?Above\s*Rs\.?\s*([\d,]+(?:\.\d+)?)\s*(lakh|lakhs|crore|crores)?.*?Above\s*Rs\.?\s*([\d,]+(?:\.\d+)?)\s*(lakh|lakhs|crore|crores)?",
                text,
                re.I,
            )

            if rule_154_match:
                amount = _parse_amount_value(rule_154_match.group(3), rule_154_match.group(4) or "")
                if amount is not None:
                    facts["direct_purchase_max"] = amount
                    facts["sources"]["direct_purchase_max"] = _source_label(doc)

            if rule_155_match:
                amount = _parse_amount_value(rule_155_match.group(3), rule_155_match.group(4) or "")
                if amount is not None:
                    facts["pc_max"] = amount
                    facts["sources"]["pc_max"] = _source_label(doc)

            if rule_162_match:
                amount = _parse_amount_value(rule_162_match.group(3), rule_162_match.group(4) or "")
                if amount is not None:
                    facts["lte_max"] = amount
                    facts["sources"]["lte_max"] = _source_label(doc)

            if rule_161_match:
                amount = _parse_amount_value(rule_161_match.group(3), rule_161_match.group(4) or "")
                if amount is not None:
                    facts["advertised_above"] = amount
                    facts["sources"]["advertised_above"] = _source_label(doc)

        if "csir manual" in source_name:
            pc_match = re.search(
                r"PC will consider procurement of all goods up to\s*[₹Rs\.\s]*([\d,]+(?:\.\d+)?)\s*(lakh|lakhs|crore|crores)?",
                text,
                re.I,
            )
            if pc_match:
                amount = _parse_amount_value(pc_match.group(1), pc_match.group(2) or "")
                if amount is not None:
                    facts["pc_max"] = amount
                    facts["sources"]["pc_max"] = _source_label(doc)

            pc_alt_match = re.search(
                r"Purchase Committee \(PC\).*?up to\s*Rs\.?\s*([\d,]+(?:\.\d+)?)\s*(lakh|lakhs|crore|crores)?.*?Technical.*?above\s*Rs\.?\s*([\d,]+(?:\.\d+)?)\s*(lakh|lakhs|crore|crores)?",
                text,
                re.I,
            )
            if pc_alt_match:
                pc_amount = _parse_amount_value(pc_alt_match.group(1), pc_alt_match.group(2) or "")
                tpc_amount = _parse_amount_value(pc_alt_match.group(3), pc_alt_match.group(4) or "")
                if pc_amount is not None:
                    facts["pc_max"] = pc_amount
                    facts["sources"]["pc_max"] = _source_label(doc)
                if tpc_amount is not None:
                    facts["tpc_above"] = tpc_amount
                    facts["sources"]["tpc_above"] = _source_label(doc)

            tpc_match = re.search(
                r"T&PC will consider procurement of all goods above\s*[₹Rs\.\s]*([\d,]+(?:\.\d+)?)\s*(lakh|lakhs|crore|crores)?",
                text,
                re.I,
            )
            if tpc_match:
                amount = _parse_amount_value(tpc_match.group(1), tpc_match.group(2) or "")
                if amount is not None:
                    facts["tpc_above"] = amount
                    facts["sources"]["tpc_above"] = _source_label(doc)

            lpc_match = re.search(
                r"valued above\s*[₹Rs\.\s]*([\d,]+(?:\.\d+)?)\s*(lakh|lakhs|crore|crores|thousand)? .*? up to\s*[₹Rs\.\s]*([\d,]+(?:\.\d+)?)\s*(lakh|lakhs|crore|crores|thousand)?",
                text,
                re.I,
            )
            if "local purchase committee" in text.lower() and lpc_match:
                upper = _parse_amount_value(lpc_match.group(3), lpc_match.group(4) or "")
                if upper is not None:
                    facts["lpc_max"] = upper
                    facts["sources"]["lpc_max"] = _source_label(doc)

    return facts


def _format_rupees(amount: int):
    digits = str(amount)
    if len(digits) <= 3:
        return digits
    last_three = digits[-3:]
    rest = digits[:-3]
    parts = []
    while len(rest) > 2:
        parts.insert(0, rest[-2:])
        rest = rest[:-2]
    if rest:
        parts.insert(0, rest)
    return ",".join(parts + [last_three])


def _source_priority(label: str):
    lowered = label.lower()
    if "special provisions" in lowered or "amendment" in lowered:
        return 0
    if "csir manual" in lowered:
        return 1
    if "gfr" in lowered:
        return 2
    return 3


def _ordered_source_lines(lines: list[str]):
    return sorted(dict.fromkeys(lines), key=_source_priority)


def _load_threshold_docs_from_files():
    docs = []
    if not DATA_DIR.exists():
        return docs

    patterns = [
        "rule 154",
        "rule 155",
        "rule 161",
        "rule 162",
        "purchase committee",
        "technical & purchase committee",
        "technical and purchase committee",
        "local purchase committee",
        "limited tender",
        "advertised tender",
    ]

    for path in DATA_DIR.glob("*.pdf"):
        try:
            reader = PdfReader(str(path))
        except Exception:
            continue

        for index, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            lowered = text.lower()
            if any(pattern in lowered for pattern in patterns):
                docs.append(
                    SimpleNamespace(
                        page_content=text,
                        metadata={
                            "source": str(path),
                            "page": index + 1,
                        },
                    )
                )
    return docs


def _build_amount_answer(user_text: str, docs: list):
    amount = extract_amount(user_text)
    if amount is None:
        return None

    combined_docs = docs + _load_threshold_docs_from_files()
    facts = _extract_procurement_facts_v2(combined_docs)
    language = _detect_response_language(user_text)

    direct_purchase_max = facts["direct_purchase_max"]
    lpc_max = facts["lpc_max"]
    pc_max = facts["pc_max"]
    tpc_above = facts["tpc_above"]
    lte_max = facts["lte_max"]
    advertised_above = facts["advertised_above"]

    committee = None
    mode = None
    controlling_level = []
    why_lines = []
    source_lines = []
    process_steps = []
    documents = []

    if direct_purchase_max is not None and amount <= direct_purchase_max:
        mode = "Direct Purchase"
        committee = "No committee"
        why_lines.append(f"Amount Rs { _format_rupees(amount) } direct purchase limit ke andar hai.")
        if "direct_purchase_max" in facts["sources"]:
            source_lines.append(f"- {facts['sources']['direct_purchase_max']}: direct purchase limit.")
            controlling_level.append(facts["sources"]["direct_purchase_max"])

    if committee is None and lpc_max is not None and amount <= lpc_max:
        mode = "Local Purchase Committee (LPC)"
        committee = "Local Purchase Committee (LPC)"
        why_lines.append(f"Amount LPC upper limit Rs { _format_rupees(lpc_max) } ke andar hai.")
        if "lpc_max" in facts["sources"]:
            source_lines.append(f"- {facts['sources']['lpc_max']}: LPC upper limit.")
            controlling_level.append(facts["sources"]["lpc_max"])

    if committee is None and pc_max is not None and amount <= pc_max:
        committee = "Purchase Committee (PC)"
        why_lines.append(f"Amount Rs { _format_rupees(amount) } PC limit Rs { _format_rupees(pc_max) } ke andar hai.")
        if "pc_max" in facts["sources"]:
            source_lines.append(f"- {facts['sources']['pc_max']}: PC up to threshold.")
            controlling_level.append(facts["sources"]["pc_max"])

    if committee is None and tpc_above is not None and amount > tpc_above:
        committee = "Technical & Purchase Committee (T&PC)"
        why_lines.append(f"Amount Rs { _format_rupees(amount) } T&PC threshold Rs { _format_rupees(tpc_above) } se upar hai.")
        if "tpc_above" in facts["sources"]:
            source_lines.append(f"- {facts['sources']['tpc_above']}: T&PC above threshold.")
            controlling_level.append(facts["sources"]["tpc_above"])

    if advertised_above is not None and amount > advertised_above:
        mode = "Advertised Tender / Open Tender"
        why_lines.append(f"Amount advertised tender threshold Rs { _format_rupees(advertised_above) } se upar hai.")
        if "advertised_above" in facts["sources"]:
            source_lines.append(f"- {facts['sources']['advertised_above']}: advertised tender threshold.")
            controlling_level.append(facts["sources"]["advertised_above"])
    elif lte_max is not None and tpc_above is not None and amount > tpc_above and amount <= lte_max:
        mode = "Limited Tender Enquiry (LTE)"
        why_lines.append(
            f"Amount T&PC threshold Rs { _format_rupees(tpc_above) } se upar aur LTE upper limit Rs { _format_rupees(lte_max) } ke andar hai."
        )
        if "lte_max" in facts["sources"]:
            source_lines.append(f"- {facts['sources']['lte_max']}: LTE upper limit.")
            controlling_level.append(facts["sources"]["lte_max"])
    elif mode is None and committee == "Purchase Committee (PC)":
        mode = "Mode to be decided by PC under Chapter 4"
        why_lines.append("Retrieved context PC ka threshold clearly batata hai, lekin is amount ke liye ek single mandatory mode clearly force nahin karta.")

    if committee is None and mode is None:
        return None

    if committee == "Purchase Committee (PC)":
        process_steps = [
            "Step 1: Need identify karo aur technical specifications final karo.",
            "Step 2: Administrative approval, budget confirmation, aur signed indent prepare karo.",
            "Step 3: Indent ko Procuring Authority / Purchase Section ko submit karo.",
            "Step 4: Purchase Committee meeting me case place karo; PC is amount tak ke goods ko consider karta hai.",
            "Step 5: PC Chapter 4 ke ambit me appropriate procurement method decide karega.",
            "Step 6: Purchase Section progress monitor karega aur final order tabhi place hoga jab funds available hon.",
        ]
        documents = [
            "Need note / requirement note",
            "Technical specifications",
            "Fund availability certificate",
            "Signed indent / purchase requisition",
            "PC proceedings / recommendation",
            "Purchase Section processing record",
        ]
    elif committee == "Technical & Purchase Committee (T&PC)":
        if mode == "Advertised Tender / Open Tender":
            process_steps = [
                "Step 1: Need identify karo aur technical specifications final karo.",
                "Step 2: Administrative approval, budget confirmation, aur signed indent prepare karo.",
                "Step 3: Case T&PC ke saamne place karo, kyunki value retrieved threshold se upar hai.",
                "Step 4: Open / advertised tender initiate karo aur bid documents float karo as per applicable rules.",
                "Step 5: Bid evaluation, T&PC recommendation, aur competent authority approval lo.",
                "Step 6: Approval ke baad order placement aur procurement record complete karo.",
            ]
            documents = [
                "Need note / requirement note",
                "Technical specifications",
                "Fund availability certificate",
                "Signed indent / purchase requisition",
                "Tender / bid documents",
                "T&PC proceedings / recommendation",
                "Approval and order record",
            ]
        else:
            process_steps = [
                "Step 1: Need identify karo aur technical specifications final karo.",
                "Step 2: Administrative approval, budget confirmation, aur signed indent prepare karo.",
                "Step 3: Case T&PC ke saamne place karo, kyunki value retrieved threshold se upar hai.",
                "Step 4: T&PC applicable procurement method decide karega; agar LTE limit ke andar hai to LTE adopt kiya ja sakta hai.",
                "Step 5: Detailed modus operandi Chapter 4 ke hisaab se follow hoga.",
                "Step 6: Recommendation ke baad competent authority approval aur order placement hoga.",
            ]
            documents = [
                "Need note / requirement note",
                "Technical specifications",
                "Fund availability certificate",
                "Signed indent / purchase requisition",
                "T&PC proceedings / recommendation",
                "Tender / LTE papers if applicable",
                "Approval and order record",
            ]
    elif committee == "Local Purchase Committee (LPC)":
        process_steps = [
            "Step 1: Need aur specs final karo.",
            "Step 2: Budget confirmation aur indent prepare karo.",
            "Step 3: LPC market survey / quotation comparison karegi.",
            "Step 4: LPC reasonable rate aur suitable supplier identify karegi.",
            "Step 5: Recommendation ke baad purchase process complete hoga.",
        ]
        documents = [
            "Indent",
            "Specifications",
            "Quotation / market survey record",
            "LPC proceedings",
            "Approval and purchase record",
        ]
    else:
        process_steps = [
            "Step 1: Need aur specifications final karo.",
            "Step 2: Budget approval aur indent prepare karo.",
            "Step 3: Applicable procurement mode ke hisaab se आगे process follow karo.",
        ]
        documents = [
            "Indent",
            "Specifications",
            "Approval note",
            "Purchase record",
        ]

    purchase_value = _format_rupees(amount)
    ordered_sources = _ordered_source_lines(source_lines)
    controlling_source = min(controlling_level, key=_source_priority) if controlling_level else None

    if controlling_source:
        lowered = controlling_source.lower()
        if "special provisions" in lowered or "amendment" in lowered:
            controlling_rule = "OM / Special Provisions"
        elif "csir manual" in lowered:
            controlling_rule = "CSIR Manual 2019"
        elif "gfr" in lowered:
            controlling_rule = "GFR 2017"
        else:
            controlling_rule = "Retrieved context"
    else:
        controlling_rule = "Retrieved context"

    flow_nodes = [
        "A[Need Assessment]",
        "B[Specs + Fund Confirmation]",
        "C[Indent / PR Submission]",
        "D[Committee Review]",
        "E[Mode Decision]",
        "F[Approval / Order]",
    ]
    flowchart = (
        "```mermaid\n"
        "flowchart TD\n"
        f"    {flow_nodes[0]} --> {flow_nodes[1]}\n"
        f"    {flow_nodes[1]} --> {flow_nodes[2]}\n"
        f"    {flow_nodes[2]} --> {flow_nodes[3]}\n"
        f"    {flow_nodes[3]} --> {flow_nodes[4]}\n"
        f"    {flow_nodes[4]} --> {flow_nodes[5]}\n"
        "```"
    )

    if language == "English":
        answer_lines = [
            "## Quick Answer",
            f"- Purchase value: Rs {purchase_value}",
            f"- Applicable mode: {mode or 'Not clearly established from retrieved context'}",
            f"- Committee: {committee or 'Not clearly established from retrieved context'}",
            "",
            "## Rule Priority Applied",
            "- Priority order used: OM / Special Provisions -> CSIR Manual 2019 -> GFR 2017.",
            f"- Controlling source level in this answer: {controlling_rule}.",
            "",
            "## Why This Applies",
        ]
        answer_lines.extend(f"- {line}" for line in why_lines[:6])
        answer_lines.extend(
            [
                "",
                "## Detailed Process",
                f"- Total steps: {len(process_steps)}",
            ]
        )
        answer_lines.extend(f"{index}. {step}" for index, step in enumerate(process_steps, 1))
        answer_lines.extend(
            [
                "",
                "## Key Documents / Outputs",
            ]
        )
        answer_lines.extend(f"- {item}" for item in documents)
        answer_lines.extend(
            [
                "",
                "## FLOWCHART (Mermaid)",
                flowchart,
                "",
                "## Source Basis",
            ]
        )
        answer_lines.extend(ordered_sources[:6] or ["- Retrieved context does not contain enough direct threshold evidence."])
        answer_lines.extend(
            [
                "",
                "## TL;DR",
                f"- For Rs {purchase_value}, use `{mode}` with `{committee}`." if mode and committee else "- The retrieved context is not sufficient for a complete classification.",
                f"- Rule priority followed: {controlling_rule}.",
            ]
        )
        return "\n".join(answer_lines)

    answer_lines = [
        "## Quick Answer",
        f"- Purchase value: Rs {purchase_value}",
        f"- Applicable mode: {mode or 'Retrieved context se clear nahin hai'}",
        f"- Committee: {committee or 'Retrieved context se clear nahin hai'}",
        "",
        "## Rule Priority Applied",
        "- Priority order used: OM / Special Provisions -> CSIR Manual 2019 -> GFR 2017.",
        f"- Is answer me controlling source level: {controlling_rule}.",
        "",
        "## Why This Applies",
    ]
    answer_lines.extend(f"- {line}" for line in why_lines[:6])
    answer_lines.extend(
        [
            "",
            "## Detailed Process",
            f"- Total steps: {len(process_steps)}",
        ]
    )
    answer_lines.extend(f"{index}. {step}" for index, step in enumerate(process_steps, 1))
    answer_lines.extend(
        [
            "",
            "## Key Documents / Outputs",
        ]
    )
    answer_lines.extend(f"- {item}" for item in documents)
    answer_lines.extend(
        [
            "",
            "## FLOWCHART (Mermaid)",
            flowchart,
            "",
            "## Source Basis",
        ]
    )
    answer_lines.extend(ordered_sources[:6] or ["- Retrieved context me direct threshold evidence enough nahin mila."])
    answer_lines.extend(
        [
            "",
            "## TL;DR",
            f"- Rs {purchase_value} ke case me `{mode}` aur `{committee}` apply hoga." if mode and committee else "- Retrieved context se complete classification clear nahin hai.",
            f"- Rule priority followed: {controlling_rule}.",
        ]
    )
    return "\n".join(answer_lines)


def _build_table_answer(docs: list):
    combined_docs = docs + _load_threshold_docs_from_files()
    facts = _extract_procurement_facts_v2(combined_docs)

    direct_purchase_max = facts["direct_purchase_max"]
    lpc_max = facts["lpc_max"]
    pc_max = facts["pc_max"]
    tpc_above = facts["tpc_above"] or pc_max
    lte_max = facts["lte_max"]
    advertised_above = facts["advertised_above"] or lte_max

    rows = []

    if direct_purchase_max is not None:
        rows.append(
            [
                f"Up to Rs {_format_rupees(direct_purchase_max)}",
                "Direct Purchase",
                "No",
                "-",
                facts["sources"].get("direct_purchase_max", "-"),
            ]
        )

    if lpc_max is not None and direct_purchase_max is not None and lpc_max > direct_purchase_max:
        rows.append(
            [
                f"Above Rs {_format_rupees(direct_purchase_max)} to Rs {_format_rupees(lpc_max)}",
                "Local Purchase Committee (LPC)",
                "Yes",
                "Local Purchase Committee (LPC)",
                facts["sources"].get("lpc_max", "-"),
            ]
        )

    pc_lower = lpc_max or direct_purchase_max
    if pc_max is not None and (pc_lower is None or pc_max > pc_lower):
        lower_label = f"Above Rs {_format_rupees(pc_lower)}" if pc_lower is not None else "Up to current PC threshold"
        rows.append(
            [
                f"{lower_label} to Rs {_format_rupees(pc_max)}",
                "Purchase Committee (PC)",
                "Yes",
                "Purchase Committee (PC)",
                facts["sources"].get("pc_max", "-"),
            ]
        )

    if tpc_above is not None and lte_max is not None and lte_max > tpc_above:
        rows.append(
            [
                f"Above Rs {_format_rupees(tpc_above)} to Rs {_format_rupees(lte_max)}",
                "Limited Tender Enquiry (LTE)",
                "Yes",
                "Technical & Purchase Committee (T&PC)",
                facts["sources"].get("lte_max", facts["sources"].get("tpc_above", "-")),
            ]
        )

    if advertised_above is not None:
        rows.append(
            [
                f"Above Rs {_format_rupees(advertised_above)}",
                "Advertised Tender / Open Tender",
                "Yes",
                "Technical & Purchase Committee (T&PC)",
                facts["sources"].get("advertised_above", "-"),
            ]
        )

    if not rows:
        return NO_CONTEXT_MESSAGE

    table = (
        "| Cost Category | Procurement Mode | Committee Required | Which Committee | Source |\n"
        "| --- | --- | --- | --- | --- |\n"
    )
    for row in rows:
        table += f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} |\n"
    return table


def _append_table_if_requested(answer_text: str, user_text: str, docs: list):
    if not _query_flags(user_text)["table"]:
        return answer_text

    table_answer = _build_table_answer(docs)
    if table_answer == NO_CONTEXT_MESSAGE:
        return answer_text
    if answer_text == NO_CONTEXT_MESSAGE:
        return table_answer
    if table_answer in answer_text:
        return answer_text
    return f"{answer_text}\n\n## Threshold Table\n{table_answer}"


def rag_answer(
    system_prompt: str,
    user_text: str,
    chat_history: list[dict] | None = None,
    docs: list | None = None,
):
    retriever = _get_resources()
    docs = docs or _prepare_answer_docs(
        user_text,
        retriever,
        include_threshold_docs=_query_flags(user_text)["amount_process"],
    )

    if not docs:
        return NO_CONTEXT_MESSAGE

    context_blocks = []
    used_sources = []
    for index, doc in enumerate(docs, start=1):
        source_name = Path(doc.metadata.get("source", "Unknown")).name
        page = doc.metadata.get("page", "N/A")
        doc_type = doc.metadata.get("doc_type", "Rule/Manual")
        year = doc.metadata.get("year", "Unknown")
        context_blocks.append(
            f"[Source {index} | {source_name} | Type: {doc_type} | Year: {year} | Page {page}]\n"
            f"{doc.page_content[:MAX_CHARS_PER_DOC]}"
        )
        used_sources.append(f"Source {index} - {source_name} (Page {page})")

    context_text = "\n\n".join(context_blocks)
    response_language = _detect_response_language(user_text)
    amount_instruction = _build_amount_instruction(user_text)
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"Reply language/style: {response_language}.\n"
                "- If the user asks in Hinglish, answer in Hinglish.\n"
                "- If the user asks in Hindi, answer in Hindi.\n"
                "- If the user asks in English, answer in English.\n\n"
                f"Conversation history:\n{_format_chat_history(chat_history)}\n\n"
                f"{amount_instruction}"
                f"Retrieved context:\n{context_text}\n\n"
                f"User question:\n{user_text}\n\n"
                "Rules:\n"
                "- Answer strictly from the retrieved context.\n"
                "- Mention only the sources you actually used.\n"
                "- Do not add outside knowledge.\n"
                "- For amount-based questions, compare the amount numerically against threshold text that is explicitly present in the context.\n"
                "- If the context gives both a lower-bound and upper-bound rule, choose the slab that actually contains the amount and explicitly reject non-matching slabs.\n"
                "- Do not use special-provision thresholds unless the retrieved text itself clearly applies to the user's procurement type.\n"
                "- Keep the amount consistent everywhere in the answer.\n"
                "- If the context contains conflicting thresholds, prefer the latest amendment or special provision over the older manual, and the older manual over GFR.\n"
                "- If you cannot support the conclusion directly from context, do not guess.\n"
                f"- If the answer is not supported, reply exactly: {NO_CONTEXT_MESSAGE}"
            ),
        },
    ]

    response = _chat_completion(messages, temperature=0.0)
    answer_text = _extract_message_content(response) or NO_CONTEXT_MESSAGE

    if answer_text == NO_CONTEXT_MESSAGE:
        return answer_text

    unique_sources = "\n".join(f"- {source}" for source in dict.fromkeys(used_sources))
    return f"{answer_text}\n\n---\nRetrieved Sources:\n{unique_sources}"


def handle_query(user_text: str, chat_history: list[dict] | None):
    intent = detect_intent(user_text)
    amount = extract_amount(user_text)
    wants_table = _query_flags(user_text)["table"]

    if intent == "HELP":
        return {"intent": intent, "amount": None, "answer": _build_help_answer()}

    if intent == "TABLE":
        retriever = _get_resources()
        docs = _prepare_answer_docs(user_text, retriever, include_threshold_docs=True)
        table_answer = _build_table_answer(docs)
        return {"intent": intent, "amount": None, "answer": table_answer}
    else:
        system_prompt = PROCESS_PROMPT if intent == "PROCESS" else POLICY_PROMPT

    if intent == "PROCESS" and amount is not None:
        retriever = _get_resources()
        docs = _prepare_answer_docs(user_text, retriever, include_threshold_docs=True)
        answer = rag_answer(system_prompt, user_text, chat_history, docs=docs)
        if wants_table:
            answer = _append_table_if_requested(answer, user_text, docs)
    else:
        answer = rag_answer(system_prompt, user_text, chat_history)
        if wants_table:
            retriever = _get_resources()
            docs = _prepare_answer_docs(user_text, retriever, include_threshold_docs=True)
            answer = _append_table_if_requested(answer, user_text, docs)
    return {"intent": intent, "amount": amount, "answer": answer}


def ask_question(user_text: str, chat_history: list[dict] | None) -> str:
    return handle_query(user_text, chat_history)["answer"]
