from __future__ import annotations

import gc
import hashlib
import logging
import os
import re
import time
import zipfile
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any

import faiss
import numpy as np
import pdfplumber
from cachetools import TTLCache
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from groq import Groq, RateLimitError, APITimeoutError, APIConnectionError
from pydantic import BaseModel, ConfigDict, Field
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

load_dotenv(BASE_DIR / ".env", override=True)
load_dotenv(ROOT_DIR / ".env", override=True)

print("ENV BASE:", BASE_DIR)
print("ENV ROOT:", ROOT_DIR)
_raw_key = os.getenv("GROQ_API_KEY") or ""
print("GROQ_API_KEY:", f"****{_raw_key[-4:]}" if len(_raw_key) > 4 else "<NOT SET>")

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("GROQ_API_KEY is not loaded. Check .env file path.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("procurebuddy-ai")
RATE_LIMIT_MAX_RETRIES = int(os.getenv("PROCUREBUDDY_LLM_RATE_LIMIT_RETRIES", "2"))
RATE_LIMIT_BACKOFF_BASE_SECONDS = float(os.getenv("PROCUREBUDDY_LLM_RATE_LIMIT_BACKOFF_BASE_SECONDS", "8"))
RATE_LIMIT_BACKOFF_CAP_SECONDS = float(os.getenv("PROCUREBUDDY_LLM_RATE_LIMIT_BACKOFF_CAP_SECONDS", "20"))
_rate_limit_cooldown_until = 0.0
_client: Groq | None = None
_client_api_key = ""

NO_RULE_FOUND = "No rule found in knowledge base"
NO_STRONG_MATCH_FOUND = "No strong match found, but here is related information from the knowledge base."
SECTION_MARKER = "\U0001F539"
GENERATION_CACHE_VERSION = "llm-hybrid-v1"
# ── Procurement Threshold Reference Table (injected into every prompt) ──────────
PROCUREMENT_THRESHOLDS_TABLE = """
## CSIR / GFR 2025 – Standard Procurement Thresholds (Master Reference)

| Value Band                    | Procurement Method          | GFR Rule | Authority / Notes                             |
|-------------------------------|-----------------------------|----------|-----------------------------------------------|
| Up to Rs. 50,000              | Direct Purchase             | Rule 154 | No quotation required; market rate sufficient |
| Rs. 50,001 – Rs. 5,00,000    | LPC (Local Purchase Comm.)  | Rule 155 | Minimum 3 quotations; LPC approval            |
| Rs. 5,00,001 – Rs. 25,00,000 | LTE (Limited Tender)        | Rule 162 | Minimum 3 firms; purchase committee approval  |
| Above Rs. 25,00,000           | OTE (Open Tender)           | Rule 161 | GeM / CPP Portal; wide publicity mandatory    |
| Proprietary / Single Source   | STE (Single Tender)         | Rule 166 | PAC certificate mandatory; DG approval >25L   |

> Source: UpdatedGFR31July2025_0.pdf (Master Reference) and CSIR Manual 2019
> IMPORTANT: If older documents mention Rs. 2.5 lakh as LTE threshold, use Rs. 5 lakh (GFR 2025).
""".strip()

SYSTEM_PROMPT = f"""You are ProcureBuddy — an expert Procurement Consultant for CSIR.
Your audience is Senior Scientists and Administrative Officers who need authoritative, audit-ready answers.

Never give one-liner answers. Always explain the legal and financial context using GFR 2025 and CSIR Manual 2019 rules.
Use **bold** for key figures and terms. Use Markdown tables when comparing thresholds or routes.

Answer ONLY from the provided knowledge-base context.
Do not invent facts. Do not expose chunk IDs, scores, or raw retrieval metadata.
Do not copy passages verbatim — rewrite in professional language.

## MANDATORY RESPONSE FORMAT

Every response MUST use exactly these five sections:

{SECTION_MARKER} Direct Answer:
<One clear, decisive sentence stating the main point. Bold the key figure or rule.>

{SECTION_MARKER} Detailed Explanation:
* <Rule or threshold with legal reference (e.g., GFR Rule 149)>
* <Conditions and exceptions>
* <Comparison with related thresholds if useful — use a Markdown table for 2+ comparisons>

{SECTION_MARKER} Procedural Steps:
1. <Step 1>
2. <Step 2>
3. <Step 3>
(If the question is not about a process, write "Not applicable — this is a rule/policy query.")

{SECTION_MARKER} Sources:
* <Document name — Chapter/Section/Para if available in context>

{SECTION_MARKER} Pro-Tip:
<One sentence of expert-level insight for a Senior Scientist. E.g., "Note: Proprietary items require a PAC certificate from the competent authority.">

Rules:
- Start with the direct answer immediately. Never open with "Based on the knowledge base" or "According to the documents".
- Rewrite source material in your own professional words.
- For amount/threshold questions, ALWAYS verify against the GFR 2025 threshold table provided in context.
- If the context is partial or the exact rule is not found, say so clearly and offer the closest related guidance.
- Keep the tone authoritative yet readable — suitable for a CSIR note file.
- MANDATORY: Procurement through GeM (Government e-Marketplace) is mandatory as per GFR Rule 149. Always mention GeM applicability unless the item is explicitly exempt.
- ANTI-HALLUCINATION: The LTE threshold is Rs. 5 lakh (NOT Rs. 2.5 lakh). If context mentions 2.5 lakh as LTE limit, override with Rs. 5 lakh per GFR 2025.
- Source priority when documents conflict: Latest OM/Special Provisions > CSIR Manual 2019 > GFR 2025 > GFR 2017.
"""

PROCESS_PROMPT = f"""You are ProcureBuddy — an expert Procurement Consultant for CSIR.
Your audience is Senior Scientists and Administrative Officers.

You will receive:
1. The user question
2. Retrieved document context
3. Extracted procurement facts from a deterministic analyzer
4. The official CSIR/GFR threshold table

For threshold and amount questions, ALWAYS cross-check with the threshold table. The table is the ground truth.
Use the extracted facts as supporting evidence, but write the final answer in natural language.

## MANDATORY RESPONSE FORMAT

{SECTION_MARKER} Direct Answer:
<State which procurement method applies and why. Bold the threshold band.>

{SECTION_MARKER} Detailed Explanation:
* <Applicable threshold band and rule reference>
* <Why this route applies (or does not apply)>
* <Any conditions, exceptions, or approval requirements>
(Use a Markdown table to compare bands if the question involves multiple thresholds.)

{SECTION_MARKER} Procedural Steps:
1. <Step 1: e.g., Check GeM availability>
2. <Step 2: e.g., Obtain quotations / prepare tender document>
3. <Step 3: e.g., Committee evaluation>
4. <Step 4: e.g., Approval from competent authority>

{SECTION_MARKER} Sources:
* <Document name — Chapter/Section/Para if available>

{SECTION_MARKER} Pro-Tip:
<Expert caution for a Senior Scientist, e.g., "For proprietary items, PAC certificate is mandatory even within the LTE band.">

Rules:
- For amount-based questions, clearly state the threshold band and method.
- If a route does not apply, say that directly and name the correct route.
- Combine retrieved rules into one readable explanation — never dump raw text.
- Always bold key figures like **Rs. 25 lakh** or **LTE**.
- Keep the tone authoritative and useful for note-file documentation.
- MANDATORY: Procurement through GeM (Government e-Marketplace) is mandatory as per GFR Rule 149. Always mention GeM applicability.
- ANTI-HALLUCINATION: The LTE threshold is Rs. 5 lakh (NOT Rs. 2.5 lakh). If context mentions 2.5 lakh as LTE limit, override with Rs. 5 lakh per GFR 2025.
- Source priority when documents conflict: Latest OM/Special Provisions > CSIR Manual 2019 > GFR 2025 > GFR 2017.
"""

MOJIBAKE_REPLACEMENTS = {
    "â": "–",   # – en dash
    "â": "—",   # — em dash
    "â": "’",   # ’ right single quote
    "â": "‘",   # ‘ left single quote
    "â": "“",   # “ left double quote
    "â": "”",   # ” right double quote
    "â¹": "₹",   # ₹ Indian Rupee sign
    "Â": "",              # stray Latin-1 padding byte
}


def resolve_data_dir() -> Path:
    configured = os.getenv("PROCUREBUDDY_DATA_DIR")
    if configured:
        return Path(configured).expanduser()

    service_dir = Path(__file__).resolve().parent
    repo_dir = service_dir.parent
    candidates = [
        repo_dir / "data",
        Path.cwd() / "data",
    ]

    for candidate in candidates:
        if candidate.exists():
            logger.info("Using discovered knowledge base directory: %s", candidate)
            return candidate

    fallback = Path("/home/ec2-user/procurebuddy-data")
    logger.info("Using fallback knowledge base directory: %s", fallback)
    return fallback


class Settings:
    def __init__(self) -> None:
        self.data_dir = resolve_data_dir()
        self.index_dir = Path(os.getenv("PROCUREBUDDY_INDEX_DIR", str(self.data_dir / ".index")))
        self.embedding_model = os.getenv("PROCUREBUDDY_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.chunk_size_words = int(os.getenv("PROCUREBUDDY_CHUNK_SIZE_WORDS", "350"))
        self.chunk_overlap_words = int(os.getenv("PROCUREBUDDY_CHUNK_OVERLAP_WORDS", "50"))
        self.top_k = min(5, max(3, int(os.getenv("PROCUREBUDDY_TOP_K", "5"))))
        self.min_score = float(os.getenv("PROCUREBUDDY_MIN_SCORE", "0.15"))
        self.answer_cache_size = max(100, int(os.getenv("PROCUREBUDDY_ANSWER_CACHE_SIZE", "500")))
        self.answer_cache_ttl_seconds = max(60, int(os.getenv("PROCUREBUDDY_ANSWER_CACHE_TTL_SECONDS", "3600")))
        self.llm_api_key = os.getenv("GROQ_API_KEY")
        if not self.llm_api_key:
            raise RuntimeError("GROQ_API_KEY is not loaded. Check .env file path.")
        self.llm_model = os.getenv("GROQ_MODEL") or "llama3-8b-8192"


settings = Settings()
os.makedirs(settings.index_dir, exist_ok=True)
def _reload_legacy_env() -> None:
    """Reload .env files so updated API keys can be picked up."""

    load_dotenv(BASE_DIR / ".env", override=True)
    load_dotenv(ROOT_DIR / ".env", override=True)


def _get_legacy_client() -> Groq:
    """Return a Groq client bound to the latest configured API key."""

    global _client, _client_api_key, api_key

    _reload_legacy_env()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not loaded. Check .env file path.")

    if _client is None or _client_api_key != api_key:
        _client = Groq(api_key=api_key)
        _client_api_key = api_key
        logger.info("Rebound legacy Groq client using the latest configured API key")
    return _client


class ChatRequest(BaseModel):
    # Accept camelCase keys sent by the frontend / Spring backend.
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
    response: str
    response_id: str
    response_hash: str
    source_chunk_ids: list[int] = Field(default_factory=list)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)


class SearchMatch(BaseModel):
    chunk_id: int
    document_id: int
    file_name: str
    chunk_index: int
    content: str
    token_count: int
    score: float


class SearchResponse(BaseModel):
    matches: list[SearchMatch]
    count: int


@dataclass
class ChunkRecord:
    chunk_id: int
    document_id: int
    file_name: str
    chunk_index: int
    content: str
    token_count: int


class KnowledgeBase:
    def __init__(self) -> None:
        self._lock = RLock()
        self._embedder = SentenceTransformer(settings.embedding_model)
        self._index: faiss.IndexFlatIP | None = None
        self._chunks: list[ChunkRecord] = []
        self._last_reload: str | None = None
        self._index_file = settings.index_dir / "faiss.index"
        self._metadata_file = settings.index_dir / "chunks.json"

    def initialize(self) -> dict[str, Any]:
        with self._lock:
            os.makedirs(settings.index_dir, exist_ok=True)
            if self._index_file.exists() and self._metadata_file.exists():
                try:
                    self._load_from_disk()
                    logger.info("Loaded persisted FAISS index from %s", settings.index_dir)
                    return self.status()
                except Exception:
                    logger.exception("Failed to load persisted FAISS index, rebuilding from documents")
            return self.reload()

    def reload(self) -> dict[str, Any]:
        with self._lock:
            settings.data_dir.mkdir(parents=True, exist_ok=True)
            os.makedirs(settings.index_dir, exist_ok=True)
            files = sorted(
                [
                    path
                    for path in settings.data_dir.iterdir()
                    if path.is_file() and path.suffix.lower() in {".pdf", ".txt", ".docx"}
                ],
                key=lambda path: path.name.lower(),
            )

            chunks: list[ChunkRecord] = []
            failures: list[dict[str, str]] = []
            next_chunk_id = 1

            for document_id, path in enumerate(files, start=1):
                try:
                    text = self._extract_text(path)
                    for chunk_index, chunk_text in enumerate(self._chunk_text(text), start=1):
                        chunks.append(
                            ChunkRecord(
                                chunk_id=next_chunk_id,
                                document_id=document_id,
                                file_name=path.name,
                                chunk_index=chunk_index,
                                content=chunk_text,
                                token_count=len(chunk_text.split()),
                            )
                        )
                        next_chunk_id += 1
                except Exception as exc:
                    logger.exception("Failed to index %s", path)
                    failures.append({"file_name": path.name, "error": str(exc)})

            if chunks:
                embeddings = self._embed([chunk.content for chunk in chunks])
                index = faiss.IndexFlatIP(embeddings.shape[1])
                index.add(embeddings)
                self._index = index
                del embeddings
            else:
                self._index = None

            self._chunks = chunks
            self._last_reload = datetime.now(timezone.utc).isoformat()
            self._persist_to_disk()
            gc.collect()

            return {
                "success": len(failures) == 0,
                "search_backend": "faiss-cosine",
                "embedding_model": settings.embedding_model,
                "data_dir": str(settings.data_dir),
                "document_count": len(files),
                "chunk_count": len(chunks),
                "indexed_count": len(files) - len(failures),
                "failed_count": len(failures),
                "failures": failures,
                "refreshed_at": self._last_reload,
            }

    def search(
        self,
        query: str,
        top_k: int | None = None,
        min_score_override: float | None = None,
    ) -> list[SearchMatch]:
        normalized_query = query.strip()
        if not normalized_query:
            return []

        with self._lock:
            if self._index is None or not self._chunks:
                logger.info("Search skipped because the FAISS index is not loaded")
                return []

            k = min(top_k or settings.top_k, len(self._chunks))
            query_embedding = self._embed([normalized_query])
            scores, indices = self._index.search(query_embedding, k)
            raw_scores = [float(score) for score in scores[0] if score >= 0]
            threshold = settings.min_score if min_score_override is None else float(min_score_override)
            logger.info(
                "Search query='%s' threshold=%.4f top_scores=%s",
                normalized_query,
                threshold,
                [round(score, 4) for score in raw_scores[: min(5, len(raw_scores))]],
            )

            matches: list[SearchMatch] = []
            for score, index in zip(scores[0], indices[0], strict=False):
                if index < 0 or score < threshold:
                    continue
                chunk = self._chunks[int(index)]
                matches.append(
                    SearchMatch(
                        chunk_id=chunk.chunk_id,
                        document_id=chunk.document_id,
                        file_name=chunk.file_name,
                        chunk_index=chunk.chunk_index,
                        content=chunk.content,
                        token_count=chunk.token_count,
                        score=float(score),
                    )
                )
            logger.info("Search query='%s' accepted_matches=%s", normalized_query, len(matches))
            return matches

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "status": "ok",
                "data_dir": str(settings.data_dir),
                "index_dir": str(settings.index_dir),
                "embedding_model": settings.embedding_model,
                "document_count": len({chunk.document_id for chunk in self._chunks}),
                "chunk_count": len(self._chunks),
                "index_loaded": self._index is not None and bool(self._chunks),
                "llm_enabled": bool(settings.llm_api_key and settings.llm_model),
                "model_name": settings.llm_model,
                "refreshed_at": self._last_reload,
            }

    def version_token(self) -> str:
        with self._lock:
            return self._last_reload or "uninitialized"

    def _embed(self, texts: list[str]) -> np.ndarray:
        embeddings = self._embedder.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype="float32")

    def _extract_text(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self._extract_pdf_text(path)
        if suffix == ".txt":
            return self._normalize_text(path.read_text(encoding="utf-8", errors="ignore"))
        if suffix == ".docx":
            return self._extract_docx_text(path)
        raise ValueError(f"Unsupported document type: {path.suffix}")

    def _extract_pdf_text(self, path: Path) -> str:
        text_parts: list[str] = []

        try:
            reader = PdfReader(str(path))
            for page in reader.pages:
                text_parts.append(page.extract_text() or "")
        except Exception:
            logger.warning("pypdf extraction failed for %s, trying pdfplumber", path)

        text = self._normalize_text("\n".join(text_parts))
        if text:
            return text

        plumber_parts: list[str] = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                plumber_parts.append(page.extract_text() or "")
        return self._normalize_text("\n".join(plumber_parts))

    def _extract_docx_text(self, path: Path) -> str:
        with zipfile.ZipFile(path) as archive:
            xml = archive.read("word/document.xml").decode("utf-8", errors="ignore")
        text = re.sub(r"</w:p>", "\n\n", xml)
        text = re.sub(r"<[^>]+>", " ", text)
        return self._normalize_text(text)

    def _chunk_text(self, text: str) -> list[str]:
        words = text.split()
        if not words:
            return []

        chunk_size = max(100, settings.chunk_size_words)
        overlap = max(0, min(settings.chunk_overlap_words, chunk_size // 2))

        chunks: list[str] = []
        start = 0
        while start < len(words):
            end = min(len(words), start + chunk_size)
            chunk = " ".join(words[start:end]).strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(words):
                break
            start = max(start + 1, end - overlap)
        return chunks

    def _normalize_text(self, value: str) -> str:
        normalized = value.replace("\x00", " ")
        normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
        normalized = re.sub(r"[ \t]+", " ", normalized)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        return normalized.strip()

    def _persist_to_disk(self) -> None:
        if self._index is None:
            return
        os.makedirs(settings.index_dir, exist_ok=True)
        faiss.write_index(self._index, str(self._index_file))
        payload = {
            "last_reload": self._last_reload,
            "chunks": [asdict(chunk) for chunk in self._chunks],
        }
        self._metadata_file.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
        logger.info("Persisted FAISS index with %s chunks to %s", len(self._chunks), settings.index_dir)

    def _load_from_disk(self) -> None:
        self._index = faiss.read_index(str(self._index_file))
        payload = json.loads(self._metadata_file.read_text(encoding="utf-8"))
        self._last_reload = payload.get("last_reload")
        self._chunks = [ChunkRecord(**chunk) for chunk in payload.get("chunks", [])]


knowledge_base = KnowledgeBase()
answer_cache = TTLCache(maxsize=settings.answer_cache_size, ttl=settings.answer_cache_ttl_seconds)
answer_cache_lock = RLock()
app = FastAPI(title="ProcureBuddy AI Service")


@app.on_event("startup")
def startup() -> None:
    logger.info("Initializing knowledge base from %s", settings.data_dir)
    logger.info(
        "Config: model=%s top_k=%d min_score=%.2f cache_size=%d cache_ttl=%ds",
        settings.llm_model, settings.top_k, settings.min_score,
        settings.answer_cache_size, settings.answer_cache_ttl_seconds,
    )
    print("LLM ENABLED:", bool(api_key))
    knowledge_base.initialize()


@app.get("/health")
def health() -> dict[str, Any]:
    status = dict(knowledge_base.status())
    status["service_variant"] = "legacy"
    status["service_version"] = "1.x-reference"
    return status


@app.post("/reload")
def reload_index() -> dict[str, Any]:
    result = knowledge_base.reload()
    with answer_cache_lock:
        answer_cache.clear()
    logger.info("Cleared answer cache after knowledge base reload")
    return result


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    matches = knowledge_base.search(request.query, settings.top_k)
    return SearchResponse(matches=matches, count=len(matches))


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    message = request.message.strip()
    user = request.user.strip()
    if not message:
        raise HTTPException(status_code=400, detail="message is required")
    if not user:
        raise HTTPException(status_code=400, detail="user is required")

    logger.info("Chat request user='%s' message='%s'", user, message[:120])

    try:
        # ── Greeting short-circuit ────────────────────────────────────────────
        if detect_intent(message) == "GREETING":
            resolved_name = extract_username(
                {
                    "displayName": request.display_name,
                    "username": request.username,
                    "email": request.email or user,
                }
            )
            greeting_text = (
                f"Namaste {resolved_name}! \U0001f60a\n"
                "How can I help you with procurement today?"
            )
            logger.info(
                "Greeting detected for user='%s' resolved_name='%s' — skipping RAG pipeline",
                user, resolved_name,
            )
            return build_chat_response(greeting_text, [])
        # ─────────────────────────────────────────────────────────────────────

        blocked_response_hashes = normalize_response_hashes(request.blocked_response_hashes)
        matches = knowledge_base.search(message, max(settings.top_k * 3, settings.top_k))
        matches = filter_matches(matches, request.blocked_chunk_ids)
        weak_match = False

        if not matches:
            logger.info("No strong retrieval match found for user='%s'; trying relaxed search", user)
            relaxed = knowledge_base.search(
                message,
                max(settings.top_k * 2, 5),
                min_score_override=0.0,
            )
            matches = filter_matches(relaxed, request.blocked_chunk_ids, query=message, require_domain=True)
            weak_match = True

        if not matches:
            if not knowledge_base.status().get("index_loaded"):
                logger.info("Knowledge base is empty or not loaded for user='%s'", user)
                empty_response = build_empty_knowledge_base_response()
                return build_chat_response(empty_response, [])
            logger.info("No relevant procurement content found after domain gate for user='%s'", user)
            no_match_response = build_no_match_response(message)
            return build_chat_response(no_match_response, [])

        result = handle_query(
            message=message,
            user=user,
            matches=matches,
            bypass_cache=request.bypass_cache,
            blocked_response_hashes=blocked_response_hashes,
            weak_match=weak_match,
        )
        payload = build_chat_response(result["answer"], matches).model_dump()
        with answer_cache_lock:
            answer_cache[build_cache_key(message)] = payload
        return ChatResponse(**payload)

    except HTTPException:
        raise  # let FastAPI handle validation errors
    except Exception:
        logger.exception("Unhandled error in /chat for user='%s' message='%s'", user, message[:120])
        error_response = (
            "I encountered an internal processing error while generating your answer. "
            "Please try again in a moment, or rephrase your question."
        )
        return build_chat_response(error_response, [])


def handle_query(
    message: str,
    user: str,
    matches: list[SearchMatch],
    bypass_cache: bool,
    blocked_response_hashes: set[str],
    weak_match: bool = False,
) -> dict[str, Any]:
    intent = detect_intent(message)
    amount = extract_amount_lakhs(message)
    prompt = build_prompt(message, user, matches, bypass_cache, weak_match=weak_match)
    system_prompt = SYSTEM_PROMPT

    if intent == "PROCESS":
        extracted_facts = _build_amount_answer(message, matches)
        prompt = build_process_prompt(
            message=message,
            user=user,
            matches=matches,
            facts=extracted_facts,
            bypass_cache=bypass_cache,
            weak_match=weak_match,
        )
        system_prompt = PROCESS_PROMPT
        logger.info("Using PROCESS prompt for user='%s' amount=%s", user, amount)

    llm_response = generate_llm_response(prompt, system_prompt=system_prompt)
    if llm_response:
        formatted_response = post_process_llm_output(
            llm_response,
            message,
            matches,
            weak_match=weak_match,
            intent=intent,
        )
        if response_hash(formatted_response) not in blocked_response_hashes:
            logger.info("Using LLM answer for user='%s' intent='%s'", user, intent)
            return {"intent": intent, "amount": amount, "answer": formatted_response}
        logger.info("Rejected previously disliked LLM response for user='%s' intent='%s'", user, intent)

    fallback_response = build_rule_based_answer(
        message,
        matches,
        blocked_response_hashes,
        weak_match=weak_match,
        intent=intent,
    )
    logger.info("Using fallback answer for user='%s' intent='%s'", user, intent)
    return {"intent": intent, "amount": amount, "answer": fallback_response}


def build_prompt(
    message: str,
    user: str,
    matches: list[SearchMatch],
    bypass_cache: bool,
    weak_match: bool = False,
) -> str:
    context_blocks = prepare_context_blocks(message, matches)
    lines = [
        "User question:",
        message,
        "",
        f"User identifier: {user}",
        "",
        "Official Procurement Threshold Reference:",
        PROCUREMENT_THRESHOLDS_TABLE,
        "",
        "Knowledge-base context:",
        context_blocks,
        "",
        "Instruction priority:",
        "1. Use only the knowledge-base context and the threshold table above.",
        "2. Start with a Direct Answer — one decisive sentence.",
        "3. Rewrite the context in professional natural language.",
        "4. Do not copy raw text, broken lines, extraction artifacts, or incomplete clauses.",
        "5. For any amount or threshold question, cross-check with the threshold table.",
        "6. Include all five sections: Direct Answer, Detailed Explanation, Procedural Steps, Sources, Pro-Tip.",
        "7. Use **bold** for key figures. Use Markdown tables for comparisons.",
        "8. Cite document name AND chapter/section/para when available in context.",
    ]
    if weak_match:
        lines.append("9. Mention that the answer is based on related guidance rather than an exact match.")
    if bypass_cache:
        lines.append("10. Use fresh wording and avoid repeating a previously rejected response.")
    return "\n".join(lines)


def build_process_prompt(
    message: str,
    user: str,
    matches: list[SearchMatch],
    facts: str,
    bypass_cache: bool,
    weak_match: bool = False,
) -> str:
    context_blocks = prepare_context_blocks(message, matches)
    lines = [
        "User question:",
        message,
        "",
        f"User identifier: {user}",
        "",
        "Official Procurement Threshold Reference:",
        PROCUREMENT_THRESHOLDS_TABLE,
        "",
        "Extracted procurement facts:",
        facts or "No structured facts were extracted.",
        "",
        "Retrieved document context:",
        context_blocks,
        "",
        "Writing guidance:",
        "1. Cross-check amounts against the threshold table above — it is the ground truth.",
        "2. State the applicable band and procurement method clearly.",
        "3. If a route does not apply, name the correct one.",
        "4. Provide step-by-step procedural steps (SOP) for the applicable method.",
        "5. Bold key figures like **Rs. 25 lakh** or **LTE**.",
        "6. Cite document name AND chapter/section/para when available.",
        "7. Include a Pro-Tip with expert-level caution for Senior Scientists.",
    ]
    if weak_match:
        lines.append("8. Briefly note that the answer is based on related guidance if the match is not exact.")
    if bypass_cache:
        lines.append("9. Use fresh wording and avoid repeating a previously rejected response.")
    return "\n".join(lines)


# Greeting words that should bypass RAG entirely
_GREETING_WORDS: frozenset[str] = frozenset({
    "hello", "hi", "hey", "namaste", "namaskar",
    "good morning", "good afternoon", "good evening", "good night",
    "howdy", "greetings", "sup", "hiya",
})


def detect_intent(message: str) -> str:
    lowered = message.strip().lower()
    # Check greeting first — single-word or short phrase match
    if any(word in lowered for word in _GREETING_WORDS):
        # Avoid misfires: only treat as greeting if the message is short and
        # contains no domain-specific procurement nouns.
        procurement_signals = (
            "tender", "purchase", "procurement", "budget", "committee",
            "approval", "lakh", "rupee", "crore", "threshold", "limit",
            "table", "rule", "regulation", "act", "gfr", "csir", "bid",
        )
        has_procurement_signal = any(sig in lowered for sig in procurement_signals)
        if not has_procurement_signal:
            return "GREETING"

    clean = clean_text(message).lower()
    process_terms = ("process", "procedure", "steps", "how", "route", "approval")
    if extract_amount_lakhs(message) is not None:
        return "PROCESS"
    if any(term in clean for term in process_terms):
        return "PROCESS"
    return "GENERAL"


def extract_username(data: Any) -> str:
    if not isinstance(data, dict):
        return "User"

    if data.get("displayName"):
        return str(data["displayName"]).strip()

    if data.get("username"):
        return str(data["username"]).strip()

    if data.get("email"):
        return str(data["email"]).split("@")[0].capitalize()

    return "User"


def _build_amount_answer(message: str, matches: list[SearchMatch]) -> str:
    amount_lakhs = extract_amount_lakhs(message)
    explanation_points = build_explanation_points(message, matches)
    sources = extract_source_names(matches)
    facts: list[str] = []

    if amount_lakhs is not None:
        facts.append(f"Asked amount: {format_lakh_amount(amount_lakhs)}")

    for point in explanation_points[:3]:
        cleaned_point = cleanup_generated_sentence(point)
        if cleaned_point:
            facts.append(cleaned_point)

    if sources:
        facts.append("Relevant sources: " + ", ".join(sources[:3]))

    return "\n".join(f"- {fact}" for fact in facts) if facts else "- No structured procurement facts extracted."


def build_cache_key(message: str) -> str:
    normalized_message = re.sub(r"\s+", " ", message.strip().lower())
    return f"{knowledge_base.version_token()}::{GENERATION_CACHE_VERSION}::{normalized_message}"


def normalize_response_hashes(values: list[str]) -> set[str]:
    return {value.strip().lower() for value in values if value and value.strip()}


# Procurement domain terms used to check chunk relevance.
_PROCUREMENT_DOMAIN_TERMS: frozenset[str] = frozenset({
    "tender", "procurement", "purchase", "supplier", "vendor", "bid", "quotation",
    "committee", "approval", "lpc", "lte", "ste", "gem", "pat", "dge",
    "rupee", "lakh", "crore", "threshold", "limit", "sanction", "authority",
    "csir", "gfr", "rate", "contract", "order", "specification", "estimate",
    "store", "indent", "rfq", "work", "service", "goods", "manual",
})

# Section-heading patterns that should rank higher in procurement context.
_SECTION_HEADING_RE = re.compile(
    r"\b(single\s+tender|limited\s+tender|open\s+tender"
    r"|advertised\s+tender|direct\s+purchase"
    r"|ste|lte|ote|lpc|gem"
    r"|pat|dge&d|rate\s+contract|local\s+purchase"
    r"|rule\s+1(?:49|54|55|61|62|66)"
    r"|purchase\s+committee|technical.*purchase\s+committee"
    r"|local\s+purchase\s+committee"
    r"|\d+\.\d+(?:\.\d+)?\s+[A-Z])",
    re.IGNORECASE,
)


def _chunk_procurement_score(content: str) -> int:
    """Count how many domain terms appear in content. Used for relevance gating."""
    lowered = content.lower()
    return sum(1 for term in _PROCUREMENT_DOMAIN_TERMS if term in lowered)


def _is_clean_chunk(content: str) -> bool:
    """Return True only if the chunk has enough readable, domain-relevant text."""
    if not content or len(content.strip()) < 50:
        return False
    alpha = sum(1 for c in content if c.isalpha())
    if alpha < max(20, len(content) * 0.25):   # at least 25 % alpha chars
        return False
    if looks_like_table_of_contents(content):
        return False
    return True


def filter_matches(
    matches: list[SearchMatch],
    blocked_chunk_ids: list[int],
    query: str = "",
    require_domain: bool = False,
) -> list[SearchMatch]:
    """Filter retrieval matches.

    Steps:
        1. Remove blocked chunk IDs (feedback-disliked).
        2. Remove chunks that are too short, too noisy, or look like a TOC.
        3. Optionally (require_domain=True) remove chunks with no procurement
           domain signal at all — used after the relaxed zero-score fallback
           to prevent garbage from reaching the LLM.
    """
    blocked = {int(chunk_id) for chunk_id in blocked_chunk_ids}

    filtered: list[SearchMatch] = []
    for match in matches:
        if match.chunk_id in blocked:
            continue
        cleaned = clean_text(match.content)
        if not _is_clean_chunk(cleaned):
            logger.debug("Rejected noisy/short chunk_id=%s file=%s", match.chunk_id, match.file_name)
            continue
        if require_domain and _chunk_procurement_score(cleaned) == 0:
            logger.debug("Rejected off-domain chunk_id=%s file=%s", match.chunk_id, match.file_name)
            continue
        filtered.append(match)
    return filtered


def response_hash(value: str) -> str:
    normalized = re.sub(r"\s+", " ", value.strip())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def build_chat_response(response: str, matches: list[SearchMatch]) -> ChatResponse:
    answer = response.strip() or build_no_match_response("the request")
    answer_hash = response_hash(answer)
    return ChatResponse(
        response=answer,
        response_id=f"resp_{answer_hash[:16]}",
        response_hash=answer_hash,
        source_chunk_ids=[match.chunk_id for match in matches[:settings.top_k]],
    )


def _legacy_rate_limit_backoff_seconds(attempt: int) -> float:
    return min(RATE_LIMIT_BACKOFF_CAP_SECONDS, RATE_LIMIT_BACKOFF_BASE_SECONDS + (attempt * 4.0))


def _wait_for_legacy_cooldown() -> None:
    global _rate_limit_cooldown_until
    remaining = _rate_limit_cooldown_until - time.monotonic()
    if remaining > 0:
        logger.warning("Honoring shared legacy Groq cooldown for %.1fs before next request", remaining)
        time.sleep(remaining)


def generate_llm_response(prompt: str, system_prompt: str = SYSTEM_PROMPT) -> str | None:
    global _rate_limit_cooldown_until
    try:
        logger.info(
            "Calling LLM for retrieved answer generation model='%s' prompt_length=%s",
            settings.llm_model,
            len(prompt),
        )
        for attempt in range(RATE_LIMIT_MAX_RETRIES + 1):
            try:
                _wait_for_legacy_cooldown()
                response = _get_legacy_client().chat.completions.create(
                    model=settings.llm_model,
                    temperature=0,
                    timeout=90,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                )
                content = (response.choices[0].message.content or "").strip()
                if not content:
                    logger.info("LLM returned an empty response")
                    return None
                logger.info("LLM returned content_length=%s", len(content))
                return content
            except RateLimitError as exc:
                backoff_seconds = _legacy_rate_limit_backoff_seconds(attempt)
                _rate_limit_cooldown_until = time.monotonic() + backoff_seconds
                logger.warning(
                    "Legacy Groq rate limit hit attempt=%d/%d; backing off %.1fs",
                    attempt + 1,
                    RATE_LIMIT_MAX_RETRIES + 1,
                    backoff_seconds,
                )
                if attempt >= RATE_LIMIT_MAX_RETRIES:
                    logger.error("Legacy Groq rate limit persisted: %s. Returning None for fallback.", exc)
                    return None
                time.sleep(backoff_seconds)
    except APITimeoutError:
        logger.error("Groq API call timed out after 90s; returning None for fallback")
        return None
    except APIConnectionError as exc:
        logger.error("Groq API connection error: %s. Returning None for fallback.", exc)
        return None
    except Exception:
        logger.exception("LLM generation failed, falling back to rule-based response")
        return None


def build_rule_based_answer(
    message: str,
    matches: list[SearchMatch],
    blocked_response_hashes: set[str],
    weak_match: bool = False,
    intent: str = "GENERAL",
) -> str:
    if not matches:
        return build_no_match_response(message)

    explanation_points = build_explanation_points(message, matches)
    summary_answer = compose_direct_answer(message, explanation_points, weak_match, matches)
    candidate = render_structured_response(
        answer=summary_answer,
        explanation_points=explanation_points,
        sources=extract_source_names(matches),
    )
    if response_hash(candidate) in blocked_response_hashes:
        candidate = render_structured_response(
            answer=f"Here is an alternate summary for: {message.strip()}",
            explanation_points=explanation_points,
            sources=extract_source_names(matches),
        )
    return candidate


def build_rule_based_intro(message: str, weak_match: bool) -> str:
    normalized_message = describe_user_request(message)
    amount_lakhs = extract_amount_lakhs(message)
    if amount_lakhs is not None:
        amount_label = format_lakh_amount(amount_lakhs)
        if weak_match:
            return f"For a procurement value of {amount_label}, the retrieved guidance is related but not fully exact."
        return f"For a procurement value of {amount_label}, the retrieved guidance highlights the relevant value thresholds and applicable procedure."
    if weak_match:
        return f"The documents provide related guidance for {normalized_message}, although the match is not exact."
    return f"The retrieved procurement guidance addresses {normalized_message}."


def post_process_llm_output(
    response: str,
    message: str,
    matches: list[SearchMatch],
    weak_match: bool = False,
    intent: str = "GENERAL",
) -> str:
    sections = parse_llm_sections(response)
    fallback_explanation_points = build_explanation_points(message, matches)

    answer = compact_answer_text(
        sections.get("direct answer")
        or sections.get("answer")
        or compose_direct_answer(message, fallback_explanation_points, weak_match, matches)
    )
    explanation_points = normalize_points(
        sections.get("detailed explanation", "")
        or sections.get("explanation", ""),
        fallback_points=fallback_explanation_points,
        max_points=4,
    )
    procedural_steps = sections.get("procedural steps", "").strip()
    pro_tip = sections.get("pro-tip", "").strip() or sections.get("pro tip", "").strip()

    # ── Anti-hallucination: auto-correct stale "2.5 lakh" LTE threshold ──────
    assembled = "\n".join([
        answer,
        "\n".join(explanation_points),
        procedural_steps,
        pro_tip,
    ])
    if re.search(r"(?:2\.?5|2,50,000)\s*(?:lakh|lac).*(?:LTE|limited\s+tender)", assembled, re.IGNORECASE) \
       or re.search(r"(?:LTE|limited\s+tender).*(?:2\.?5|2,50,000)\s*(?:lakh|lac)", assembled, re.IGNORECASE):
        logger.warning("Anti-hallucination: correcting stale 2.5 lakh LTE threshold to 5 lakh (GFR 2025)")
        answer = re.sub(r"Rs\.?\s*2[.,]?5\s*lakh", "Rs. 5 lakh", answer, flags=re.IGNORECASE)
        answer = re.sub(r"Rs\.?\s*2,50,000", "Rs. 5,00,000", answer, flags=re.IGNORECASE)
        explanation_points = [
            re.sub(r"Rs\.?\s*2[.,]?5\s*lakh", "Rs. 5 lakh", p, flags=re.IGNORECASE)
            for p in explanation_points
        ]
        explanation_points = [
            re.sub(r"Rs\.?\s*2,50,000", "Rs. 5,00,000", p, flags=re.IGNORECASE)
            for p in explanation_points
        ]
    # ─────────────────────────────────────────────────────────────────────────

    return render_structured_response(
        answer=answer,
        explanation_points=explanation_points,
        sources=extract_source_names(matches),
        procedural_steps=procedural_steps,
        pro_tip=pro_tip,
    )


def parse_structured_sections(response: str) -> dict[str, str]:
    normalized = clean_text(response)
    pattern = re.compile(
        rf"(?:{re.escape(SECTION_MARKER)}|ðŸ”¹)\s*(Answer|Detailed Explanation|Supporting Context|Sources|Confidence)\s*:\s*",
        re.IGNORECASE,
    )
    matches = list(pattern.finditer(normalized))
    if not matches:
        return {}

    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        key = match.group(1).strip().lower()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(normalized)
        sections[key] = normalized[start:end].strip()
    return sections


def parse_llm_sections(response: str) -> dict[str, str]:
    normalized = clean_text(response)
    heading_pattern = rf"(?:{re.escape(SECTION_MARKER)}|[\u25b9\U0001F539])"
    section_names = (
        "Direct Answer|Answer"
        "|Detailed Explanation|Explanation"
        "|Procedural Steps"
        "|Supporting Context"
        "|Sources|Source Citation"
        "|Pro-Tip|Pro Tip|Caution"
        "|Confidence"
    )
    pattern = re.compile(
        heading_pattern + r"\s*(" + section_names + r")\s*:\s*",
        re.IGNORECASE,
    )
    matches = list(pattern.finditer(normalized))
    if not matches:
        return {}

    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        key = match.group(1).strip().lower()
        # Normalize aliases: "answer" → "direct answer", "source citation" → "sources"
        if key == "answer":
            key = "direct answer"
        elif key in ("source citation", "source"):
            key = "sources"
        elif key == "caution":
            key = "pro-tip"
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(normalized)
        sections[key] = normalized[start:end].strip()
    return sections


def prepare_context_blocks(message: str, matches: list[SearchMatch]) -> str:
    ranked_matches = prioritize_matches(message, matches)[: min(5, len(matches))]
    blocks: list[str] = []
    for index, match in enumerate(ranked_matches, start=1):
        summary = summarize_match_for_context(message, match)
        blocks.append(f"{index}. Document: {match.file_name}\n   Summary: {summary}")
    return "\n".join(blocks)


def _source_priority_bonus(file_name: str) -> float:
    """Port of core.py _score_doc source-level weighting.

    Priority: SnT Special Provisions / Amendments > CSIR Manual > GFR > other.
    """
    lowered = file_name.lower()
    if "special provisions" in lowered or "amendment" in lowered:
        return 0.12
    if "csir manual" in lowered:
        return 0.08
    if "gfr" in lowered or "updatedgfr" in lowered:
        return 0.06
    return 0.0


def _query_relevance_bonus(query: str, content: str) -> float:
    """Port of core.py _query_flags concept: give extra bonus to chunks
    that match the specific retrieval intent of the query."""
    lowered_query = query.lower()
    lowered_content = content.lower()
    bonus = 0.0

    # Amount / process queries: boost chunks with threshold markers
    if any(kw in lowered_query for kw in ("lakh", "crore", "rs", "purchase process", "procurement process", "committee")):
        for marker in (
            "purchase committee", "technical & purchase committee",
            "local purchase committee", "limited tender", "advertised tender",
            "rule 155", "rule 161", "rule 162", "up to rs", "above rs",
        ):
            if marker in lowered_content:
                bonus += 0.03

    # Single tender queries: boost STE-relevant chunks
    if any(kw in lowered_query for kw in ("single tender", "ste", "proprietary", "pac")):
        for marker in ("single tender", "rule 166", "proprietary article", "standardisation", "emergency"):
            if marker in lowered_content:
                bonus += 0.04

    # Table / overview queries: boost comprehensive rule chunks
    if any(kw in lowered_query for kw in ("table", "slab", "overview", "matrix")):
        for marker in ("rule 154", "rule 155", "rule 161", "rule 162", "direct purchase", "limited tender enquiry"):
            if marker in lowered_content:
                bonus += 0.04

    return min(bonus, 0.20)  # cap to avoid overriding semantic score


def prioritize_matches(query: str, matches: list[SearchMatch]) -> list[SearchMatch]:
    query_terms = tokenize(query)

    def score(match: SearchMatch) -> tuple[float, float]:
        try:
            cleaned = clean_text(match.content)
            content_terms = tokenize(cleaned)

            # Keyword overlap bonus — weighted so exact-term matches win.
            overlap = len(query_terms.intersection(content_terms))
            keyword_bonus = overlap * 0.07

            # Section-heading bonus: chunks with Rule X / tender type headings.
            heading_bonus = 0.10 if _SECTION_HEADING_RE.search(cleaned) else 0.0

            # Procurement domain density bonus.
            domain_score = min(_chunk_procurement_score(cleaned) * 0.02, 0.10)

            # Source-level priority (from core.py _score_doc logic).
            source_bonus = _source_priority_bonus(match.file_name)

            # Query-flag relevance boost (from core.py _query_flags concept).
            query_bonus = _query_relevance_bonus(query, cleaned)

            # Text quality: prefer chunks with more alpha chars than digit chars.
            alpha_chars = sum(1 for char in cleaned if char.isalpha())
            digit_chars = sum(1 for char in cleaned if char.isdigit())
            quality = (alpha_chars / max(1, len(cleaned))) - (digit_chars / max(1, len(cleaned)))
            quality_bonus = quality * 0.02

            combined = match.score + keyword_bonus + heading_bonus + domain_score + source_bonus + query_bonus + quality_bonus
            return (combined, match.score)
        except Exception:
            logger.warning(
                "Scoring failed for chunk_id=%s file=%s, using base score",
                match.chunk_id, match.file_name, exc_info=True,
            )
            return (match.score, match.score)

    return sorted(matches, key=score, reverse=True)


def clean_text(value: str) -> str:
    cleaned = value or ""
    for source, target in MOJIBAKE_REPLACEMENTS.items():
        cleaned = cleaned.replace(source, target)
    cleaned = cleaned.replace("\x00", " ")
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"-\s*\n\s*", "", cleaned)
    cleaned = re.sub(r"\n+", " ", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    return cleaned.strip()


def summarize_match_for_context(query: str, match: SearchMatch) -> str:
    cleaned = clean_text(match.content)
    if looks_like_table_of_contents(cleaned):
        return ""
    selected = select_relevant_sentences(query, cleaned, max_sentences=2)
    if not selected:
        return cleanup_generated_sentence(rewrite_procurement_sentence(cleaned))
    return cleanup_generated_sentence(" ".join(selected))


def select_relevant_sentences(query: str, content: str, max_sentences: int = 2) -> list[str]:
    if looks_like_table_of_contents(content):
        return []

    candidates = [candidate for candidate in extract_readable_units(content) if not looks_like_table_of_contents(candidate)]
    if not candidates:
        return []

    query_terms = tokenize(query)
    amount_lakhs = extract_amount_lakhs(query)
    ranked: list[tuple[int, int, str]] = []
    for candidate in candidates:
        sentence_terms = tokenize(candidate)
        overlap = len(query_terms.intersection(sentence_terms))
        amount_bonus = score_amount_relevance(candidate, amount_lakhs)
        process_bonus = score_process_relevance(query, candidate)
        alpha_chars = sum(1 for char in candidate if char.isalpha())
        digit_chars = sum(1 for char in candidate if char.isdigit())
        readability = alpha_chars - digit_chars
        ranked.append((overlap + amount_bonus + process_bonus, readability, candidate))

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


def compact_answer_text(value: str) -> str:
    cleaned = clean_text(value)
    protected = cleaned.replace("Rs.", "Rs<dot>")
    sentences = [sentence.replace("Rs<dot>", "Rs.") for sentence in split_sentences(protected)]
    if not sentences:
        return smooth_summary_text(cleaned, max_length=320)
    answer = " ".join(sentences[: min(2, len(sentences))])
    return smooth_summary_text(answer, max_length=320)


def normalize_points(raw_value: str, fallback_points: list[str], max_points: int = 4) -> list[str]:
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
        normalized = semantic_dedup_key(point)
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(smooth_summary_text(point, max_length=180))
    if not deduped:
        deduped = [cleanup_generated_sentence(point) for point in fallback_points]
        deduped = [point for point in deduped if point]
    return [point for point in deduped[:max_points] if point]


def build_explanation_points(message: str, matches: list[SearchMatch]) -> list[str]:
    amount_specific_points = build_amount_specific_points(message, matches)
    if amount_specific_points:
        return amount_specific_points

    ranked = prioritize_matches(message, matches)[: min(5, len(matches))]
    points: list[str] = []
    seen: set[str] = set()
    for match in ranked:
        if looks_like_table_of_contents(match.content):
            continue
        for summary in select_relevant_sentences(message, clean_text(match.content), max_sentences=3):
            rewritten = cleanup_generated_sentence(summary)
            if not rewritten:
                continue
            key = semantic_dedup_key(rewritten)
            if key in seen:
                continue
            seen.add(key)
            points.append(rewritten)
            if len(points) >= 3:
                return points
    return points or ["The retrieved documents provide related procurement guidance, but the wording is partial."]


def build_supporting_context_points(message: str, matches: list[SearchMatch]) -> list[str]:
    explanation_points = build_explanation_points(message, matches)
    if not explanation_points:
        return ["The knowledge base contains related procurement material, but the exact section is limited."]

    amount_lakhs = extract_amount_lakhs(message)
    points: list[str] = []
    if amount_lakhs is not None:
        points.append(build_amount_context_summary(amount_lakhs, explanation_points))

    sources = extract_source_names(matches)
    if sources:
        points.append(
            cleanup_generated_sentence(
                f"This summary comes from {sources[0]} and focuses on value thresholds and supplier eligibility."
            )
        )
    else:
        points.append("The retrieved context focuses on procurement thresholds and procedure notes relevant to the question.")

    deduped: list[str] = []
    seen: set[str] = set()
    for point in points:
        cleaned_point = cleanup_generated_sentence(point)
        if not cleaned_point:
            continue
        key = semantic_dedup_key(cleaned_point)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(cleaned_point)
    return deduped[:2] or ["The knowledge base contains related procurement material, but the exact section is limited."]


def extract_source_names(matches: list[SearchMatch]) -> list[str]:
    sources: list[str] = []
    seen: set[str] = set()
    for match in prioritize_matches("", matches):
        source_name = clean_text(match.file_name)
        normalized = source_name.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        sources.append(source_name)
    return sources[:5]


def infer_confidence(matches: list[SearchMatch], weak_match: bool = False) -> str:
    if weak_match or not matches:
        return "Low"
    top_score = matches[0].score
    if top_score >= 0.55 and len(matches) >= 3:
        return "High"
    if top_score >= 0.25:
        return "Medium"
    return "Low"


def normalize_confidence(value: str | None, fallback: str) -> str:
    cleaned = clean_text(value or "").lower()
    if cleaned.startswith("high"):
        return "High"
    if cleaned.startswith("medium"):
        return "Medium"
    if cleaned.startswith("low"):
        return "Low"
    return fallback


def render_structured_response(
    answer: str,
    explanation_points: list[str],
    sources: list[str],
    procedural_steps: str = "",
    pro_tip: str = "",
) -> str:
    lines = [
        f"{SECTION_MARKER} Direct Answer:",
        compact_answer_text(answer),
        "",
        f"{SECTION_MARKER} Detailed Explanation:",
    ]
    for point in explanation_points[:4]:
        cleaned_point = cleanup_generated_sentence(point)
        if cleaned_point:
            lines.append(f"* {cleaned_point}")

    # Procedural Steps section
    lines.append("")
    lines.append(f"{SECTION_MARKER} Procedural Steps:")
    if procedural_steps:
        lines.append(procedural_steps)
    else:
        lines.append("Not applicable — this is a rule/policy query.")

    # Sources section
    lines.append("")
    lines.append(f"{SECTION_MARKER} Sources:")
    for source in sources or ["Knowledge base documents"]:
        lines.append(f"* {clean_text(source)}")

    # Pro-Tip section
    lines.append("")
    lines.append(f"{SECTION_MARKER} Pro-Tip:")
    if pro_tip:
        cleaned_tip = cleanup_generated_sentence(pro_tip)
        lines.append(cleaned_tip or pro_tip)
    else:
        lines.append("Always verify the current threshold limits from the latest GFR circular before initiating procurement.")

    return "\n".join(lines).strip()


def compose_direct_answer(
    message: str,
    explanation_points: list[str],
    weak_match: bool = False,
    matches: list[SearchMatch] | None = None,
) -> str:
    request_label = describe_user_request(message)
    amount_lakhs = extract_amount_lakhs(message)
    match_features = extract_match_features(matches or [])
    if amount_lakhs is not None:
        amount_label = format_lakh_amount(amount_lakhs)
        mentions_small_purchase = match_features["has_lpc_limit"] or any(
            "2.5 lakhs" in point.lower() or "local purchase committee" in point.lower()
            for point in explanation_points
        )
        mentions_fifty_lakhs = match_features["has_under_50_rule"] or any("50 lakhs" in point.lower() for point in explanation_points)
        if amount_lakhs > 2.5 and mentions_small_purchase and mentions_fifty_lakhs:
            return (
                f"For {amount_label}, this is not a Local Purchase Committee purchase. "
                "It is above the Rs. 2.5 lakh LPC limit and falls within the up-to-Rs. 50 lakh band referenced in the retrieved guidance."
            )
        if mentions_fifty_lakhs:
            return f"For {amount_label}, the value falls within the up-to-Rs. 50 lakh band used for certain supplier eligibility and purchase preference rules."
    if explanation_points:
        primary = cleanup_generated_sentence(explanation_points[0])
        if primary:
            if weak_match:
                return f"For {request_label}, the closest related guidance is: {primary}"
            return f"For {request_label}, {primary[:1].lower() + primary[1:]}"
    if weak_match:
        return f"The documents provide related guidance for {request_label}, although the match is not exact."
    return f"The retrieved procurement guidance addresses {request_label}."


def split_candidate_points(raw_value: str) -> list[str]:
    if "* " in raw_value and "\n" not in raw_value:
        return [segment.strip() for segment in raw_value.split("* ") if segment.strip()]
    parts: list[str] = []
    for line in raw_value.splitlines():
        if not line.strip():
            continue
        normalized_line = re.sub(r"^[\-\*\d.)\s]+", "", line.strip())
        if normalized_line:
            parts.append(normalized_line)
    if parts:
        return parts
    return [segment for segment in re.split(r"(?<=[.!?])\s+", raw_value) if segment.strip()]


def cleanup_generated_sentence(value: str) -> str:
    cleaned = rewrite_procurement_sentence(value)
    cleaned = smooth_summary_text(cleaned, max_length=180)
    cleaned = strip_leading_noise(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = cleaned.replace("..", ".")
    cleaned = re.sub(r"\b(?:where|that|which|subject to|provided that)\s+\.$", ".", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(?:as the case may be|etc|and so on)\s*\.$", ".", cleaned, flags=re.IGNORECASE)
    if not cleaned:
        return ""
    if not is_useful_sentence(cleaned):
        return ""
    return ensure_sentence_punctuation(cleaned)


def strip_leading_noise(value: str) -> str:
    cleaned = value
    cleaned = re.sub(r"^(?:chapter|annexure|appendix|table|form)\s*[-:0-9a-zA-Z.() ]+\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^(?:\d+(?:\.\d+){0,4}\s*)+", "", cleaned)
    cleaned = re.sub(r"^(?:[A-Z]{1,4}\s+){2,}", "", cleaned)
    cleaned = re.sub(r"^\W+", "", cleaned)
    return cleaned.strip()


def ensure_sentence_punctuation(value: str) -> str:
    cleaned = value.strip().rstrip(",;:-")
    if not cleaned:
        return ""
    if cleaned[-1] not in ".!?":
        cleaned = f"{cleaned}."
    return cleaned


def is_useful_sentence(value: str) -> bool:
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


def extract_readable_units(content: str) -> list[str]:
    cleaned = clean_text(content)
    units = re.split(r"(?<=[.!?;:])\s+|,\s+(?=(?:and|but|if|when|where|while|provided|however)\b)", cleaned)
    extracted: list[str] = []
    for unit in units:
        normalized = rewrite_procurement_sentence(unit)
        normalized = strip_leading_noise(normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        if not normalized:
            continue
        if len(normalized.split()) < 7:
            continue
        if not any(char.isalpha() for char in normalized):
            continue
        cleaned_unit = cleanup_generated_sentence(normalized)
        if cleaned_unit:
            extracted.append(cleaned_unit)
    return extracted


def smooth_summary_text(value: str, max_length: int = 220) -> str:
    cleaned = clean_text(value)
    cleaned = strip_leading_noise(cleaned)
    cleaned = cleaned.replace("₹", "Rs. ")
    cleaned = re.sub(r"\b\d+\s*$", "", cleaned).strip()
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    if len(cleaned) <= max_length:
        return ensure_sentence_punctuation(cleaned)
    truncated = cleaned[:max_length].rsplit(" ", 1)[0].rstrip(",;:-")
    return ensure_sentence_punctuation(truncated)


def extract_amount_lakhs(message: str) -> float | None:
    cleaned = clean_text(message).lower().replace(",", "")
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:lakh|lakhs|lac|lacs)\b", cleaned)
    if match:
        return float(match.group(1))
    return None


def looks_like_table_of_contents(value: str) -> bool:
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


def format_lakh_amount(amount_lakhs: float) -> str:
    if float(amount_lakhs).is_integer():
        return f"Rs. {int(amount_lakhs)} lakhs"
    return f"Rs. {amount_lakhs:g} lakhs"


def score_amount_relevance(candidate: str, amount_lakhs: float | None) -> int:
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
    if amount_lakhs > 2.5 and ("2,50,000" in candidate or "2.5 lakhs" in lowered or "local purchase committee" in lowered):
        score += 2
    return score


def score_process_relevance(query: str, candidate: str) -> int:
    query_lower = query.lower()
    candidate_lower = candidate.lower()
    score = 0
    if "process" in query_lower and any(term in candidate_lower for term in ("procedure", "procurement", "committee", "eligible", "award", "l1")):
        score += 2
    if "table" in query_lower and "|" in candidate:
        score += 1
    return score


def semantic_dedup_key(value: str) -> str:
    lowered = clean_text(value).lower()
    if "local purchase committee" in lowered or "2,50,000" in lowered or "2.5 lakhs" in lowered:
        return "lpc-threshold"
    if "only local suppliers" in lowered and "50 lakhs" in lowered:
        return "local-supplier-under-50"
    if "above rs. 50 lakh" in lowered or "more than 50 lakhs" in lowered or "next purchase-preference rules" in lowered:
        return "above-50-rule"
    if "l1" in lowered and "50%" in lowered:
        return "l1-supplier-split"
    return lowered


def rewrite_procurement_sentence(value: str) -> str:
    cleaned = clean_text(value)
    lowered = cleaned.lower()

    if "local purchase committee" in lowered and "25,000" in cleaned and ("2,50,000" in cleaned or "two lakh fifty thousand" in lowered):
        return (
            "Local Purchase Committee procurement is meant only for smaller purchases above Rs. 25,000 and up to Rs. 2.5 lakhs, "
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


def build_amount_context_summary(amount_lakhs: float, explanation_points: list[str]) -> str:
    amount_label = format_lakh_amount(amount_lakhs)
    mentions_small_purchase = any("2.5 lakhs" in point.lower() or "local purchase committee" in point.lower() for point in explanation_points)
    mentions_fifty_lakhs = any("50 lakhs" in point.lower() for point in explanation_points)
    if mentions_small_purchase and mentions_fifty_lakhs:
        return f"{amount_label} sits above the small Local Purchase Committee range but below the Rs. 50 lakh threshold mentioned in the retrieved guidance."
    if mentions_fifty_lakhs:
        return f"{amount_label} falls within the up-to-Rs. 50 lakh value band discussed in the retrieved guidance."
    return f"{amount_label} should be read against the value thresholds described in the retrieved procurement guidance."


def extract_match_features(matches: list[SearchMatch]) -> dict[str, bool]:
    combined = " ".join(clean_text(match.content).lower() for match in matches[:8])
    return {
        "has_lpc_limit": (
            "local purchase committee" in combined
            and ("2,50,000" in combined or "two lakh fifty thousand" in combined)
        ),
        "has_under_50_rule": (
            ("50 lakhs or less" in combined or "up to rs. 50 lakhs" in combined or "up to 50 lakhs" in combined)
            and ("local suppliers" in combined or "local capacity" in combined)
        ),
        "has_divisible_preference": (
            "50% of the order quantity" in combined
            or "remaining 50% quantity" in combined
            or "match the l1 price" in combined
        ),
    }


def build_amount_specific_points(message: str, matches: list[SearchMatch]) -> list[str]:
    amount_lakhs = extract_amount_lakhs(message)
    if amount_lakhs is None:
        return []

    features = extract_match_features(matches)
    points: list[str] = []
    if amount_lakhs > 2.5 and features["has_lpc_limit"]:
        points.append(
            f"A purchase value of {format_lakh_amount(amount_lakhs)} is above the Local Purchase Committee limit of Rs. 2.5 lakhs, so the LPC route does not apply here."
        )
    if amount_lakhs <= 50 and features["has_under_50_rule"]:
        points.append(
            "Where the nodal ministry has confirmed enough local capacity and competition, procurements up to Rs. 50 lakhs may be restricted to local suppliers."
        )
    if features["has_divisible_preference"]:
        points.append(
            "If the tender is divisible and the lowest bid is not from a local supplier, an eligible local supplier may be allowed to match the L1 price for part of the order."
        )
    return points[:3]


def describe_user_request(message: str) -> str:
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


def build_empty_knowledge_base_response() -> str:
    return render_structured_response(
        answer="The knowledge base is currently empty or not loaded, so I cannot provide a grounded procurement answer yet.",
        explanation_points=[
            "No indexed documents are available for retrieval right now.",
            "The FAISS index needs loaded procurement documents before grounded answers can be generated.",
        ],
        sources=[],
    )


def build_no_match_response(message: str) -> str:
    normalized_message = clean_text(message)
    amount = extract_amount_lakhs(message)
    # Instead of "I don't know", offer the closest general guidance
    threshold_hint = ""
    if amount is not None:
        if amount <= 0.5:
            threshold_hint = " General GFR guidelines suggest **Direct Purchase** for values up to Rs. 50,000."
        elif amount <= 5.0:
            threshold_hint = " General GFR guidelines suggest **LPC (Local Purchase Committee)** for values between Rs. 50,001 and Rs. 5 lakh."
        elif amount <= 25.0:
            threshold_hint = " General GFR guidelines suggest **LTE (Limited Tender Enquiry)** for values between Rs. 5 lakh and Rs. 25 lakh."
        else:
            threshold_hint = " General GFR guidelines suggest **OTE (Open Tender Enquiry)** for values above Rs. 25 lakh."

    return render_structured_response(
        answer=(
            f"The specific document section for \"{normalized_message}\" was not found in the knowledge base."
            f"{threshold_hint}"
        ),
        explanation_points=[
            "The retriever did not find a strong enough section that directly answers this question.",
            "This may mean the exact rule wording differs from the query, or the specific provision is not covered in the indexed documents.",
            "⚠️ Disclaimer: The above is general GFR guidance. Please verify with the specific CSIR Manual chapter or latest office circular for your institute.",
        ],
        sources=["UpdatedGFR31July2025_0.pdf (General Reference)"],
        pro_tip="When the exact rule is not found, cross-check with your institute's S&T Purchase Committee or Finance & Accounts section for the latest applicable thresholds.",
    )


def select_best_sentence(query: str, content: str) -> str:
    sentences = split_sentences(content)
    if not sentences:
        return content[:280].strip()

    query_terms = tokenize(query)
    best_sentence = sentences[0]
    best_score = -1

    for sentence in sentences:
        sentence_terms = tokenize(sentence)
        overlap = len(query_terms.intersection(sentence_terms))
        if overlap > best_score:
            best_score = overlap
            best_sentence = sentence

    compact = re.sub(r"\s+", " ", best_sentence).strip()
    return compact[:280].rstrip()


def split_sentences(content: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", content)
    return [part.strip() for part in parts if part.strip()]


def tokenize(value: str) -> set[str]:
    normalized = re.sub(r"[^a-zA-Z0-9\s]", " ", value.lower())
    return {token for token in normalized.split() if len(token) >= 2}
