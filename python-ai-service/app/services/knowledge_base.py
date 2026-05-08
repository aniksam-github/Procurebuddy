"""FAISS vector index management, document extraction, and chunking.

This module owns the KnowledgeBase lifecycle: loading, reloading,
searching, and persisting the FAISS index to disk.
"""

from __future__ import annotations

import gc
import json
import logging
import math
import os
import re
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any

import faiss
import numpy as np
import pdfplumber
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings
from app.utils.text_cleaner import remove_pdf_artifacts

logger = logging.getLogger("procurebuddy-ai")


# ── Data classes ────────────────────────────────────────────────────────────

@dataclass
class ChunkRecord:
    chunk_id: int
    document_id: int
    file_name: str
    chunk_index: int
    content: str
    token_count: int


# ── Pydantic models for search results ──────────────────────────────────────

from pydantic import BaseModel, Field


class SearchMatch(BaseModel):
    chunk_id: int
    document_id: int
    file_name: str
    chunk_index: int
    content: str
    token_count: int
    score: float
    metadata: dict[str, str] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    matches: list[SearchMatch]
    count: int


# ── Knowledge Base ──────────────────────────────────────────────────────────

class KnowledgeBase:
    """FAISS-backed knowledge base with PDF/TXT/DOCX support."""

    def __init__(self) -> None:
        self._lock = RLock()
        # sentence-transformers pulls in transformers; transformers may attempt to import
        # TensorFlow/Keras integrations if Keras is present without TF. Force-disable TF/Flax
        # to keep imports reliable in minimal production envs.
        os.environ.setdefault("TRANSFORMERS_NO_TF_IMPORT", "1")
        os.environ.setdefault("USE_TF", "0")
        os.environ.setdefault("USE_FLAX", "0")

        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._embedder = SentenceTransformer(settings.embedding_model)
        except Exception as exc:
            logger.exception("Failed to initialize SentenceTransformer embedder; semantic search disabled: %s", exc)
            self._embedder = None
        self._index: faiss.IndexFlatIP | None = None
        self._chunks: list[ChunkRecord] = []
        self._chunk_term_freqs: list[Counter[str]] = []
        self._keyword_doc_freq: Counter[str] = Counter()
        self._avg_chunk_length: float = 0.0
        self._last_reload: str | None = None
        self._index_file = settings.index_dir / "faiss.index"
        self._metadata_file = settings.index_dir / "chunks.json"

    def initialize(self) -> dict[str, Any]:
        """Load persisted index or rebuild from documents."""
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
        """Rebuild the FAISS index from all documents in the data directory."""
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
            self._rebuild_keyword_index()
            self._last_reload = datetime.now(timezone.utc).isoformat()
            self._persist_to_disk()
            gc.collect()

            return {
                "success": len(failures) == 0,
                "search_backend": "hybrid-faiss-keyword",
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
        """Backward-compatible semantic search wrapper."""
        return self.search_semantic(query, top_k=top_k, min_score_override=min_score_override)

    def search_semantic(
        self,
        query: str,
        top_k: int | None = None,
        min_score_override: float | None = None,
    ) -> list[SearchMatch]:
        """Perform cosine-similarity search against the FAISS index."""
        normalized_query = query.strip()
        if not normalized_query:
            return []

        with self._lock:
            if self._embedder is None:
                return []
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
                        metadata=self._build_chunk_metadata(chunk.file_name, chunk.content),
                    )
                )
            logger.info("Semantic search query='%s' accepted_matches=%s", normalized_query, len(matches))
            return matches

    def search_keyword(
        self,
        query: str,
        top_k: int | None = None,
        min_score_override: float | None = None,
    ) -> list[SearchMatch]:
        """Perform lightweight BM25-style keyword search across chunk text."""
        normalized_query = query.strip()
        if not normalized_query:
            return []

        with self._lock:
            if not self._chunks or not self._chunk_term_freqs:
                logger.info("Keyword search skipped because the keyword index is not loaded")
                return []

            query_terms = self._tokenize(normalized_query)
            if not query_terms:
                return []

            scored: list[tuple[float, int]] = []
            total_docs = len(self._chunk_term_freqs)
            avg_length = self._avg_chunk_length or 1.0
            for index, term_freqs in enumerate(self._chunk_term_freqs):
                doc_length = sum(term_freqs.values()) or 1
                score = 0.0
                chunk_text = self._chunks[index].content.lower()
                metadata = self._build_chunk_metadata(self._chunks[index].file_name, self._chunks[index].content)
                metadata_blob = " ".join(
                    [
                        str(metadata.get("document_name", "")),
                        str(metadata.get("rule_number", "")),
                        str(metadata.get("topic", "")),
                    ]
                ).lower()
                for term in query_terms:
                    tf = term_freqs.get(term, 0)
                    if tf == 0:
                        continue
                    doc_freq = self._keyword_doc_freq.get(term, 0)
                    idf = math.log((total_docs + 1) / (doc_freq + 1)) + 1.0
                    score += idf * ((tf * 2.2) / (tf + 1.2 + 0.75 * (doc_length / avg_length)))
                    if term in chunk_text:
                        score += 0.05
                    if term in metadata_blob:
                        score += 0.20
                if normalized_query.lower() in chunk_text:
                    score += 0.35
                explicit_rules = re.findall(r"\brule\s+\d{3}\b", normalized_query, flags=re.IGNORECASE)
                if explicit_rules:
                    for explicit_rule in explicit_rules:
                        lowered_rule = explicit_rule.lower()
                        if lowered_rule in chunk_text:
                            score += 0.80
                        if lowered_rule in metadata_blob:
                            score += 1.00
                if score > 0:
                    scored.append((score, index))

            if not scored:
                return []

            scored.sort(key=lambda item: item[0], reverse=True)
            top_scored = scored[: min(top_k or settings.top_k, len(scored))]
            threshold = 0.0 if min_score_override is None else float(min_score_override)
            logger.info(
                "Keyword search query='%s' threshold=%.4f top_scores=%s",
                normalized_query,
                threshold,
                [round(score, 4) for score, _ in top_scored[:5]],
            )

            matches: list[SearchMatch] = []
            for score, index in top_scored:
                if score < threshold:
                    continue
                chunk = self._chunks[index]
                matches.append(
                    SearchMatch(
                        chunk_id=chunk.chunk_id,
                        document_id=chunk.document_id,
                        file_name=chunk.file_name,
                        chunk_index=chunk.chunk_index,
                        content=chunk.content,
                        token_count=chunk.token_count,
                        score=float(score),
                        metadata=self._build_chunk_metadata(chunk.file_name, chunk.content),
                    )
                )
            logger.info("Keyword search query='%s' accepted_matches=%s", normalized_query, len(matches))
            return matches

    def status(self) -> dict[str, Any]:
        """Return current knowledge base status."""
        with self._lock:
            return {
                "status": "ok",
                "search_backend": "hybrid-faiss-keyword",
                "data_dir": str(settings.data_dir),
                "index_dir": str(settings.index_dir),
                "embedding_model": settings.embedding_model,
                "document_count": len({chunk.document_id for chunk in self._chunks}),
                "chunk_count": len(self._chunks),
                "index_loaded": self._index is not None and bool(self._chunks),
                "keyword_index_loaded": bool(self._chunk_term_freqs),
                "llm_enabled": bool(settings.llm_api_key and settings.llm_model),
                "model_name": settings.llm_model,
                "refreshed_at": self._last_reload,
            }

    def version_token(self) -> str:
        with self._lock:
            return self._last_reload or "uninitialized"

    # ── Private methods ─────────────────────────────────────────────────────

    def _embed(self, texts: list[str]) -> np.ndarray:
        embeddings = self._embedder.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype="float32")

    def _build_chunk_metadata(self, file_name: str, content: str) -> dict[str, str]:
        document_name = self._canonical_document_name(file_name)
        rule_number = self._extract_rule_number(content)
        topic = self._infer_topic(content, document_name)
        return {
            "document_name": document_name,
            "rule_number": rule_number,
            "topic": topic,
        }

    def _canonical_document_name(self, file_name: str) -> str:
        lowered = file_name.lower()
        if "updatedgfr" in lowered or ("gfr" in lowered and "2025" in lowered):
            return "GFR 2025"
        if "csir manual" in lowered:
            return "CSIR Manual 2019"
        if "make in india" in lowered or "local content" in lowered or "purchase preference" in lowered:
            return "Make in India Policy"
        if "snt" in lowered or "special provisions" in lowered or "scientific procurement" in lowered:
            return "Scientific Procurement Provisions"
        if "amend" in lowered:
            return "Amendments"
        if any(token in lowered for token in ("write-off", "condemn", "inventory", "stores")):
            return "Compendium"
        return file_name

    def _extract_rule_number(self, content: str) -> str:
        match = re.search(r"\bRule\s+\d{3}\b", content, flags=re.IGNORECASE)
        return match.group(0).title() if match else ""

    def _infer_topic(self, content: str, document_name: str) -> str:
        lowered = f" {content.lower()} "
        topic_signals = (
            ("GeM Sourcing", (" rule 149 ", " gem ", "government e-marketplace", "government e marketplace")),
            ("Direct Purchase", (" rule 154 ", "direct purchase")),
            ("LPC Procurement", (" rule 155 ", "local purchase committee", " lpc ")),
            ("OTE Procurement", (" rule 161 ", "open tender", " ote ")),
            ("LTE Procurement", (" rule 162 ", "limited tender", " lte ")),
            ("STE Procurement", (" rule 166 ", "single tender", " pac ", " proprietary ")),
            ("Make in India", ("make in india", "local content", "purchase preference")),
            ("Write-off / Disposal", ("write-off", "condemn", "obsolete", "surplus", "scrap")),
        )
        for topic, aliases in topic_signals:
            if any(alias in lowered for alias in aliases):
                return topic
        return document_name

    def _extract_text(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self._extract_pdf_text(path)
        if suffix == ".txt":
            raw = path.read_text(encoding="utf-8", errors="ignore")
            return self._normalize_text(remove_pdf_artifacts(raw))
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
            return remove_pdf_artifacts(text)

        plumber_parts: list[str] = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                plumber_parts.append(page.extract_text() or "")
        return remove_pdf_artifacts(self._normalize_text("\n".join(plumber_parts)))

    def _extract_docx_text(self, path: Path) -> str:
        with zipfile.ZipFile(path) as archive:
            xml = archive.read("word/document.xml").decode("utf-8", errors="ignore")
        text = re.sub(r"</w:p>", "\n\n", xml)
        text = re.sub(r"<[^>]+>", " ", text)
        return self._normalize_text(text)

    def _chunk_text(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=max(200, settings.chunk_size),
            chunk_overlap=max(0, settings.chunk_overlap),
            separators=[
                "\n\nRule ",
                "\n\nChapter ",
                "\n\nSection ",
                "\n\n",
                "\n",
                " ",
            ],
        )
        return [chunk for chunk in splitter.split_text(text) if chunk.strip()]

    def _normalize_text(self, value: str) -> str:
        normalized = value.replace("\x00", " ")
        normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
        normalized = re.sub(r"[ \t]+", " ", normalized)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        return normalized.strip()

    def _tokenize(self, value: str) -> list[str]:
        normalized = re.sub(r"[^a-zA-Z0-9\s]", " ", value.lower())
        return [token for token in normalized.split() if len(token) >= 2]

    def _rebuild_keyword_index(self) -> None:
        term_freqs: list[Counter[str]] = []
        doc_freq: Counter[str] = Counter()
        total_length = 0

        for chunk in self._chunks:
            tokens = self._tokenize(chunk.content)
            frequencies = Counter(tokens)
            term_freqs.append(frequencies)
            doc_freq.update(frequencies.keys())
            total_length += sum(frequencies.values())

        self._chunk_term_freqs = term_freqs
        self._keyword_doc_freq = doc_freq
        self._avg_chunk_length = (total_length / len(term_freqs)) if term_freqs else 0.0

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
        self._rebuild_keyword_index()


# ── Singleton ───────────────────────────────────────────────────────────────
knowledge_base = KnowledgeBase()
