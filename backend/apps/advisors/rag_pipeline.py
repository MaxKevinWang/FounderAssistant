"""Local RAG pipeline implementation for the thesis demo.

Pipeline stages:
1) Document ingestion from `knowledge_base/` (PDF + HTML + plain text).
2) Text normalization and sliding-window chunking.
3) Embedding generation through local Ollama embedding model.
4) Index persistence (`metadata.json` + `embeddings.npy`).
5) Query-time retrieval (dense; with keyword fallback on embedding failure).
6) Context-grounded generation with source markers.

This module exposes:
- `build_rag_index`: build/refresh index from local corpus.
- `get_rag_status`: inspect index + model readiness.
- `rag_answer`: run retrieval + generation and return citations.
- `rag_answer_stream`: run retrieval once, then stream grounded generation.
"""

from __future__ import annotations

import json
import math
import re
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field as dataclass_field
from datetime import UTC, datetime
from functools import lru_cache
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from django.conf import settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from .local_llm import LocalLLMError, check_status, generate_text, stream_generate_text
from .model_profiles import DEFAULT_ANSWER_MAX_TOKENS
from .legal_corpus import (
    LEGAL_EXTRACTOR_VERSION,
    LegalCorpusError,
    extract_legal_pdf,
    get_legal_source_profile,
    legal_scope_sha256,
)

INDEX_METADATA_FILENAME = "metadata.json"
INDEX_EMBEDDINGS_FILENAME = "embeddings.npy"
SOURCE_MANIFEST_FILENAME = "source_manifest.json"
EXTRACTED_TEXT_DIRNAME = "extracted_texts"
ARCHIVE_DIRNAME = "archive"
FIGURE_DESCRIPTION_DIRNAME = "figure_descriptions"
FIGURE_DESCRIPTION_JSON_FILENAME = "figure_descriptions.json"
DEFAULT_RERANKER_MODEL = str(
    getattr(settings, "LOCAL_RAG_RERANKER_MODEL", "")
).strip()
DEFAULT_RERANKER_DEVICE = str(
    getattr(settings, "LOCAL_RAG_RERANKER_DEVICE", "auto")
).strip()
DEFAULT_RERANKER_REVISION = str(
    getattr(settings, "LOCAL_RAG_RERANKER_REVISION", "")
).strip()
DEFAULT_RERANKER_BATCH_SIZE = int(
    getattr(settings, "LOCAL_RAG_RERANKER_BATCH_SIZE", 4)
)
DEFAULT_RERANKER_MAX_LENGTH = int(
    getattr(settings, "LOCAL_RAG_RERANKER_MAX_LENGTH", 512)
)
DEFAULT_RERANK_CANDIDATE_K = 20
DEFAULT_MAX_CONTEXT_TOKENS = 3000
DEFAULT_CONTEXT_METADATA_TOKEN_ALLOWANCE = 64
TOKEN_ESTIMATOR_ID = "normalized_characters_div_3_v1"
DEFAULT_HISTORY_TURN_LIMIT = 3
DEFAULT_HISTORY_TOKEN_BUDGET = 1500
DEFAULT_HISTORY_MESSAGE_TOKEN_LIMIT = 600
HISTORY_TOKEN_ESTIMATOR_ID = "unicode_weighted_characters_v1"
LEGACY_RAG_PROMPT_VERSION = "legacy_grounded_v1"
RAG_PROMPT_VERSION = "evidence_checklist_v2"
SUPPORTED_RAG_PROMPT_VERSIONS = {
    LEGACY_RAG_PROMPT_VERSION,
    RAG_PROMPT_VERSION,
}
RAG_CONFIDENCE_POLICY_VERSION = "reranker_nonnegative_count_v1"
LANGCHAIN_TEXT_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ": ", ") ", " ", ""]
CHUNK_METADATA_FIELDS = [
    "chunk_type",
    "topic",
    "legal_area",
    "business_type",
    "authority",
    "source_quality",
    "language",
    "jurisdiction",
    "document_type",
    "section_title",
    "heading_path",
    "page_start",
    "page_end",
    "url",
    "publication_date",
    "law_name",
    "law_abbreviation",
    "law_section",
    "effective_from",
    "effective_to",
    "retrieved_at",
    "imported_at",
    "canonical_url",
    "temporal_status",
    "supersedes",
    "metadata_reviewed",
]

PDF_PUBLICATION_SECTION_HEADINGS = (
    "impressum",
    "herausgeber",
    "bildnachweis",
    "bestellmöglichkeit",
    "broschüren und infoletter",
)
PDF_PUBLICATION_SECTION_PREFIXES = (
    "zentraler bestellservice für publikationen",
    "publikationsversand der bundesregierung",
    "bestellung über das gebärdentelefon",
    "online-bestellung:",
)
PDF_PUBLICATION_END_PHRASES = (
    "europäischen parlament",
    "politischer parteien oder gruppen eingesetzt werden",
    "zwecke der wahlwerbung verwendet werden",
)
PDF_PUBLISHER_LINK_DOMAINS = (
    "www.bmwi.de",
    "www.existenzgruender.de",
    "www.existenzgruenderinnen.de",
    "www.kultur-kreativ-wirtschaft.de",
    "www.unternehmergeist-macht-schule.de",
    "gruenderplattform.de",
    "www.bundesregierung.de",
)
PDF_PUBLICATION_CHUNK_MARKERS = (
    "publikationen@bundesregierung.de",
    "gebaerdentelefon@sip.bundesregierung.de",
    "gebärdentelefon",
    "bestellfax",
    "servicetelefon: 030 18 272 2721",
    "zentraler bestellservice für publikationen",
    "publikationsversand der bundesregierung",
    "diese publikation wird von der bundesregierung",
    "diese publikation wird vom bundesministerium",
    "herausgeber bundesministerium",
    "bildnachweis",
)


@dataclass(frozen=True)
class RAGConfig:
    """All tunable settings for index build and retrieval/runtime."""

    data_dir: Path
    index_dir: Path
    embedding_model: str
    llm_provider: str
    llm_base_url: str
    llm_api_key: str
    verify_ssl: bool
    timeout_seconds: int
    chunk_size: int
    chunk_overlap: int
    min_chunk_chars: int


DEFAULT_RAG_CONFIG = RAGConfig(
    data_dir=Path(getattr(settings, "LOCAL_RAG_DATA_DIR", Path(settings.BASE_DIR).parent / "knowledge_base")),
    index_dir=Path(getattr(settings, "LOCAL_RAG_INDEX_DIR", Path(settings.BASE_DIR) / "rag_index")),
    embedding_model=getattr(settings, "LOCAL_RAG_EMBED_MODEL", "bge-m3"),
    llm_provider=getattr(settings, "LOCAL_LLM_PROVIDER", "ollama"),
    llm_base_url=str(getattr(settings, "LOCAL_LLM_BASE_URL", "http://127.0.0.1:11434")).rstrip("/"),
    llm_api_key=getattr(settings, "LOCAL_LLM_API_KEY", ""),
    verify_ssl=bool(getattr(settings, "LOCAL_LLM_VERIFY_SSL", True)),
    timeout_seconds=int(getattr(settings, "LOCAL_LLM_TIMEOUT_SECONDS", 360)),
    chunk_size=int(getattr(settings, "LOCAL_RAG_CHUNK_SIZE", 1200)),
    chunk_overlap=int(getattr(settings, "LOCAL_RAG_CHUNK_OVERLAP", 180)),
    min_chunk_chars=int(getattr(settings, "LOCAL_RAG_MIN_CHUNK_CHARS", 180)),
)


class RAGError(Exception):
    """Raised for corpus/index/retrieval failures in the RAG pipeline."""

    pass


@dataclass(frozen=True)
class RetrievalSelection:
    """Auditable result of candidate retrieval, reranking, and selection."""

    hits: list[dict[str, Any]]
    candidates: list[dict[str, Any]]
    retrieval_mode: str
    candidate_k: int
    context_k: int
    candidate_retrieval_latency_ms: int
    reranker_latency_ms: int
    selection_latency_ms: int
    estimated_context_tokens: int
    token_estimator: str
    reranker_model: str
    reranker_revision: str
    expansion_latency_ms: int = 0
    expanded_neighbor_count: int = 0

    @property
    def total_retrieval_latency_ms(self) -> int:
        return (
            self.candidate_retrieval_latency_ms
            + self.reranker_latency_ms
            + self.selection_latency_ms
            + self.expansion_latency_ms
        )


class _VisibleTextHTMLParser(HTMLParser):
    """Extract visible text from HTML while skipping noisy/non-content tags."""

    _SKIP_TAGS = {"script", "style", "noscript", "template", "svg"}
    _BLOCK_TAGS = {
        "article",
        "aside",
        "blockquote",
        "br",
        "dd",
        "div",
        "dl",
        "dt",
        "figcaption",
        "footer",
        "form",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hr",
        "li",
        "main",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "td",
        "th",
        "tr",
        "ul",
    }

    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ANN001
        tag_name = tag.lower()
        if tag_name in self._SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag_name in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag_name = tag.lower()
        if tag_name in self._SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth:
            return
        if tag_name in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = unescape(data or "")
        if text.strip():
            self._parts.append(text)

    def extracted_text(self) -> str:
        return "".join(self._parts)


@dataclass(frozen=True)
class HtmlBlock:
    """One structure-aware text block extracted from an HTML document."""

    text: str
    heading_path: str = ""
    block_type: str = "section_text"


class _StructuredHTMLParser(HTMLParser):
    """Extract semantic HTML blocks while preserving heading/list/table signals."""

    _SKIP_TAGS = {"script", "style", "noscript", "template", "svg"}
    _NOISY_TAGS = {"nav", "header", "footer", "form", "button"}
    _NOISY_ATTR_PATTERNS = (
        "breadcrumb",
        "cookie",
        "footer",
        "inhaltsverzeichnis",
        "menu",
        "navigation",
        "navbar",
        "pagination",
        "share",
        "social",
        "toc",
    )
    _TEXT_BLOCK_TAGS = {"p", "blockquote", "figcaption", "pre"}
    _HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}

    def __init__(self) -> None:
        super().__init__()
        self.blocks: list[HtmlBlock] = []
        self._skip_depth = 0
        self._skip_stack: list[str] = []
        self._heading_stack: list[tuple[int, str]] = []
        self._active_tag = ""
        self._active_heading_level: int | None = None
        self._buffer: list[str] = []
        self._in_table = False
        self._table_rows: list[list[str]] = []
        self._current_row: list[str] | None = None
        self._current_cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ANN001
        tag_name = tag.lower()
        if tag_name in self._SKIP_TAGS or self._is_noisy_element(tag_name, attrs):
            self._skip_depth += 1
            self._skip_stack.append(tag_name)
            return
        if self._skip_depth:
            return
        if tag_name in self._HEADING_TAGS:
            self._start_buffer(tag_name)
            self._active_heading_level = int(tag_name[1])
        elif tag_name in self._TEXT_BLOCK_TAGS:
            self._start_buffer(tag_name)
        elif tag_name == "li":
            self._start_buffer("li")
        elif tag_name == "table":
            self._flush_buffer()
            self._in_table = True
            self._table_rows = []
        elif self._in_table and tag_name == "tr":
            self._current_row = []
        elif self._in_table and tag_name in {"th", "td"}:
            self._current_cell = []
        elif tag_name == "br":
            self._append_text("\n")

    def handle_endtag(self, tag: str) -> None:
        tag_name = tag.lower()
        if self._skip_depth:
            if self._skip_stack and tag_name == self._skip_stack[-1]:
                self._skip_stack.pop()
                self._skip_depth = max(0, self._skip_depth - 1)
            return
        if tag_name in self._SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._in_table and tag_name in {"th", "td"} and self._current_cell is not None:
            cell_text = _normalize_inline_text(" ".join(self._current_cell))
            if self._current_row is not None and cell_text:
                self._current_row.append(cell_text)
            self._current_cell = None
        elif self._in_table and tag_name == "tr":
            if self._current_row:
                self._table_rows.append(self._current_row)
            self._current_row = None
        elif tag_name == "table" and self._in_table:
            self._emit_table()
        elif tag_name in self._HEADING_TAGS and self._active_tag == tag_name:
            heading_text = _normalize_inline_text(" ".join(self._buffer))
            if heading_text and self._active_heading_level is not None:
                self._set_heading(self._active_heading_level, heading_text)
                self.blocks.append(
                    HtmlBlock(text=heading_text, heading_path=self._heading_path(), block_type="heading")
                )
            self._reset_buffer()
        elif tag_name == self._active_tag:
            self._flush_buffer()

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = unescape(data or "")
        if not text.strip():
            return
        if self._current_cell is not None:
            self._current_cell.append(text)
        elif self._active_tag:
            self._buffer.append(text)
        elif len(text.strip()) >= 30:
            self.blocks.append(
                HtmlBlock(
                    text=_normalize_inline_text(text),
                    heading_path=self._heading_path(),
                    block_type="section_text",
                )
            )

    def _is_noisy_element(self, tag_name: str, attrs) -> bool:  # noqa: ANN001
        if tag_name in self._NOISY_TAGS:
            return True
        attr_text = " ".join(str(value).casefold() for _, value in attrs if value)
        return any(pattern in attr_text for pattern in self._NOISY_ATTR_PATTERNS)

    def _start_buffer(self, tag_name: str) -> None:
        self._flush_buffer()
        self._active_tag = tag_name
        self._buffer = []

    def _append_text(self, text: str) -> None:
        if self._current_cell is not None:
            self._current_cell.append(text)
        elif self._active_tag:
            self._buffer.append(text)

    def _flush_buffer(self) -> None:
        if not self._active_tag:
            return
        text = _normalize_inline_text(" ".join(self._buffer))
        if text:
            block_type = "list_item" if self._active_tag == "li" else "section_text"
            if block_type == "list_item":
                text = f"- {text}"
            self.blocks.append(HtmlBlock(text=text, heading_path=self._heading_path(), block_type=block_type))
        self._reset_buffer()

    def _reset_buffer(self) -> None:
        self._active_tag = ""
        self._active_heading_level = None
        self._buffer = []

    def _set_heading(self, level: int, text: str) -> None:
        self._heading_stack = [(heading_level, title) for heading_level, title in self._heading_stack if heading_level < level]
        self._heading_stack.append((level, text))

    def _heading_path(self) -> str:
        return " > ".join(title for _, title in self._heading_stack)

    def _emit_table(self) -> None:
        rows = [" | ".join(row) for row in self._table_rows if row]
        text = _normalize_text("\n".join(rows))
        if text:
            self.blocks.append(HtmlBlock(text=text, heading_path=self._heading_path(), block_type="table_like_text"))
        self._in_table = False
        self._table_rows = []
        self._current_row = None
        self._current_cell = None


@dataclass(frozen=True)
class ChunkRecord:
    """Normalized metadata row for one indexed text chunk."""

    chunk_id: str
    domain: str
    source_path: str
    source_name: str
    chunk_index: int
    text: str
    char_count: int
    topic: str = ""
    legal_area: str = ""
    business_type: str = ""
    authority: str = ""
    source_quality: str = ""
    language: str = ""
    jurisdiction: str = ""
    document_type: str = ""
    section_title: str = ""
    heading_path: str = ""
    chunk_type: str = "section_text"
    page_start: str = ""
    page_end: str = ""
    url: str = ""
    publication_date: str = ""
    law_name: str = ""
    law_abbreviation: str = ""
    law_section: str = ""
    effective_from: str = ""
    effective_to: str = ""
    retrieved_at: str = ""
    imported_at: str = ""
    canonical_url: str = ""
    temporal_status: str = ""
    supersedes: str = ""
    metadata_reviewed: bool = False


@dataclass(frozen=True)
class ExtractedPage:
    """One extracted document page or page-like block with source location."""

    text: str
    page_number: int | None = None


@dataclass(frozen=True)
class ExtractedDocument:
    """Normalized document text plus page spans for chunk metadata."""

    text: str
    pages: list[ExtractedPage]
    page_spans: list[tuple[int, int, int]]
    structured_chunks: list[ChunkCandidate] | None = None
    metadata: dict[str, Any] = dataclass_field(default_factory=dict)


@dataclass(frozen=True)
class ChunkCandidate:
    """Boundary-aware chunk before it is enriched into a persisted record."""

    text: str
    page_start: str = ""
    page_end: str = ""
    section_title: str = ""
    heading_path: str = ""
    chunk_type: str = "section_text"
    metadata: dict[str, Any] = dataclass_field(default_factory=dict)


def _resolve_config(overrides: dict[str, Any] | None = None) -> RAGConfig:
    """Merge runtime overrides with defaults and validate chunk parameters."""

    if not overrides:
        return DEFAULT_RAG_CONFIG
    payload = {
        "data_dir": Path(overrides.get("data_dir", DEFAULT_RAG_CONFIG.data_dir)),
        "index_dir": Path(overrides.get("index_dir", DEFAULT_RAG_CONFIG.index_dir)),
        "embedding_model": str(overrides.get("embedding_model", DEFAULT_RAG_CONFIG.embedding_model)),
        "llm_provider": str(overrides.get("llm_provider", DEFAULT_RAG_CONFIG.llm_provider)),
        "llm_base_url": str(overrides.get("llm_base_url", DEFAULT_RAG_CONFIG.llm_base_url)),
        "llm_api_key": str(overrides.get("llm_api_key", DEFAULT_RAG_CONFIG.llm_api_key)),
        "verify_ssl": bool(overrides.get("verify_ssl", DEFAULT_RAG_CONFIG.verify_ssl)),
        "timeout_seconds": int(overrides.get("timeout_seconds", DEFAULT_RAG_CONFIG.timeout_seconds)),
        "chunk_size": int(overrides.get("chunk_size", DEFAULT_RAG_CONFIG.chunk_size)),
        "chunk_overlap": int(overrides.get("chunk_overlap", DEFAULT_RAG_CONFIG.chunk_overlap)),
        "min_chunk_chars": int(overrides.get("min_chunk_chars", DEFAULT_RAG_CONFIG.min_chunk_chars)),
    }
    if payload["chunk_size"] <= 0:
        raise RAGError("chunk_size must be greater than 0.")
    if payload["chunk_overlap"] < 0:
        raise RAGError("chunk_overlap must be 0 or greater.")
    if payload["chunk_overlap"] >= payload["chunk_size"]:
        raise RAGError("chunk_overlap must be smaller than chunk_size.")
    return RAGConfig(**payload)


def _normalize_openwebui_base_url(base_url: str) -> str:
    """Return the Open WebUI root URL even if `/ollama` was configured."""

    normalized = base_url.rstrip("/")
    if normalized.endswith("/ollama"):
        return normalized[: -len("/ollama")]
    return normalized


def _embedding_endpoint(config: RAGConfig) -> str:
    """Return the embedding endpoint for the configured LLM provider."""

    provider = config.llm_provider.strip().lower()
    if provider == "ollama":
        return f"{config.llm_base_url.rstrip('/')}/api/embed"
    if provider == "openwebui":
        base_url = _normalize_openwebui_base_url(config.llm_base_url)
        return f"{base_url}/ollama/api/embed"
    raise RAGError(f"Unsupported LOCAL_LLM_PROVIDER for embeddings: {config.llm_provider}")


def _request_headers(config: RAGConfig) -> dict[str, str]:
    """Build embedding request headers, including Open WebUI API auth if set."""

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    if config.llm_api_key:
        headers["Authorization"] = f"Bearer {config.llm_api_key}"
    return headers


def _ssl_context(config: RAGConfig) -> ssl.SSLContext | None:
    """Return a custom SSL context only when certificate verification is disabled."""

    if config.verify_ssl:
        return None
    return ssl._create_unverified_context()


def _post_json(url: str, payload: dict[str, Any], config: RAGConfig) -> dict[str, Any]:
    """HTTP JSON helper with consistent embedding-service error mapping."""

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=_request_headers(config),
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=config.timeout_seconds, context=_ssl_context(config)) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="ignore")
        except Exception:
            body = ""
        detail = f"{exc} {body}".strip()
        raise RAGError(f"Failed to call local embedding service: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RAGError(f"Failed to call local embedding service: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RAGError(f"Embedding service returned invalid JSON: {exc}") from exc


def _normalize_text(text: str) -> str:
    """Normalize whitespace and control characters before chunking."""

    normalized = text.replace("\x00", " ").replace("\r", "\n")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def _normalize_inline_text(text: str) -> str:
    """Collapse all inline whitespace for HTML paragraph/list/table cells."""

    return re.sub(r"\s+", " ", text.replace("\x00", " ")).strip()


def _normalized_line_key(line: str) -> str:
    """Normalize a line for conservative boilerplate heading comparisons."""

    return _normalize_inline_text(line).casefold().strip(" :")


def _is_pdf_publication_section_start(line: str) -> bool:
    """Detect PDF publication/ordering sections that should not enter RAG."""

    key = _normalized_line_key(line)
    if re.fullmatch(r"(?:\d+\s+)?impressum", key):
        return True
    if key in PDF_PUBLICATION_SECTION_HEADINGS:
        return True
    return any(key.startswith(prefix) for prefix in PDF_PUBLICATION_SECTION_PREFIXES)


def _starts_generic_publisher_link_block(lines: list[str], index: int) -> bool:
    """Detect generic publisher link lists without removing topical source links."""

    if _normalized_line_key(lines[index]) != "internet":
        return False
    lookahead = "\n".join(lines[index : index + 10]).casefold()
    domain_hits = sum(1 for domain in PDF_PUBLISHER_LINK_DOMAINS if domain in lookahead)
    return domain_hits >= 2


def _is_publisher_link_line(line: str) -> bool:
    """Return true for short lines that are part of generic publisher URL blocks."""

    key = _normalized_line_key(line)
    if not key:
        return True
    return key == "internet" or key.startswith("www.") or key in PDF_PUBLISHER_LINK_DOMAINS


def _strip_pdf_publication_boilerplate(text: str) -> str:
    """Remove PDF imprint/order/link boilerplate while keeping nearby article text."""

    lines = text.splitlines()
    kept: list[str] = []
    skip_publication_block = False
    skip_link_block = False

    for index, line in enumerate(lines):
        haystack = _normalize_inline_text(line).casefold()

        if skip_link_block:
            if _is_publisher_link_line(line):
                continue
            skip_link_block = False

        if skip_publication_block:
            if any(phrase in haystack for phrase in PDF_PUBLICATION_END_PHRASES):
                skip_publication_block = False
            continue

        if _starts_generic_publisher_link_block(lines, index):
            skip_link_block = True
            continue

        if _is_pdf_publication_section_start(line):
            # "Internet" link lists are handled separately above; the remaining
            # sections are imprint/order/photo-credit blocks from PDF back pages.
            skip_publication_block = True
            continue

        kept.append(line)

    return _normalize_text("\n".join(kept))


def _repair_pdf_hyphenation(text: str) -> str:
    """Merge common PDF line-break hyphenation before generic normalization."""

    # German PDFs often split words visually at line endings, e.g.
    # "Unternehmens-\ngründung". Keeping that split hurts both embedding and
    # generation, so we only merge when letters are present on both sides.
    repaired = re.sub(r"([A-Za-zÄÖÜäöüß])- *\n+ *([A-Za-zÄÖÜäöüß])", r"\1\2", text)
    return repaired


def _read_text_file(file_path: Path) -> str:
    """Best-effort text file reader with fallback encodings."""

    for encoding in ("utf-8", "cp1252", "latin-1"):
        try:
            return file_path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return file_path.read_text(encoding="utf-8", errors="ignore")


def _build_page_spans(pages: list[ExtractedPage]) -> tuple[str, list[tuple[int, int, int]]]:
    """Concatenate pages and record character spans for page lookup."""

    parts: list[str] = []
    spans: list[tuple[int, int, int]] = []
    cursor = 0
    for page in pages:
        if not page.text:
            continue
        if parts:
            parts.append("\n\n")
            cursor += 2
        start = cursor
        parts.append(page.text)
        cursor += len(page.text)
        end = cursor
        if page.page_number is not None:
            spans.append((start, end, page.page_number))
    return "".join(parts).strip(), spans


def _extract_pdf_document(
    file_path: Path,
    reviewed_figures_by_page: dict[int, list[dict[str, Any]]] | None = None,
) -> ExtractedDocument:
    """Extract page-aware PDF text and append reviewed visual descriptions by page."""

    try:
        reader = PdfReader(str(file_path))
    except Exception as exc:  # pragma: no cover - defensive parsing for broken PDFs
        raise RAGError(f"Failed to read PDF {file_path}: {exc}") from exc

    pages: list[ExtractedPage] = []
    for page_number, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        normalized = _strip_pdf_publication_boilerplate(_normalize_text(_repair_pdf_hyphenation(raw)))
        reviewed_figure_text = _format_reviewed_figures_for_page_text(
            reviewed_figures_by_page.get(page_number, []) if reviewed_figures_by_page else []
        )
        if reviewed_figure_text:
            normalized = _normalize_text("\n\n".join(part for part in (normalized, reviewed_figure_text) if part))
        if normalized:
            pages.append(ExtractedPage(text=normalized, page_number=page_number))
    text, spans = _build_page_spans(pages)
    return ExtractedDocument(text=text, pages=pages, page_spans=spans)


def _extract_html_text(file_path: Path) -> str:
    """Extract visible text from HTML source and return plain text payload."""

    html_content = _read_text_file(file_path)
    parser = _VisibleTextHTMLParser()
    try:
        parser.feed(html_content)
        parser.close()
    except Exception as exc:  # pragma: no cover - defensive parsing
        raise RAGError(f"Failed to parse HTML {file_path}: {exc}") from exc
    return parser.extracted_text()


def _extract_html_document(file_path: Path, chunk_size: int, min_chunk_chars: int) -> ExtractedDocument:
    """Extract HTML as structure-aware chunks plus a plain-text fallback."""

    html_content = _read_text_file(file_path)
    parser = _StructuredHTMLParser()
    try:
        parser.feed(html_content)
        parser.close()
    except Exception as exc:  # pragma: no cover - defensive parsing
        raise RAGError(f"Failed to parse structured HTML {file_path}: {exc}") from exc

    blocks = [block for block in parser.blocks if block.text.strip()]
    if not blocks:
        raw = _extract_html_text(file_path)
        text = _normalize_text(raw)
        page = ExtractedPage(text=text, page_number=None) if text else None
        pages = [page] if page else []
        return ExtractedDocument(text=text, pages=pages, page_spans=[])

    content_blocks = [block for block in blocks if block.block_type != "heading" and not _is_noisy_html_block(block)]
    text = "\n\n".join(block.text for block in content_blocks).strip()
    structured_chunks = _chunk_html_blocks(
        blocks=blocks,
        chunk_size=chunk_size,
        min_chunk_chars=min_chunk_chars,
    )
    page = ExtractedPage(text=text, page_number=None) if text else None
    pages = [page] if page else []
    return ExtractedDocument(text=text, pages=pages, page_spans=[], structured_chunks=structured_chunks)


def _extract_document(
    file_path: Path,
    *,
    chunk_size: int,
    min_chunk_chars: int,
    reviewed_figures_by_page: dict[int, list[dict[str, Any]]] | None = None,
) -> ExtractedDocument:
    """Route extraction by suffix and return normalized document payload."""

    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf_document(file_path, reviewed_figures_by_page=reviewed_figures_by_page)
    elif suffix in {".html", ".htm"}:
        return _extract_html_document(
            file_path=file_path,
            chunk_size=chunk_size,
            min_chunk_chars=min_chunk_chars,
        )
    else:
        raw = _read_text_file(file_path)
    text = _normalize_text(raw)
    page = ExtractedPage(text=text, page_number=None) if text else None
    pages = [page] if page else []
    return ExtractedDocument(text=text, pages=pages, page_spans=[])


def _extract_corpus_document(
    file_path: Path,
    relative_path: Path,
    *,
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_chars: int,
    reviewed_figures_by_page: dict[int, list[dict[str, Any]]] | None = None,
) -> ExtractedDocument:
    """Use the legal section parser for scoped laws and generic extraction otherwise."""

    legal_profile = get_legal_source_profile(relative_path)
    if legal_profile is None:
        return _extract_document(
            file_path,
            chunk_size=chunk_size,
            min_chunk_chars=min_chunk_chars,
            reviewed_figures_by_page=reviewed_figures_by_page,
        )

    try:
        legal_result = extract_legal_pdf(
            file_path,
            legal_profile,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    except LegalCorpusError as exc:
        raise RAGError(str(exc)) from exc
    if not legal_result.chunks:
        raise RAGError(f"No selected legal sections extracted from {relative_path.as_posix()}")

    return ExtractedDocument(
        text=legal_result.text,
        pages=[],
        page_spans=[],
        structured_chunks=[
            ChunkCandidate(
                text=chunk.text,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                section_title=chunk.section_title,
                heading_path=chunk.heading_path,
                chunk_type=chunk.chunk_type,
                metadata=chunk.metadata,
            )
            for chunk in legal_result.chunks
        ],
        metadata=legal_result.document_metadata,
    )


def _split_text_with_langchain(text: str, chunk_size: int, chunk_overlap: int) -> list[tuple[str, int, int]]:
    """Split text with LangChain while preserving source character offsets."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=LANGCHAIN_TEXT_SEPARATORS,
        add_start_index=True,
    )
    chunks_with_offsets: list[tuple[str, int, int]] = []
    search_cursor = 0

    for document in splitter.create_documents([text]):
        raw_chunk = document.page_content
        chunk = raw_chunk.strip()
        if not chunk:
            continue

        start_index = document.metadata.get("start_index")
        if isinstance(start_index, int):
            leading_ws = len(raw_chunk) - len(raw_chunk.lstrip())
            start = start_index + leading_ws
        else:
            start = text.find(chunk, search_cursor)
            if start < 0:
                start = text.find(chunk)
            if start < 0:
                start = search_cursor

        end = min(len(text), start + len(chunk))
        chunks_with_offsets.append((chunk, start, end))
        search_cursor = max(search_cursor, end)

    return chunks_with_offsets


def _page_range_for_span(page_spans: list[tuple[int, int, int]], start: int, end: int) -> tuple[str, str]:
    """Return page range touched by a character span."""

    touched = [page for page_start, page_end, page in page_spans if page_start < end and page_end > start]
    if not touched:
        return "", ""
    return str(min(touched)), str(max(touched))


def _chunk_type_from_text(chunk_text: str) -> str:
    """Classify simple chunk shapes for review and retrieval diagnostics."""

    stripped = chunk_text.strip()
    bullet_lines = len(re.findall(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+", stripped))
    lines = [line for line in stripped.splitlines() if line.strip()]
    if bullet_lines >= 3:
        return "list"
    if lines and sum(1 for line in lines if "|" in line or "\t" in line) >= 3:
        return "table_like_text"
    return "section_text"


def _is_pdf_publication_boilerplate_chunk(chunk_text: str) -> bool:
    """Catch residual PDF publication boilerplate before embedding."""

    haystack = _normalize_inline_text(chunk_text).casefold()
    if any(marker in haystack for marker in PDF_PUBLICATION_CHUNK_MARKERS):
        return True

    domain_hits = sum(1 for domain in PDF_PUBLISHER_LINK_DOMAINS if domain in haystack)
    return domain_hits >= 3 and (" internet " in f" {haystack} " or len(chunk_text) < 900)


def _section_title_for_chunk(chunk_text: str, previous_section: str = "") -> str:
    """Infer section title for a chunk, falling back to previous title."""

    inferred = _infer_section_title(chunk_text)
    return inferred or previous_section


def _is_noisy_html_block(block: HtmlBlock) -> bool:
    """Filter recurring consent/media/legal boilerplate from cleaned HTML pages."""

    haystack = f"{block.heading_path}\n{block.text}".casefold()
    noisy_phrases = (
        "einwilligung zum videodienst",
        "datenverarbeitung durch den youtube-player",
        "inhaltsverzeichnis",
        "jw-player",
        "zum inhalt",
        "zum inhaltsverzeichnis",
        "youtube-player",
        "cookies auf ihrem endgerät",
        "video-anbieter",
        "technische daten an den",
    )
    if any(phrase in haystack for phrase in noisy_phrases):
        return True
    if block.text.startswith("©") and len(block.text) < 160:
        return True
    return False


def _chunk_html_blocks(blocks: list[HtmlBlock], chunk_size: int, min_chunk_chars: int) -> list[ChunkCandidate]:
    """Group structure-aware HTML blocks into chunks without flattening headings."""

    chunks: list[ChunkCandidate] = []
    current_blocks: list[str] = []
    current_heading = ""
    current_types: list[str] = []
    min_html_chunk_chars = max(80, min_chunk_chars // 2)

    def flush() -> None:
        nonlocal current_blocks, current_heading, current_types
        if not current_blocks:
            return
        body = "\n".join(current_blocks).strip()
        if not body:
            current_blocks = []
            current_types = []
            return
        text = f"{current_heading}\n\n{body}".strip() if current_heading and current_heading not in body[:160] else body
        if len(text) >= min_html_chunk_chars:
            section_title = current_heading.split(" > ")[-1] if current_heading else _infer_section_title(text)
            chunk_type = "list" if current_types.count("list_item") >= 2 else _chunk_type_from_text(text)
            chunks.append(
                ChunkCandidate(
                    text=text,
                    section_title=section_title,
                    heading_path=current_heading or section_title,
                    chunk_type=chunk_type,
                )
            )
        current_blocks = []
        current_types = []

    for block in blocks:
        if _is_noisy_html_block(block):
            continue
        if block.block_type == "heading":
            flush()
            current_heading = block.heading_path or block.text
            continue
        if not block.text.strip():
            continue

        block_heading = block.heading_path or current_heading
        if block.block_type == "table_like_text":
            flush()
            table_text = f"{block_heading}\n\n{block.text}".strip() if block_heading else block.text
            if len(table_text) >= min_html_chunk_chars:
                chunks.append(
                    ChunkCandidate(
                        text=table_text,
                        section_title=block_heading.split(" > ")[-1] if block_heading else "",
                        heading_path=block_heading,
                        chunk_type="table_like_text",
                    )
                )
            current_heading = block_heading
            continue

        projected_length = sum(len(item) for item in current_blocks) + len(block.text) + max(0, len(current_blocks) - 1)
        if current_blocks and (block_heading != current_heading or projected_length > chunk_size):
            flush()
        current_heading = block_heading
        current_blocks.append(block.text)
        current_types.append(block.block_type)

    flush()

    expanded_chunks: list[ChunkCandidate] = []
    for chunk in chunks:
        if len(chunk.text) <= int(chunk_size * 1.35):
            expanded_chunks.append(chunk)
            continue
        for split_chunk in _chunk_document_text(
            text=chunk.text,
            page_spans=[],
            chunk_size=chunk_size,
            chunk_overlap=0,
            min_chunk_chars=min_html_chunk_chars,
        ):
            expanded_chunks.append(
                ChunkCandidate(
                    text=split_chunk.text,
                    section_title=chunk.section_title,
                    heading_path=chunk.heading_path,
                    chunk_type=chunk.chunk_type,
                )
            )
    return expanded_chunks


def _chunk_document_text(
    text: str,
    page_spans: list[tuple[int, int, int]],
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_chars: int,
) -> list[ChunkCandidate]:
    """Create LangChain-split chunks with page and lightweight section metadata."""

    if not text:
        return []
    chunks: list[ChunkCandidate] = []
    previous_section = ""

    for chunk, start, end in _split_text_with_langchain(
        text=text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    ):
        if len(chunk) >= min_chunk_chars:
            section_title = _section_title_for_chunk(chunk, previous_section=previous_section)
            if section_title:
                previous_section = section_title
            page_start, page_end = _page_range_for_span(page_spans, start, end)
            chunks.append(
                ChunkCandidate(
                    text=chunk,
                    page_start=page_start,
                    page_end=page_end,
                    section_title=section_title,
                    heading_path=section_title,
                    chunk_type=_chunk_type_from_text(chunk),
                )
            )
    return chunks


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int, min_chunk_chars: int) -> list[str]:
    """Compatibility wrapper returning only chunk text."""

    return [
        chunk.text
        for chunk in _chunk_document_text(
            text=text,
            page_spans=[],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_chars=min_chunk_chars,
        )
    ]


def _collect_source_files(data_dir: Path) -> list[Path]:
    """Collect non-hidden, non-archived files from corpus directory."""

    if not data_dir.exists():
        raise RAGError(f"Data directory not found: {data_dir}")

    files: list[Path] = []
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file():
            continue
        relative_path = path.relative_to(data_dir)
        if any(part.startswith(".") for part in relative_path.parts):
            continue
        if relative_path.parts[0].casefold() == ARCHIVE_DIRNAME:
            continue
        files.append(path)
    return files


def _domain_from_relative_path(relative_path: Path) -> str:
    """Use top-level folder name as domain label for reporting/filtering."""

    if len(relative_path.parts) == 0:
        return "unknown"
    return relative_path.parts[0]


def _first_matching_labels(text: str, rules: dict[str, tuple[str, ...]], limit: int = 4) -> str:
    """Return comma-separated labels whose keyword rules match the text."""

    lowered = text.casefold()
    labels = []
    for label, keywords in rules.items():
        if any(keyword.casefold() in lowered for keyword in keywords):
            labels.append(label)
        if len(labels) >= limit:
            break
    return ", ".join(labels)


def _extract_source_url(file_path: Path) -> str:
    """Read crawler preamble or filename host hints to recover original URL."""

    if file_path.suffix.lower() not in {".html", ".htm", ".txt", ".md"}:
        return ""
    try:
        head = file_path.read_text(encoding="utf-8", errors="ignore")[:4000]
    except Exception:
        return ""

    match = re.search(r"^\s*source_url:\s*(https?://\S+)\s*$", head, flags=re.IGNORECASE | re.MULTILINE)
    if match:
        return match.group(1).strip()

    canonical = re.search(
        r"""<link[^>]+rel=["']canonical["'][^>]+href=["']([^"']+)["']""",
        head,
        flags=re.IGNORECASE,
    )
    if canonical:
        return unescape(canonical.group(1).strip())
    return ""


def _extract_publication_date(file_path: Path) -> str:
    """Publication dates require reviewed metadata, not filesystem timestamps."""

    return ""


def _compact_utc_timestamp_to_iso(value: str) -> str:
    """Convert crawler stamps such as `20260622T133205Z` to ISO 8601."""

    match = re.fullmatch(r"(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})Z", value.strip())
    if not match:
        return ""
    year, month, day, hour, minute, second = match.groups()
    return f"{year}-{month}-{day}T{hour}:{minute}:{second}Z"


def _extract_collection_date(file_path: Path) -> str:
    """Read the explicit crawler timestamp, never a file creation/modification date."""

    if file_path.suffix.lower() not in {".html", ".htm", ".txt", ".md"}:
        return ""
    try:
        head = file_path.read_text(encoding="utf-8", errors="ignore")[:2000]
    except Exception:
        return ""

    match = re.search(r"^\s*fetched_at_utc:\s*([^\s]+)\s*$", head, flags=re.IGNORECASE | re.MULTILINE)
    if match:
        value = match.group(1).strip()
        return _compact_utc_timestamp_to_iso(value) or value
    return ""


def _source_fingerprint(relative_path: Path, source_name: str, url: str) -> str:
    """Build one searchable string from path, filename, and URL."""

    return " ".join([relative_path.as_posix(), source_name, url]).casefold()


def _infer_language(text: str) -> str:
    """Best-effort language tag for retrieved chunks."""

    lowered = text.casefold()
    german_hits = sum(
        lowered.count(token)
        for token in (" der ", " die ", " das ", " und ", " nicht ", " gewerbe", " gründung", " unternehmen")
    )
    english_hits = sum(
        lowered.count(token)
        for token in (" the ", " and ", " business", " company", " consumer", " contract", " startup")
    )
    if german_hits == 0 and english_hits == 0:
        return "unknown"
    if german_hits >= english_hits * 1.5:
        return "de"
    if english_hits >= german_hits * 1.5:
        return "en"
    return "mixed"


def _infer_document_type(file_path: Path, fingerprint: str) -> str:
    """Classify the physical/source document shape."""

    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return "pdf_guide"
    if suffix in {".html", ".htm"}:
        if any(host in fingerprint for host in ("existenzgruendungsportal", "europa.eu", "bmas.de")):
            return "official_web_page"
        if "gruenderplattform.de" in fingerprint:
            return "platform_article"
        return "web_page"
    if suffix in {".txt", ".md"}:
        return "plain_text"
    return suffix.lstrip(".") or "unknown"


def _infer_source_quality(fingerprint: str) -> str:
    """Estimate source authority from host/path hints."""

    if "europa.eu" in fingerprint:
        return "official_eu"
    if any(
        host in fingerprint
        for host in ("existenzgruendungsportal", "gruenderzeiten", "gründerzeiten", "bmwk", "bmas.de", "gesetze-im-internet")
    ):
        return "official_government"
    if any(marker in fingerprint for marker in ("ihk", "hwk", "handwerkskammer", "arbeitsagentur")):
        return "institutional"
    if "gruenderplattform.de" in fingerprint:
        return "startup_platform"
    if any(marker in fingerprint for marker in ("lexware", "firma.de", "businessplan-lektorat")):
        return "commercial_secondary"
    return "unknown"


def _infer_authority(fingerprint: str, text: str) -> str:
    """Identify the most likely institutional authority/source family."""

    probe = f"{fingerprint} {text[:1500]}".casefold()
    authority_rules = {
        "European Union": ("europa.eu", "european union", "europäische union"),
        "BMWK / Existenzgruendungsportal": ("existenzgruendungsportal", "gruenderzeiten", "gründerzeiten", "bmwk"),
        "BMAS": ("bmas.de", "bundesministerium für arbeit", "mindestlohn"),
        "IHK": ("ihk", "industrie- und handelskammer"),
        "HWK / Handwerkskammer": ("hwk", "handwerkskammer", "handwerksrolle"),
        "Finanzamt": ("finanzamt", "steuer"),
        "Gewerbeamt": ("gewerbeamt", "gewerbeanmeldung"),
        "Gesundheitsamt": ("gesundheitsamt", "infektionsschutz", "hygiene"),
        "Bundesagentur fuer Arbeit": ("arbeitsagentur", "gründungszuschuss"),
        "Gruenderplattform": ("gruenderplattform.de", "gründerplattform"),
    }
    return _first_matching_labels(probe, authority_rules, limit=3) or "unknown"


def _infer_jurisdiction(fingerprint: str, text: str) -> str:
    """Infer legal/geographical scope."""

    probe = f"{fingerprint} {text[:1000]}".casefold()
    if "europa.eu" in probe or " eu-" in probe or "european union" in probe:
        return "EU"
    if any(term in probe for term in ("bundesweit", "deutschland", "germany", "bundesministerium", "gesetze-im-internet")):
        return "Germany/federal"
    if any(term in probe for term in ("gewerbeamt", "ordnungsamt", "gemeinde", "kommune", "vor ort")):
        return "Germany/local"
    return "Germany"


def _infer_topic(domain: str, fingerprint: str, text: str) -> str:
    """Infer founder decision topics from path and chunk text."""

    probe = f"{domain} {fingerprint} {text[:2500]}"
    topic_rules = {
        "legal_form_selection": ("rechtsform", "gmbh", "ug", "gbr", "einzelunternehmen", "haftung"),
        "trade_registration": ("gewerbeanmeldung", "gewerbe anmelden", "gewerbeschein", "gewerbeamt"),
        "professional_status": ("freiberuf", "freiberuflich", "selbständig", "selbstständig"),
        "business_plan": ("businessplan", "executive summary", "geschäftsplan"),
        "funding": ("finanzierung", "förderung", "kredit", "gründungszuschuss", "zuschuss"),
        "financial_planning": ("liquiditätsplanung", "break-even", "rentabilität", "umsatzplanung", "kapitalbedarf"),
        "tax": ("finanzamt", "umsatzsteuer", "einkommensteuer", "körperschaftsteuer", "eür", "steuer"),
        "market_validation": ("zielgruppenanalyse", "marktanalyse", "mvp", "minimum viable product", "design thinking"),
        "business_model": ("geschäftsmodell", "value proposition", "disruption", "canvas"),
        "ecommerce": ("e-commerce", "online-shop", "fernabsatz", "distance selling", "verbrauchervertrag"),
        "food_hygiene": ("lebensmittel", "hygiene", "gesundheitsamt", "infektionsschutz", "haccp"),
        "youth_protection": ("jugendschutz", "tabak", "alkohol", "spirituosen"),
        "insurance": ("versicherung", "betriebshaftpflicht", "krankenversicherung"),
        "marketing": ("marketing", "social media", "affiliate", "kommunikationspolitik"),
    }
    inferred = _first_matching_labels(probe, topic_rules)
    return inferred or domain


def _infer_legal_area(fingerprint: str, text: str) -> str:
    """Infer legal areas that can be used for filtering/review."""

    probe = f"{fingerprint} {text[:3000]}"
    legal_rules = {
        "company_law": ("gmbh", "ug", "gbr", "handelsregister", "gesellschaft", "haftung"),
        "trade_law": ("gewerbeordnung", "gewerbeanmeldung", "gewerbeamt", "gewerbeschein"),
        "tax_law": ("umsatzsteuer", "einkommensteuer", "körperschaftsteuer", "finanzamt", "eür"),
        "employment_social_law": ("mindestlohn", "sozialversicherung", "arbeitslosenversicherung"),
        "food_hygiene_law": ("lebensmittelhygiene", "gesundheitszeugnis", "infektionsschutz", "haccp"),
        "hospitality_law": ("gaststätt", "schanklizenz", "konzession"),
        "youth_protection_law": ("jugendschutz", "juschg", "tabak", "alkohol"),
        "consumer_contract_law": ("verbrauchervertrag", "widerrufsrecht", "garantie", "gewährleistung"),
        "data_ip_law": ("datenschutz", "dsgvo", "marke", "urheber"),
    }
    return _first_matching_labels(probe, legal_rules, limit=5)


def _infer_business_type(fingerprint: str, text: str) -> str:
    """Infer business model/industry labels for scenario matching."""

    probe = f"{fingerprint} {text[:3000]}"
    business_rules = {
        "retail": ("einzelhandel", "verkauf", "laden", "shop", "warensortiment"),
        "ecommerce": ("online-shop", "e-commerce", "fernabsatz", "plattform"),
        "food_and_beverage": ("lebensmittel", "restaurant", "café", "getränke", "pizza", "gastronomie"),
        "health_therapy": ("therapie", "gesundheit", "heil", "pflege", "ergotherapie"),
        "craft_trade": ("handwerk", "handwerksrolle", "meister", "friseur"),
        "service_business": ("dienstleistung", "beratung", "coach", "agentur"),
        "freelance": ("freiberuf", "freelancer", "selbständig", "selbstständig"),
        "digital_startup": ("startup", "software", "app", "mvp", "plattform"),
    }
    return _first_matching_labels(probe, business_rules, limit=5)


def _infer_section_title(text: str) -> str:
    """Use the first short line as a lightweight section/title hint."""

    weak_starts = {
        "der",
        "die",
        "das",
        "ein",
        "eine",
        "einer",
        "eines",
        "er",
        "sie",
        "es",
        "dazu",
        "und",
        "aber",
    }
    for raw_line in text.splitlines():
        line = raw_line.strip(" -•\t")
        if not (8 <= len(line) <= 140):
            continue
        if line.endswith(".") or "." in line:
            continue
        if line and line[0].islower():
            continue
        words = line.split()
        if len(words) > 10:
            continue
        if words and words[0].casefold() in weak_starts and not line.endswith(("?", ":")):
            continue
        if "," in line and not line.endswith(":"):
            continue
        if re.match(r"^\d+$", line):
            continue
        if not (
            line.endswith(("?", ":"))
            or "|" in line
            or line.isupper()
            or len(words) <= 6
            or any(keyword in line.casefold() for keyword in ("tipp", "checkliste", "gründerzeiten", "businessplan"))
        ):
            continue
        if line:
            return line
    return ""


def _infer_chunk_metadata(
    *,
    source_file: Path,
    relative_path: Path,
    domain: str,
    source_name: str,
    chunk_text: str,
    source_url: str,
    publication_date: str,
) -> dict[str, Any]:
    """Infer editable chunk metadata from stable local signals."""

    fingerprint = _source_fingerprint(relative_path, source_name, source_url)
    return {
        "topic": _infer_topic(domain, fingerprint, chunk_text),
        "legal_area": _infer_legal_area(fingerprint, chunk_text),
        "business_type": _infer_business_type(fingerprint, chunk_text),
        "authority": _infer_authority(fingerprint, chunk_text),
        "source_quality": _infer_source_quality(fingerprint),
        "language": _infer_language(chunk_text),
        "jurisdiction": _infer_jurisdiction(fingerprint, chunk_text),
        "document_type": _infer_document_type(source_file, fingerprint),
        "section_title": _infer_section_title(chunk_text),
        "url": source_url,
        "publication_date": publication_date,
        "metadata_reviewed": False,
    }


def _figure_description_json_path(config: RAGConfig) -> Path:
    """Return the optional reviewed figure-description artifact path."""

    return config.index_dir / FIGURE_DESCRIPTION_DIRNAME / FIGURE_DESCRIPTION_JSON_FILENAME


def _format_single_reviewed_figure_for_page_text(figure: dict[str, Any]) -> str:
    """Convert one reviewed figure description into text appended to its PDF page."""

    title = str(figure.get("title") or figure.get("final_figure_id") or "Reviewed figure").strip()
    figure_type = str(figure.get("figure_type") or "figure").strip()
    description = str(figure.get("description_markdown") or "").strip()
    values = figure.get("extracted_values") if isinstance(figure.get("extracted_values"), list) else []

    lines = [
        "[Reviewed figure description inserted from multimodal PDF review]",
        f"Figure title: {title}",
        f"Figure type: {figure_type}",
    ]
    if description:
        lines.extend(["", "Description:", description])
    if values:
        lines.extend(["", "Extracted values:"])
        lines.extend(f"- {str(value).strip()}" for value in values if str(value).strip())
    return _normalize_text("\n".join(lines))


def _format_reviewed_figures_for_page_text(figures: list[dict[str, Any]]) -> str:
    """Format all reviewed figures for insertion into one extracted PDF page."""

    formatted = [
        _format_single_reviewed_figure_for_page_text(figure)
        for figure in figures
        if isinstance(figure, dict)
    ]
    formatted = [text for text in formatted if text]
    if not formatted:
        return ""
    return _normalize_text("\n\n".join(["[Reviewed visual content from this PDF page]", *formatted]))


def _load_reviewed_figure_descriptions_by_source(
    config: RAGConfig,
) -> dict[str, dict[int, list[dict[str, Any]]]]:
    """Load reviewed figure descriptions grouped by source PDF and page.

    The descriptions are not indexed as standalone documents. They are used as
    page-level enrichment during PDF text extraction, so chunking still happens
    on the combined page text.
    """

    figure_json_path = _figure_description_json_path(config)
    if not figure_json_path.exists():
        return {}

    try:
        figures = json.loads(figure_json_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RAGError(f"Failed to read reviewed figure descriptions: {exc}") from exc
    if not isinstance(figures, list):
        raise RAGError(f"Reviewed figure descriptions must be a JSON list: {figure_json_path}")

    grouped: dict[str, dict[int, list[dict[str, Any]]]] = {}
    for figure in figures:
        if not isinstance(figure, dict):
            continue
        source_path = str(figure.get("source_path") or "").strip()
        if not source_path.lower().endswith(".pdf"):
            continue
        try:
            page_number = int(figure.get("page_number") or 0)
        except (TypeError, ValueError):
            continue
        if page_number <= 0:
            continue
        grouped.setdefault(source_path, {}).setdefault(page_number, []).append(figure)
    return grouped


def _is_reviewed_figure_enriched_text(chunk_text: str) -> bool:
    """Return true when a chunk contains reviewed figure-description material."""

    lowered = chunk_text.casefold()
    return any(
        marker in lowered
        for marker in (
            "[reviewed visual content from this pdf page]",
            "[reviewed figure description inserted from multimodal pdf review]",
            "figure title:",
            "figure type:",
            "extracted values:",
        )
    )


def _build_chunks_for_source_files(
    config: RAGConfig,
    source_files: list[Path],
) -> tuple[list[ChunkRecord], list[dict[str, Any]]]:
    """Build fully enriched chunks for an explicit source-file collection."""

    reviewed_figures_by_source = _load_reviewed_figure_descriptions_by_source(config)
    previous_path = config.index_dir / INDEX_METADATA_FILENAME
    previous = (
        _normalize_index_metadata(
            json.loads(previous_path.read_text(encoding="utf-8")),
            data_dir=config.data_dir,
        )
        if previous_path.exists() else {}
    )
    imported_dates = {
        item["source_path"]: item.get("imported_at", "")
        for item in previous.get("documents", [])
    }
    import_time = datetime.now(UTC).isoformat()
    all_chunks: list[ChunkRecord] = []
    extracted_docs: list[dict[str, Any]] = []

    for source_file in source_files:
        relative_path = source_file.relative_to(config.data_dir)
        domain = _domain_from_relative_path(relative_path)
        source_name = source_file.stem if source_file.suffix else source_file.name
        source_url = _extract_source_url(source_file)
        publication_date = _extract_publication_date(source_file)
        source_dates = {"retrieved_at": _extract_collection_date(source_file)}
        if source_file.suffix.lower() == ".pdf":
            source_dates["imported_at"] = imported_dates.get(relative_path.as_posix()) or import_time
        try:
            document = _extract_corpus_document(
                source_file,
                relative_path,
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
                min_chunk_chars=config.min_chunk_chars,
                reviewed_figures_by_page=reviewed_figures_by_source.get(relative_path.as_posix(), {}),
            )
        except RAGError:
            raise
        except Exception as exc:  # pragma: no cover - defensive handling
            raise RAGError(f"Failed to parse {source_file}: {exc}") from exc

        text = document.text
        if not text:
            continue

        chunk_candidates = document.structured_chunks or _chunk_document_text(
            text=text,
            page_spans=document.page_spans,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            min_chunk_chars=config.min_chunk_chars,
        )
        if source_file.suffix.lower() == ".pdf":
            chunk_candidates = [
                chunk_candidate
                for chunk_candidate in chunk_candidates
                if not _is_pdf_publication_boilerplate_chunk(chunk_candidate.text)
            ]
        for idx, chunk_candidate in enumerate(chunk_candidates):
            inferred_metadata = _infer_chunk_metadata(
                source_file=source_file,
                relative_path=relative_path,
                domain=domain,
                source_name=source_name,
                chunk_text=chunk_candidate.text,
                source_url=source_url,
                publication_date=publication_date,
            )
            inferred_metadata.update(document.metadata)
            inferred_metadata.update({key: value for key, value in source_dates.items() if value})
            inferred_metadata.update(chunk_candidate.metadata)
            inferred_metadata.update(
                {
                    "section_title": chunk_candidate.section_title or inferred_metadata.get("section_title", ""),
                    "heading_path": chunk_candidate.heading_path,
                    "chunk_type": "figure_enriched_text"
                    if _is_reviewed_figure_enriched_text(chunk_candidate.text)
                    else chunk_candidate.chunk_type,
                    "page_start": chunk_candidate.page_start,
                    "page_end": chunk_candidate.page_end,
                }
            )
            all_chunks.append(
                ChunkRecord(
                    chunk_id=f"{relative_path.as_posix()}::{idx}",
                    domain=domain,
                    source_path=relative_path.as_posix(),
                    source_name=source_name,
                    chunk_index=idx,
                    text=chunk_candidate.text,
                    char_count=len(chunk_candidate.text),
                    **inferred_metadata,
                )
            )

        document_metadata = _infer_chunk_metadata(
            source_file=source_file,
            relative_path=relative_path,
            domain=domain,
            source_name=source_name,
            chunk_text=text[:5000],
            source_url=source_url,
            publication_date=publication_date,
        )
        document_metadata.update(document.metadata)
        document_metadata.update({key: value for key, value in source_dates.items() if value})
        extracted_docs.append(
            {
                "domain": domain,
                "source_path": relative_path.as_posix(),
                "source_name": source_name,
                "char_count": len(text),
                "chunk_count": len(chunk_candidates),
                "text": text,
                **document_metadata,
            }
        )
    return all_chunks, extracted_docs


def _build_chunks(config: RAGConfig) -> tuple[list[ChunkRecord], list[dict[str, Any]]]:
    """Extract the complete configured corpus into chunks and document rows."""

    return _build_chunks_for_source_files(config, _collect_source_files(config.data_dir))


def _embed_texts(texts: list[str], config: RAGConfig, batch_size: int = 16) -> np.ndarray:
    """Embed texts in batches and return row-normalized float32 matrix."""

    vectors: list[list[float]] = []
    endpoint = _embedding_endpoint(config)

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        payload = {"model": config.embedding_model, "input": batch}
        response = None
        for attempt in range(3):
            try:
                response = _post_json(endpoint, payload, config)
                break
            except RAGError:
                if attempt == 2:
                    raise
                time.sleep(1.2 * (attempt + 1))
        embeddings = response.get("embeddings", [])
        if len(embeddings) != len(batch):
            raise RAGError(
                f"Embedding response size mismatch: expected {len(batch)}, got {len(embeddings)}."
            )
        vectors.extend(embeddings)

    array = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return array / norms


def embed_queries(queries: list[str], config: RAGConfig) -> np.ndarray:
    """Return normalized query embeddings for explicit evaluation freezing."""

    if not queries:
        raise RAGError("At least one query is required for embedding.")
    return _embed_texts(queries, config=config)


def embed_query(query: str, config: RAGConfig) -> np.ndarray:
    """Return one normalized query embedding for explicit evaluation freezing."""

    return embed_queries([query], config=config)[0]


def _index_paths(index_dir: Path) -> tuple[Path, Path]:
    """Return `(metadata_path, embedding_path)` for an index directory."""

    return index_dir / INDEX_METADATA_FILENAME, index_dir / INDEX_EMBEDDINGS_FILENAME


def _metadata_defaults() -> dict[str, Any]:
    """Return default values for optional editable metadata fields."""

    return {field: False if field == "metadata_reviewed" else "" for field in CHUNK_METADATA_FIELDS}


def _normalize_index_metadata(
    metadata: dict[str, Any], data_dir: Path | None = None
) -> dict[str, Any]:
    """Backfill editable metadata fields for older index files."""

    defaults = _metadata_defaults()
    collection_dates: dict[str, str] = {}
    source_root = data_dir or (Path(metadata["data_dir"]) if metadata.get("data_dir") else None)
    for collection_name in ("documents", "chunks"):
        for item in metadata.get(collection_name, []):
            for field, default_value in defaults.items():
                item.setdefault(field, default_value)
            source_path = str(item.get("source_path", ""))
            if source_path.lower().endswith(".pdf"):
                # This is a known indexing time, not an inferred download date.
                if not item["imported_at"]:
                    item["imported_at"] = metadata.get("generated_at", "")
            elif source_root is not None and source_path:
                if source_path not in collection_dates:
                    collection_dates[source_path] = _extract_collection_date(source_root / source_path)
                collected = collection_dates[source_path]
                if collected:
                    if not item["retrieved_at"]:
                        item["retrieved_at"] = collected
                    # Correct only legacy values demonstrably copied from the crawler.
                    normalized_publication = (
                        _compact_utc_timestamp_to_iso(item["publication_date"])
                        or item["publication_date"]
                    )
                    if normalized_publication == collected:
                        item["publication_date"] = ""
            legacy_collection_date = _compact_utc_timestamp_to_iso(item["publication_date"])
            if legacy_collection_date:
                if not item["retrieved_at"]:
                    item["retrieved_at"] = legacy_collection_date
                item["publication_date"] = ""
    return metadata


def _load_metadata_only(config: RAGConfig) -> tuple[Path, dict[str, Any]]:
    """Load index metadata without loading the embedding matrix."""

    metadata_path, _ = _index_paths(config.index_dir)
    if not metadata_path.exists():
        raise RAGError(f"RAG metadata not found in {config.index_dir}. Build the index first.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return metadata_path, _normalize_index_metadata(metadata, data_dir=config.data_dir)


def _write_metadata(metadata_path: Path, metadata: dict[str, Any]) -> None:
    """Write metadata atomically enough for local demo use."""

    metadata["updated_at"] = datetime.now(UTC).isoformat()
    temp_path = metadata_path.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(metadata_path)


def _save_index(
    config: RAGConfig,
    chunks: list[ChunkRecord],
    extracted_docs: list[dict[str, Any]],
    embeddings: np.ndarray,
) -> None:
    """Persist extracted text snapshots, metadata JSON, and embedding matrix."""

    config.index_dir.mkdir(parents=True, exist_ok=True)
    metadata_path, embedding_path = _index_paths(config.index_dir)
    extracted_dir = config.index_dir / EXTRACTED_TEXT_DIRNAME
    extracted_dir.mkdir(parents=True, exist_ok=True)
    for old_snapshot in extracted_dir.glob("*.txt"):
        old_snapshot.unlink()

    for doc in extracted_docs:
        out_name = doc["source_path"].replace("/", "__").replace("\\", "__")
        (extracted_dir / f"{out_name}.txt").write_text(doc["text"], encoding="utf-8")

    metadata = {
        "version": "v2",
        "generated_at": datetime.now(UTC).isoformat(),
        "embedding_model": config.embedding_model,
        "embedding_provider": config.llm_provider,
        "legal_corpus_scope_sha256": legal_scope_sha256(),
        "legal_extractor_version": LEGAL_EXTRACTOR_VERSION,
        "data_dir": str(config.data_dir),
        "index_dir": str(config.index_dir),
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "min_chunk_chars": config.min_chunk_chars,
        "document_count": len(extracted_docs),
        "chunk_count": len(chunks),
        "documents": [
            {
                "domain": doc["domain"],
                "source_path": doc["source_path"],
                "source_name": doc["source_name"],
                "char_count": doc["char_count"],
                "chunk_count": doc["chunk_count"],
                **{field: doc.get(field, False if field == "metadata_reviewed" else "") for field in CHUNK_METADATA_FIELDS},
            }
            for doc in extracted_docs
        ],
        "chunks": [
            {
                "chunk_id": chunk.chunk_id,
                "domain": chunk.domain,
                "source_path": chunk.source_path,
                "source_name": chunk.source_name,
                "chunk_index": chunk.chunk_index,
                "char_count": chunk.char_count,
                "text": chunk.text,
                **{field: getattr(chunk, field) for field in CHUNK_METADATA_FIELDS},
            }
            for chunk in chunks
        ],
    }

    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    np.save(embedding_path, embeddings)


def _save_current_source_manifest(config: RAGConfig) -> None:
    """Persist source signatures after every successful full index build."""

    sources: dict[str, dict[str, int]] = {}
    for source_file in _collect_source_files(config.data_dir):
        stat = source_file.stat()
        source_path = source_file.relative_to(config.data_dir).as_posix()
        sources[source_path] = {
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "data_dir": str(config.data_dir),
        "source_count": len(sources),
        "sources": sources,
    }
    (config.index_dir / SOURCE_MANIFEST_FILENAME).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_rag_index(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Public entrypoint to build or refresh local RAG index artifacts."""

    config = _resolve_config(overrides)
    started = time.perf_counter()

    chunks, extracted_docs = _build_chunks(config)
    if not chunks:
        raise RAGError(f"No valid chunks extracted from data directory: {config.data_dir}")

    embeddings = _embed_texts([chunk.text for chunk in chunks], config=config)
    _save_index(config=config, chunks=chunks, extracted_docs=extracted_docs, embeddings=embeddings)
    _save_current_source_manifest(config)

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    return {
        "status": "ok",
        "data_dir": str(config.data_dir),
        "index_dir": str(config.index_dir),
        "embedding_model": config.embedding_model,
        "document_count": len(extracted_docs),
        "chunk_count": len(chunks),
        "duration_ms": elapsed_ms,
    }


def _load_index(config: RAGConfig) -> tuple[np.ndarray, dict[str, Any]]:
    """Load and validate index files before query-time retrieval."""

    metadata_path, embedding_path = _index_paths(config.index_dir)
    if not metadata_path.exists() or not embedding_path.exists():
        raise RAGError(
            f"RAG index not found in {config.index_dir}. Build it first with 'python manage.py build_rag_index'."
        )

    metadata = _normalize_index_metadata(
        json.loads(metadata_path.read_text(encoding="utf-8")),
        data_dir=config.data_dir,
    )
    embeddings = np.load(embedding_path)
    if embeddings.ndim != 2:
        raise RAGError("Invalid embedding matrix shape in index.")
    if len(metadata.get("chunks", [])) != embeddings.shape[0]:
        raise RAGError("Chunk metadata and embedding rows are inconsistent.")
    return embeddings, metadata


def list_rag_chunks(
    page: int = 1,
    page_size: int = 20,
    query: str = "",
    domain: str = "",
    reviewed: str = "",
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return paginated chunk metadata rows for expert review UI."""

    config = _resolve_config(overrides)
    _, metadata = _load_metadata_only(config)
    chunks = list(metadata.get("chunks", []))

    if domain:
        chunks = [chunk for chunk in chunks if chunk.get("domain") == domain]

    reviewed_value = reviewed.strip().lower()
    if reviewed_value in {"true", "false"}:
        expected = reviewed_value == "true"
        chunks = [chunk for chunk in chunks if bool(chunk.get("metadata_reviewed")) is expected]

    query_text = query.strip().casefold()
    if query_text:
        searchable_fields = [
            "chunk_id",
            "domain",
            "source_path",
            "source_name",
            "topic",
            "legal_area",
            "business_type",
            "authority",
            "source_quality",
            "language",
            "jurisdiction",
            "document_type",
            "section_title",
            "url",
            "law_name",
            "law_abbreviation",
            "law_section",
            "canonical_url",
            "temporal_status",
            "text",
        ]
        chunks = [
            chunk
            for chunk in chunks
            if query_text in " ".join(str(chunk.get(field, "")) for field in searchable_fields).casefold()
        ]

    page_size = max(1, min(int(page_size), 100))
    total = len(chunks)
    total_pages = max(1, (total + page_size - 1) // page_size)
    page = max(1, min(int(page), total_pages))
    start = (page - 1) * page_size
    end = start + page_size

    return {
        "page": page,
        "page_size": page_size,
        "total": total,
        "total_pages": total_pages,
        "domains": sorted({chunk.get("domain", "unknown") for chunk in metadata.get("chunks", [])}),
        "metadata_fields": CHUNK_METADATA_FIELDS,
        "items": [
            {
                **chunk,
                "text_preview": chunk.get("text", "")[:800],
            }
            for chunk in chunks[start:end]
        ],
    }


def update_rag_chunk_metadata(
    chunk_id: str,
    metadata_patch: dict[str, Any],
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist manual metadata edits for one chunk without changing embeddings."""

    config = _resolve_config(overrides)
    metadata_path, metadata = _load_metadata_only(config)
    allowed_fields = set(CHUNK_METADATA_FIELDS)
    invalid_fields = sorted(set(metadata_patch) - allowed_fields)
    if invalid_fields:
        raise RAGError(f"Unsupported metadata fields: {', '.join(invalid_fields)}")

    target = None
    for chunk in metadata.get("chunks", []):
        if chunk.get("chunk_id") == chunk_id:
            target = chunk
            break
    if target is None:
        raise RAGError(f"Chunk not found: {chunk_id}")

    for field in CHUNK_METADATA_FIELDS:
        if field not in metadata_patch:
            continue
        if field == "metadata_reviewed":
            target[field] = bool(metadata_patch[field])
        else:
            target[field] = str(metadata_patch[field]).strip()

    if "metadata_reviewed" not in metadata_patch:
        target["metadata_reviewed"] = True

    _write_metadata(metadata_path, metadata)
    return {
        "status": "ok",
        "chunk": {
            **target,
            "text_preview": target.get("text", "")[:800],
        },
    }


def get_rag_status(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return readiness status for corpus index and local model environment."""

    config = _resolve_config(overrides)
    status_payload: dict[str, Any] = {
        "data_dir": str(config.data_dir),
        "index_dir": str(config.index_dir),
        "embedding_model": config.embedding_model,
        "index_ready": False,
        "document_count": 0,
        "chunk_count": 0,
    }

    metadata_path, embedding_path = _index_paths(config.index_dir)
    if metadata_path.exists() and embedding_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            status_payload.update(
                {
                    "index_ready": True,
                    "document_count": metadata.get("document_count", 0),
                    "chunk_count": metadata.get("chunk_count", 0),
                    "generated_at": metadata.get("generated_at", ""),
                }
            )
        except Exception:
            status_payload["index_error"] = "Index files exist but metadata could not be read."

    try:
        llm_status = check_status()
        available_models = llm_status.get("available_models", [])
        status_payload.update(
            {
                "llm_reachable": bool(llm_status.get("reachable")),
                "generation_model": llm_status.get("configured_model", ""),
                "embedding_model_installed": config.embedding_model in available_models,
            }
        )
    except LocalLLMError as exc:
        status_payload.update(
            {
                "llm_reachable": False,
                "embedding_model_installed": False,
                "llm_error": str(exc),
            }
        )
    return status_payload


def _chunk_hit_payload(rank: int, score: float, chunk_meta: dict[str, Any]) -> dict[str, Any]:
    """Build a retrieval hit payload including editable metadata fields."""

    return {
        "rank": rank,
        "score": float(score),
        "chunk_id": chunk_meta["chunk_id"],
        "domain": chunk_meta["domain"],
        "source_path": chunk_meta["source_path"],
        "source_name": chunk_meta["source_name"],
        "chunk_index": chunk_meta["chunk_index"],
        "text": chunk_meta["text"],
        "text_preview": chunk_meta["text"][:320],
        **{field: chunk_meta.get(field, False if field == "metadata_reviewed" else "") for field in CHUNK_METADATA_FIELDS},
    }


def _retrieve_chunks(
    query: str,
    config: RAGConfig,
    top_k: int = 5,
    domain: str = "",
    allow_keyword_fallback: bool = True,
    query_embedding: np.ndarray | None = None,
) -> tuple[list[dict[str, Any]], int, str]:
    """Retrieve top chunks, optionally constrained to one decision domain."""

    def _tokenize(text: str) -> list[str]:
        # Unicode-safe tokenizer for fallback ranking.
        return [token for token in re.split(r"[^\w]+", text.lower()) if len(token) > 1]

    def _keyword_fallback_hits(chunks: list[dict[str, Any]], query_text: str, limit: int) -> list[dict[str, Any]]:
        query_tokens = set(_tokenize(query_text))
        scored: list[tuple[float, dict[str, Any]]] = []

        for chunk_meta in chunks:
            chunk_tokens = _tokenize(chunk_meta["text"])
            if not chunk_tokens:
                score = 0.0
            else:
                overlap = sum(1 for token in chunk_tokens if token in query_tokens)
                score = overlap / max(1, len(set(chunk_tokens)))
            scored.append((score, chunk_meta))

        scored.sort(key=lambda item: item[0], reverse=True)
        selected = scored[:limit]
        hits: list[dict[str, Any]] = []
        for rank, (score, chunk_meta) in enumerate(selected, start=1):
            hits.append(_chunk_hit_payload(rank=rank, score=score, chunk_meta=chunk_meta))
        return hits

    started = time.perf_counter()
    embeddings, metadata = _load_index(config)
    chunks = metadata["chunks"]
    normalized_domain = domain.strip()
    eligible_indices = np.asarray(
        [
            idx
            for idx, chunk in enumerate(chunks)
            if not normalized_domain or chunk.get("domain") == normalized_domain
        ],
        dtype=np.int64,
    )
    if not len(eligible_indices):
        raise RAGError(f"No RAG chunks found for domain: {normalized_domain}")
    eligible_chunks = [chunks[idx] for idx in eligible_indices.tolist()]
    top_k = max(1, min(int(top_k), len(eligible_chunks)))

    frozen_query_vector: np.ndarray | None = None
    if query_embedding is not None:
        frozen_query_vector = np.asarray(query_embedding, dtype=np.float32)
        if (
            frozen_query_vector.ndim != 1
            or not frozen_query_vector.size
            or not np.isfinite(frozen_query_vector).all()
        ):
            raise RAGError(
                "Provided query embedding must be a finite one-dimensional vector."
            )
        norm = float(np.linalg.norm(frozen_query_vector))
        if norm == 0.0:
            raise RAGError("Provided query embedding must be non-zero.")
        frozen_query_vector = frozen_query_vector / norm

    try:
        # Preferred path: dense retrieval using query embedding.
        query_vector = (
            frozen_query_vector
            if frozen_query_vector is not None
            else embed_query(query, config=config)
        )
        if embeddings.shape[1] != query_vector.shape[0]:
            raise RAGError(
                "Index embedding dimension does not match query embedding dimension. "
                "Rebuild the RAG index with the currently configured embedding model."
            )
        similarities = embeddings @ query_vector
        eligible_scores = similarities[eligible_indices]
        eligible_chunk_ids = np.asarray(
            [str(chunk["chunk_id"]) for chunk in eligible_chunks]
        )
        ranked_positions = np.lexsort(
            (eligible_chunk_ids, -eligible_scores)
        )[:top_k]
        top_indices = eligible_indices[ranked_positions]

        hits: list[dict[str, Any]] = []
        for rank, idx in enumerate(top_indices.tolist(), start=1):
            chunk_meta = chunks[idx]
            hits.append(_chunk_hit_payload(rank=rank, score=float(similarities[idx]), chunk_meta=chunk_meta))
        retrieval_mode = (
            "dense_embedding_frozen_query"
            if frozen_query_vector is not None
            else "dense_embedding"
        )
    except RAGError:
        if frozen_query_vector is not None or not allow_keyword_fallback:
            raise
        # Fallback path: keyword overlap retrieval when embedding endpoint is unstable.
        hits = _keyword_fallback_hits(chunks=eligible_chunks, query_text=query, limit=top_k)
        retrieval_mode = "keyword_fallback"

    retrieval_ms = int((time.perf_counter() - started) * 1000)
    return hits, retrieval_ms, retrieval_mode


def _resolve_reranker_device(torch_module: Any, requested_device: str) -> str:
    requested = requested_device.strip().lower() or "auto"
    if requested == "auto":
        return "cuda" if torch_module.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch_module.cuda.is_available():
        raise RAGError(
            f"Reranker device {requested_device!r} requires CUDA, but CUDA is unavailable."
        )
    if requested != "cpu" and not requested.startswith("cuda"):
        raise RAGError(f"Unsupported reranker device: {requested_device}")
    return requested


@lru_cache(maxsize=2)
def _load_cross_encoder(
    model_name: str,
    revision: str,
    requested_device: str,
) -> tuple[Any, Any, Any, str]:
    """Load one Hugging Face sequence-classification reranker lazily."""

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RAGError(
            "Cross-encoder reranking requires the optional reranker dependencies. "
            "Install backend/requirements/reranker.txt."
        ) from exc

    device = _resolve_reranker_device(torch, requested_device)
    try:
        model_revision = revision or None
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, revision=model_revision
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, revision=model_revision
        )
        if device.startswith("cuda"):
            model = model.to(device=device, dtype=torch.float16)
        else:
            model = model.to(device)
        model.eval()
    except Exception as exc:
        raise RAGError(f"Could not load reranker model {model_name}: {exc}") from exc
    return tokenizer, model, torch, device


def _score_reranker_pairs(
    *,
    query: str,
    passages: list[str],
    model_name: str,
    revision: str = DEFAULT_RERANKER_REVISION,
    batch_size: int,
    max_length: int,
    device: str = DEFAULT_RERANKER_DEVICE,
) -> list[float]:
    """Return one raw cross-encoder relevance logit per query-passage pair."""

    if not passages:
        return []
    if batch_size < 1:
        raise RAGError("reranker batch_size must be at least 1")
    if max_length < 16:
        raise RAGError("reranker max_length must be at least 16")
    tokenizer, model, torch, resolved_device = _load_cross_encoder(
        model_name, revision, device
    )
    scores: list[float] = []
    for start in range(0, len(passages), batch_size):
        batch_passages = passages[start : start + batch_size]
        try:
            encoded = tokenizer(
                [query] * len(batch_passages),
                batch_passages,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {
                key: value.to(resolved_device) for key, value in encoded.items()
            }
            with torch.inference_mode():
                logits = model(**encoded).logits
            if logits.ndim == 2 and logits.shape[-1] == 1:
                batch_scores = logits[:, 0]
            elif logits.ndim == 2 and logits.shape[-1] == 2:
                batch_scores = logits[:, 1]
            else:
                raise RAGError(
                    f"Unsupported reranker logits shape: {tuple(logits.shape)}"
                )
            scores.extend(float(value) for value in batch_scores.detach().cpu())
        except RAGError:
            raise
        except Exception as exc:
            raise RAGError(f"Reranker inference failed: {exc}") from exc
    if len(scores) != len(passages):
        raise RAGError(
            f"Reranker returned {len(scores)} scores for {len(passages)} passages"
        )
    return scores


def _rerank_candidates(
    *,
    query: str,
    candidates: list[dict[str, Any]],
    model_name: str,
    revision: str = DEFAULT_RERANKER_REVISION,
    batch_size: int = DEFAULT_RERANKER_BATCH_SIZE,
    max_length: int = DEFAULT_RERANKER_MAX_LENGTH,
    device: str = DEFAULT_RERANKER_DEVICE,
) -> tuple[list[dict[str, Any]], int]:
    """Rerank dense candidates while preserving their first-stage provenance."""

    if not model_name.strip():
        raise RAGError("A reranker model name is required")
    started = time.perf_counter()
    scores = _score_reranker_pairs(
        query=query,
        passages=[str(candidate.get("text", "")) for candidate in candidates],
        model_name=model_name,
        revision=revision,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
    )
    reranked = []
    for candidate, score in zip(candidates, scores, strict=True):
        item = dict(candidate)
        item["dense_rank"] = int(candidate.get("rank", 0))
        item["dense_score"] = float(candidate.get("score", 0.0))
        item["reranker_score"] = float(score)
        item["reranker_model"] = model_name
        item["reranker_revision"] = revision
        item["score"] = float(score)
        item["score_type"] = "cross_encoder_logit"
        reranked.append(item)
    reranked.sort(
        key=lambda item: (-float(item["reranker_score"]), int(item["dense_rank"]))
    )
    for rank, item in enumerate(reranked, start=1):
        item["rank"] = rank
    latency_ms = int((time.perf_counter() - started) * 1000)
    return reranked, latency_ms


def _estimate_text_tokens(text: str) -> int:
    """Conservatively estimate multilingual generator tokens without model coupling."""

    normalized = re.sub(r"\s+", " ", text).strip()
    return max(1, math.ceil(len(normalized) / 3)) if normalized else 0


def _estimate_history_tokens(text: str) -> int:
    """Estimate chat-history tokens conservatively for Latin and CJK text."""

    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return 0
    weighted_units = sum(1 if ord(char) < 128 else 4 for char in normalized)
    return max(1, math.ceil(weighted_units / 4))


def _truncate_history_text(text: str, max_tokens: int) -> str:
    """Trim one history message to a conservative model-independent budget."""

    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized or max_tokens < 1:
        return ""
    if _estimate_history_tokens(normalized) <= max_tokens:
        return normalized

    unit_budget = max_tokens * 4 - 4
    used_units = 0
    characters: list[str] = []
    for char in normalized:
        char_units = 1 if ord(char) < 128 else 4
        if used_units + char_units > unit_budget:
            break
        characters.append(char)
        used_units += char_units
    return f"{''.join(characters).rstrip()}..." if characters else ""


def _normalize_conversation_history(
    history: list[dict[str, Any]] | None,
    *,
    max_turns: int = DEFAULT_HISTORY_TURN_LIMIT,
    max_tokens: int = DEFAULT_HISTORY_TOKEN_BUDGET,
    max_message_tokens: int = DEFAULT_HISTORY_MESSAGE_TOKEN_LIMIT,
) -> tuple[list[dict[str, str]], int]:
    """Keep only the newest valid chat messages under the history budget."""

    if not history or max_turns < 1 or max_tokens < 1:
        return [], 0

    valid_messages: list[dict[str, str]] = []
    for item in history:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip().lower()
        content = str(item.get("content", "")).strip()
        if role not in {"user", "assistant"} or not content:
            continue
        valid_messages.append({"role": role, "content": content})

    recent_messages = valid_messages[-(max_turns * 2) :]
    selected_reversed: list[dict[str, str]] = []
    used_tokens = 0
    for item in reversed(recent_messages):
        remaining_tokens = max_tokens - used_tokens
        message_budget = min(max_message_tokens, remaining_tokens - 4)
        if message_budget < 1:
            break
        content = _truncate_history_text(item["content"], message_budget)
        if not content:
            continue
        message_tokens = _estimate_history_tokens(content) + 4
        selected_reversed.append({"role": item["role"], "content": content})
        used_tokens += message_tokens

    return list(reversed(selected_reversed)), used_tokens


def _build_history_aware_retrieval_query(
    question: str,
    conversation_history: list[dict[str, str]],
) -> str:
    """Add bounded conversational context to retrieval without another LLM call."""

    if not conversation_history:
        return question
    history_lines = [
        f"{item['role'].upper()}: {item['content']}" for item in conversation_history
    ]
    return (
        "PREVIOUS CONVERSATION:\n"
        f"{'\n'.join(history_lines)}\n\n"
        f"CURRENT USER QUESTION:\n{question}"
    )


def _select_context_hits(
    candidates: list[dict[str, Any]],
    *,
    context_k: int,
    max_context_tokens: int | None,
    metadata_token_allowance: int = DEFAULT_CONTEXT_METADATA_TOKEN_ALLOWANCE,
) -> tuple[list[dict[str, Any]], int, int]:
    """Greedily select ranked complete chunks under count and token limits."""

    if context_k < 1:
        raise RAGError("context_k must be at least 1")
    if max_context_tokens is not None and max_context_tokens < 1:
        raise RAGError("max_context_tokens must be at least 1 when provided")
    if metadata_token_allowance < 0:
        raise RAGError("metadata_token_allowance cannot be negative")

    started = time.perf_counter()
    selected: list[dict[str, Any]] = []
    total_tokens = 0
    for candidate in candidates:
        text_tokens = _estimate_text_tokens(str(candidate.get("text", "")))
        candidate_tokens = text_tokens + metadata_token_allowance
        if (
            max_context_tokens is not None
            and total_tokens + candidate_tokens > max_context_tokens
        ):
            continue
        item = dict(candidate)
        item["pre_selection_rank"] = int(candidate.get("rank", 0))
        item["estimated_text_tokens"] = text_tokens
        item["estimated_context_tokens"] = candidate_tokens
        selected.append(item)
        total_tokens += candidate_tokens
        if len(selected) >= context_k:
            break
    if not selected:
        raise RAGError(
            "No retrieved chunk fits the configured context token budget"
        )
    for rank, item in enumerate(selected, start=1):
        item["rank"] = rank
    latency_ms = int((time.perf_counter() - started) * 1000)
    return selected, total_tokens, latency_ms


def _expand_context_hits_with_neighbors(
    hits: list[dict[str, Any]],
    *,
    config: RAGConfig,
    max_context_tokens: int,
    max_neighbor_distance: int = 1,
    metadata_token_allowance: int = DEFAULT_CONTEXT_METADATA_TOKEN_ALLOWANCE,
) -> tuple[list[dict[str, Any]], int, int]:
    """Merge same-document neighbor chunks into selected evidence cards."""

    if not hits:
        raise RAGError("At least one selected hit is required for neighbor expansion")
    if max_context_tokens < 1 or max_neighbor_distance < 1:
        raise RAGError(
            "Neighbor expansion requires a positive budget and distance"
        )
    if metadata_token_allowance < 0:
        raise RAGError("metadata_token_allowance cannot be negative")

    started = time.perf_counter()
    _, metadata = _load_index(config)
    chunks = list(metadata.get("chunks", []))
    by_source_index = {
        (str(chunk.get("source_path", "")), int(chunk.get("chunk_index", -1))): chunk
        for chunk in chunks
    }
    expanded = [dict(hit) for hit in hits]
    per_hit_tokens = [
        _estimate_text_tokens(str(hit.get("text", "")))
        + metadata_token_allowance
        for hit in expanded
    ]
    total_tokens = sum(per_hit_tokens)
    if total_tokens > max_context_tokens:
        raise RAGError("Selected hits already exceed the neighbor expansion budget")
    used = {
        (str(hit.get("source_path", "")), int(hit.get("chunk_index", -1)))
        for hit in hits
    }
    for item in expanded:
        item["expanded_neighbor_chunk_ids"] = []
        item["expanded_neighbor_chunks"] = []

    for distance in range(1, max_neighbor_distance + 1):
        for direction in (-1, 1):
            for index, original in enumerate(hits):
                source_path = str(original.get("source_path", ""))
                neighbor_index = int(original.get("chunk_index", -1)) + (
                    direction * distance
                )
                key = (source_path, neighbor_index)
                neighbor = by_source_index.get(key)
                if neighbor is None or key in used:
                    continue
                neighbor_text = str(neighbor.get("text", "")).strip()
                if not neighbor_text:
                    continue
                separator = (
                    "\n\n[Adjacent context: "
                    f"{neighbor.get('chunk_id', '')}]\n"
                )
                proposed_text = (
                    str(expanded[index].get("text", ""))
                    + separator
                    + neighbor_text
                )
                proposed_tokens = (
                    _estimate_text_tokens(proposed_text)
                    + metadata_token_allowance
                )
                delta = proposed_tokens - per_hit_tokens[index]
                if total_tokens + delta > max_context_tokens:
                    continue
                expanded[index]["text"] = proposed_text
                expanded[index]["expanded_neighbor_chunk_ids"].append(
                    str(neighbor.get("chunk_id", ""))
                )
                expanded[index]["expanded_neighbor_chunks"].append(
                    {
                        "chunk_id": str(neighbor.get("chunk_id", "")),
                        "chunk_index": int(neighbor.get("chunk_index", -1)),
                        "page_start": str(neighbor.get("page_start", "")),
                        "page_end": str(neighbor.get("page_end", "")),
                    }
                )
                used.add(key)
                total_tokens += delta
                per_hit_tokens[index] = proposed_tokens

    for item, item_tokens in zip(expanded, per_hit_tokens, strict=True):
        item["expanded_neighbor_count"] = len(
            item["expanded_neighbor_chunk_ids"]
        )
        item["estimated_text_tokens"] = max(
            0, item_tokens - metadata_token_allowance
        )
        item["estimated_context_tokens"] = item_tokens
    latency_ms = int((time.perf_counter() - started) * 1000)
    return expanded, total_tokens, latency_ms


def _assess_retrieval_confidence(
    hits: list[dict[str, Any]],
    *,
    reranker_model: str,
) -> dict[str, Any]:
    """Return a transparent heuristic confidence label for reranked context."""

    scores = [
        float(hit["reranker_score"])
        for hit in hits
        if isinstance(hit.get("reranker_score"), (int, float))
    ]
    source_count = len(
        {str(hit.get("source_path", "")) for hit in hits if hit.get("source_path")}
    )
    if not reranker_model or not scores:
        return {
            "policy_version": RAG_CONFIDENCE_POLICY_VERSION,
            "level": "unrated",
            "reason": "No cross-encoder scores are available for confidence gating.",
            "reranker_model": reranker_model,
            "selected_hit_count": len(hits),
            "nonnegative_reranker_hit_count": 0,
            "max_reranker_score": None,
            "source_count": source_count,
        }
    nonnegative_count = sum(score >= 0 for score in scores)
    if nonnegative_count == 0:
        level = "insufficient"
        reason = "No selected chunk reached a nonnegative cross-encoder logit."
    elif nonnegative_count == 1:
        level = "limited"
        reason = "Only one selected chunk reached a nonnegative cross-encoder logit."
    else:
        level = "adequate"
        reason = (
            f"{nonnegative_count} selected chunks reached a nonnegative "
            "cross-encoder logit."
        )
    return {
        "policy_version": RAG_CONFIDENCE_POLICY_VERSION,
        "level": level,
        "reason": reason,
        "reranker_model": reranker_model,
        "selected_hit_count": len(hits),
        "nonnegative_reranker_hit_count": nonnegative_count,
        "max_reranker_score": max(scores),
        "source_count": source_count,
    }


def _retrieve_context(
    *,
    query: str,
    config: RAGConfig,
    context_k: int = 5,
    candidate_k: int | None = None,
    domain: str = "",
    reranker_model: str | None = None,
    reranker_revision: str | None = None,
    max_context_tokens: int | None = None,
    allow_keyword_fallback: bool = True,
    reranker_batch_size: int = DEFAULT_RERANKER_BATCH_SIZE,
    reranker_max_length: int = DEFAULT_RERANKER_MAX_LENGTH,
    reranker_device: str = DEFAULT_RERANKER_DEVICE,
    neighbor_expansion_distance: int = 0,
) -> RetrievalSelection:
    """Retrieve candidates, optionally rerank them, and build bounded context."""

    selected_reranker = (
        DEFAULT_RERANKER_MODEL if reranker_model is None else reranker_model.strip()
    )
    reranking_enabled = bool(selected_reranker)
    selected_revision = (
        DEFAULT_RERANKER_REVISION
        if reranker_revision is None
        else reranker_revision.strip()
    )
    resolved_candidate_k = (
        int(candidate_k)
        if candidate_k is not None
        else (DEFAULT_RERANK_CANDIDATE_K if reranking_enabled else int(context_k))
    )
    if resolved_candidate_k < context_k:
        raise RAGError("candidate_k must be greater than or equal to context_k")

    candidates, retrieval_ms, first_stage_mode = _retrieve_chunks(
        query=query,
        config=config,
        top_k=resolved_candidate_k,
        domain=domain,
        allow_keyword_fallback=allow_keyword_fallback,
    )
    ranked_candidates = candidates
    reranker_ms = 0
    retrieval_mode = first_stage_mode
    if reranking_enabled:
        ranked_candidates, reranker_ms = _rerank_candidates(
            query=query,
            candidates=candidates,
            model_name=selected_reranker,
            revision=selected_revision,
            batch_size=reranker_batch_size,
            max_length=reranker_max_length,
            device=reranker_device,
        )
        retrieval_mode = f"{first_stage_mode}_rerank"
    else:
        ranked_candidates = []
        for candidate in candidates:
            item = dict(candidate)
            item["dense_rank"] = int(candidate.get("rank", 0))
            item["dense_score"] = float(candidate.get("score", 0.0))
            item["score_type"] = "dense_cosine_similarity"
            ranked_candidates.append(item)

    hits, estimated_tokens, selection_ms = _select_context_hits(
        ranked_candidates,
        context_k=context_k,
        max_context_tokens=max_context_tokens,
    )
    expansion_ms = 0
    if neighbor_expansion_distance:
        if max_context_tokens is None:
            raise RAGError(
                "Neighbor expansion requires an explicit max_context_tokens budget"
            )
        hits, estimated_tokens, expansion_ms = (
            _expand_context_hits_with_neighbors(
                hits,
                config=config,
                max_context_tokens=max_context_tokens,
                max_neighbor_distance=neighbor_expansion_distance,
            )
        )
        retrieval_mode = f"{retrieval_mode}_neighbor_expand"
    return RetrievalSelection(
        hits=hits,
        candidates=ranked_candidates,
        retrieval_mode=retrieval_mode,
        candidate_k=resolved_candidate_k,
        context_k=context_k,
        candidate_retrieval_latency_ms=retrieval_ms,
        reranker_latency_ms=reranker_ms,
        selection_latency_ms=selection_ms,
        estimated_context_tokens=estimated_tokens,
        token_estimator=TOKEN_ESTIMATOR_ID,
        reranker_model=selected_reranker,
        reranker_revision=selected_revision,
        expansion_latency_ms=expansion_ms,
        expanded_neighbor_count=sum(
            int(hit.get("expanded_neighbor_count", 0)) for hit in hits
        ),
    )


def _build_rag_prompt(
    question: str,
    hits: list[dict[str, Any]],
    conversation_history: list[dict[str, str]] | None = None,
    retrieval_confidence: dict[str, Any] | None = None,
    prompt_version: str = RAG_PROMPT_VERSION,
) -> str:
    """Construct grounded generation prompt with explicit citation markers."""

    if prompt_version not in SUPPORTED_RAG_PROMPT_VERSIONS:
        raise RAGError(f"Unsupported RAG prompt version: {prompt_version}")

    context_lines = []
    for hit in hits:
        metadata_bits = [
            f"domain={hit['domain']}",
            f"topic={hit.get('topic', '')}",
            f"legal_area={hit.get('legal_area', '')}",
            f"business_type={hit.get('business_type', '')}",
            f"authority={hit.get('authority', '')}",
            f"source_quality={hit.get('source_quality', '')}",
            f"jurisdiction={hit.get('jurisdiction', '')}",
            f"document_type={hit.get('document_type', '')}",
            f"language={hit.get('language', '')}",
            f"url={hit.get('url', '')}",
        ]
        context_lines.append(
            "\n".join(
                [
                    f"[S{hit['rank']}] source={hit['source_path']} chunk={hit['chunk_index']} "
                    + " ".join(bit for bit in metadata_bits if not bit.endswith("=")),
                    hit["text"],
                ]
            )
        )

    confidence_block = ""
    if retrieval_confidence:
        level = str(retrieval_confidence.get("level", "unrated"))
        reason = str(retrieval_confidence.get("reason", ""))
        confidence_block = (
            f"EVIDENCE SUFFICIENCY: {level}. {reason}\n"
            "When evidence sufficiency is limited or insufficient, answer only the "
            "parts directly established by the CONTEXT. Do not fill missing items "
            "from general knowledge or plausible analogy. For each unsupported item, "
            "state that the retrieved sources do not establish it.\n\n"
        )

    checklist_block = ""
    if prompt_version == RAG_PROMPT_VERSION:
        checklist_block = (
            "Before drafting, silently build an evidence checklist. Map every requested "
            "entity, dimension, stage, and comparison in the QUESTION to one or more "
            "source markers. Mark any item without adequate evidence as unresolved. Use "
            "this checklist to ensure that no subquestion is skipped, but do not print the "
            "internal checklist.\n"
            "For numerical comparisons, show the formula or arithmetic, preserve units, "
            "state any rounding, and do not infer missing inputs. Distinguish "
            "source-supported facts from your own inference or recommendation.\n\n"
        )

    history_block = ""
    if conversation_history:
        history_lines = [
            f"{item['role'].upper()}: {item['content']}"
            for item in conversation_history
        ]
        history_block = (
            "CONVERSATION HISTORY (context only; not factual evidence):\n"
            f"{'\n'.join(history_lines)}\n\n"
            "Use this history only to resolve references, constraints, and the user's "
            "ongoing intent. Previous assistant statements may be wrong: never cite "
            "them or treat them as evidence. Factual claims must remain grounded in "
            "the current retrieved CONTEXT.\n\n"
        )

    return (
        "You are a founder-assistance system for early-stage entrepreneurs in Germany.\n"
        "Use the retrieved CONTEXT as the primary evidence base. Prefer precise, "
        "context-supported statements over broad general advice.\n"
        "If the CONTEXT is insufficient for part of the question, say so clearly and "
        "explain what should be confirmed with the relevant authority or professional.\n"
        "Do not transfer a source's business-model-specific conclusion to a different "
        "business model merely because the cases use similar terms. If applicability "
        "is uncertain, state that limitation instead of treating the analogy as evidence.\n"
        "When you state a factual, legal, administrative, or numerical claim from the "
        "CONTEXT, cite at least one source marker like [S1]. Copy exact numbers and "
        "units from the CONTEXT when they are available.\n"
        "Do not invent legal requirements that are not supported by the CONTEXT.\n\n"
        f"{confidence_block}"
        "Language rule: answer strictly in the same language as the QUESTION. If the "
        "QUESTION is German, write fully in German and do not add English translations "
        "in parentheses. If the QUESTION is English, write fully in English; keep German "
        "legal and administrative terms such as GmbH, UG, GbR, Gewerbeamt, Finanzamt, "
        "IHK, and Handwerkskammer unchanged and explain them briefly in English only "
        "when needed. Do not mix German and English unless the user does so.\n\n"
        f"{checklist_block}"
        f"{history_block}"
        "Give a detailed, practical, and readable answer. For complex founder questions, "
        "cover the relevant conditions, exceptions, risks, and next checks instead of "
        "being overly brief. Do not force a step-by-step structure; use numbered actions "
        "only when the question asks for a process or when the answer naturally benefits "
        "from ordered actions. Use clean Markdown formatting.\n\n"
        f"QUESTION:\n{question}\n\n"
        f"CONTEXT:\n{'\n\n'.join(context_lines)}\n\n"
        "OUTPUT GUIDELINES:\n"
        "- Start with a direct answer to the user's concrete question.\n"
        "- Then explain the relevant requirements and reasoning in enough detail.\n"
        "- If the user asks several subquestions, answer each one explicitly.\n"
        "- Place citations close to the claims they support, using markers such as [S1].\n"
        "- Add a separate uncertainty/confirmation note only when something cannot be "
        "settled from the retrieved context.\n"
        "- End with a short final section in the answer language, for example **Fazit** "
        "for German or **Conclusion** for English."
    )


def _citation_payload(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return UI-safe citation metadata without full chunk text."""

    return [
        {
            "rank": hit["rank"],
            "score": hit["score"],
            "chunk_id": hit["chunk_id"],
            "domain": hit["domain"],
            "source_path": hit["source_path"],
            "source_name": hit["source_name"],
            "chunk_index": hit["chunk_index"],
            "text_preview": hit["text_preview"],
            "score_type": hit.get("score_type", "dense_cosine_similarity"),
            "dense_rank": hit.get("dense_rank", hit["rank"]),
            "dense_score": hit.get("dense_score", hit["score"]),
            "reranker_score": hit.get("reranker_score"),
            "reranker_model": hit.get("reranker_model", ""),
            "reranker_revision": hit.get("reranker_revision", ""),
            "estimated_context_tokens": hit.get("estimated_context_tokens", 0),
            "expanded_neighbor_count": hit.get("expanded_neighbor_count", 0),
            "expanded_neighbor_chunk_ids": hit.get(
                "expanded_neighbor_chunk_ids", []
            ),
            "expanded_neighbor_chunks": hit.get("expanded_neighbor_chunks", []),
            **{field: hit.get(field, False if field == "metadata_reviewed" else "") for field in CHUNK_METADATA_FIELDS},
        }
        for hit in hits
    ]


def rag_answer(
    question: str,
    top_k: int = 5,
    candidate_k: int | None = None,
    domain: str = "",
    system_prompt: str = "",
    temperature: float = 0.2,
    max_tokens: int = DEFAULT_ANSWER_MAX_TOKENS,
    model: str | None = None,
    reasoning_effort: str | None = None,
    overrides: dict[str, Any] | None = None,
    include_retrieved_text: bool = False,
    include_retrieval_candidates: bool = False,
    reranker_model: str | None = None,
    reranker_revision: str | None = None,
    max_context_tokens: int | None = None,
    allow_keyword_fallback: bool = True,
    neighbor_expansion_distance: int = 0,
    use_confidence_gating: bool = False,
    rag_prompt_version: str = RAG_PROMPT_VERSION,
    conversation_history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Public inference entrypoint for retrieval-augmented answering.

    Full retrieved chunk text is excluded from normal API responses. Formal
    evaluation can opt in so retrieval quality remains independently auditable.
    """

    config = _resolve_config(overrides)
    bounded_history, estimated_history_tokens = _normalize_conversation_history(
        conversation_history
    )
    retrieval_query = _build_history_aware_retrieval_query(
        question,
        bounded_history,
    )
    retrieval = _retrieve_context(
        query=retrieval_query,
        config=config,
        context_k=top_k,
        candidate_k=candidate_k,
        domain=domain,
        reranker_model=reranker_model,
        reranker_revision=reranker_revision,
        max_context_tokens=max_context_tokens,
        allow_keyword_fallback=allow_keyword_fallback,
        neighbor_expansion_distance=neighbor_expansion_distance,
    )
    hits = retrieval.hits
    retrieval_confidence = _assess_retrieval_confidence(
        hits,
        reranker_model=retrieval.reranker_model,
    )
    rag_prompt = _build_rag_prompt(
        question=question,
        hits=hits,
        conversation_history=bounded_history,
        retrieval_confidence=(
            retrieval_confidence if use_confidence_gating else None
        ),
        prompt_version=rag_prompt_version,
    )

    generation = generate_text(
        prompt=rag_prompt,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        model=model,
        reasoning_effort=reasoning_effort,
    )
    result = {
        **generation,
        "used_rag": True,
        "retrieval_latency_ms": retrieval.total_retrieval_latency_ms,
        "candidate_retrieval_latency_ms": (
            retrieval.candidate_retrieval_latency_ms
        ),
        "reranker_latency_ms": retrieval.reranker_latency_ms,
        "selection_latency_ms": retrieval.selection_latency_ms,
        "expansion_latency_ms": retrieval.expansion_latency_ms,
        "retrieval_mode": retrieval.retrieval_mode,
        "retrieval_domain": domain.strip(),
        "history_message_count": len(bounded_history),
        "estimated_history_tokens": estimated_history_tokens,
        "history_used_for_retrieval": bool(bounded_history),
        "history_token_estimator": HISTORY_TOKEN_ESTIMATOR_ID,
        "candidate_k": retrieval.candidate_k,
        "candidate_count": len(retrieval.candidates),
        "context_k": retrieval.context_k,
        "selected_chunk_count": len(hits),
        "estimated_context_tokens": retrieval.estimated_context_tokens,
        "token_estimator": retrieval.token_estimator,
        "reranker_model": retrieval.reranker_model,
        "reranker_revision": retrieval.reranker_revision,
        "expanded_neighbor_count": retrieval.expanded_neighbor_count,
        "rag_prompt_version": rag_prompt_version,
        "confidence_gating_applied": use_confidence_gating,
        "retrieval_confidence": retrieval_confidence,
        "citations": _citation_payload(hits),
    }
    if include_retrieved_text:
        result["retrieved_chunks"] = hits
    if include_retrieval_candidates:
        result["retrieval_candidates"] = retrieval.candidates
    return result


def rag_answer_stream(
    question: str,
    top_k: int = 5,
    candidate_k: int | None = None,
    domain: str = "",
    system_prompt: str = "",
    temperature: float = 0.2,
    max_tokens: int = DEFAULT_ANSWER_MAX_TOKENS,
    model: str | None = None,
    overrides: dict[str, Any] | None = None,
    reranker_model: str | None = None,
    reranker_revision: str | None = None,
    max_context_tokens: int | None = None,
    allow_keyword_fallback: bool = True,
    neighbor_expansion_distance: int = 0,
    use_confidence_gating: bool = False,
    rag_prompt_version: str = RAG_PROMPT_VERSION,
    conversation_history: list[dict[str, Any]] | None = None,
) -> Iterator[dict[str, Any]]:
    """Stream retrieval-augmented generation events for chat-style UIs."""

    config = _resolve_config(overrides)
    bounded_history, estimated_history_tokens = _normalize_conversation_history(
        conversation_history
    )
    retrieval_query = _build_history_aware_retrieval_query(
        question,
        bounded_history,
    )
    retrieval = _retrieve_context(
        query=retrieval_query,
        config=config,
        context_k=top_k,
        candidate_k=candidate_k,
        domain=domain,
        reranker_model=reranker_model,
        reranker_revision=reranker_revision,
        max_context_tokens=max_context_tokens,
        allow_keyword_fallback=allow_keyword_fallback,
        neighbor_expansion_distance=neighbor_expansion_distance,
    )
    hits = retrieval.hits
    retrieval_confidence = _assess_retrieval_confidence(
        hits,
        reranker_model=retrieval.reranker_model,
    )
    rag_prompt = _build_rag_prompt(
        question=question,
        hits=hits,
        conversation_history=bounded_history,
        retrieval_confidence=(
            retrieval_confidence if use_confidence_gating else None
        ),
        prompt_version=rag_prompt_version,
    )

    yield {
        "type": "metadata",
        "used_rag": True,
        "retrieval_latency_ms": retrieval.total_retrieval_latency_ms,
        "candidate_retrieval_latency_ms": (
            retrieval.candidate_retrieval_latency_ms
        ),
        "reranker_latency_ms": retrieval.reranker_latency_ms,
        "selection_latency_ms": retrieval.selection_latency_ms,
        "expansion_latency_ms": retrieval.expansion_latency_ms,
        "retrieval_mode": retrieval.retrieval_mode,
        "retrieval_domain": domain.strip(),
        "history_message_count": len(bounded_history),
        "estimated_history_tokens": estimated_history_tokens,
        "history_used_for_retrieval": bool(bounded_history),
        "history_token_estimator": HISTORY_TOKEN_ESTIMATOR_ID,
        "candidate_k": retrieval.candidate_k,
        "candidate_count": len(retrieval.candidates),
        "context_k": retrieval.context_k,
        "selected_chunk_count": len(hits),
        "estimated_context_tokens": retrieval.estimated_context_tokens,
        "token_estimator": retrieval.token_estimator,
        "reranker_model": retrieval.reranker_model,
        "reranker_revision": retrieval.reranker_revision,
        "expanded_neighbor_count": retrieval.expanded_neighbor_count,
        "rag_prompt_version": rag_prompt_version,
        "confidence_gating_applied": use_confidence_gating,
        "retrieval_confidence": retrieval_confidence,
        "citations": _citation_payload(hits),
    }

    for event in stream_generate_text(
        prompt=rag_prompt,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        model=model,
    ):
        yield event
