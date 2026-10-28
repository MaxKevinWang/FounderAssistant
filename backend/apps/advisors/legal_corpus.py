"""Deterministic extraction and selection for official German-law PDFs.

The generic RAG splitter is appropriate for guides and articles, but full statutes
need stricter boundaries. This module applies a versioned allowlist, ignores front
matter and tables of contents, and emits chunks that never cross a section boundary.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

DEFAULT_LEGAL_SCOPE_PATH = Path(__file__).with_name("legal_corpus_scope_v1.json")
LEGAL_EXTRACTOR_VERSION = "1.1"
LEGAL_TEXT_SEPARATORS = ["\n\n", ". ", "; ", ": ", ", ", " ", ""]

_SECTION_HEADER_RE = re.compile(r"^\u00a7\s*(?P<section>\d+[a-zA-Z]*)\b(?:\s+(?P<title>.*))?$")
_APPENDIX_HEADER_RE = re.compile(r"^Anlage\s+(?P<appendix>[A-Z])\b(?:\s*:?[ ]*(?P<title>.*))?$", re.IGNORECASE)
_REFERENCE_TITLE_RE = re.compile(
    r"^(?:Abs(?:atz|\.)?|Satz|Nummer|Nr\.?|Buchstabe|des\b|der\b|in\s+Verbindung\b)",
    re.IGNORECASE,
)
_STRUCTURE_HEADING_RE = re.compile(
    r"^(?:(?:Buch|Teil|Abschnitt|Unterabschnitt|Titel|Untertitel|Kapitel)\s+(?:\d+|[IVXLC]+)\b|"
    r"(?:Erster|Zweiter|Dritter|Vierter|Fünfter|Sechster|Siebter|Achter|Neunter|Zehnter)\s+"
    r"(?:Abschnitt|Unterabschnitt|Titel|Kapitel)\b)",
    re.IGNORECASE,
)
_MULTI_SECTION_BOUNDARY_RE = re.compile(r"^\u00a7{2}\s*\d", re.IGNORECASE)
_LETTERED_STRUCTURE_HEADING_RE = re.compile(r"^[A-Z]\.\s+\S")
_PARAGRAPH_START_RE = re.compile(r"^(?:\(\d+[a-z]?\)|\d+[.)]|[a-z][.)])\s*", re.IGNORECASE)
_PAGE_FOOTER_RE = re.compile(r"\s*-\s*Seite\s+\d+\s+von\s+\d+\s*-\s*$", re.IGNORECASE)
_STANDALONE_FOOTNOTE_MARKER_RE = re.compile(r"^\*+\)$")
_SERVICE_HEADER_MARKERS = (
    "ein service des bundesministerium",
    "ein service des bundesministeriums",
    "sowie des bundesamts für justiz",
    "www.gesetze-im-internet.de",
)


class LegalCorpusError(Exception):
    """Raised when the legal scope or an official-law PDF cannot be processed."""


@dataclass(frozen=True)
class LegalSourceProfile:
    """One manifest entry describing an official law and its selected scope."""

    source_path: str
    domain: str
    law_name: str
    law_abbreviation: str
    canonical_url: str
    authority: str
    legal_area: str
    retrieved_at: str
    temporal_status: str
    effective_from: str
    effective_to: str
    supersedes: str
    sections: tuple[str, ...]
    section_ranges: tuple[tuple[str, str], ...]
    appendices: tuple[str, ...]
    selection_reason: str

    def includes_section(self, section_id: str) -> bool:
        """Return whether a statutory section is part of this versioned scope."""

        normalized = section_id.casefold()
        if normalized in {item.casefold() for item in self.sections}:
            return True
        key = _section_sort_key(section_id)
        return any(_section_sort_key(start) <= key <= _section_sort_key(end) for start, end in self.section_ranges)

    def requested_section_labels(self, detected_ids: set[str]) -> list[str]:
        """Expand exact sections and ranges against IDs found in the source PDF."""

        selected = {section_id for section_id in detected_ids if self.includes_section(section_id)}
        selected.update(self.sections)
        return sorted(selected, key=_section_sort_key)

    def document_metadata(self) -> dict[str, Any]:
        """Return stable metadata shared by every chunk from this law."""

        return {
            "topic": "primary law for founder decisions",
            "legal_area": self.legal_area,
            "authority": self.authority,
            "source_quality": "primary_law",
            "language": "de",
            "jurisdiction": "Germany",
            "document_type": "primary_law",
            "url": self.canonical_url,
            "law_name": self.law_name,
            "law_abbreviation": self.law_abbreviation,
            "law_section": "",
            "effective_from": self.effective_from,
            "effective_to": self.effective_to,
            "retrieved_at": self.retrieved_at,
            "canonical_url": self.canonical_url,
            "temporal_status": self.temporal_status,
            "supersedes": self.supersedes,
        }


@dataclass(frozen=True)
class _LineRecord:
    text: str
    page_number: int


@dataclass(frozen=True)
class LegalUnit:
    """One selected statutory section or appendix before embedding splits."""

    unit_type: str
    unit_id: str
    title: str
    text: str
    page_start: int
    page_end: int

    @property
    def score(self) -> tuple[int, int, int, int]:
        """Prefer substantive body occurrences over short table-of-contents rows."""

        body_markers = len(re.findall(r"(?m)^\(\d+[a-z]?\)", self.text))
        sentence_markers = self.text.count(".")
        page_span = self.page_end - self.page_start
        return body_markers, page_span, len(self.text), sentence_markers


@dataclass(frozen=True)
class LegalChunk:
    """One selected statutory chunk with explicit legal provenance."""

    text: str
    page_start: str
    page_end: str
    section_title: str
    heading_path: str
    chunk_type: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class LegalExtractionResult:
    """Selected text, chunks, document metadata, and an extraction audit."""

    text: str
    units: list[LegalUnit]
    chunks: list[LegalChunk]
    document_metadata: dict[str, Any]
    audit: dict[str, Any]


def _section_sort_key(section_id: str) -> tuple[int, str]:
    match = re.fullmatch(r"(\d+)([a-zA-Z]*)", section_id.strip())
    if not match:
        raise LegalCorpusError(f"Invalid statutory section identifier: {section_id}")
    return int(match.group(1)), match.group(2).casefold()


@lru_cache(maxsize=4)
def _load_legal_scope_cached(manifest_path: str) -> tuple[str, dict[str, LegalSourceProfile]]:
    path = Path(manifest_path)
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LegalCorpusError(f"Failed to load legal corpus scope {path}: {exc}") from exc

    profiles: dict[str, LegalSourceProfile] = {}
    for item in payload.get("documents", []):
        profile = LegalSourceProfile(
            source_path=str(item["source_path"]),
            domain=str(item["domain"]),
            law_name=str(item["law_name"]),
            law_abbreviation=str(item["law_abbreviation"]),
            canonical_url=str(item["canonical_url"]),
            authority=str(item["authority"]),
            legal_area=str(item["legal_area"]),
            retrieved_at=str(item["retrieved_at"]),
            temporal_status=str(item["temporal_status"]),
            effective_from=str(item.get("effective_from", "")),
            effective_to=str(item.get("effective_to", "")),
            supersedes=str(item.get("supersedes", "")),
            sections=tuple(str(value) for value in item.get("sections", [])),
            section_ranges=tuple((str(start), str(end)) for start, end in item.get("section_ranges", [])),
            appendices=tuple(str(value).upper() for value in item.get("appendices", [])),
            selection_reason=str(item.get("selection_reason", "")),
        )
        if profile.source_path in profiles:
            raise LegalCorpusError(f"Duplicate source_path in legal corpus scope: {profile.source_path}")
        profiles[profile.source_path] = profile
    return hashlib.sha256(raw).hexdigest(), profiles


def load_legal_scope(manifest_path: Path | None = None) -> tuple[str, dict[str, LegalSourceProfile]]:
    """Load and validate the versioned legal-source manifest."""

    path = (manifest_path or DEFAULT_LEGAL_SCOPE_PATH).resolve()
    return _load_legal_scope_cached(str(path))


def legal_scope_sha256(manifest_path: Path | None = None) -> str:
    """Return the manifest hash stored with built indexes for reproducibility."""

    scope_hash, _ = load_legal_scope(manifest_path)
    return scope_hash


def get_legal_source_profile(
    relative_path: Path | str,
    manifest_path: Path | None = None,
) -> LegalSourceProfile | None:
    """Return the legal profile for a corpus-relative path, if configured."""

    _, profiles = load_legal_scope(manifest_path)
    normalized = Path(relative_path).as_posix()
    return profiles.get(normalized)


def _clean_pdf_page_lines(raw_text: str, page_number: int) -> list[_LineRecord]:
    """Remove recurring service/page furniture while preserving statutory text."""

    repaired = raw_text.replace("\x00", " ").replace("\u00ad", "").replace("\r", "\n")
    repaired = re.sub(r"([^\W\d_])-\s*\n\s*([^\W\d_])", r"\1\2", repaired, flags=re.UNICODE)
    records: list[_LineRecord] = []
    for raw_line in repaired.splitlines():
        line = re.sub(r"[ \t]+", " ", raw_line).strip()
        line = _PAGE_FOOTER_RE.sub("", line).strip()
        if not line:
            continue
        folded = line.casefold()
        if any(marker in folded for marker in _SERVICE_HEADER_MARKERS):
            continue
        records.append(_LineRecord(text=line, page_number=page_number))
    return records


def _section_header(line: str) -> tuple[str, str] | None:
    match = _SECTION_HEADER_RE.match(line)
    if not match:
        return None
    title = (match.group("title") or "").strip()
    if title and _REFERENCE_TITLE_RE.match(title):
        return None
    if title and (title[0].islower() or title.endswith(".")):
        return None
    return match.group("section"), title


def _appendix_header(line: str) -> tuple[str, str] | None:
    match = _APPENDIX_HEADER_RE.match(line)
    if not match:
        return None
    title = (match.group("title") or "").strip()
    if title.casefold().startswith(
        ("oder ", "und ", "nach ", "in der ", "in dem ", "abschnitt ")
    ):
        return None
    return match.group("appendix").upper(), title


def _trim_non_normative_tail(lines: list[_LineRecord]) -> list[_LineRecord]:
    """Discard amendment footnotes and hierarchy headings attached to the prior section."""

    for index, record in enumerate(lines):
        footnote_match = re.search(r"\bFußnote\b", record.text, flags=re.IGNORECASE)
        if footnote_match:
            prefix = record.text[: footnote_match.start()].strip()
            lines = lines[:index]
            if prefix:
                lines.append(_LineRecord(text=prefix, page_number=record.page_number))
            break
    while lines and (
        _STRUCTURE_HEADING_RE.match(lines[-1].text)
        or _LETTERED_STRUCTURE_HEADING_RE.match(lines[-1].text)
        or lines[-1].text.casefold() in {"nichtamtliches inhaltsverzeichnis", "inhaltsübersicht"}
    ):
        lines.pop()
    return lines


def _is_wrapped_title_continuation(title: str, candidate: str) -> bool:
    """Recognize a PDF line that visually continues a long statutory heading."""

    if not title or not candidate or _PARAGRAPH_START_RE.match(candidate):
        return False
    if candidate[0].islower():
        return True
    folded_title = title.casefold().rstrip()
    if title.endswith((",", ";", ":")):
        return True
    return folded_title.endswith((" der", " des", " die", " von", " für", " und", " sowie"))


def _format_legal_unit(
    profile: LegalSourceProfile,
    unit_type: str,
    unit_id: str,
    title: str,
    records: list[_LineRecord],
) -> tuple[str, str]:
    """Join visual PDF lines into stable statutory paragraphs."""

    body_records = list(records[1:])
    if unit_type == "section":
        body_records = _trim_non_normative_tail(body_records)
    else:
        body_records = _remove_appendix_source_note(body_records)
        while body_records and _is_appendix_title_continuation(body_records[0].text):
            title = f"{title} {body_records.pop(0).text}".strip()
    while body_records and _is_wrapped_title_continuation(title, body_records[0].text):
        title = f"{title} {body_records.pop(0).text}".strip()
    paragraphs: list[str] = []
    current = ""
    for record in body_records:
        line = record.text.strip()
        if not line or line.casefold() == "nichtamtliches inhaltsverzeichnis":
            continue
        if _STANDALONE_FOOTNOTE_MARKER_RE.fullmatch(line):
            continue
        starts_new = bool(_PARAGRAPH_START_RE.match(line))
        if starts_new and current:
            paragraphs.append(current.strip())
            current = line
        elif current:
            current = f"{current} {line}"
        else:
            current = line
    if current:
        paragraphs.append(current.strip())

    if unit_type == "section":
        heading = f"{profile.law_abbreviation} \u00a7 {unit_id}"
    else:
        heading = f"{profile.law_abbreviation} Anlage {unit_id}"
    if title:
        heading = f"{heading} {title}"
    return "\n\n".join([heading, *paragraphs]).strip(), title


def _remove_appendix_source_note(lines: list[_LineRecord]) -> list[_LineRecord]:
    """Remove a wrapped ``Fundstelle`` note without dropping the appendix table."""

    cleaned: list[_LineRecord] = []
    inside_source_note = False
    for record in lines:
        line = record.text
        marker_index = line.casefold().find("(fundstelle:")
        if marker_index >= 0:
            prefix = line[:marker_index].strip()
            if prefix:
                cleaned.append(_LineRecord(text=prefix, page_number=record.page_number))
            inside_source_note = "fussnote" not in line.casefold()
            continue
        if inside_source_note:
            if "fussnote" in line.casefold():
                inside_source_note = False
            continue
        cleaned.append(record)
    return cleaned


def _is_appendix_title_continuation(candidate: str) -> bool:
    """Recognize an uppercase continuation of a wrapped appendix heading."""

    folded = candidate.casefold().strip()
    if not folded or candidate.startswith("(") or candidate[:1].isdigit():
        return False
    return not folded.startswith(("nummer", "abschnitt"))


def _find_occurrences(
    records: list[_LineRecord],
    profile: LegalSourceProfile,
) -> tuple[dict[str, list[LegalUnit]], dict[str, list[LegalUnit]]]:
    """Find all section/appendix occurrences, including short TOC duplicates."""

    markers: list[tuple[int, str, str, str]] = []
    inside_appendix = False
    for index, record in enumerate(records):
        appendix = _appendix_header(record.text)
        if appendix:
            markers.append((index, "appendix", appendix[0], appendix[1]))
            inside_appendix = True
            continue
        section = _section_header(record.text)
        if section:
            markers.append((index, "section", section[0], section[1]))
            inside_appendix = False
            continue
        if not inside_appendix and (
            _MULTI_SECTION_BOUNDARY_RE.match(record.text)
            or _STRUCTURE_HEADING_RE.match(record.text)
        ):
            markers.append((index, "boundary", "", ""))

    sections: dict[str, list[LegalUnit]] = {}
    appendices: dict[str, list[LegalUnit]] = {}
    for marker_index, (start, unit_type, unit_id, title) in enumerate(markers):
        end = markers[marker_index + 1][0] if marker_index + 1 < len(markers) else len(records)
        if unit_type == "boundary":
            continue
        unit_records = records[start:end]
        if not unit_records:
            continue
        text, normalized_title = _format_legal_unit(profile, unit_type, unit_id, title, unit_records)
        occurrence = LegalUnit(
            unit_type=unit_type,
            unit_id=unit_id,
            title=normalized_title,
            text=text,
            page_start=unit_records[0].page_number,
            page_end=unit_records[-1].page_number,
        )
        target = sections if unit_type == "section" else appendices
        target.setdefault(unit_id, []).append(occurrence)
    return sections, appendices


def _best_occurrence(occurrences: list[LegalUnit]) -> LegalUnit:
    """Resolve TOC/body duplicates by choosing the most substantive occurrence."""

    return max(occurrences, key=lambda occurrence: occurrence.score)


def _split_legal_occurrence(
    occurrence: LegalUnit,
    profile: LegalSourceProfile,
    chunk_size: int,
    chunk_overlap: int,
) -> list[LegalChunk]:
    """Split only inside an oversized section and repeat its legal heading."""

    heading, _, body = occurrence.text.partition("\n\n")
    if not body or len(occurrence.text) <= int(chunk_size * 1.15):
        parts = [body] if body else [""]
    else:
        body_size = max(240, chunk_size - len(heading) - 2)
        overlap = min(chunk_overlap, max(0, body_size // 5))
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=body_size,
            chunk_overlap=overlap,
            separators=LEGAL_TEXT_SEPARATORS,
        )
        parts = [part.strip() for part in splitter.split_text(body) if part.strip()]

    if occurrence.unit_type == "section":
        law_section = f"\u00a7 {occurrence.unit_id}"
        chunk_type = "legal_section"
    else:
        law_section = f"Anlage {occurrence.unit_id}"
        chunk_type = "legal_appendix"
    section_title = heading.removeprefix(f"{profile.law_abbreviation} ").strip()
    metadata = {
        **profile.document_metadata(),
        "law_section": law_section,
    }
    return [
        LegalChunk(
            text=f"{heading}\n\n{part}".strip(),
            page_start=str(occurrence.page_start),
            page_end=str(occurrence.page_end),
            section_title=section_title,
            heading_path=f"{profile.law_abbreviation} > {section_title}",
            chunk_type=chunk_type,
            metadata=metadata,
        )
        for part in parts
    ]


def extract_legal_pdf(
    file_path: Path,
    profile: LegalSourceProfile,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> LegalExtractionResult:
    """Extract only selected sections from one official German-law PDF."""

    try:
        reader = PdfReader(str(file_path))
    except Exception as exc:  # pragma: no cover - defensive handling for broken PDFs
        raise LegalCorpusError(f"Failed to read legal PDF {file_path}: {exc}") from exc

    records: list[_LineRecord] = []
    raw_char_count = 0
    for page_number, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        raw_char_count += len(raw)
        records.extend(_clean_pdf_page_lines(raw, page_number))

    section_occurrences, appendix_occurrences = _find_occurrences(records, profile)
    detected_ids = set(section_occurrences)
    requested_ids = profile.requested_section_labels(detected_ids)
    missing_sections = [section_id for section_id in requested_ids if section_id not in section_occurrences]

    selected_occurrences: list[LegalUnit] = []
    skipped_repealed: list[str] = []
    for section_id in requested_ids:
        occurrences = section_occurrences.get(section_id)
        if not occurrences:
            continue
        selected = _best_occurrence(occurrences)
        if "weggefallen" in selected.text.casefold() and len(selected.text) < 250:
            skipped_repealed.append(section_id)
            continue
        selected_occurrences.append(selected)

    missing_appendices: list[str] = []
    for appendix_id in profile.appendices:
        occurrences = appendix_occurrences.get(appendix_id)
        if not occurrences:
            missing_appendices.append(appendix_id)
            continue
        selected_occurrences.append(_best_occurrence(occurrences))

    selected_occurrences.sort(
        key=lambda occurrence: (
            occurrence.page_start,
            0 if occurrence.unit_type == "section" else 1,
            _section_sort_key(occurrence.unit_id) if occurrence.unit_type == "section" else (0, occurrence.unit_id),
        )
    )
    chunks = [
        chunk
        for occurrence in selected_occurrences
        for chunk in _split_legal_occurrence(
            occurrence,
            profile,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    ]
    selected_text = "\n\n".join(occurrence.text for occurrence in selected_occurrences).strip()
    duplicate_occurrences = sum(
        max(0, len(section_occurrences.get(section_id, [])) - 1) for section_id in requested_ids
    ) + sum(max(0, len(appendix_occurrences.get(appendix_id, [])) - 1) for appendix_id in profile.appendices)
    audit = {
        "source_path": profile.source_path,
        "source_sha256": hashlib.sha256(file_path.read_bytes()).hexdigest(),
        "law_name": profile.law_name,
        "law_abbreviation": profile.law_abbreviation,
        "page_count": len(reader.pages),
        "raw_char_count": raw_char_count,
        "selected_char_count": len(selected_text),
        "detected_section_count": len(detected_ids),
        "selected_section_count": sum(1 for item in selected_occurrences if item.unit_type == "section"),
        "selected_sections": [item.unit_id for item in selected_occurrences if item.unit_type == "section"],
        "selected_appendices": [item.unit_id for item in selected_occurrences if item.unit_type == "appendix"],
        "missing_selected_sections": missing_sections,
        "missing_selected_appendices": missing_appendices,
        "skipped_repealed_sections": skipped_repealed,
        "duplicate_toc_or_reference_occurrences_resolved": duplicate_occurrences,
        "chunk_count": len(chunks),
        "front_matter_or_toc_indexed": False,
        "selection_reason": profile.selection_reason,
    }
    return LegalExtractionResult(
        text=selected_text,
        units=selected_occurrences,
        chunks=chunks,
        document_metadata=profile.document_metadata(),
        audit=audit,
    )


def build_legal_corpus_audit(
    data_dir: Path,
    *,
    chunk_size: int,
    chunk_overlap: int,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Run extraction without embedding and return a corpus-level audit payload."""

    scope_hash, profiles = load_legal_scope(manifest_path)
    documents: list[dict[str, Any]] = []
    for profile in profiles.values():
        source_file = data_dir / Path(profile.source_path)
        if not source_file.exists():
            documents.append(
                {
                    "source_path": profile.source_path,
                    "law_abbreviation": profile.law_abbreviation,
                    "error": "source_file_missing",
                }
            )
            continue
        result = extract_legal_pdf(
            source_file,
            profile,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        documents.append(result.audit)

    return {
        "scope_id": "startup_primary_law_scope_v1",
        "scope_sha256": scope_hash,
        "generated_at": datetime.now(UTC).isoformat(),
        "data_dir": str(data_dir),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "document_count": len(documents),
        "successful_document_count": sum("error" not in document for document in documents),
        "selected_section_count": sum(document.get("selected_section_count", 0) for document in documents),
        "chunk_count": sum(document.get("chunk_count", 0) for document in documents),
        "documents": documents,
    }
