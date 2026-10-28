"""Utility helpers for a minimal RAG web-ingestion notebook workflow.

Design goal:
- Keep notebook cells minimal (import + URL + run).
- Reuse existing backend crawler command.
- Separate "crawl + auto classify" from "RAG incremental indexing".
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import numpy as np
from django.core.management import call_command

from .rag_pipeline import (
    CHUNK_METADATA_FIELDS,
    EXTRACTED_TEXT_DIRNAME,
    INDEX_EMBEDDINGS_FILENAME,
    INDEX_METADATA_FILENAME,
    SOURCE_MANIFEST_FILENAME,
    RAGError,
    _build_chunks_for_source_files,
    _collect_source_files,
    _embed_texts,
    _extract_document,
    _resolve_config,
    build_rag_index,
)
from .legal_corpus import LEGAL_EXTRACTOR_VERSION, legal_scope_sha256

_TRACKING_QUERY_KEYS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "utm_id",
    "gclid",
    "fbclid",
    "mc_cid",
    "mc_eid",
}

_DOMAIN_KEYWORDS = {
    "legal_form_selection": [
        "gmbh",
        "unternehmergesellschaft",
        "ug (haftungsbeschränkt)",
        "legal form",
        "limited liability",
        "company registration",
        "amtsgericht",
        "commercial register",
    ],
    "funding_preparation": [
        "funding",
        "grant",
        "subsidy",
        "investor",
        "loan",
        "financing",
        "cashflow",
        "venture capital",
        "seed round",
    ],
    "market_validation": [
        "consumer",
        "customers",
        "ecommerce",
        "market",
        "demand",
        "selling",
        "contracts",
        "guarantees",
        "product safety",
        "distance selling",
        "dispute",
    ],
}

_ALLOWED_TOP_LEVEL_DOMAINS = {
    "legal_form_selection",
    "funding_preparation",
    "market_validation",
}

def _canonicalize_url(url: str) -> str:
    parsed = urlparse(url)
    query_pairs = []
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        if key.lower() in _TRACKING_QUERY_KEYS:
            continue
        query_pairs.append((key, value))
    query = urlencode(sorted(query_pairs)) if query_pairs else ""
    return urlunparse((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path, "", query, ""))


def _extract_source_url_from_html(html_path: Path) -> str | None:
    text_head = html_path.read_text(encoding="utf-8", errors="ignore")[:6000]
    match = re.search(r"source_url:\s*(\S+)", text_head)
    if not match:
        return None
    return match.group(1).strip()


def _guess_domain_from_content(source_url: str, extracted_text: str) -> str:
    url_lc = source_url.lower()
    text_lc = extracted_text.lower()

    scores = {domain: 0 for domain in _DOMAIN_KEYWORDS}
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        for kw in keywords:
            if kw in url_lc:
                scores[domain] += 2
            if kw in text_lc:
                scores[domain] += 1

    best_domain = max(scores, key=scores.get)
    if scores[best_domain] == 0:
        return "market_validation"
    return best_domain


def _find_new_staging_file(staging_dir: Path, before: set[Path]) -> Path:
    after = set(staging_dir.glob("*.html"))
    candidates = [path for path in after if path not in before]
    if not candidates:
        raise RAGError("Crawler did not produce a new HTML file in staging folder.")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _cleanup_empty_staging_dir(staging_dir: Path) -> None:
    if not staging_dir.exists():
        return
    if any(staging_dir.iterdir()):
        return
    try:
        staging_dir.rmdir()
    except OSError:
        pass


def _unique_destination_path(destination_dir: Path, filename: str) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    target = destination_dir / filename
    if not target.exists():
        return target
    stem = target.stem
    suffix = target.suffix
    index = 2
    while True:
        candidate = destination_dir / f"{stem}__{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def _remove_older_duplicates(target_root_dir: Path, canonical_source_url: str) -> list[Path]:
    matches: list[tuple[Path, float]] = []
    for html_file in target_root_dir.rglob("*.html"):
        source_url = _extract_source_url_from_html(html_file)
        if not source_url:
            continue
        if _canonicalize_url(source_url) == canonical_source_url:
            matches.append((html_file, html_file.stat().st_mtime))

    if len(matches) <= 1:
        return []

    matches.sort(key=lambda item: item[1], reverse=True)
    removed: list[Path] = []
    for old_file, _ in matches[1:]:
        old_file.unlink(missing_ok=True)
        removed.append(old_file)
    return removed


def _find_existing_canonical_file(data_dir: Path, canonical_source_url: str) -> Path | None:
    matches: list[tuple[Path, float]] = []
    for html_file in data_dir.rglob("*.html"):
        source_url = _extract_source_url_from_html(html_file)
        if not source_url:
            continue
        if _canonicalize_url(source_url) == canonical_source_url:
            matches.append((html_file, html_file.stat().st_mtime))
    if not matches:
        return None
    matches.sort(key=lambda item: item[1], reverse=True)
    return matches[0][0]


def crawl_and_auto_classify_url(
    url: str,
    staging_domain: str = "market_validation",
    force_refresh: bool = False,
) -> dict[str, Any]:
    """Crawl one URL and auto-classify it into the corpus structure.

    Input:
    - `url`: page URL provided by user.
    Output:
    - dictionary with saved path, inferred domain/subfolder, and dedup info.
    """

    config = _resolve_config()
    data_dir = config.data_dir
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise RAGError(f"Invalid URL: {url}")
    if staging_domain not in _ALLOWED_TOP_LEVEL_DOMAINS:
        allowed = ", ".join(sorted(_ALLOWED_TOP_LEVEL_DOMAINS))
        raise RAGError(f"staging_domain must be one of: {allowed}")

    canonical_input = _canonicalize_url(url)
    existing = _find_existing_canonical_file(data_dir=data_dir, canonical_source_url=canonical_input)
    if existing is not None and not force_refresh:
        relative = existing.relative_to(data_dir)
        inferred_subfolder = relative.parts[1] if len(relative.parts) >= 3 else ""
        return {
            "status": "ok",
            "input_url": url,
            "canonical_source_url": canonical_input,
            "already_exists": True,
            "saved_path": str(existing),
            "saved_bytes": existing.stat().st_size,
            "inferred_domain": relative.parts[0] if relative.parts else "",
            "inferred_subfolder": inferred_subfolder,
            "dedup_removed_count": 0,
            "dedup_removed_paths": [],
        }

    staging_dir = data_dir / staging_domain / "web_sources"
    staging_dir.mkdir(parents=True, exist_ok=True)

    before = set(staging_dir.glob("*.html"))
    call_command("crawl_web_pages", url, domain=staging_domain)
    new_file = _find_new_staging_file(staging_dir, before)

    source_url = _extract_source_url_from_html(new_file) or url
    extracted_text = _extract_document(
        new_file,
        chunk_size=config.chunk_size,
        min_chunk_chars=config.min_chunk_chars,
    ).text
    inferred_domain = _guess_domain_from_content(source_url=source_url, extracted_text=extracted_text)
    if inferred_domain not in _ALLOWED_TOP_LEVEL_DOMAINS:
        inferred_domain = "market_validation"
    inferred_subfolder = ""

    destination_dir = data_dir / inferred_domain
    destination_path = _unique_destination_path(destination_dir, new_file.name)
    new_file.rename(destination_path)
    _cleanup_empty_staging_dir(staging_dir)

    removed = _remove_older_duplicates(
        target_root_dir=data_dir,
        canonical_source_url=_canonicalize_url(source_url),
    )

    return {
        "status": "ok",
        "input_url": url,
        "source_url": source_url,
        "canonical_source_url": _canonicalize_url(source_url),
        "inferred_domain": inferred_domain,
        "inferred_subfolder": inferred_subfolder,
        "saved_path": str(destination_path),
        "saved_bytes": destination_path.stat().st_size,
        "dedup_removed_count": len(removed),
        "dedup_removed_paths": [str(path) for path in removed],
    }


def _load_existing_index(config) -> tuple[dict[str, Any], np.ndarray] | None:
    metadata_path = config.index_dir / INDEX_METADATA_FILENAME
    embeddings_path = config.index_dir / INDEX_EMBEDDINGS_FILENAME
    if not metadata_path.exists() or not embeddings_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    embeddings = np.load(embeddings_path)
    chunk_count = len(metadata.get("chunks", []))
    if embeddings.ndim != 2 or embeddings.shape[0] != chunk_count:
        raise RAGError("Existing index files are inconsistent. Run full rebuild with build_rag_index().")
    return metadata, embeddings


def _manifest_path(config) -> Path:
    return config.index_dir / SOURCE_MANIFEST_FILENAME


def _source_signature(file_path: Path) -> dict[str, int]:
    stat = file_path.stat()
    return {
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _build_source_manifest(config) -> dict[str, dict[str, int]]:
    manifest: dict[str, dict[str, int]] = {}
    for source_file in _collect_source_files(config.data_dir):
        source_path = source_file.relative_to(config.data_dir).as_posix()
        manifest[source_path] = _source_signature(source_file)
    return manifest


def _load_source_manifest(config) -> dict[str, dict[str, int]] | None:
    path = _manifest_path(config)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    return payload.get("sources", None)


def _save_source_manifest(config, sources: dict[str, dict[str, int]]) -> None:
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "data_dir": str(config.data_dir),
        "source_count": len(sources),
        "sources": sources,
    }
    _manifest_path(config).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _detect_manifest_delta(
    current: dict[str, dict[str, int]],
    recorded: dict[str, dict[str, int]],
    indexed_paths: set[str] | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """Compare corpus state without treating a stale manifest as unindexed data.

    `metadata.json` is authoritative for whether a document already has chunks.
    The source manifest is authoritative only for change detection, because older
    full builds did not always refresh it.
    """

    current_paths = set(current.keys())
    recorded_paths = set(recorded.keys())
    baseline_paths = indexed_paths if indexed_paths is not None else recorded_paths
    added = sorted(current_paths - baseline_paths)
    removed = sorted(baseline_paths - current_paths)
    changed = sorted(
        source_path
        for source_path in (current_paths & recorded_paths & baseline_paths)
        if current[source_path] != recorded[source_path]
    )
    return added, removed, changed


def _write_new_extracted_texts(config, docs_with_text: list[dict[str, Any]]) -> None:
    extracted_dir = config.index_dir / EXTRACTED_TEXT_DIRNAME
    extracted_dir.mkdir(parents=True, exist_ok=True)
    for doc in docs_with_text:
        out_name = doc["source_path"].replace("/", "__").replace("\\", "__")
        (extracted_dir / f"{out_name}.txt").write_text(doc["text"], encoding="utf-8")


def _run_full_rebuild(
    *,
    config,
    overrides: dict[str, Any] | None,
    reason: str,
    delta: dict[str, int] | None = None,
) -> dict[str, Any]:
    full_result = build_rag_index(overrides=overrides)
    current_manifest = _build_source_manifest(config)
    _save_source_manifest(config=config, sources=current_manifest)
    return {
        "status": "ok",
        "mode": "full_rebuild",
        "reason": reason,
        "delta": delta or {"added": 0, "removed": 0, "changed": 0},
        **full_result,
    }


def build_rag_index_for_new_documents(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run RAG indexing for new data, with safe fallback to full rebuild.

    Behavior:
    - No index yet: full initial build.
    - Only new source files: append new chunks/embeddings incrementally.
    - Removed or changed existing source files: full rebuild to keep consistency.
    """

    config = _resolve_config(overrides)
    started = datetime.now(UTC)
    current_manifest = _build_source_manifest(config)

    loaded = _load_existing_index(config)
    if loaded is None:
        return _run_full_rebuild(
            config=config,
            overrides=overrides,
            reason="index_missing",
            delta={"added": len(current_manifest), "removed": 0, "changed": 0},
        )

    metadata, embeddings = loaded
    current_legal_scope_hash = legal_scope_sha256()
    has_indexed_primary_law = any(
        document.get("document_type") == "primary_law" for document in metadata.get("documents", [])
    )
    if has_indexed_primary_law and metadata.get("legal_corpus_scope_sha256") != current_legal_scope_hash:
        return _run_full_rebuild(
            config=config,
            overrides=overrides,
            reason="legal_corpus_scope_changed",
        )
    if has_indexed_primary_law and metadata.get("legal_extractor_version") != LEGAL_EXTRACTOR_VERSION:
        return _run_full_rebuild(
            config=config,
            overrides=overrides,
            reason="legal_extractor_changed",
        )
    if metadata.get("embedding_model") != config.embedding_model:
        return _run_full_rebuild(
            config=config,
            overrides=overrides,
            reason="embedding_model_changed",
        )
    if int(metadata.get("chunk_size", -1)) != config.chunk_size:
        return _run_full_rebuild(
            config=config,
            overrides=overrides,
            reason="chunk_size_changed",
        )
    if int(metadata.get("chunk_overlap", -1)) != config.chunk_overlap:
        return _run_full_rebuild(
            config=config,
            overrides=overrides,
            reason="chunk_overlap_changed",
        )
    if int(metadata.get("min_chunk_chars", -1)) != config.min_chunk_chars:
        return _run_full_rebuild(
            config=config,
            overrides=overrides,
            reason="min_chunk_chars_changed",
        )

    recorded_manifest = _load_source_manifest(config)
    if recorded_manifest is None:
        return _run_full_rebuild(
            config=config,
            overrides=overrides,
            reason="manifest_missing",
            delta={"added": len(current_manifest), "removed": 0, "changed": 0},
        )

    indexed_paths = {
        str(document.get("source_path"))
        for document in metadata.get("documents", [])
        if document.get("source_path")
    }
    added_paths, removed_paths, changed_paths = _detect_manifest_delta(
        current=current_manifest,
        recorded=recorded_manifest,
        indexed_paths=indexed_paths,
    )
    manifest_healed_count = len((set(current_manifest) & indexed_paths) - set(recorded_manifest))
    delta = {
        "added": len(added_paths),
        "removed": len(removed_paths),
        "changed": len(changed_paths),
        "manifest_healed": manifest_healed_count,
    }

    if removed_paths or changed_paths:
        return _run_full_rebuild(
            config=config,
            overrides=overrides,
            reason="source_removed_or_changed",
            delta=delta,
        )

    if not added_paths:
        if manifest_healed_count:
            _save_source_manifest(config=config, sources=current_manifest)
        return {
            "status": "ok",
            "mode": "incremental",
            "reason": "no_new_sources_manifest_healed" if manifest_healed_count else "no_new_sources",
            "delta": delta,
            "new_document_count": 0,
            "new_chunk_count": 0,
            "document_count": metadata.get("document_count", 0),
            "chunk_count": metadata.get("chunk_count", 0),
            "duration_ms": 0,
        }

    source_files_by_path = {
        path.relative_to(config.data_dir).as_posix(): path for path in _collect_source_files(config.data_dir)
    }
    new_files = [source_files_by_path[source_path] for source_path in added_paths if source_path in source_files_by_path]

    new_chunks, new_docs_with_text = _build_chunks_for_source_files(config, new_files)

    if not new_chunks:
        _save_source_manifest(config=config, sources=current_manifest)
        return {
            "status": "ok",
            "mode": "incremental",
            "reason": "new_sources_but_no_chunks",
            "delta": delta,
            "new_document_count": 0,
            "new_chunk_count": 0,
            "document_count": metadata.get("document_count", 0),
            "chunk_count": metadata.get("chunk_count", 0),
            "duration_ms": 0,
        }

    new_embeddings = _embed_texts([chunk.text for chunk in new_chunks], config=config)
    merged_embeddings = np.vstack([embeddings, new_embeddings]).astype(np.float32)

    _write_new_extracted_texts(config=config, docs_with_text=new_docs_with_text)

    metadata.setdefault("documents", [])
    metadata.setdefault("chunks", [])
    metadata["documents"].extend(
        {
            "domain": doc["domain"],
            "source_path": doc["source_path"],
            "source_name": doc["source_name"],
            "char_count": doc["char_count"],
            "chunk_count": doc["chunk_count"],
            **{
                field: doc.get(field, False if field == "metadata_reviewed" else "")
                for field in CHUNK_METADATA_FIELDS
            },
        }
        for doc in new_docs_with_text
    )
    metadata["chunks"].extend(
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
        for chunk in new_chunks
    )
    metadata["generated_at"] = datetime.now(UTC).isoformat()
    metadata["legal_corpus_scope_sha256"] = legal_scope_sha256()
    metadata["legal_extractor_version"] = LEGAL_EXTRACTOR_VERSION
    metadata["document_count"] = len(metadata["documents"])
    metadata["chunk_count"] = len(metadata["chunks"])

    metadata_path = config.index_dir / INDEX_METADATA_FILENAME
    embeddings_path = config.index_dir / INDEX_EMBEDDINGS_FILENAME
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    np.save(embeddings_path, merged_embeddings)
    _save_source_manifest(config=config, sources=current_manifest)

    duration_ms = int((datetime.now(UTC) - started).total_seconds() * 1000)
    return {
        "status": "ok",
        "mode": "incremental",
        "reason": "new_sources_indexed",
        "delta": delta,
        "new_document_count": len(new_docs_with_text),
        "new_chunk_count": len(new_chunks),
        "document_count": metadata["document_count"],
        "chunk_count": metadata["chunk_count"],
        "duration_ms": duration_ms,
    }
