"""FAQ dataset deduplication utilities.

This module applies a three-layer dedup strategy to the crawled
Existenzgruendungsportal FAQ dataset:
1) Exact duplicates (canonical URL + normalized QA hash)
2) Lexical near-duplicates (token-overlap candidate generation + sequence ratio)
3) Semantic near-duplicates (embedding cosine similarity)

Outputs are designed for traceability in thesis reporting:
- cleaned dataset JSON
- dedup map JSON (removed -> kept)
- review CSV for manual inspection of near-duplicate pairs
- summary Markdown
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import numpy as np
from django.conf import settings

from .rag_pipeline import RAGError


@dataclass(frozen=True)
class FAQDedupConfig:
    """Runtime options for FAQ deduplication."""

    input_json_path: Path
    output_clean_json_path: Path
    output_map_json_path: Path
    output_review_csv_path: Path
    output_summary_md_path: Path
    embedding_model: str
    llm_base_url: str
    timeout_seconds: int
    lexical_threshold: float
    lexical_review_margin: float
    semantic_threshold: float
    semantic_review_margin: float
    min_shared_tokens: int
    max_token_doc_frequency: int
    max_length_delta_ratio: float
    require_same_domain_for_semantic: bool


class _UnionFind:
    """Simple union-find structure for duplicate cluster building."""

    def __init__(self, size: int) -> None:
        self._parent = list(range(size))
        self._rank = [0] * size

    def find(self, idx: int) -> int:
        parent = self._parent[idx]
        if parent != idx:
            self._parent[idx] = self.find(parent)
        return self._parent[idx]

    def union(self, a: int, b: int) -> bool:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return False
        rank_a = self._rank[ra]
        rank_b = self._rank[rb]
        if rank_a < rank_b:
            self._parent[ra] = rb
        elif rank_a > rank_b:
            self._parent[rb] = ra
        else:
            self._parent[rb] = ra
            self._rank[ra] += 1
        return True


def _default_input_json_path() -> Path:
    return (
        Path(settings.BASE_DIR).parent
        / "docs"
        / "problem_datasets"
        / "existenzgruendung_faq_problem_dataset.json"
    )


def _default_output_clean_json_path() -> Path:
    return (
        Path(settings.BASE_DIR).parent
        / "docs"
        / "problem_datasets"
        / "dedup"
        / "existenzgruendung_faq_problem_dataset_dedup_clean.json"
    )


def _default_output_map_json_path() -> Path:
    return (
        Path(settings.BASE_DIR).parent
        / "docs"
        / "problem_datasets"
        / "dedup"
        / "existenzgruendung_faq_problem_dataset_dedup_map.json"
    )


def _default_output_review_csv_path() -> Path:
    return (
        Path(settings.BASE_DIR).parent
        / "docs"
        / "problem_datasets"
        / "dedup"
        / "existenzgruendung_faq_problem_dataset_dedup_review.csv"
    )


def _default_output_summary_md_path() -> Path:
    return (
        Path(settings.BASE_DIR).parent
        / "docs"
        / "problem_datasets"
        / "dedup"
        / "existenzgruendung_faq_problem_dataset_dedup_summary.md"
    )


def default_faq_dedup_config() -> FAQDedupConfig:
    """Build default config from project settings."""

    return FAQDedupConfig(
        input_json_path=_default_input_json_path(),
        output_clean_json_path=_default_output_clean_json_path(),
        output_map_json_path=_default_output_map_json_path(),
        output_review_csv_path=_default_output_review_csv_path(),
        output_summary_md_path=_default_output_summary_md_path(),
        embedding_model=str(getattr(settings, "LOCAL_RAG_EMBED_MODEL", "bge-m3")),
        llm_base_url=str(getattr(settings, "LOCAL_LLM_BASE_URL", "http://127.0.0.1:11434")),
        timeout_seconds=int(getattr(settings, "LOCAL_LLM_TIMEOUT_SECONDS", 360)),
        lexical_threshold=0.90,
        lexical_review_margin=0.05,
        semantic_threshold=0.92,
        semantic_review_margin=0.03,
        min_shared_tokens=3,
        max_token_doc_frequency=350,
        max_length_delta_ratio=0.25,
        require_same_domain_for_semantic=True,
    )


def _load_dataset(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RAGError(f"FAQ dataset not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise RAGError("Invalid FAQ dataset format: expected top-level object with 'items' list.")
    return payload


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _canonicalize_url(url: str) -> str:
    if not url:
        return ""
    parts = urlsplit(url.strip())
    normalized = urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path, "", ""))
    return normalized


def _normalize_text(text: str) -> str:
    if not text:
        return ""
    value = text.lower()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _tokenize(text: str) -> set[str]:
    tokens = [token for token in re.split(r"[^\w]+", text.lower()) if len(token) >= 3]
    return set(tokens)


def _qa_hash(question: str, answer: str) -> str:
    payload = f"{_normalize_text(question)}\n{_normalize_text(answer)}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _post_json(url: str, payload: dict[str, Any], timeout_seconds: int) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="ignore")
        except Exception:
            body = ""
        raise RAGError(f"Embedding service HTTP error: {exc} {body}".strip()) from exc
    except urllib.error.URLError as exc:
        raise RAGError(f"Embedding service connection error: {exc}") from exc


def _embed_questions(texts: list[str], config: FAQDedupConfig, batch_size: int = 32) -> np.ndarray:
    def _sanitize_for_embedding(text: str) -> str:
        cleaned = re.sub(r"\s+", " ", text or "").strip()
        cleaned = cleaned[:800]
        if cleaned:
            return cleaned
        return "startup founder faq question"

    def _is_finite_vector(vector: list[float]) -> bool:
        array = np.asarray(vector, dtype=np.float32)
        return bool(np.isfinite(array).all())

    def _embed_single_with_fallback(text: str) -> list[float]:
        candidates = [
            text,
            _sanitize_for_embedding(text),
            re.sub(r"[^\w\s]", " ", _sanitize_for_embedding(text)),
            "startup founder faq question",
        ]
        seen: set[str] = set()
        for candidate in candidates:
            value = re.sub(r"\s+", " ", candidate).strip()
            if not value:
                value = "startup founder faq question"
            if value in seen:
                continue
            seen.add(value)
            try:
                response = _post_json(
                    f"{config.llm_base_url}/api/embed",
                    {"model": config.embedding_model, "input": [value]},
                    config.timeout_seconds,
                )
            except RAGError:
                continue
            embeddings = response.get("embeddings", [])
            if len(embeddings) != 1:
                continue
            vector = embeddings[0]
            if _is_finite_vector(vector):
                return vector
        raise RAGError("Failed to embed one FAQ question after fallback attempts.")

    def _embed_batch_with_fallback(batch: list[str]) -> list[list[float]]:
        payload = {"model": config.embedding_model, "input": batch}
        try:
            response = _post_json(
                f"{config.llm_base_url}/api/embed",
                payload,
                config.timeout_seconds,
            )
            embeddings = response.get("embeddings", [])
            if len(embeddings) != len(batch):
                raise RAGError(
                    f"Embedding size mismatch: expected {len(batch)}, got {len(embeddings)}."
                )
            for vector in embeddings:
                if not _is_finite_vector(vector):
                    raise RAGError("Embedding response contains non-finite values.")
            return embeddings
        except RAGError:
            # If batch embedding fails (e.g., one bad sample causing provider NaN),
            # split recursively to isolate problematic rows and recover with fallback.
            if len(batch) == 1:
                return [_embed_single_with_fallback(batch[0])]
            mid = len(batch) // 2
            left = _embed_batch_with_fallback(batch[:mid])
            right = _embed_batch_with_fallback(batch[mid:])
            return left + right

    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        vectors.extend(_embed_batch_with_fallback(batch))

    array = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return array / norms


def _item_priority(item: dict[str, Any], original_index: int) -> tuple[int, int, int, int]:
    """Higher tuple value means better representative candidate."""

    stand_year = item.get("stand_year")
    try:
        year_value = int(stand_year) if stand_year is not None else -1
    except Exception:
        year_value = -1

    answer_len = len(str(item.get("reference_answer", "")))
    question_len = len(str(item.get("question", "")))
    # Prefer older index only as final tie-breaker for deterministic output.
    return (year_value, answer_len, question_len, -original_index)


def _build_markdown_summary(payload: dict[str, Any]) -> str:
    stats = payload.get("stats", {})
    lines = [
        "# FAQ Deduplication Summary",
        "",
        f"- Generated at (UTC): {payload.get('generated_at_utc', '')}",
        f"- Input dataset: {payload.get('input_dataset_path', '')}",
        f"- Output clean dataset: {payload.get('output_clean_dataset_path', '')}",
        "",
        "## Counts",
        "",
        f"- Original items: {stats.get('original_items', 0)}",
        f"- Cleaned items: {stats.get('cleaned_items', 0)}",
        f"- Removed items: {stats.get('removed_items', 0)}",
        f"- Duplicate clusters (>1 item): {stats.get('cluster_count_multi', 0)}",
        "",
        "## Merge Reasons",
        "",
        f"- Exact URL merges: {stats.get('exact_url_merges', 0)}",
        f"- Exact QA merges: {stats.get('exact_qa_merges', 0)}",
        f"- Lexical merges: {stats.get('lexical_merges', 0)}",
        f"- Semantic merges: {stats.get('semantic_merges', 0)}",
        "",
        "## Review Rows",
        "",
        f"- Lexical review rows: {stats.get('lexical_review_rows', 0)}",
        f"- Semantic review rows: {stats.get('semantic_review_rows', 0)}",
        "",
    ]
    return "\n".join(lines) + "\n"


def deduplicate_faq_dataset(config: FAQDedupConfig) -> dict[str, Any]:
    """Run full FAQ dedup pipeline and persist all outputs."""

    started = time.perf_counter()
    dataset = _load_dataset(config.input_json_path)
    items = list(dataset.get("items", []))
    if not items:
        raise RAGError("FAQ dataset has no items to deduplicate.")

    total = len(items)
    uf = _UnionFind(total)

    # Precompute normalized fields for all records.
    normalized_questions = [_normalize_text(str(item.get("question", ""))) for item in items]
    normalized_answers = [_normalize_text(str(item.get("reference_answer", ""))) for item in items]
    canonical_urls = [_canonicalize_url(str(item.get("source_url", ""))) for item in items]
    token_sets = [_tokenize(question) for question in normalized_questions]

    merge_events: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    stats = {
        "original_items": total,
        "exact_url_merges": 0,
        "exact_qa_merges": 0,
        "lexical_merges": 0,
        "semantic_merges": 0,
        "lexical_review_rows": 0,
        "semantic_review_rows": 0,
    }

    def register_merge(i: int, j: int, reason: str, score: float | None = None) -> None:
        merged = uf.union(i, j)
        if merged:
            if reason == "exact_url":
                stats["exact_url_merges"] += 1
            elif reason == "exact_qa":
                stats["exact_qa_merges"] += 1
            elif reason == "lexical":
                stats["lexical_merges"] += 1
            elif reason == "semantic":
                stats["semantic_merges"] += 1
        merge_events.append(
            {
                "left_index": i,
                "right_index": j,
                "left_id": str(items[i].get("id", "")),
                "right_id": str(items[j].get("id", "")),
                "reason": reason,
                "score": score,
                "merged_cluster": merged,
            }
        )

    # Layer 1a: exact duplicate by canonical URL.
    first_by_url: dict[str, int] = {}
    for idx, url in enumerate(canonical_urls):
        if not url:
            continue
        leader = first_by_url.get(url)
        if leader is None:
            first_by_url[url] = idx
            continue
        register_merge(leader, idx, reason="exact_url", score=1.0)

    # Layer 1b: exact duplicate by normalized QA hash.
    first_by_hash: dict[str, int] = {}
    for idx in range(total):
        digest = _qa_hash(normalized_questions[idx], normalized_answers[idx])
        leader = first_by_hash.get(digest)
        if leader is None:
            first_by_hash[digest] = idx
            continue
        register_merge(leader, idx, reason="exact_qa", score=1.0)

    # Layer 2: lexical near-duplicates.
    token_document_frequency: Counter[str] = Counter()
    for tokens in token_sets:
        token_document_frequency.update(tokens)

    token_index: dict[str, list[int]] = defaultdict(list)
    for idx, tokens in enumerate(token_sets):
        for token in tokens:
            if token_document_frequency[token] <= config.max_token_doc_frequency:
                token_index[token].append(idx)

    for idx in range(total):
        if not normalized_questions[idx]:
            continue
        candidate_overlap: Counter[int] = Counter()
        for token in token_sets[idx]:
            if token_document_frequency[token] > config.max_token_doc_frequency:
                continue
            for other in token_index[token]:
                if other <= idx:
                    continue
                candidate_overlap[other] += 1

        base_length = max(1, len(normalized_questions[idx]))
        for other, shared_tokens in candidate_overlap.items():
            if shared_tokens < config.min_shared_tokens:
                continue

            other_length = max(1, len(normalized_questions[other]))
            length_delta = abs(base_length - other_length) / max(base_length, other_length)
            if length_delta > config.max_length_delta_ratio:
                continue

            ratio = SequenceMatcher(None, normalized_questions[idx], normalized_questions[other]).ratio()
            if ratio >= (config.lexical_threshold - config.lexical_review_margin):
                review_rows.append(
                    {
                        "pair_type": "lexical",
                        "score": round(float(ratio), 6),
                        "status": "auto_merged" if ratio >= config.lexical_threshold else "review_candidate",
                        "left_id": str(items[idx].get("id", "")),
                        "right_id": str(items[other].get("id", "")),
                        "same_domain": str(items[idx].get("decision_domain", ""))
                        == str(items[other].get("decision_domain", "")),
                        "left_question": str(items[idx].get("question", "")),
                        "right_question": str(items[other].get("question", "")),
                        "left_url": str(items[idx].get("source_url", "")),
                        "right_url": str(items[other].get("source_url", "")),
                    }
                )
                stats["lexical_review_rows"] += 1

            if ratio >= config.lexical_threshold:
                register_merge(idx, other, reason="lexical", score=float(ratio))

    # Layer 3: semantic near-duplicates.
    embeddings = _embed_questions(normalized_questions, config=config)
    similarities = embeddings @ embeddings.T

    for idx in range(total):
        candidate_offsets = np.where(similarities[idx, idx + 1 :] >= (config.semantic_threshold - config.semantic_review_margin))[0]
        if len(candidate_offsets) == 0:
            continue
        candidates = (candidate_offsets + idx + 1).tolist()

        for other in candidates:
            score = float(similarities[idx, other])
            same_domain = str(items[idx].get("decision_domain", "")) == str(
                items[other].get("decision_domain", "")
            )

            if config.require_same_domain_for_semantic and not same_domain:
                continue

            if score >= (config.semantic_threshold - config.semantic_review_margin):
                review_rows.append(
                    {
                        "pair_type": "semantic",
                        "score": round(score, 6),
                        "status": "auto_merged" if score >= config.semantic_threshold else "review_candidate",
                        "left_id": str(items[idx].get("id", "")),
                        "right_id": str(items[other].get("id", "")),
                        "same_domain": same_domain,
                        "left_question": str(items[idx].get("question", "")),
                        "right_question": str(items[other].get("question", "")),
                        "left_url": str(items[idx].get("source_url", "")),
                        "right_url": str(items[other].get("source_url", "")),
                    }
                )
                stats["semantic_review_rows"] += 1

            if score >= config.semantic_threshold:
                register_merge(idx, other, reason="semantic", score=score)

    # Build clusters.
    root_to_indices: dict[int, list[int]] = defaultdict(list)
    for idx in range(total):
        root_to_indices[uf.find(idx)].append(idx)

    representative_by_root: dict[int, int] = {}
    for root, member_indices in root_to_indices.items():
        best_idx = max(member_indices, key=lambda index: _item_priority(items[index], index))
        representative_by_root[root] = best_idx

    cleaned_items: list[dict[str, Any]] = []
    removed_rows: list[dict[str, Any]] = []

    # Build quick lookup for reasons by index pair.
    reasons_by_pair: dict[tuple[int, int], set[str]] = defaultdict(set)
    scores_by_pair: dict[tuple[int, int], list[float]] = defaultdict(list)
    for event in merge_events:
        a = int(event["left_index"])
        b = int(event["right_index"])
        pair = (min(a, b), max(a, b))
        reasons_by_pair[pair].add(str(event["reason"]))
        score = event.get("score")
        if isinstance(score, (float, int)):
            scores_by_pair[pair].append(float(score))

    for root, member_indices in root_to_indices.items():
        keep_idx = representative_by_root[root]
        cleaned_items.append(items[keep_idx])

        for member in member_indices:
            if member == keep_idx:
                continue

            pair = (min(member, keep_idx), max(member, keep_idx))
            reasons = sorted(reasons_by_pair.get(pair, set()))
            if not reasons:
                reasons = ["cluster_link"]

            score_values = scores_by_pair.get(pair, [])
            removed_rows.append(
                {
                    "removed_id": str(items[member].get("id", "")),
                    "kept_id": str(items[keep_idx].get("id", "")),
                    "removed_index": member,
                    "kept_index": keep_idx,
                    "reasons": reasons,
                    "max_score": (max(score_values) if score_values else None),
                    "removed_source_url": str(items[member].get("source_url", "")),
                    "kept_source_url": str(items[keep_idx].get("source_url", "")),
                }
            )

    cleaned_items = sorted(cleaned_items, key=lambda row: str(row.get("id", "")))
    removed_rows = sorted(removed_rows, key=lambda row: str(row.get("removed_id", "")))

    cleaned_dataset = {
        "metadata": {
            **dataset.get("metadata", {}),
            "deduplication": {
                "generated_at_utc": datetime.now(UTC).isoformat(),
                "method": "exact_url + exact_qa + lexical + semantic",
                "lexical_threshold": config.lexical_threshold,
                "semantic_threshold": config.semantic_threshold,
                "min_shared_tokens": config.min_shared_tokens,
                "max_token_doc_frequency": config.max_token_doc_frequency,
                "max_length_delta_ratio": config.max_length_delta_ratio,
                "require_same_domain_for_semantic": config.require_same_domain_for_semantic,
            },
            "original_record_count": total,
            "record_count": len(cleaned_items),
            "removed_record_count": len(removed_rows),
        },
        "items": cleaned_items,
    }

    dedup_map_payload = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "input_dataset_path": str(config.input_json_path),
        "output_clean_dataset_path": str(config.output_clean_json_path),
        "stats": {
            **stats,
            "cleaned_items": len(cleaned_items),
            "removed_items": len(removed_rows),
            "cluster_count_total": len(root_to_indices),
            "cluster_count_multi": sum(1 for members in root_to_indices.values() if len(members) > 1),
        },
        "removed_mappings": removed_rows,
        "merge_events": merge_events,
    }

    summary_payload = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "input_dataset_path": str(config.input_json_path),
        "output_clean_dataset_path": str(config.output_clean_json_path),
        "stats": dedup_map_payload["stats"],
    }

    _save_json(config.output_clean_json_path, cleaned_dataset)
    _save_json(config.output_map_json_path, dedup_map_payload)

    config.output_review_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with config.output_review_csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        fieldnames = [
            "pair_type",
            "score",
            "status",
            "left_id",
            "right_id",
            "same_domain",
            "left_question",
            "right_question",
            "left_url",
            "right_url",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in review_rows:
            writer.writerow(row)

    config.output_summary_md_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_summary_md_path.write_text(
        _build_markdown_summary(summary_payload),
        encoding="utf-8",
    )

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    return {
        "status": "ok",
        "input_json_path": str(config.input_json_path),
        "output_clean_json_path": str(config.output_clean_json_path),
        "output_map_json_path": str(config.output_map_json_path),
        "output_review_csv_path": str(config.output_review_csv_path),
        "output_summary_md_path": str(config.output_summary_md_path),
        "original_items": total,
        "cleaned_items": len(cleaned_items),
        "removed_items": len(removed_rows),
        "cluster_count_total": len(root_to_indices),
        "cluster_count_multi": sum(1 for members in root_to_indices.values() if len(members) > 1),
        "exact_url_merges": stats["exact_url_merges"],
        "exact_qa_merges": stats["exact_qa_merges"],
        "lexical_merges": stats["lexical_merges"],
        "semantic_merges": stats["semantic_merges"],
        "lexical_review_rows": stats["lexical_review_rows"],
        "semantic_review_rows": stats["semantic_review_rows"],
        "duration_ms": elapsed_ms,
    }
