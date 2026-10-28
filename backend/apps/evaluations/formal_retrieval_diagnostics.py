"""Build reproducible retrieval diagnostics from frozen formal RAG traces."""

from __future__ import annotations

import hashlib
import json
import re
import statistics
from collections import Counter, defaultdict
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

from django.conf import settings


DEFAULT_SOURCE_NAMES = (
    "FORMAL-CORE-REMAINING-RAG-001.json",
    "FORMAL-GEMMA4-31B-RAG-001.json",
    "FORMAL-KIMI-K3-RAG-001.json",
)


class FormalRetrievalDiagnosticsError(RuntimeError):
    """Raised when formal retrieval traces cannot be compared safely."""


def project_root() -> Path:
    return Path(settings.BASE_DIR).parent


def default_source_paths() -> list[Path]:
    root = project_root() / "docs/evaluation/runs/formal"
    return [root / name for name in DEFAULT_SOURCE_NAMES]


def default_rag_scores_path() -> Path:
    root = project_root() / "docs/evaluation/runs/formal/judge/codex"
    run_id = "FORMAL-CODEX56-JUDGE-001"
    return root / run_id / f"{run_id}_scores.json"


def default_baseline_scores_path() -> Path:
    root = project_root() / "docs/evaluation/runs/formal/judge/codex"
    run_id = "FORMAL-CODEX56-BASELINE-JUDGE-001"
    return root / run_id / f"{run_id}_scores.json"


def default_output_dir() -> Path:
    return project_root() / "docs/evaluation/runs/formal/retrieval"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FormalRetrievalDiagnosticsError(
            f"Could not read JSON artifact {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise FormalRetrievalDiagnosticsError(f"Expected a JSON object: {path}")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(project_root().resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _round(value: float) -> float:
    return round(float(value), 4)


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return _round(statistics.mean(materialized)) if materialized else 0.0


def _rate(numerator: int, denominator: int) -> float:
    return _round(numerator / denominator) if denominator else 0.0


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _normalize_source_path(value: Any) -> str:
    normalized = str(value or "").strip().replace("\\", "/").lstrip("./")
    for prefix in ("knowledge_base/", "data/"):
        if normalized.casefold().startswith(prefix):
            normalized = normalized[len(prefix):]
            break
    return normalized.casefold()


def _find_subsequence(haystack: list[str], needle: list[str]) -> list[int]:
    if not needle or len(needle) > len(haystack):
        return []
    positions: list[int] = []
    width = len(needle)
    for start in range(len(haystack) - width + 1):
        if haystack[start : start + width] == needle:
            positions.append(start)
    return positions


def _reference_status(
    reference: dict[str, Any], chunks: list[dict[str, Any]]
) -> str:
    source_path = _normalize_source_path(reference.get("raw_source_path"))
    quote = _normalize_text(reference.get("quote"))
    same_document = [
        chunk
        for chunk in chunks
        if _normalize_source_path(chunk.get("source_path")) == source_path
    ]
    if quote and any(quote in _normalize_text(chunk.get("text")) for chunk in same_document):
        return "exact"
    if same_document:
        return "right_document_wrong_chunk"
    return "right_document_absent"


def _exact_evidence_words(
    references: list[dict[str, Any]], chunks: list[dict[str, Any]]
) -> tuple[int, int]:
    """Return exact-reference word positions and all retrieved word positions.

    This intentionally conservative EIR proxy marks only complete frozen quotes
    found as contiguous, whitespace-normalized words in the referenced source.
    """

    chunk_tokens: list[list[str]] = [
        _normalize_text(chunk.get("text")).split() for chunk in chunks
    ]
    matched_positions: list[set[int]] = [set() for _ in chunks]
    for reference in references:
        source_path = _normalize_source_path(reference.get("raw_source_path"))
        quote_tokens = _normalize_text(reference.get("quote")).split()
        for index, chunk in enumerate(chunks):
            if _normalize_source_path(chunk.get("source_path")) != source_path:
                continue
            for start in _find_subsequence(chunk_tokens[index], quote_tokens):
                matched_positions[index].update(
                    range(start, start + len(quote_tokens))
                )
    numerator = sum(len(positions) for positions in matched_positions)
    denominator = sum(len(tokens) for tokens in chunk_tokens)
    return numerator, denominator


def _load_panel(
    source_paths: Iterable[Path],
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
    str,
]:
    paths = [path.resolve() for path in source_paths]
    if not paths:
        raise FormalRetrievalDiagnosticsError("At least one source run is required")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FormalRetrievalDiagnosticsError(f"Source runs are missing: {missing}")

    runs = [(path, _load_json(path)) for path in paths]
    for path, run in runs:
        if run.get("status") != "completed":
            raise FormalRetrievalDiagnosticsError(f"Source run is incomplete: {path}")

    canonical_path, canonical_run = max(
        runs,
        key=lambda item: len(item[1].get("dataset", {}).get("items", [])),
    )
    dataset_items = canonical_run.get("dataset", {}).get("items", [])
    if not dataset_items:
        raise FormalRetrievalDiagnosticsError(
            f"Canonical source has no dataset items: {canonical_path}"
        )
    qra_by_id = {str(item.get("id", "")): item for item in dataset_items}
    if "" in qra_by_id or len(qra_by_id) != len(dataset_items):
        raise FormalRetrievalDiagnosticsError("Canonical dataset has invalid QRA IDs")

    manifest_sha = str(canonical_run.get("dataset", {}).get("manifest_sha256", ""))
    if not manifest_sha:
        raise FormalRetrievalDiagnosticsError("Dataset manifest SHA-256 is missing")
    canonical_fingerprint = _canonical_sha256(dataset_items)
    source_manifest: list[dict[str, Any]] = []
    panel_results: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for path, run in runs:
        other_items = run.get("dataset", {}).get("items", [])
        other_by_id = {str(item.get("id", "")): item for item in other_items}
        if "" in other_by_id or len(other_by_id) != len(other_items):
            raise FormalRetrievalDiagnosticsError(f"Invalid dataset IDs in {path}")
        for qra_id, item in other_by_id.items():
            if qra_id not in qra_by_id or _canonical_sha256(item) != _canonical_sha256(
                qra_by_id[qra_id]
            ):
                raise FormalRetrievalDiagnosticsError(
                    f"Dataset item {qra_id} differs from the canonical item in {path}"
                )
        if len(other_items) == len(dataset_items):
            other_manifest_sha = str(
                run.get("dataset", {}).get("manifest_sha256", "")
            )
            if (
                other_manifest_sha != manifest_sha
                or _canonical_sha256(other_items) != canonical_fingerprint
            ):
                raise FormalRetrievalDiagnosticsError(f"Full dataset differs in {path}")

        included = 0
        for result in run.get("results", []):
            if str(result.get("mode", "")) != "rag":
                continue
            if int(result.get("repeat", 1)) != 1:
                raise FormalRetrievalDiagnosticsError(
                    f"Only repeat 1 is supported: {path}"
                )
            generation = result.get("generation", {})
            if generation.get("status") != "success":
                raise FormalRetrievalDiagnosticsError(
                    f"Unsuccessful generation for {result.get('model')} / "
                    f"{result.get('qra_id')} in {path}"
                )
            model = str(result.get("model", ""))
            qra_id = str(result.get("qra_id", ""))
            key = (model, qra_id)
            if not model or qra_id not in qra_by_id:
                raise FormalRetrievalDiagnosticsError(f"Invalid result identity in {path}")
            if key in seen:
                raise FormalRetrievalDiagnosticsError(
                    f"Duplicate RAG result for {model} / {qra_id}"
                )
            chunks = generation.get("retrieved_chunks", [])
            if not isinstance(chunks, list) or not chunks:
                raise FormalRetrievalDiagnosticsError(
                    f"Missing retrieval trace for {model} / {qra_id}"
                )
            seen.add(key)
            included += 1
            panel_results.append(result)

        source_manifest.append(
            {
                "run_id": str(run.get("run_id", "")),
                "path": _display_path(path),
                "sha256": _file_sha256(path),
                "included_result_count": included,
                "configured_top_k": run.get("config", {}).get("top_k"),
                "index": run.get("config", {}).get("index"),
            }
        )

    models = sorted({str(result["model"]) for result in panel_results})
    expected_ids = set(qra_by_id)
    for model in models:
        actual_ids = {
            str(result["qra_id"])
            for result in panel_results
            if str(result["model"]) == model
        }
        if actual_ids != expected_ids:
            raise FormalRetrievalDiagnosticsError(
                f"Model {model} does not cover the complete canonical QRA panel"
            )

    return panel_results, qra_by_id, source_manifest, manifest_sha


def _answer_observation(
    result: dict[str, Any], qra: dict[str, Any]
) -> dict[str, Any]:
    generation = result["generation"]
    chunks = generation["retrieved_chunks"]
    references = qra.get("references", [])
    statuses = {
        str(reference["reference_id"]): _reference_status(reference, chunks)
        for reference in references
    }
    counts = Counter(statuses.values())
    keypoint_support: list[dict[str, Any]] = []
    for keypoint in qra.get("keypoints", []):
        reference_ids = [str(value) for value in keypoint.get("reference_ids", [])]
        exact_count = sum(statuses.get(value) == "exact" for value in reference_ids)
        if reference_ids and exact_count == len(reference_ids):
            support = "all"
        elif exact_count:
            support = "partial"
        else:
            support = "none"
        keypoint_support.append(
            {
                "keypoint_id": str(keypoint.get("id", "")),
                "support": support,
                "reference_count": len(reference_ids),
                "exact_reference_count": exact_count,
            }
        )

    evidence_words, retrieved_words = _exact_evidence_words(references, chunks)
    chunk_ids = [str(chunk.get("chunk_id", "")) for chunk in chunks]
    return {
        "model": str(result["model"]),
        "model_type": str(result.get("model_type", "")),
        "qra_id": str(result["qra_id"]),
        "domain": str(qra.get("domain", result.get("domain", ""))),
        "question_type": str(
            qra.get("question_type", result.get("question_type", ""))
        ),
        "question": str(qra.get("question", "")),
        "reference_count": len(references),
        "exact_reference_count": counts["exact"],
        "reference_recall": _rate(counts["exact"], len(references)),
        "reference_status_counts": {
            "exact": counts["exact"],
            "right_document_wrong_chunk": counts["right_document_wrong_chunk"],
            "right_document_absent": counts["right_document_absent"],
        },
        "keypoint_support": keypoint_support,
        "exact_evidence_word_count": evidence_words,
        "retrieved_word_count": retrieved_words,
        "exact_evidence_word_rate_proxy": _rate(evidence_words, retrieved_words),
        "retrieved_chunk_count": len(chunks),
        "retrieved_context_char_count": sum(
            len(str(chunk.get("text", ""))) for chunk in chunks
        ),
        "chunk_ids": chunk_ids,
        "finish_reason": str(generation.get("finish_reason", "")),
        "retrieval_mode": str(generation.get("retrieval_mode", "")),
    }


def _aggregate_answers(items: list[dict[str, Any]]) -> dict[str, Any]:
    reference_count = sum(int(item["reference_count"]) for item in items)
    exact_count = sum(int(item["exact_reference_count"]) for item in items)
    failures = {
        key: sum(int(item["reference_status_counts"][key]) for item in items)
        for key in (
            "exact",
            "right_document_wrong_chunk",
            "right_document_absent",
        )
    }
    support = Counter(
        keypoint["support"]
        for item in items
        for keypoint in item["keypoint_support"]
    )
    keypoint_count = sum(support.values())
    evidence_words = sum(int(item["exact_evidence_word_count"]) for item in items)
    retrieved_words = sum(int(item["retrieved_word_count"]) for item in items)
    return {
        "answer_count": len(items),
        "reference_count": reference_count,
        "exact_reference_count": exact_count,
        "macro_reference_recall_at_k": _mean(
            float(item["reference_recall"]) for item in items
        ),
        "evidence_weighted_reference_recall_at_k": _rate(
            exact_count, reference_count
        ),
        "reference_failure_layers": {
            key: {"count": value, "rate": _rate(value, reference_count)}
            for key, value in failures.items()
        },
        "keypoint_evidence_availability": {
            "keypoint_count": keypoint_count,
            **{
                key: {"count": support[key], "rate": _rate(support[key], keypoint_count)}
                for key in ("all", "partial", "none")
            },
        },
        "exact_evidence_word_rate_proxy": {
            "matched_word_count": evidence_words,
            "retrieved_word_count": retrieved_words,
            "pooled_rate": _rate(evidence_words, retrieved_words),
            "macro_answer_rate": _mean(
                float(item["exact_evidence_word_rate_proxy"]) for item in items
            ),
        },
        "retrieval_context": {
            "mean_chunk_count": _mean(
                float(item["retrieved_chunk_count"]) for item in items
            ),
            "mean_word_count": _mean(
                float(item["retrieved_word_count"]) for item in items
            ),
            "mean_char_count": _mean(
                float(item["retrieved_context_char_count"]) for item in items
            ),
        },
    }


def _grouped_answers(
    items: list[dict[str, Any]], group_fields: tuple[str, ...]
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        groups[tuple(str(item[field]) for field in group_fields)].append(item)
    output = []
    for key, group in sorted(groups.items()):
        output.append(
            {
                **dict(zip(group_fields, key, strict=True)),
                **_aggregate_answers(group),
            }
        )
    return output


def _load_score_index(
    path: Path | None,
    *,
    expected_mode: str,
    panel_keys: set[tuple[str, str]],
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any] | None]:
    if path is None:
        return {}, None
    resolved = path.resolve()
    payload = _load_json(resolved)
    if payload.get("status") != "completed":
        raise FormalRetrievalDiagnosticsError(f"Judge scores are incomplete: {resolved}")
    results = payload.get("results", [])
    if len(results) != int(payload.get("result_count", -1)):
        raise FormalRetrievalDiagnosticsError(
            f"Judge result count is inconsistent: {resolved}"
        )
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for item in results:
        if str(item.get("mode", "")) != expected_mode:
            raise FormalRetrievalDiagnosticsError(
                f"Expected only {expected_mode} Judge scores in {resolved}"
            )
        key = (str(item.get("model", "")), str(item.get("qra_id", "")))
        if key in index:
            raise FormalRetrievalDiagnosticsError(f"Duplicate Judge score for {key}")
        index[key] = item
    if set(index) != panel_keys:
        missing = sorted(panel_keys - set(index))[:10]
        extra = sorted(set(index) - panel_keys)[:10]
        raise FormalRetrievalDiagnosticsError(
            f"Judge scores do not match retrieval panel; missing={missing}, extra={extra}"
        )
    return index, {
        "judge_run_id": str(payload.get("judge_run_id", "")),
        "path": _display_path(resolved),
        "sha256": _file_sha256(resolved),
        "mode": expected_mode,
        "result_count": len(results),
    }


def _label_map(
    score: dict[str, Any], expected_ids: set[str]
) -> dict[str, str]:
    labels: dict[str, str] = {}
    for item in score.get("keypoint_labels", []):
        keypoint_id = str(item.get("keypoint_id", ""))
        label = str(item.get("label", ""))
        if keypoint_id in labels or label not in {"covered", "contradicted", "neither"}:
            raise FormalRetrievalDiagnosticsError(
                f"Invalid Judge keypoint label in candidate {score.get('candidate_id')}"
            )
        labels[keypoint_id] = label
    if set(labels) != expected_ids:
        raise FormalRetrievalDiagnosticsError(
            f"Judge keypoints do not match QRA {score.get('qra_id')}"
        )
    return labels


def _aggregate_labels(items: list[dict[str, Any]]) -> dict[str, Any]:
    labels = Counter(str(item["label"]) for item in items)
    total = len(items)
    return {
        "keypoint_observation_count": total,
        "covered": {"count": labels["covered"], "rate": _rate(labels["covered"], total)},
        "contradicted": {
            "count": labels["contradicted"],
            "rate": _rate(labels["contradicted"], total),
        },
        "neither": {"count": labels["neither"], "rate": _rate(labels["neither"], total)},
    }


def _grouped_labels(
    items: list[dict[str, Any]], group_fields: tuple[str, ...]
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        groups[tuple(str(item[field]) for field in group_fields)].append(item)
    return [
        {
            **dict(zip(group_fields, key, strict=True)),
            **_aggregate_labels(group),
        }
        for key, group in sorted(groups.items())
    ]


def _judge_conditioning(
    observations: list[dict[str, Any]],
    rag_scores: dict[tuple[str, str], dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    labeled: list[dict[str, Any]] = []
    for observation in observations:
        key = (observation["model"], observation["qra_id"])
        support_items = observation["keypoint_support"]
        expected_ids = {str(item["keypoint_id"]) for item in support_items}
        labels = _label_map(rag_scores[key], expected_ids)
        for item in support_items:
            labeled.append(
                {
                    "model": observation["model"],
                    "qra_id": observation["qra_id"],
                    "domain": observation["domain"],
                    "question_type": observation["question_type"],
                    "support": item["support"],
                    "keypoint_id": item["keypoint_id"],
                    "label": labels[item["keypoint_id"]],
                }
            )
    missing = [item for item in labeled if item["label"] != "covered"]
    missing_support = Counter(str(item["support"]) for item in missing)
    report = {
        "overall": _aggregate_labels(labeled),
        "by_evidence_support": _grouped_labels(labeled, ("support",)),
        "by_domain_and_support": _grouped_labels(labeled, ("domain", "support")),
        "by_question_type_and_support": _grouped_labels(
            labeled, ("question_type", "support")
        ),
        "by_model_and_support": _grouped_labels(labeled, ("model", "support")),
        "uncovered_or_contradicted_attribution": {
            "observation_count": len(missing),
            **{
                key: {
                    "count": missing_support[key],
                    "rate": _rate(missing_support[key], len(missing)),
                }
                for key in ("all", "partial", "none")
            },
        },
    }
    return report, labeled


def _aggregate_deltas(items: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(items)
    baseline_covered = sum(item["baseline_label"] == "covered" for item in items)
    rag_covered = sum(item["rag_label"] == "covered" for item in items)
    baseline_contradicted = sum(
        item["baseline_label"] == "contradicted" for item in items
    )
    rag_contradicted = sum(item["rag_label"] == "contradicted" for item in items)
    return {
        "keypoint_observation_count": count,
        "baseline_coverage_rate": _rate(baseline_covered, count),
        "rag_coverage_rate": _rate(rag_covered, count),
        "coverage_delta": _round(
            _rate(rag_covered, count) - _rate(baseline_covered, count)
        ),
        "baseline_contradiction_rate": _rate(baseline_contradicted, count),
        "rag_contradiction_rate": _rate(rag_contradicted, count),
        "contradiction_delta": _round(
            _rate(rag_contradicted, count) - _rate(baseline_contradicted, count)
        ),
    }


def _grouped_deltas(
    items: list[dict[str, Any]], group_fields: tuple[str, ...]
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        groups[tuple(str(item[field]) for field in group_fields)].append(item)
    return [
        {
            **dict(zip(group_fields, key, strict=True)),
            **_aggregate_deltas(group),
        }
        for key, group in sorted(groups.items())
    ]


def _baseline_to_rag_conditioning(
    labeled: list[dict[str, Any]],
    baseline_scores: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    baseline_maps: dict[tuple[str, str], dict[str, str]] = {}
    for item in labeled:
        key = (item["model"], item["qra_id"])
        if key not in baseline_maps:
            expected_ids = {
                row["keypoint_id"]
                for row in labeled
                if (row["model"], row["qra_id"]) == key
            }
            baseline_maps[key] = _label_map(baseline_scores[key], expected_ids)
    deltas = [
        {
            **item,
            "rag_label": item["label"],
            "baseline_label": baseline_maps[(item["model"], item["qra_id"])][
                item["keypoint_id"]
            ],
        }
        for item in labeled
    ]
    return {
        "overall": _aggregate_deltas(deltas),
        "by_evidence_support": _grouped_deltas(deltas, ("support",)),
        "by_domain_and_support": _grouped_deltas(
            deltas, ("domain", "support")
        ),
        "by_question_type_and_support": _grouped_deltas(
            deltas, ("question_type", "support")
        ),
        "by_model_and_support": _grouped_deltas(deltas, ("model", "support")),
    }


def _retrieval_agreement(observations: list[dict[str, Any]]) -> dict[str, Any]:
    by_qra: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in observations:
        by_qra[item["qra_id"]].append(item)
    details: list[dict[str, Any]] = []
    pairwise_scores: list[float] = []
    for qra_id, items in sorted(by_qra.items()):
        sequences: dict[tuple[str, ...], list[str]] = defaultdict(list)
        for item in items:
            sequences[tuple(item["chunk_ids"])].append(item["model"])
        local_scores = []
        for first, second in combinations(items, 2):
            first_set = set(first["chunk_ids"])
            second_set = set(second["chunk_ids"])
            union = first_set | second_set
            score = len(first_set & second_set) / len(union) if union else 1.0
            local_scores.append(score)
            pairwise_scores.append(score)
        details.append(
            {
                "qra_id": qra_id,
                "unique_ranked_sequence_count": len(sequences),
                "mean_pairwise_chunk_set_jaccard": _mean(local_scores),
                "sequences": [
                    {"models": sorted(models), "chunk_ids": list(sequence)}
                    for sequence, models in sorted(
                        sequences.items(), key=lambda entry: sorted(entry[1])
                    )
                ],
            }
        )
    divergent = [item for item in details if item["unique_ranked_sequence_count"] > 1]
    return {
        "qra_count": len(details),
        "identical_ranked_sequence_qra_count": len(details) - len(divergent),
        "divergent_qra_count": len(divergent),
        "mean_pairwise_chunk_set_jaccard": _mean(pairwise_scores),
        "divergent_qras": divergent,
    }


def _run_health(observations: list[dict[str, Any]]) -> dict[str, Any]:
    chunks = Counter(int(item["retrieved_chunk_count"]) for item in observations)
    reasons = Counter(str(item["finish_reason"]) for item in observations)
    modes = Counter(str(item["retrieval_mode"]) for item in observations)
    return {
        "successful_answer_count": len(observations),
        "retrieved_chunk_count_distribution": dict(sorted(chunks.items())),
        "finish_reason_distribution": dict(sorted(reasons.items())),
        "retrieval_mode_distribution": dict(sorted(modes.items())),
        "mean_context_char_count": _mean(
            float(item["retrieved_context_char_count"]) for item in observations
        ),
    }


def _question_diagnostics(
    observations: list[dict[str, Any]],
    labeled: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    label_by_qra: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in labeled:
        label_by_qra[item["qra_id"]].append(item)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in observations:
        groups[item["qra_id"]].append(item)
    output = []
    for qra_id, items in groups.items():
        summary = _aggregate_answers(items)
        row = {
            "qra_id": qra_id,
            "domain": items[0]["domain"],
            "question_type": items[0]["question_type"],
            "question": items[0]["question"],
            **summary,
        }
        if label_by_qra[qra_id]:
            row["rag_judge"] = _aggregate_labels(label_by_qra[qra_id])
        output.append(row)
    return sorted(
        output,
        key=lambda item: (
            item["macro_reference_recall_at_k"],
            item["domain"],
            item["qra_id"],
        ),
    )


def _percent(value: Any) -> str:
    return f"{float(value) * 100:.2f}%"


def _markdown(report: dict[str, Any]) -> str:
    panel = report["panel"]
    overall = report["retrieval"]["overall"]
    layers = overall["reference_failure_layers"]
    lines = [
        f"# {report['report_id']}: Formal Retrieval Diagnostics",
        "",
        "> Status: post-hoc exploratory diagnostic. The exposed formal set may guide",
        "> engineering, but it must not be reused as the sole confirmatory test set.",
        "",
        "## Panel and method",
        "",
        f"- {panel['model_count']} models 脳 {panel['qra_count']} QRA = "
        f"{panel['answer_count']} successful RAG traces.",
        f"- Frozen dataset manifest: `{panel['dataset_manifest_sha256']}`.",
        "- Exact recall requires the complete whitespace-normalized frozen quote in a",
        "  retrieved chunk from the referenced source document.",
        "- The EIR proxy marks only words belonging to those complete exact quotes;",
        "  paraphrases and alternative valid evidence are intentionally not counted.",
        "",
        "## Main retrieval result",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Macro Reference Recall@K | {_percent(overall['macro_reference_recall_at_k'])} |",
        "| Evidence-weighted Reference Recall@K | "
        f"{_percent(overall['evidence_weighted_reference_recall_at_k'])} |",
        f"| Exact reference layer | {_percent(layers['exact']['rate'])} |",
        "| Right document, wrong chunk | "
        f"{_percent(layers['right_document_wrong_chunk']['rate'])} |",
        "| Referenced document absent | "
        f"{_percent(layers['right_document_absent']['rate'])} |",
        "| Exact-evidence word-rate proxy | "
        f"{_percent(overall['exact_evidence_word_rate_proxy']['pooled_rate'])} |",
        "",
        "## By domain",
        "",
        "| Domain | Answers | Macro Recall@K | Weighted Recall@K | Wrong chunk | Document absent | All keypoint refs | EIR proxy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["retrieval"]["by_domain"]:
        availability = row["keypoint_evidence_availability"]
        failure = row["reference_failure_layers"]
        lines.append(
            f"| {row['domain']} | {row['answer_count']} | "
            f"{_percent(row['macro_reference_recall_at_k'])} | "
            f"{_percent(row['evidence_weighted_reference_recall_at_k'])} | "
            f"{_percent(failure['right_document_wrong_chunk']['rate'])} | "
            f"{_percent(failure['right_document_absent']['rate'])} | "
            f"{_percent(availability['all']['rate'])} | "
            f"{_percent(row['exact_evidence_word_rate_proxy']['pooled_rate'])} |"
        )

    lines.extend(
        [
            "",
            "## By question type",
            "",
            "| Question type | Macro Recall@K | Weighted Recall@K | Wrong chunk | Document absent |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in report["retrieval"]["by_question_type"]:
        failure = row["reference_failure_layers"]
        lines.append(
            f"| {row['question_type']} | "
            f"{_percent(row['macro_reference_recall_at_k'])} | "
            f"{_percent(row['evidence_weighted_reference_recall_at_k'])} | "
            f"{_percent(failure['right_document_wrong_chunk']['rate'])} | "
            f"{_percent(failure['right_document_absent']['rate'])} |"
        )

    judge = report.get("rag_judge_conditioned_on_evidence")
    if judge:
        lines.extend(
            [
                "",
                "## RAG answer quality conditioned on retrieved evidence",
                "",
                "| Exact keypoint-reference support | Observations | Covered | Contradicted |",
                "|---|---:|---:|---:|",
            ]
        )
        for row in judge["by_evidence_support"]:
            lines.append(
                f"| {row['support']} | {row['keypoint_observation_count']} | "
                f"{_percent(row['covered']['rate'])} | "
                f"{_percent(row['contradicted']['rate'])} |"
            )
        attribution = judge["uncovered_or_contradicted_attribution"]
        lines.extend(
            [
                "",
                f"Of {attribution['observation_count']} uncovered or contradicted keypoint "
                f"observations, {_percent(attribution['none']['rate'])} had no exact "
                "reference evidence and "
                f"{_percent(attribution['partial']['rate'])} had only partial evidence.",
            ]
        )

    delta = report.get("baseline_to_rag_conditioned_on_evidence")
    if delta:
        lines.extend(
            [
                "",
                "## Baseline to RAG change conditioned on retrieved evidence",
                "",
                "| Exact keypoint-reference support | Baseline coverage | RAG coverage | Delta | Baseline contradiction | RAG contradiction |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in delta["by_evidence_support"]:
            lines.append(
                f"| {row['support']} | {_percent(row['baseline_coverage_rate'])} | "
                f"{_percent(row['rag_coverage_rate'])} | "
                f"{float(row['coverage_delta']) * 100:+.2f} pp | "
                f"{_percent(row['baseline_contradiction_rate'])} | "
                f"{_percent(row['rag_contradiction_rate'])} |"
            )

    agreement = report["retrieval_agreement"]
    lines.extend(
        [
            "",
            "## Cross-model retrieval consistency",
            "",
            f"- Identical ranked chunk sequence: "
            f"{agreement['identical_ranked_sequence_qra_count']}/{agreement['qra_count']} QRA.",
            f"- Divergent QRA: {agreement['divergent_qra_count']}.",
            "- Mean pairwise chunk-set Jaccard: "
            f"{agreement['mean_pairwise_chunk_set_jaccard']:.4f}.",
            "",
            "## Lowest-recall questions",
            "",
            "| QRA | Domain | Type | Macro Recall@K | RAG keypoint coverage |",
            "|---|---|---|---:|---:|",
        ]
    )
    for row in report["question_diagnostics"][:15]:
        coverage = row.get("rag_judge", {}).get("covered", {}).get("rate")
        coverage_text = _percent(coverage) if coverage is not None else "n/a"
        lines.append(
            f"| {row['qra_id']} | {row['domain']} | {row['question_type']} | "
            f"{_percent(row['macro_reference_recall_at_k'])} | {coverage_text} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This report separates retrieval availability from answer use. An exact miss",
            "does not prove that every retrieved passage is irrelevant, because the metric",
            "does not credit paraphrases or alternative valid evidence. Conversely, exact",
            "evidence availability does not guarantee that a generator will integrate or",
            "calculate every required keypoint. These diagnostics therefore guide ablations;",
            "they do not replace blinded multi-Judge or human evaluation.",
            "",
        ]
    )
    return "\n".join(lines)


def build_formal_retrieval_diagnostics(
    *,
    source_paths: Iterable[Path],
    rag_scores_path: Path | None,
    baseline_scores_path: Path | None,
    output_dir: Path,
    report_id: str,
) -> dict[str, Path]:
    results, qra_by_id, sources, manifest_sha = _load_panel(source_paths)
    observations = [
        _answer_observation(result, qra_by_id[str(result["qra_id"])])
        for result in results
    ]
    panel_keys = {(item["model"], item["qra_id"]) for item in observations}
    rag_scores, rag_manifest = _load_score_index(
        rag_scores_path, expected_mode="rag", panel_keys=panel_keys
    )
    baseline_scores, baseline_manifest = _load_score_index(
        baseline_scores_path, expected_mode="baseline", panel_keys=panel_keys
    )
    if baseline_scores and not rag_scores:
        raise FormalRetrievalDiagnosticsError(
            "Baseline conditioning requires matching RAG Judge scores"
        )

    judge_report: dict[str, Any] | None = None
    labeled: list[dict[str, Any]] = []
    if rag_scores:
        judge_report, labeled = _judge_conditioning(observations, rag_scores)

    report: dict[str, Any] = {
        "report_id": report_id,
        "status": "completed",
        "evaluation_status": "posthoc_exploratory_retrieval_diagnostic",
        "created_at": datetime.now(UTC).isoformat(),
        "method": {
            "reference_match": (
                "Complete whitespace-normalized frozen quote in a retrieved chunk "
                "from the same normalized source path"
            ),
            "failure_layers": [
                "exact",
                "right_document_wrong_chunk",
                "right_document_absent",
            ],
            "eir_proxy": (
                "Fraction of retrieved whitespace-token positions covered by complete "
                "exact frozen quotes from the same source document"
            ),
            "known_conservatism": (
                "Paraphrases, partial quotes, and alternative valid evidence are not credited"
            ),
        },
        "inputs": {
            "source_runs": sources,
            "rag_judge_scores": rag_manifest,
            "baseline_judge_scores": baseline_manifest,
        },
        "panel": {
            "model_count": len({item["model"] for item in observations}),
            "models": sorted({item["model"] for item in observations}),
            "qra_count": len(qra_by_id),
            "answer_count": len(observations),
            "dataset_manifest_sha256": manifest_sha,
        },
        "run_health": _run_health(observations),
        "retrieval": {
            "overall": _aggregate_answers(observations),
            "by_model": _grouped_answers(observations, ("model",)),
            "by_domain": _grouped_answers(observations, ("domain",)),
            "by_question_type": _grouped_answers(
                observations, ("question_type",)
            ),
        },
        "retrieval_agreement": _retrieval_agreement(observations),
        "question_diagnostics": _question_diagnostics(observations, labeled),
    }
    if judge_report is not None:
        report["rag_judge_conditioned_on_evidence"] = judge_report
    if baseline_scores:
        report["baseline_to_rag_conditioned_on_evidence"] = (
            _baseline_to_rag_conditioning(labeled, baseline_scores)
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{report_id}.json"
    markdown_path = output_dir / f"{report_id}.md"
    _write_json(json_path, report)
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path}
