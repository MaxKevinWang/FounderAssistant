"""Freeze non-deterministic LLM query decompositions for reproducible ablations."""

from __future__ import annotations

import json
import statistics
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from apps.advisors.query_decomposition import (
    QUERY_DECOMPOSITION_PROMPT_VERSION,
    QueryDecomposition,
    decompose_query,
)
from apps.evaluations.formal_development_benchmark import (
    build_dataset_manifest,
    project_root,
    sha256_file,
)
from apps.evaluations.formal_rerank_ablation import (
    _canonical_sha256,
    _display_path,
    _round,
    _write_json_atomic,
)


DEFAULT_DECOMPOSER_MODEL = "gpt-oss:20b"
DecompositionFunction = Callable[..., QueryDecomposition]
ProgressCallback = Callable[[dict[str, Any]], None]


class FormalQueryDecompositionManifestError(RuntimeError):
    """Raised when a frozen decomposition manifest is invalid or incomplete."""


def default_output_dir() -> Path:
    return project_root() / "docs/evaluation/runs/formal/retrieval"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FormalQueryDecompositionManifestError(
            f"Cannot load decomposition manifest {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise FormalQueryDecompositionManifestError(
            f"Expected JSON object in decomposition manifest {path}"
        )
    return payload


def _entry_from_decomposition(
    qra: dict[str, Any], decomposition: QueryDecomposition
) -> dict[str, Any]:
    return {
        "status": "success",
        "qra_id": str(qra["id"]),
        "domain": str(qra["domain"]),
        "question_type": str(qra["question_type"]),
        "question": str(qra["question"]),
        "generated_at": datetime.now(UTC).isoformat(),
        "decomposition": {
            "needs_decomposition": decomposition.needs_decomposition,
            "subqueries": decomposition.subqueries,
            "retrieval_queries": decomposition.retrieval_queries,
            "provider": decomposition.provider,
            "model": decomposition.model,
            "raw_response": decomposition.raw_response,
            "latency_ms": decomposition.latency_ms,
            "finish_reason": decomposition.finish_reason,
            "usage": decomposition.usage,
            "prompt_version": decomposition.prompt_version,
            "prompt_sha256": decomposition.prompt_sha256,
            "seed": decomposition.seed,
        },
        "error_type": "",
        "error": "",
    }


def _as_decomposition(entry: dict[str, Any]) -> QueryDecomposition:
    value = entry.get("decomposition", {})
    if not isinstance(value, dict):
        raise FormalQueryDecompositionManifestError(
            f"Invalid decomposition payload for {entry.get('qra_id', '')}"
        )
    raw_subqueries = value.get("subqueries", [])
    if not isinstance(raw_subqueries, list) or not all(
        isinstance(item, str) for item in raw_subqueries
    ):
        raise FormalQueryDecompositionManifestError(
            f"Invalid subqueries for {entry.get('qra_id', '')}"
        )
    raw_usage = value.get("usage", {})
    usage = raw_usage if isinstance(raw_usage, dict) else {}
    return QueryDecomposition(
        original_query=str(entry["question"]),
        needs_decomposition=bool(value.get("needs_decomposition", False)),
        subqueries=list(raw_subqueries),
        provider=str(value.get("provider", "")),
        model=str(value.get("model", "")),
        raw_response=str(value.get("raw_response", "")),
        latency_ms=int(value.get("latency_ms", 0)),
        finish_reason=str(value.get("finish_reason", "")),
        usage=usage,
        prompt_version=str(value.get("prompt_version", "")),
        prompt_sha256=str(value.get("prompt_sha256", "")),
        seed=int(value.get("seed", 0)),
    )


def load_query_decomposition_manifest(
    path: Path,
    *,
    qra_items: list[dict[str, Any]],
) -> dict[str, QueryDecomposition]:
    """Load a completed manifest and bind every entry to the exact QRA text."""

    payload = _load_json(path)
    if payload.get("status") != "completed":
        raise FormalQueryDecompositionManifestError(
            f"Decomposition manifest is not completed: {path}"
        )
    entries = payload.get("entries", [])
    if not isinstance(entries, list):
        raise FormalQueryDecompositionManifestError(
            "Decomposition manifest entries must be an array"
        )
    by_id = {
        str(entry.get("qra_id", "")): entry
        for entry in entries
        if isinstance(entry, dict) and entry.get("status") == "success"
    }
    expected_ids = {str(qra["id"]) for qra in qra_items}
    if set(by_id) != expected_ids:
        missing = sorted(expected_ids - set(by_id))
        extra = sorted(set(by_id) - expected_ids)
        raise FormalQueryDecompositionManifestError(
            f"Manifest/QRA ID mismatch; missing={missing}, extra={extra}"
        )
    output: dict[str, QueryDecomposition] = {}
    for qra in qra_items:
        qra_id = str(qra["id"])
        entry = by_id[qra_id]
        if str(entry.get("question", "")) != str(qra["question"]):
            raise FormalQueryDecompositionManifestError(
                f"Manifest question differs from frozen QRA for {qra_id}"
            )
        output[qra_id] = _as_decomposition(entry)
    return output


def _summary(entries: list[dict[str, Any]]) -> dict[str, Any]:
    decomposed = [
        entry
        for entry in entries
        if entry["decomposition"]["needs_decomposition"]
    ]
    latencies = [float(entry["decomposition"]["latency_ms"]) for entry in entries]
    return {
        "question_count": len(entries),
        "decomposed_question_count": len(decomposed),
        "decomposed_question_rate": _round(len(decomposed) / len(entries)),
        "mean_subquery_count": _round(
            statistics.mean(
                len(entry["decomposition"]["subqueries"]) for entry in entries
            )
        ),
        "mean_generation_latency_ms": _round(statistics.mean(latencies)),
    }


def run_query_decomposition_freeze(
    *,
    qra_items: list[dict[str, Any]],
    output_dir: Path,
    manifest_id: str,
    model: str = DEFAULT_DECOMPOSER_MODEL,
    reasoning_effort: str | None = "low",
    seed: int = 0,
    max_subqueries: int = 4,
    resume: bool = False,
    progress_callback: ProgressCallback | None = None,
    decompose_fn: DecompositionFunction = decompose_query,
) -> Path:
    """Generate each decomposition once and freeze it as a reusable artifact."""

    if not qra_items:
        raise FormalQueryDecompositionManifestError(
            "At least one QRA item is required"
        )
    if not model.strip() or max_subqueries < 2:
        raise FormalQueryDecompositionManifestError(
            "A model and at least two allowed subqueries are required"
        )
    implementation_path = Path(__file__).resolve()
    decomposition_path = (
        project_root() / "backend/apps/advisors/query_decomposition.py"
    ).resolve()
    dataset = build_dataset_manifest(qra_items)
    config = {
        "model": model,
        "reasoning_effort": reasoning_effort,
        "temperature": 0.0,
        "seed_requested": seed,
        "max_subqueries": max_subqueries,
        "prompt_version": QUERY_DECOMPOSITION_PROMPT_VERSION,
        "implementation": {
            "path": _display_path(implementation_path),
            "sha256": sha256_file(implementation_path),
        },
        "decomposition_implementation": {
            "path": _display_path(decomposition_path),
            "sha256": sha256_file(decomposition_path),
        },
    }
    fingerprint = _canonical_sha256(
        {
            "dataset_manifest_sha256": dataset["manifest_sha256"],
            "config": config,
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = output_dir / f"{manifest_id}.partial.json"
    output_path = output_dir / f"{manifest_id}.json"
    if resume:
        if not partial_path.is_file():
            raise FormalQueryDecompositionManifestError(
                f"Resume checkpoint does not exist: {partial_path}"
            )
        report = _load_json(partial_path)
        if report.get("resume_validation_sha256") != fingerprint:
            raise FormalQueryDecompositionManifestError(
                "Resume dataset, implementation, or configuration differs"
            )
        report["prior_attempts"] = list(report.get("prior_attempts", [])) + [
            entry
            for entry in report.get("entries", [])
            if entry.get("status") != "success"
        ]
        report["entries"] = [
            entry
            for entry in report.get("entries", [])
            if entry.get("status") == "success"
        ]
    else:
        for path in (partial_path, output_path):
            if path.exists():
                raise FormalQueryDecompositionManifestError(
                    f"Output already exists: {path}"
                )
        report = {
            "manifest_id": manifest_id,
            "status": "running",
            "created_at": datetime.now(UTC).isoformat(),
            "completed_at": None,
            "resume_validation_sha256": fingerprint,
            "config": config,
            "dataset": dataset,
            "entries": [],
            "prior_attempts": [],
        }
        _write_json_atomic(partial_path, report)

    completed = {
        str(entry["qra_id"])
        for entry in report["entries"]
        if entry.get("status") == "success"
    }
    for position, qra in enumerate(qra_items, start=1):
        qra_id = str(qra["id"])
        if qra_id in completed:
            continue
        if progress_callback:
            progress_callback(
                {
                    "type": "start",
                    "position": position,
                    "total": len(qra_items),
                    "qra_id": qra_id,
                }
            )
        try:
            decomposition = decompose_fn(
                str(qra["question"]),
                model=model,
                reasoning_effort=reasoning_effort,
                seed=seed,
                max_subqueries=max_subqueries,
            )
            entry = _entry_from_decomposition(qra, decomposition)
        except Exception as exc:
            entry = {
                "status": "error",
                "qra_id": qra_id,
                "domain": str(qra["domain"]),
                "question_type": str(qra["question_type"]),
                "question": str(qra["question"]),
                "generated_at": datetime.now(UTC).isoformat(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        report["entries"].append(entry)
        _write_json_atomic(partial_path, report)
        if progress_callback:
            progress_callback(
                {
                    "type": "complete",
                    "position": position,
                    "total": len(qra_items),
                    "qra_id": qra_id,
                    "status": entry["status"],
                    "latency_ms": int(
                        entry.get("decomposition", {}).get("latency_ms", 0)
                    ),
                    "error": str(entry.get("error", "")),
                }
            )
        if entry["status"] != "success":
            raise FormalQueryDecompositionManifestError(
                f"Decomposition failed for {qra_id}: {entry['error']}"
            )

    successful = [
        entry for entry in report["entries"] if entry["status"] == "success"
    ]
    if len(successful) != len(qra_items):
        raise FormalQueryDecompositionManifestError(
            "Decomposition manifest is incomplete"
        )
    report["status"] = "completed"
    report["completed_at"] = datetime.now(UTC).isoformat()
    report["summary"] = _summary(successful)
    _write_json_atomic(output_path, report)
    partial_path.unlink()
    return output_path
