"""Run a resumable retrieval-only dense-versus-reranker formal ablation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import statistics
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from apps.advisors.rag_pipeline import (
    TOKEN_ESTIMATOR_ID,
    RetrievalSelection,
    _resolve_config,
    _retrieve_context,
)
from apps.evaluations.formal_development_benchmark import (
    build_dataset_manifest,
    project_root,
    sha256_file,
    snapshot_rag_index,
)
from apps.evaluations.formal_retrieval_diagnostics import (
    _aggregate_answers,
    _exact_evidence_words,
    _grouped_answers,
    _reference_status,
)


DENSE_K_VALUES = (5, 8, 10, 20)
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_RERANKER_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
RetrievalFunction = Callable[..., RetrievalSelection]
ProgressCallback = Callable[[dict[str, Any]], None]


class FormalRerankAblationError(RuntimeError):
    """Raised when a reranking ablation cannot be reproduced safely."""


def default_output_dir() -> Path:
    return project_root() / "docs/evaluation/runs/formal/retrieval"


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FormalRerankAblationError(f"Cannot load {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise FormalRerankAblationError(f"Expected JSON object in {path}")
    return value


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(project_root().resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _round(value: float) -> float:
    return round(float(value), 4)


def _rate(numerator: int, denominator: int) -> float:
    return _round(numerator / denominator) if denominator else 0.0


def _mean(values: list[float]) -> float:
    return _round(statistics.mean(values)) if values else 0.0


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _metric_observation(
    qra: dict[str, Any],
    chunks: list[dict[str, Any]],
    *,
    profile: str,
) -> dict[str, Any]:
    references = list(qra.get("references", []))
    statuses = {
        str(reference["reference_id"]): _reference_status(reference, chunks)
        for reference in references
    }
    counts = Counter(statuses.values())
    supports: list[dict[str, Any]] = []
    for keypoint in qra.get("keypoints", []):
        reference_ids = [str(value) for value in keypoint.get("reference_ids", [])]
        exact_count = sum(statuses.get(value) == "exact" for value in reference_ids)
        if reference_ids and exact_count == len(reference_ids):
            support = "all"
        elif exact_count:
            support = "partial"
        else:
            support = "none"
        supports.append(
            {
                "keypoint_id": str(keypoint.get("id", "")),
                "support": support,
                "reference_count": len(reference_ids),
                "exact_reference_count": exact_count,
            }
        )
    evidence_words, retrieved_words = _exact_evidence_words(references, chunks)
    return {
        "profile": profile,
        "model": "retrieval_only",
        "qra_id": str(qra["id"]),
        "domain": str(qra["domain"]),
        "question_type": str(qra["question_type"]),
        "question": str(qra["question"]),
        "reference_count": len(references),
        "exact_reference_count": counts["exact"],
        "reference_recall": _rate(counts["exact"], len(references)),
        "reference_status_counts": {
            "exact": counts["exact"],
            "right_document_wrong_chunk": counts["right_document_wrong_chunk"],
            "right_document_absent": counts["right_document_absent"],
        },
        "keypoint_support": supports,
        "exact_evidence_word_count": evidence_words,
        "retrieved_word_count": retrieved_words,
        "exact_evidence_word_rate_proxy": _rate(
            evidence_words, retrieved_words
        ),
        "retrieved_chunk_count": len(chunks),
        "retrieved_context_char_count": sum(
            len(str(chunk.get("text", ""))) for chunk in chunks
        ),
        "chunk_ids": [str(chunk.get("chunk_id", "")) for chunk in chunks],
        "finish_reason": "retrieval_only",
        "retrieval_mode": profile,
    }


def _dense_chunks(
    candidates: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    return sorted(candidates, key=lambda item: int(item["dense_rank"]))[:limit]


def _profile_observations(
    qra: dict[str, Any], retrieval: RetrievalSelection
) -> dict[str, dict[str, Any]]:
    observations = {
        f"dense_top_{limit}": _metric_observation(
            qra,
            _dense_chunks(retrieval.candidates, limit),
            profile=f"dense_top_{limit}",
        )
        for limit in DENSE_K_VALUES
        if limit <= retrieval.candidate_k
    }
    observations["rerank_top_5"] = _metric_observation(
        qra,
        retrieval.hits,
        profile="rerank_top_5",
    )
    return observations


def _artifact_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "rank",
        "pre_selection_rank",
        "dense_rank",
        "score",
        "score_type",
        "dense_score",
        "reranker_score",
        "reranker_model",
        "reranker_revision",
        "chunk_id",
        "domain",
        "source_path",
        "source_name",
        "chunk_index",
        "text",
        "chunk_type",
        "section_title",
        "heading_path",
        "page_start",
        "page_end",
        "estimated_text_tokens",
        "estimated_context_tokens",
    )
    return {key: chunk.get(key) for key in keys if key in chunk}


def _summaries(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_profile: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        for profile, observation in result["profile_metrics"].items():
            by_profile.setdefault(profile, []).append(observation)
    output: dict[str, Any] = {}
    for profile, observations in sorted(by_profile.items()):
        output[profile] = {
            "overall": _aggregate_answers(observations),
            "by_domain": _grouped_answers(observations, ("domain",)),
            "by_question_type": _grouped_answers(
                observations, ("question_type",)
            ),
        }
    return output


def _delta(
    summaries: dict[str, Any], first: str, second: str, metric: str
) -> float:
    return _round(
        float(summaries[second]["overall"][metric])
        - float(summaries[first]["overall"][metric])
    )


def _domain_row(
    summaries: dict[str, Any], profile: str, domain: str
) -> dict[str, Any]:
    return next(
        row
        for row in summaries[profile]["by_domain"]
        if row["domain"] == domain
    )


def _acceptance(summaries: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    macro_delta = _delta(
        summaries,
        "dense_top_5",
        "rerank_top_5",
        "macro_reference_recall_at_k",
    )
    market = _domain_row(summaries, "rerank_top_5", "market_validation")
    selected_counts = [len(item["selected_chunks"]) for item in results]
    modes = Counter(str(item["retrieval_mode"]) for item in results)
    gates = {
        "overall_macro_recall_gain_at_least_10pp": {
            "threshold": 0.10,
            "observed": macro_delta,
            "passed": macro_delta >= 0.10,
        },
        "market_weighted_recall_at_least_75pct": {
            "threshold": 0.75,
            "observed": market["evidence_weighted_reference_recall_at_k"],
            "passed": market["evidence_weighted_reference_recall_at_k"] >= 0.75,
        },
        "mean_selected_chunks_at_most_5": {
            "threshold": 5,
            "observed": _mean([float(value) for value in selected_counts]),
            "passed": _mean([float(value) for value in selected_counts]) <= 5,
        },
        "keyword_fallback_count_zero": {
            "threshold": 0,
            "observed": sum(
                count for mode, count in modes.items() if "keyword_fallback" in mode
            ),
            "passed": not any("keyword_fallback" in mode for mode in modes),
        },
    }
    return {
        "all_gates_passed": all(value["passed"] for value in gates.values()),
        "gates": gates,
    }


def _latency_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_values = [int(item["candidate_retrieval_latency_ms"]) for item in results]
    reranker_values = [int(item["reranker_latency_ms"]) for item in results]
    warm_reranker = reranker_values[1:]
    return {
        "question_count": len(results),
        "candidate_retrieval_mean_ms": _mean(
            [float(value) for value in candidate_values]
        ),
        "reranker_mean_ms_including_cold_start": _mean(
            [float(value) for value in reranker_values]
        ),
        "reranker_median_ms_including_cold_start": _round(
            statistics.median(reranker_values)
        )
        if reranker_values
        else 0.0,
        "first_reranker_ms_including_model_load": reranker_values[0]
        if reranker_values
        else 0,
        "warm_reranker_mean_ms_excluding_first": _mean(
            [float(value) for value in warm_reranker]
        ),
        "total_wall_component_seconds": _round(
            sum(candidate_values + reranker_values) / 1000
        ),
    }


def _percent(value: Any) -> str:
    return f"{float(value) * 100:.2f}%"


def _markdown(report: dict[str, Any]) -> str:
    summaries = report["summaries"]
    latency = report["latency"]
    lines = [
        f"# {report['report_id']}: Dense Retrieval vs Cross-Encoder Reranking",
        "",
        "> Post-hoc exploratory retrieval-only ablation. No answers were generated.",
        "",
        "## Configuration",
        "",
        f"- QRA items: {report['dataset']['item_count']}",
        f"- Candidate K: {report['config']['candidate_k']}",
        f"- Context K: {report['config']['context_k']}",
        f"- Context budget: {report['config']['max_context_tokens']} estimated tokens",
        f"- Reranker: `{report['config']['reranker_model']}`",
        f"- Revision: `{report['config']['reranker_revision']}`",
        f"- Device: `{report['config']['reranker_device']}`",
        "- Keyword fallback: forbidden",
        "",
        "## Overall retrieval",
        "",
        "| Profile | Macro recall | Weighted recall | Wrong chunk | Document absent | All keypoint refs | EIR proxy | Mean chunks |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for profile in ("dense_top_5", "dense_top_8", "dense_top_10", "dense_top_20", "rerank_top_5"):
        if profile not in summaries:
            continue
        row = summaries[profile]["overall"]
        failure = row["reference_failure_layers"]
        support = row["keypoint_evidence_availability"]
        lines.append(
            f"| {profile} | {_percent(row['macro_reference_recall_at_k'])} | "
            f"{_percent(row['evidence_weighted_reference_recall_at_k'])} | "
            f"{_percent(failure['right_document_wrong_chunk']['rate'])} | "
            f"{_percent(failure['right_document_absent']['rate'])} | "
            f"{_percent(support['all']['rate'])} | "
            f"{_percent(row['exact_evidence_word_rate_proxy']['pooled_rate'])} | "
            f"{row['retrieval_context']['mean_chunk_count']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Rerank@5 by domain",
            "",
            "| Domain | Macro recall | Weighted recall | All keypoint refs |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in summaries["rerank_top_5"]["by_domain"]:
        lines.append(
            f"| {row['domain']} | {_percent(row['macro_reference_recall_at_k'])} | "
            f"{_percent(row['evidence_weighted_reference_recall_at_k'])} | "
            f"{_percent(row['keypoint_evidence_availability']['all']['rate'])} |"
        )
    lines.extend(
        [
            "",
            "## Acceptance gates",
            "",
            "| Gate | Observed | Passed |",
            "|---|---:|:---:|",
        ]
    )
    for name, gate in report["acceptance"]["gates"].items():
        observed = gate["observed"]
        if "recall" in name or "gain" in name:
            observed_text = _percent(observed)
        else:
            observed_text = str(observed)
        lines.append(
            f"| {name} | {observed_text} | {'yes' if gate['passed'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Latency",
            "",
            f"- Candidate retrieval mean: {latency['candidate_retrieval_mean_ms'] / 1000:.3f} s/question.",
            "- First reranker call including model load: "
            f"{latency['first_reranker_ms_including_model_load'] / 1000:.3f} s.",
            "- Warm reranker mean excluding first call: "
            f"{latency['warm_reranker_mean_ms_excluding_first'] / 1000:.3f} s/question.",
            "",
            "## Interpretation",
            "",
            "Dense@20 measures candidate availability; Rerank@5 measures how much of",
            "that evidence the cross-encoder retained under the final prompt budget.",
            "The strict metric requires a complete frozen quote in the same source",
            "document and therefore does not credit paraphrases or alternative evidence.",
            "The exposed formal set is used only for exploratory engineering decisions.",
            "",
        ]
    )
    return "\n".join(lines)


def run_formal_rerank_ablation(
    *,
    qra_items: list[dict[str, Any]],
    output_dir: Path,
    report_id: str,
    candidate_k: int = 20,
    context_k: int = 5,
    max_context_tokens: int = 3000,
    reranker_model: str = DEFAULT_RERANKER_MODEL,
    reranker_revision: str = DEFAULT_RERANKER_REVISION,
    reranker_device: str = "auto",
    reranker_batch_size: int = 8,
    reranker_max_length: int = 512,
    resume: bool = False,
    progress_callback: ProgressCallback | None = None,
    retrieval_fn: RetrievalFunction = _retrieve_context,
) -> dict[str, Path]:
    if not qra_items:
        raise FormalRerankAblationError("At least one QRA item is required")
    if candidate_k < max(DENSE_K_VALUES):
        raise FormalRerankAblationError("candidate_k must be at least 20")
    if context_k != 5:
        raise FormalRerankAblationError(
            "This frozen Step-2 ablation requires context_k=5"
        )
    if max_context_tokens < 1:
        raise FormalRerankAblationError("max_context_tokens must be positive")
    if not reranker_model.strip() or not reranker_revision.strip():
        raise FormalRerankAblationError(
            "A pinned reranker model and revision are required"
        )

    root = project_root()
    implementation_path = Path(__file__).resolve()
    rag_pipeline_path = (
        root / "backend/apps/advisors/rag_pipeline.py"
    ).resolve()
    dataset = build_dataset_manifest(qra_items)
    index = snapshot_rag_index()
    config = {
        "candidate_k": candidate_k,
        "context_k": context_k,
        "max_context_tokens": max_context_tokens,
        "token_estimator": TOKEN_ESTIMATOR_ID,
        "reranker_model": reranker_model,
        "reranker_revision": reranker_revision,
        "reranker_device": reranker_device,
        "reranker_batch_size": reranker_batch_size,
        "reranker_max_length": reranker_max_length,
        "allow_keyword_fallback": False,
        "dense_k_values": list(DENSE_K_VALUES),
        "implementation": {
            "path": _display_path(implementation_path),
            "sha256": sha256_file(implementation_path),
        },
        "rag_pipeline": {
            "path": _display_path(rag_pipeline_path),
            "sha256": sha256_file(rag_pipeline_path),
        },
        "packages": {
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
        },
    }
    resume_fingerprint = _canonical_sha256(
        {
            "dataset_manifest_sha256": dataset["manifest_sha256"],
            "index_metadata_sha256": index.get("metadata_sha256", ""),
            "index_embeddings_sha256": index.get("embeddings_sha256", ""),
            "config": config,
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = output_dir / f"{report_id}.partial.json"
    json_path = output_dir / f"{report_id}.json"
    markdown_path = output_dir / f"{report_id}.md"

    if resume:
        if not partial_path.is_file():
            raise FormalRerankAblationError(
                f"Resume checkpoint does not exist: {partial_path}"
            )
        report = _load_json(partial_path)
        if report.get("resume_validation_sha256") != resume_fingerprint:
            raise FormalRerankAblationError(
                "Resume inputs, index, implementation, or configuration differ"
            )
        report["prior_attempts"] = list(report.get("prior_attempts", [])) + [
            item
            for item in report.get("results", [])
            if item.get("status") != "success"
        ]
        report["results"] = [
            item
            for item in report.get("results", [])
            if item.get("status") == "success"
        ]
    else:
        for path in (partial_path, json_path, markdown_path):
            if path.exists():
                raise FormalRerankAblationError(f"Output already exists: {path}")
        report = {
            "report_id": report_id,
            "status": "running",
            "evaluation_status": "posthoc_exploratory_retrieval_only_ablation",
            "started_at": datetime.now(UTC).isoformat(),
            "completed_at": None,
            "resume_validation_sha256": resume_fingerprint,
            "config": config,
            "dataset": dataset,
            "rag_index": index,
            "results": [],
            "prior_attempts": [],
        }
        _write_json_atomic(partial_path, report)

    completed = {
        str(item["qra_id"])
        for item in report["results"]
        if item.get("status") == "success"
    }
    rag_config = _resolve_config(None)
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
        started = datetime.now(UTC).isoformat()
        try:
            retrieval = retrieval_fn(
                query=str(qra["question"]),
                config=rag_config,
                context_k=context_k,
                candidate_k=candidate_k,
                domain=str(qra["domain"]),
                reranker_model=reranker_model,
                reranker_revision=reranker_revision,
                max_context_tokens=max_context_tokens,
                allow_keyword_fallback=False,
                reranker_batch_size=reranker_batch_size,
                reranker_max_length=reranker_max_length,
                reranker_device=reranker_device,
            )
            result = {
                "status": "success",
                "qra_id": qra_id,
                "domain": str(qra["domain"]),
                "question_type": str(qra["question_type"]),
                "started_at": started,
                "completed_at": datetime.now(UTC).isoformat(),
                "retrieval_mode": retrieval.retrieval_mode,
                "candidate_retrieval_latency_ms": (
                    retrieval.candidate_retrieval_latency_ms
                ),
                "reranker_latency_ms": retrieval.reranker_latency_ms,
                "selection_latency_ms": retrieval.selection_latency_ms,
                "estimated_context_tokens": retrieval.estimated_context_tokens,
                "dense_candidates": [
                    _artifact_chunk(chunk) for chunk in retrieval.candidates
                ],
                "selected_chunks": [
                    _artifact_chunk(chunk) for chunk in retrieval.hits
                ],
                "profile_metrics": _profile_observations(qra, retrieval),
                "error_type": "",
                "error": "",
            }
        except Exception as exc:
            result = {
                "status": "error",
                "qra_id": qra_id,
                "domain": str(qra["domain"]),
                "question_type": str(qra["question_type"]),
                "started_at": started,
                "completed_at": datetime.now(UTC).isoformat(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        report["results"].append(result)
        _write_json_atomic(partial_path, report)
        if progress_callback:
            progress_callback(
                {
                    "type": "complete",
                    "position": position,
                    "total": len(qra_items),
                    "qra_id": qra_id,
                    "status": result["status"],
                    "reranker_latency_ms": int(
                        result.get("reranker_latency_ms", 0)
                    ),
                    "error": str(result.get("error", "")),
                }
            )
        if result["status"] != "success":
            raise FormalRerankAblationError(
                f"Ablation failed for {qra_id}: {result['error']}"
            )

    successful = [item for item in report["results"] if item["status"] == "success"]
    if len(successful) != len(qra_items):
        raise FormalRerankAblationError("Ablation panel is incomplete")
    report["status"] = "completed"
    report["completed_at"] = datetime.now(UTC).isoformat()
    report["summaries"] = _summaries(successful)
    report["latency"] = _latency_summary(successful)
    report["acceptance"] = _acceptance(report["summaries"], successful)
    _write_json_atomic(json_path, report)
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    partial_path.unlink()
    return {"json": json_path, "markdown": markdown_path}
