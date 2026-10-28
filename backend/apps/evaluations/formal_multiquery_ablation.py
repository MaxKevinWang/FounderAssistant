"""Run a resumable retrieval-only multi-query ablation."""

from __future__ import annotations

import json
import statistics
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from apps.advisors.multiquery_retrieval import (
    DEFAULT_RRF_K,
    MultiQueryRetrieval,
    retrieve_multiquery_context,
)
from apps.advisors.query_decomposition import (
    QUERY_DECOMPOSITION_PROMPT_VERSION,
    QueryDecomposition,
    decompose_query,
)
from apps.advisors.rag_pipeline import (
    TOKEN_ESTIMATOR_ID,
    _resolve_config,
    _select_context_hits,
)
from apps.evaluations.formal_development_benchmark import (
    build_dataset_manifest,
    project_root,
    sha256_file,
    snapshot_rag_index,
)
from apps.evaluations.formal_rerank_ablation import (
    _canonical_sha256,
    _display_path,
    _metric_observation,
    _round,
    _write_json_atomic,
)
from apps.evaluations.formal_query_decomposition_manifest import (
    load_query_decomposition_manifest,
)
from apps.evaluations.formal_retrieval_diagnostics import (
    _aggregate_answers,
    _grouped_answers,
)


DEFAULT_DECOMPOSER_MODEL = "gpt-oss:20b"
DecompositionFunction = Callable[..., QueryDecomposition]
RetrievalFunction = Callable[..., MultiQueryRetrieval]
ProgressCallback = Callable[[dict[str, Any]], None]


class FormalMultiQueryAblationError(RuntimeError):
    """Raised when the multi-query ablation cannot be reproduced safely."""


def default_output_dir() -> Path:
    return project_root() / "docs/evaluation/runs/formal/retrieval"


def _mean(values: list[float]) -> float:
    return _round(statistics.mean(values)) if values else 0.0


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FormalMultiQueryAblationError(f"Cannot load {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise FormalMultiQueryAblationError(f"Expected JSON object in {path}")
    return value


def _artifact_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "rank",
        "pre_selection_rank",
        "dense_rank",
        "dense_score",
        "score",
        "score_type",
        "rrf_score",
        "query_ranks",
        "query_scores",
        "query_hit_count",
        "best_query_rank",
        "original_query_rank",
        "matched_queries",
        "fusion_rank",
        "query_reranker_scores",
        "query_reranker_ranks",
        "max_query_reranker_score",
        "best_reranker_query_index",
        "reranker_model",
        "reranker_revision",
        "coverage_gain",
        "query_coverage_after_selection",
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


def _query_trace(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "query": str(result["query"]),
        "retrieval_mode": str(result["retrieval_mode"]),
        "latency_ms": int(result["latency_ms"]),
        "hits": [
            {
                "rank": int(hit.get("rank", 0)),
                "score": float(hit.get("score", 0.0)),
                "chunk_id": str(hit.get("chunk_id", "")),
                "source_path": str(hit.get("source_path", "")),
            }
            for hit in result["hits"]
        ],
    }


def _profile_observations(
    qra: dict[str, Any],
    retrieval: MultiQueryRetrieval,
    *,
    max_context_tokens: int,
) -> dict[str, dict[str, Any]]:
    original_hits = list(retrieval.query_results[0]["hits"])
    rrf_candidates = sorted(
        retrieval.candidates,
        key=lambda item: int(item.get("fusion_rank", item.get("rank", 0))),
    )
    rrf_hits, _, _ = _select_context_hits(
        rrf_candidates,
        context_k=retrieval.context_k,
        max_context_tokens=max_context_tokens,
    )
    observations = {
        "original_dense_top_5": _metric_observation(
            qra,
            original_hits[:5],
            profile="original_dense_top_5",
        ),
        "original_dense_top_20": _metric_observation(
            qra,
            original_hits[:20],
            profile="original_dense_top_20",
        ),
        "multiquery_candidate_top_20": _metric_observation(
            qra,
            retrieval.candidates,
            profile="multiquery_candidate_top_20",
        ),
        "multiquery_rrf_top_5": _metric_observation(
            qra,
            rrf_hits,
            profile="multiquery_rrf_top_5",
        ),
    }
    if retrieval.reranker_model:
        max_rerank_hits, _, _ = _select_context_hits(
            retrieval.candidates,
            context_k=retrieval.context_k,
            max_context_tokens=max_context_tokens,
        )
        observations["multiquery_rerank_max_top_5"] = _metric_observation(
            qra,
            max_rerank_hits,
            profile="multiquery_rerank_max_top_5",
        )
    else:
        observations["multiquery_coverage_top_5"] = _metric_observation(
            qra,
            retrieval.hits,
            profile="multiquery_coverage_top_5",
        )
    return observations


def _summaries(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_profile: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        for profile, observation in result["profile_metrics"].items():
            by_profile.setdefault(profile, []).append(observation)
    return {
        profile: {
            "overall": _aggregate_answers(observations),
            "by_domain": _grouped_answers(observations, ("domain",)),
            "by_question_type": _grouped_answers(
                observations, ("question_type",)
            ),
        }
        for profile, observations in sorted(by_profile.items())
    }


def _domain_row(
    summaries: dict[str, Any], profile: str, domain: str
) -> dict[str, Any]:
    return next(
        row
        for row in summaries[profile]["by_domain"]
        if row["domain"] == domain
    )


def _acceptance(
    summaries: dict[str, Any], results: list[dict[str, Any]]
) -> dict[str, Any]:
    profile = (
        "multiquery_rerank_max_top_5"
        if "multiquery_rerank_max_top_5" in summaries
        else "multiquery_coverage_top_5"
    )
    baseline = summaries["original_dense_top_5"]["overall"]
    selected = summaries[profile]["overall"]
    macro_delta = _round(
        float(selected["macro_reference_recall_at_k"])
        - float(baseline["macro_reference_recall_at_k"])
    )
    market = _domain_row(summaries, profile, "market_validation")
    selected_counts = [len(item["selected_chunks"]) for item in results]
    modes = Counter(
        mode
        for item in results
        for mode in item.get("retrieval_modes", [])
    )
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


def _decomposition_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    decomposed = [
        result for result in results if result["decomposition"]["needs_decomposition"]
    ]
    return {
        "question_count": len(results),
        "decomposed_question_count": len(decomposed),
        "decomposed_question_rate": _round(len(decomposed) / len(results)),
        "mean_subquery_count_all_questions": _mean(
            [float(len(result["decomposition"]["subqueries"])) for result in results]
        ),
        "mean_subquery_count_when_decomposed": _mean(
            [
                float(len(result["decomposition"]["subqueries"]))
                for result in decomposed
            ]
        ),
    }


def _latency_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "question_count": len(results),
        "decomposition_mean_ms": _mean(
            [float(item["decomposition_latency_ms"]) for item in results]
        ),
        "decomposition_runtime_mean_ms": _mean(
            [float(item["decomposition_runtime_latency_ms"]) for item in results]
        ),
        "candidate_retrieval_mean_ms": _mean(
            [float(item["candidate_retrieval_latency_ms"]) for item in results]
        ),
        "fusion_mean_ms": _mean(
            [float(item["fusion_latency_ms"]) for item in results]
        ),
        "selection_mean_ms": _mean(
            [float(item["selection_latency_ms"]) for item in results]
        ),
        "reranker_mean_ms": _mean(
            [float(item.get("reranker_latency_ms", 0)) for item in results]
        ),
        "total_pipeline_component_mean_ms": _mean(
            [
                float(item["decomposition_runtime_latency_ms"])
                + float(item["candidate_retrieval_latency_ms"])
                + float(item["fusion_latency_ms"])
                + float(item.get("reranker_latency_ms", 0))
                + float(item["selection_latency_ms"])
                for item in results
            ]
        ),
    }


def _percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def _markdown(report: dict[str, Any]) -> str:
    config = report["config"]
    summaries = report["summaries"]
    lines = [
        f"# {report['report_id']}: Multi-Query Retrieval Ablation",
        "",
        "> Post-hoc exploratory retrieval-only ablation. No answers were generated.",
        "",
        "## Configuration",
        "",
        f"- QRA items: {len(report['results'])}",
        f"- Decomposer: `{config['decomposer_model']}`",
        f"- Decomposition source: `{config['decomposition_source']}`",
        f"- Prompt version: `{config['decomposer_prompt_version']}`",
        f"- Per-query K: {config['per_query_k']}",
        f"- Fused candidate K: {config['candidate_k']}",
        f"- Context K: {config['context_k']}",
        f"- Context budget: {config['max_context_tokens']} estimated tokens",
        f"- RRF k: {config['rrf_k']}",
        f"- Multi-query reranker: `{config['reranker_model'] or 'disabled'}`",
        "- Keyword fallback: forbidden",
        "",
        "## Overall retrieval",
        "",
        "| Profile | Macro recall | Weighted recall | All keypoint refs | EIR proxy | Mean chunks |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    profile_order = [
        "original_dense_top_5",
        "original_dense_top_20",
        "multiquery_candidate_top_20",
        "multiquery_rrf_top_5",
    ]
    if "multiquery_coverage_top_5" in summaries:
        profile_order.append("multiquery_coverage_top_5")
    if "multiquery_rerank_max_top_5" in summaries:
        profile_order.extend(
            [
                "multiquery_rerank_max_top_5",
            ]
        )
    for profile in profile_order:
        overall = summaries[profile]["overall"]
        lines.append(
            f"| {profile} | {_percent(overall['macro_reference_recall_at_k'])} | "
            f"{_percent(overall['evidence_weighted_reference_recall_at_k'])} | "
            f"{_percent(overall['keypoint_evidence_availability']['all']['rate'])} | "
            f"{_percent(overall['exact_evidence_word_rate_proxy']['pooled_rate'])} | "
            f"{overall['retrieval_context']['mean_chunk_count']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Selected profile by domain",
            "",
            "| Domain | Macro recall | Weighted recall | All keypoint refs |",
            "|---|---:|---:|---:|",
        ]
    )
    selected_profile = (
        "multiquery_rerank_max_top_5"
        if "multiquery_rerank_max_top_5" in summaries
        else "multiquery_coverage_top_5"
    )
    for row in summaries[selected_profile]["by_domain"]:
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
        rendered = (
            _percent(observed)
            if isinstance(observed, float)
            and (
                "recall" in name
                or "weighted" in name
            )
            else str(observed)
        )
        lines.append(f"| {name} | {rendered} | {'yes' if gate['passed'] else 'no'} |")
    decomposition = report["decomposition"]
    latency = report["latency"]
    lines.extend(
        [
            "",
            "## Decomposition and latency",
            "",
            f"- Decomposed questions: {decomposition['decomposed_question_count']}/{decomposition['question_count']} "
            f"({_percent(decomposition['decomposed_question_rate'])}).",
            f"- Mean subqueries when decomposed: {decomposition['mean_subquery_count_when_decomposed']:.2f}.",
            f"- Decomposition mean: {latency['decomposition_mean_ms'] / 1000:.3f} s/question.",
            f"- Decomposition runtime mean in this run: {latency['decomposition_runtime_mean_ms'] / 1000:.3f} s/question.",
            f"- Multi-query dense retrieval mean: {latency['candidate_retrieval_mean_ms'] / 1000:.3f} s/question.",
            f"- Multi-query reranker mean: {latency['reranker_mean_ms'] / 1000:.3f} s/question.",
            f"- Total measured component mean: {latency['total_pipeline_component_mean_ms'] / 1000:.3f} s/question.",
            "",
            "## Interpretation",
            "",
            "The original query is always retained. Subqueries are generated without access to references or keypoints. ",
            "RRF builds a domain-filtered candidate pool. When the cross-encoder is enabled, each candidate is scored against every query view and ranked by its maximum relevance score; otherwise the final greedy selector maximizes complementary query-view coverage. ",
            "The exposed formal set is used only for exploratory engineering decisions.",
            "",
        ]
    )
    return "\n".join(lines)


def run_formal_multiquery_ablation(
    *,
    qra_items: list[dict[str, Any]],
    output_dir: Path,
    report_id: str,
    decomposer_model: str = DEFAULT_DECOMPOSER_MODEL,
    decomposer_reasoning_effort: str | None = "low",
    decomposer_seed: int = 0,
    max_subqueries: int = 4,
    per_query_k: int = 20,
    candidate_k: int = 20,
    context_k: int = 5,
    max_context_tokens: int = 3000,
    rrf_k: int = DEFAULT_RRF_K,
    reranker_model: str = "",
    reranker_revision: str = "",
    reranker_device: str = "auto",
    reranker_batch_size: int = 16,
    reranker_max_length: int = 512,
    decomposition_manifest_path: Path | None = None,
    resume: bool = False,
    progress_callback: ProgressCallback | None = None,
    decompose_fn: DecompositionFunction = decompose_query,
    retrieval_fn: RetrievalFunction = retrieve_multiquery_context,
) -> dict[str, Path]:
    if not qra_items:
        raise FormalMultiQueryAblationError("At least one QRA item is required")
    if not decomposer_model.strip():
        raise FormalMultiQueryAblationError("A decomposition model is required")
    if max_subqueries < 2 or per_query_k < 1:
        raise FormalMultiQueryAblationError(
            "max_subqueries must be at least 2 and per_query_k must be positive"
        )
    if candidate_k < context_k or context_k != 5:
        raise FormalMultiQueryAblationError(
            "This frozen ablation requires candidate_k >= context_k and context_k=5"
        )
    if max_context_tokens < 1 or rrf_k < 0:
        raise FormalMultiQueryAblationError(
            "Context budget must be positive and rrf_k cannot be negative"
        )
    if bool(reranker_model.strip()) != bool(reranker_revision.strip()):
        raise FormalMultiQueryAblationError(
            "Reranker model and pinned revision must be configured together"
        )

    root = project_root()
    implementation_path = Path(__file__).resolve()
    decomposition_path = (
        root / "backend/apps/advisors/query_decomposition.py"
    ).resolve()
    retrieval_path = (
        root / "backend/apps/advisors/multiquery_retrieval.py"
    ).resolve()
    dataset = build_dataset_manifest(qra_items)
    frozen_decompositions = (
        load_query_decomposition_manifest(
            decomposition_manifest_path.resolve(),
            qra_items=qra_items,
        )
        if decomposition_manifest_path is not None
        else None
    )
    index = snapshot_rag_index()
    config = {
        "decomposer_model": decomposer_model,
        "decomposition_source": (
            "frozen_manifest" if frozen_decompositions is not None else "live_generation"
        ),
        "decomposition_manifest": (
            {
                "path": _display_path(decomposition_manifest_path.resolve()),
                "sha256": sha256_file(decomposition_manifest_path.resolve()),
            }
            if decomposition_manifest_path is not None
            else None
        ),
        "decomposer_reasoning_effort": decomposer_reasoning_effort,
        "decomposer_temperature": 0.0,
        "decomposer_seed": decomposer_seed,
        "decomposer_prompt_version": QUERY_DECOMPOSITION_PROMPT_VERSION,
        "max_subqueries": max_subqueries,
        "per_query_k": per_query_k,
        "candidate_k": candidate_k,
        "context_k": context_k,
        "max_context_tokens": max_context_tokens,
        "token_estimator": TOKEN_ESTIMATOR_ID,
        "rrf_k": rrf_k,
        "reranker_model": reranker_model,
        "reranker_revision": reranker_revision,
        "reranker_device": reranker_device,
        "reranker_batch_size": reranker_batch_size,
        "reranker_max_length": reranker_max_length,
        "allow_keyword_fallback": False,
        "implementation": {
            "path": _display_path(implementation_path),
            "sha256": sha256_file(implementation_path),
        },
        "decomposition_implementation": {
            "path": _display_path(decomposition_path),
            "sha256": sha256_file(decomposition_path),
        },
        "retrieval_implementation": {
            "path": _display_path(retrieval_path),
            "sha256": sha256_file(retrieval_path),
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
            raise FormalMultiQueryAblationError(
                f"Resume checkpoint does not exist: {partial_path}"
            )
        report = _load_json(partial_path)
        if report.get("resume_validation_sha256") != resume_fingerprint:
            raise FormalMultiQueryAblationError(
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
                raise FormalMultiQueryAblationError(f"Output already exists: {path}")
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
            if frozen_decompositions is not None:
                decomposition = frozen_decompositions[qra_id]
                decomposition_runtime_latency_ms = 0
            else:
                decomposition = decompose_fn(
                    str(qra["question"]),
                    model=decomposer_model,
                    reasoning_effort=decomposer_reasoning_effort,
                    seed=decomposer_seed,
                    max_subqueries=max_subqueries,
                )
                decomposition_runtime_latency_ms = decomposition.latency_ms
            retrieval = retrieval_fn(
                queries=decomposition.retrieval_queries,
                config=rag_config,
                per_query_k=per_query_k,
                candidate_k=candidate_k,
                context_k=context_k,
                domain=str(qra["domain"]),
                max_context_tokens=max_context_tokens,
                rrf_k=rrf_k,
                allow_keyword_fallback=False,
                reranker_model=reranker_model,
                reranker_revision=reranker_revision,
                reranker_device=reranker_device,
                reranker_batch_size=reranker_batch_size,
                reranker_max_length=reranker_max_length,
            )
            result = {
                "status": "success",
                "qra_id": qra_id,
                "domain": str(qra["domain"]),
                "question_type": str(qra["question_type"]),
                "started_at": started,
                "completed_at": datetime.now(UTC).isoformat(),
                "decomposition": {
                    "needs_decomposition": decomposition.needs_decomposition,
                    "subqueries": decomposition.subqueries,
                    "retrieval_queries": decomposition.retrieval_queries,
                    "provider": decomposition.provider,
                    "model": decomposition.model,
                    "raw_response": decomposition.raw_response,
                    "finish_reason": decomposition.finish_reason,
                    "usage": decomposition.usage,
                    "prompt_version": decomposition.prompt_version,
                    "prompt_sha256": decomposition.prompt_sha256,
                    "seed": decomposition.seed,
                },
                "retrieval_mode": retrieval.retrieval_mode,
                "retrieval_modes": retrieval.retrieval_modes,
                "decomposition_latency_ms": decomposition.latency_ms,
                "decomposition_runtime_latency_ms": (
                    decomposition_runtime_latency_ms
                ),
                "candidate_retrieval_latency_ms": (
                    retrieval.candidate_retrieval_latency_ms
                ),
                "fusion_latency_ms": retrieval.fusion_latency_ms,
                "selection_latency_ms": retrieval.selection_latency_ms,
                "reranker_latency_ms": retrieval.reranker_latency_ms,
                "estimated_context_tokens": retrieval.estimated_context_tokens,
                "query_results": [
                    _query_trace(item) for item in retrieval.query_results
                ],
                "fused_candidates": [
                    _artifact_chunk(chunk) for chunk in retrieval.candidates
                ],
                "selected_chunks": [
                    _artifact_chunk(chunk) for chunk in retrieval.hits
                ],
                "profile_metrics": _profile_observations(
                    qra,
                    retrieval,
                    max_context_tokens=max_context_tokens,
                ),
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
                    "decomposition_latency_ms": int(
                        result.get("decomposition_latency_ms", 0)
                    ),
                    "decomposition_runtime_latency_ms": int(
                        result.get("decomposition_runtime_latency_ms", 0)
                    ),
                    "retrieval_latency_ms": int(
                        result.get("candidate_retrieval_latency_ms", 0)
                    ),
                    "error": str(result.get("error", "")),
                }
            )
        if result["status"] != "success":
            raise FormalMultiQueryAblationError(
                f"Ablation failed for {qra_id}: {result['error']}"
            )

    successful = [item for item in report["results"] if item["status"] == "success"]
    if len(successful) != len(qra_items):
        raise FormalMultiQueryAblationError("Ablation panel is incomplete")
    report["status"] = "completed"
    report["completed_at"] = datetime.now(UTC).isoformat()
    report["summaries"] = _summaries(successful)
    report["decomposition"] = _decomposition_summary(successful)
    report["latency"] = _latency_summary(successful)
    report["acceptance"] = _acceptance(report["summaries"], successful)
    _write_json_atomic(json_path, report)
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    partial_path.unlink()
    return {"json": json_path, "markdown": markdown_path}
