"""Build a post-hoc neighbor-expansion ablation from frozen retrieval traces."""

from __future__ import annotations

import json
import statistics
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from apps.advisors.rag_pipeline import (
    RAGConfig,
    _expand_context_hits_with_neighbors,
    _resolve_config,
)
from apps.evaluations.formal_development_benchmark import (
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
from apps.evaluations.formal_retrieval_diagnostics import (
    _aggregate_answers,
    _grouped_answers,
)


ExpansionFunction = Callable[..., tuple[list[dict[str, Any]], int, int]]


class FormalNeighborExpansionAblationError(RuntimeError):
    """Raised when frozen retrieval traces cannot support the ablation."""


def default_output_dir() -> Path:
    return project_root() / "docs/evaluation/runs/formal/retrieval"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FormalNeighborExpansionAblationError(
            f"Cannot load source retrieval report {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise FormalNeighborExpansionAblationError(
            f"Expected JSON object in {path}"
        )
    return value


def _mean(values: list[float]) -> float:
    return _round(statistics.mean(values)) if values else 0.0


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


def _acceptance(
    summaries: dict[str, Any], results: list[dict[str, Any]]
) -> dict[str, Any]:
    baseline = summaries["dense_top_5"]["overall"]
    selected = summaries["rerank_neighbor_2"]["overall"]
    macro_delta = _round(
        float(selected["macro_reference_recall_at_k"])
        - float(baseline["macro_reference_recall_at_k"])
    )
    market = next(
        row
        for row in summaries["rerank_neighbor_2"]["by_domain"]
        if row["domain"] == "market_validation"
    )
    mean_chunks = _mean(
        [
            float(result["profile_metrics"]["rerank_neighbor_2"]["retrieved_chunk_count"])
            for result in results
        ]
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
        "mean_evidence_cards_at_most_5": {
            "threshold": 5,
            "observed": mean_chunks,
            "passed": mean_chunks <= 5,
        },
    }
    return {
        "all_gates_passed": all(gate["passed"] for gate in gates.values()),
        "gates": gates,
    }


def _percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def _markdown(report: dict[str, Any]) -> str:
    summaries = report["summaries"]
    lines = [
        f"# {report['report_id']}: Neighbor Expansion Ablation",
        "",
        "> Post-hoc exploratory retrieval-only ablation. No model calls or answers were generated.",
        "",
        "## Configuration",
        "",
        f"- Source retrieval report: `{report['source_report']['path']}`",
        f"- Context budget: {report['config']['max_context_tokens']} estimated tokens",
        "- Neighbor distances: 1 and 2 chunks within the same source document",
        "- Evidence cards remain fixed at five",
        "",
        "## Overall retrieval",
        "",
        "| Profile | Macro recall | Weighted recall | All keypoint refs | Mean chars |",
        "|---|---:|---:|---:|---:|",
    ]
    for profile in (
        "dense_top_5",
        "dense_neighbor_1",
        "dense_neighbor_2",
        "rerank_top_5",
        "rerank_neighbor_1",
        "rerank_neighbor_2",
    ):
        overall = summaries[profile]["overall"]
        lines.append(
            f"| {profile} | {_percent(overall['macro_reference_recall_at_k'])} | "
            f"{_percent(overall['evidence_weighted_reference_recall_at_k'])} | "
            f"{_percent(overall['keypoint_evidence_availability']['all']['rate'])} | "
            f"{overall['retrieval_context']['mean_char_count']:.0f} |"
        )
    lines.extend(
        [
            "",
            "## Rerank + distance 2 by domain",
            "",
            "| Domain | Macro recall | Weighted recall | All keypoint refs |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in summaries["rerank_neighbor_2"]["by_domain"]:
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
            if isinstance(observed, float) and "recall" in name
            else str(observed)
        )
        lines.append(f"| {name} | {rendered} | {'yes' if gate['passed'] else 'no'} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Neighbor text is merged into its selected evidence card, so the number of citations does not increase. The strict same-source quote metric can therefore measure whether local context repairs right-document/wrong-chunk failures. The 3000-token estimator budget is enforced before every merge.",
            "",
        ]
    )
    return "\n".join(lines)


def build_formal_neighbor_expansion_ablation(
    *,
    qra_items: list[dict[str, Any]],
    source_report_path: Path,
    output_dir: Path,
    report_id: str,
    max_context_tokens: int = 3000,
    expansion_distances: tuple[int, ...] = (1, 2),
    expansion_fn: ExpansionFunction = _expand_context_hits_with_neighbors,
) -> dict[str, Path]:
    """Expand frozen dense/reranked Top-5 traces without new model calls."""

    if not qra_items or expansion_distances != (1, 2):
        raise FormalNeighborExpansionAblationError(
            "This frozen ablation requires QRA items and distances (1, 2)"
        )
    if max_context_tokens < 1:
        raise FormalNeighborExpansionAblationError(
            "max_context_tokens must be positive"
        )
    source_path = source_report_path.resolve()
    source = _load_json(source_path)
    if source.get("status") != "completed":
        raise FormalNeighborExpansionAblationError(
            "Source retrieval report must be completed"
        )
    source_results = {
        str(result.get("qra_id", "")): result
        for result in source.get("results", [])
        if isinstance(result, dict) and result.get("status") == "success"
    }
    expected_ids = {str(qra["id"]) for qra in qra_items}
    if set(source_results) != expected_ids:
        raise FormalNeighborExpansionAblationError(
            "Source retrieval result IDs do not match QRA items"
        )
    current_index = snapshot_rag_index()
    source_index = source.get("rag_index", {})
    for key in ("metadata_sha256", "embeddings_sha256"):
        if source_index.get(key) and source_index.get(key) != current_index.get(key):
            raise FormalNeighborExpansionAblationError(
                f"Current RAG index differs from source report: {key}"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{report_id}.json"
    markdown_path = output_dir / f"{report_id}.md"
    for path in (json_path, markdown_path):
        if path.exists():
            raise FormalNeighborExpansionAblationError(
                f"Output already exists: {path}"
            )
    config: RAGConfig = _resolve_config(None)
    results: list[dict[str, Any]] = []
    for qra in qra_items:
        qra_id = str(qra["id"])
        source_result = source_results[qra_id]
        dense_hits = sorted(
            source_result["dense_candidates"],
            key=lambda item: int(item["dense_rank"]),
        )[:5]
        rerank_hits = list(source_result["selected_chunks"])
        observations = {
            "dense_top_5": _metric_observation(
                qra, dense_hits, profile="dense_top_5"
            ),
            "rerank_top_5": _metric_observation(
                qra, rerank_hits, profile="rerank_top_5"
            ),
        }
        expansion_artifacts: dict[str, Any] = {}
        for name, hits in (("dense", dense_hits), ("rerank", rerank_hits)):
            for distance in expansion_distances:
                expanded, estimated_tokens, latency_ms = expansion_fn(
                    hits,
                    config=config,
                    max_context_tokens=max_context_tokens,
                    max_neighbor_distance=distance,
                )
                profile = f"{name}_neighbor_{distance}"
                observations[profile] = _metric_observation(
                    qra, expanded, profile=profile
                )
                expansion_artifacts[profile] = {
                    "estimated_context_tokens": estimated_tokens,
                    "latency_ms": latency_ms,
                    "expanded_neighbor_chunk_ids": [
                        chunk_id
                        for hit in expanded
                        for chunk_id in hit.get(
                            "expanded_neighbor_chunk_ids", []
                        )
                    ],
                }
        results.append(
            {
                "status": "success",
                "qra_id": qra_id,
                "domain": str(qra["domain"]),
                "question_type": str(qra["question_type"]),
                "profile_metrics": observations,
                "expansion": expansion_artifacts,
            }
        )

    summaries = _summaries(results)
    report = {
        "report_id": report_id,
        "status": "completed",
        "evaluation_status": "posthoc_exploratory_retrieval_only_ablation",
        "completed_at": datetime.now(UTC).isoformat(),
        "source_report": {
            "path": _display_path(source_path),
            "sha256": sha256_file(source_path),
        },
        "qra_manifest_sha256": _canonical_sha256(
            [
                {
                    "id": qra["id"],
                    "question": qra["question"],
                    "references": qra.get("references", []),
                }
                for qra in qra_items
            ]
        ),
        "rag_index": current_index,
        "config": {
            "max_context_tokens": max_context_tokens,
            "expansion_distances": list(expansion_distances),
            "same_source_only": True,
            "implementation": {
                "path": _display_path(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
        },
        "results": results,
        "summaries": summaries,
    }
    report["acceptance"] = _acceptance(summaries, results)
    _write_json_atomic(json_path, report)
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path}
