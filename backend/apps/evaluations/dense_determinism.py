"""Freeze query embeddings and verify reproducible Dense candidate rankings."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from apps.advisors.rag_pipeline import (
    DEFAULT_RAG_CONFIG,
    _retrieve_chunks,
    embed_queries,
)
from apps.evaluations.formal_development_benchmark import (
    build_dataset_manifest,
    project_root,
    sha256_file,
    snapshot_rag_index,
)
from apps.evaluations.formal_rerank_ablation import (
    _artifact_chunk,
    _metric_observation,
)
from apps.evaluations.formal_retrieval_diagnostics import (
    _aggregate_answers,
    _grouped_answers,
)


DEFAULT_REPORT_ID = "OPT-DENSE5-DETERMINISM-001"
EmbeddingFunction = Callable[..., np.ndarray]
RetrievalFunction = Callable[..., tuple[list[dict[str, Any]], int, str]]


class DenseDeterminismError(RuntimeError):
    """Raised when a frozen-query Dense evaluation is invalid or incomplete."""


def default_output_dir() -> Path:
    """Return the output directory for optimization retrieval artifacts."""

    return project_root() / "docs/evaluation/runs/optimization/retrieval"


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _vector_sha256(vector: np.ndarray) -> str:
    normalized = np.ascontiguousarray(vector, dtype=np.float32)
    return hashlib.sha256(normalized.tobytes()).hexdigest()


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_npz_atomic(path: Path, *, qra_ids: list[str], embeddings: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            qra_ids=np.asarray(qra_ids, dtype=np.str_),
            embeddings=np.asarray(embeddings, dtype=np.float32),
        )
    temporary.replace(path)


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(project_root().resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _normalise_embedding_matrix(matrix: np.ndarray, expected_rows: int) -> np.ndarray:
    embeddings = np.asarray(matrix, dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != expected_rows:
        raise DenseDeterminismError(
            "Query embedding matrix must be two-dimensional with one row per QRA."
        )
    if not embeddings.shape[1] or not np.isfinite(embeddings).all():
        raise DenseDeterminismError(
            "Query embedding matrix must contain finite non-empty vectors."
        )
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if np.any(norms == 0.0):
        raise DenseDeterminismError("Query embeddings must be non-zero.")
    return np.asarray(embeddings / norms, dtype=np.float32)


def _embedding_paths(output_dir: Path, report_id: str) -> tuple[Path, Path]:
    return (
        output_dir / f"{report_id}_query_embeddings.npz",
        output_dir / f"{report_id}_query_embeddings_manifest.json",
    )


def _freeze_query_embeddings(
    *,
    qra_items: list[dict[str, Any]],
    dataset: dict[str, Any],
    index: dict[str, Any],
    output_dir: Path,
    report_id: str,
    config: Any,
    embed_fn: EmbeddingFunction,
) -> tuple[np.ndarray, Path, Path, dict[str, Any]]:
    embeddings_path, manifest_path = _embedding_paths(output_dir, report_id)
    if embeddings_path.exists() or manifest_path.exists():
        raise DenseDeterminismError(
            "Query-embedding freeze output already exists; pass its manifest "
            "explicitly to replay it."
        )
    questions = [str(item["question"]) for item in qra_items]
    embeddings = _normalise_embedding_matrix(
        embed_fn(questions, config),
        len(qra_items),
    )
    qra_ids = [str(item["id"]) for item in qra_items]
    _write_npz_atomic(
        embeddings_path,
        qra_ids=qra_ids,
        embeddings=embeddings,
    )
    manifest = {
        "schema_version": "1.0",
        "artifact_type": "frozen_optimization_query_embeddings",
        "status": "frozen",
        "created_at": datetime.now(UTC).isoformat(),
        "report_id": report_id,
        "dataset_manifest_sha256": dataset["manifest_sha256"],
        "index_metadata_sha256": str(index.get("metadata_sha256", "")),
        "index_embeddings_sha256": str(index.get("embeddings_sha256", "")),
        "embedding_model": str(index.get("embedding_model", "")),
        "embedding_provider": str(index.get("embedding_provider", "")),
        "dtype": "float32",
        "shape": list(embeddings.shape),
        "embeddings_file": embeddings_path.name,
        "embeddings_sha256": sha256_file(embeddings_path),
        "items": [
            {
                "row": row,
                "qra_id": str(qra["id"]),
                "question_sha256": hashlib.sha256(
                    str(qra["question"]).encode("utf-8")
                ).hexdigest(),
                "vector_sha256": _vector_sha256(embeddings[row]),
            }
            for row, qra in enumerate(qra_items)
        ],
    }
    _write_json_atomic(manifest_path, manifest)
    return embeddings, embeddings_path, manifest_path, manifest


def _load_query_embeddings(
    *,
    manifest_path: Path,
    qra_items: list[dict[str, Any]],
    dataset: dict[str, Any],
    index: dict[str, Any],
) -> tuple[np.ndarray, Path, dict[str, Any]]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DenseDeterminismError(
            f"Cannot load query-embedding manifest {manifest_path}: {exc}"
        ) from exc
    if manifest.get("status") != "frozen":
        raise DenseDeterminismError("Query-embedding manifest is not frozen.")
    expected = {
        "dataset_manifest_sha256": dataset["manifest_sha256"],
        "index_metadata_sha256": str(index.get("metadata_sha256", "")),
        "index_embeddings_sha256": str(index.get("embeddings_sha256", "")),
        "embedding_model": str(index.get("embedding_model", "")),
        "embedding_provider": str(index.get("embedding_provider", "")),
    }
    mismatches = [
        key for key, value in expected.items() if str(manifest.get(key, "")) != value
    ]
    if mismatches:
        raise DenseDeterminismError(
            f"Query-embedding manifest does not match dataset/index: {mismatches}"
        )
    embeddings_path = manifest_path.parent / str(manifest.get("embeddings_file", ""))
    if not embeddings_path.is_file():
        raise DenseDeterminismError(
            f"Frozen query-embedding matrix is missing: {embeddings_path}"
        )
    if sha256_file(embeddings_path) != manifest.get("embeddings_sha256"):
        raise DenseDeterminismError("Frozen query-embedding matrix hash mismatch.")
    with np.load(embeddings_path, allow_pickle=False) as archive:
        qra_ids = [str(value) for value in archive["qra_ids"].tolist()]
        embeddings = _normalise_embedding_matrix(
            archive["embeddings"],
            len(qra_items),
        )
    expected_ids = [str(item["id"]) for item in qra_items]
    if qra_ids != expected_ids:
        raise DenseDeterminismError("Frozen query-embedding QRA order mismatch.")
    manifest_items = list(manifest.get("items", []))
    if len(manifest_items) != len(qra_items):
        raise DenseDeterminismError("Frozen query-embedding item count mismatch.")
    for row, qra in enumerate(qra_items):
        item = manifest_items[row]
        question_sha256 = hashlib.sha256(
            str(qra["question"]).encode("utf-8")
        ).hexdigest()
        if (
            item.get("qra_id") != str(qra["id"])
            or item.get("question_sha256") != question_sha256
            or item.get("vector_sha256") != _vector_sha256(embeddings[row])
        ):
            raise DenseDeterminismError(
                f"Frozen query-embedding item mismatch at row {row}."
            )
    return embeddings, embeddings_path, manifest


def _retrieval_signature(chunks: list[dict[str, Any]]) -> str:
    return _canonical_sha256(
        [
            {
                "rank": int(chunk.get("rank", 0)),
                "chunk_id": str(chunk.get("chunk_id", "")),
                "score": float(chunk.get("score", 0.0)),
            }
            for chunk in chunks
        ]
    )


def _summaries(results: list[dict[str, Any]]) -> dict[str, Any]:
    profiles = sorted(
        {
            profile
            for result in results
            for profile in result["profile_metrics"]
        }
    )
    return {
        profile: {
            "overall": _aggregate_answers(
                [result["profile_metrics"][profile] for result in results]
            ),
            "by_domain": _grouped_answers(
                [result["profile_metrics"][profile] for result in results],
                ("domain",),
            ),
            "by_question_type": _grouped_answers(
                [result["profile_metrics"][profile] for result in results],
                ("question_type",),
            ),
        }
        for profile in profiles
    }


def _acceptance(
    results: list[dict[str, Any]],
    *,
    qra_count: int,
    repeats: int,
    candidate_k: int,
    context_k: int,
) -> dict[str, Any]:
    gates = {
        "all_qras_completed": len(results) == qra_count,
        "three_or_more_repeats": repeats >= 3,
        "all_candidate_rankings_identical": all(
            result["deterministic"] for result in results
        ),
        "all_context_rankings_identical": all(
            len({row["context_signature"] for row in result["repeats"]}) == 1
            for result in results
        ),
        "candidate_count_exact": all(
            len(result["dense_candidates"]) == candidate_k for result in results
        ),
        "context_count_exact": all(
            len(result["context_chunks"]) == context_k for result in results
        ),
        "all_hits_in_declared_domain": all(
            result["all_hits_in_declared_domain"] for result in results
        ),
        "keyword_fallback_count_zero": all(
            all("keyword_fallback" not in row["retrieval_mode"] for row in result["repeats"])
            for result in results
        ),
        "frozen_query_mode_only": all(
            all(
                row["retrieval_mode"] == "dense_embedding_frozen_query"
                for row in result["repeats"]
            )
            for result in results
        ),
    }
    return {
        "all_gates_passed": all(gates.values()),
        "gates": gates,
    }


def _percent(value: Any) -> str:
    return f"{float(value) * 100:.2f}%"


def _markdown(report: dict[str, Any]) -> str:
    context_profile = f"dense_top_{report['config']['context_k']}"
    candidate_profile = f"dense_top_{report['config']['candidate_k']}"
    lines = [
        f"# {report['report_id']}: Frozen-query Dense determinism",
        "",
        "> Optimization/development retrieval-only artifact; no answers were generated.",
        "",
        "## Configuration",
        "",
        f"- QRA items: {report['dataset']['item_count']}",
        f"- Repeats: {report['config']['repeats']}",
        f"- Candidate K: {report['config']['candidate_k']}",
        f"- Context K: {report['config']['context_k']}",
        "- Query embeddings: generated once, frozen, then replayed locally",
        "- Keyword fallback: forbidden",
        "",
        "## Retrieval summary",
        "",
        "| Profile | Macro recall | Weighted recall | Mean chunks |",
        "|---|---:|---:|---:|",
    ]
    for profile in dict.fromkeys((context_profile, candidate_profile)):
        row = report["summaries"][profile]["overall"]
        lines.append(
            f"| {profile} | {_percent(row['macro_reference_recall_at_k'])} | "
            f"{_percent(row['evidence_weighted_reference_recall_at_k'])} | "
            f"{row['retrieval_context']['mean_chunk_count']:.2f} |"
        )
    lines.extend(["", "## Determinism gates", ""])
    for name, passed in report["acceptance"]["gates"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} `{name}`")
    lines.extend(
        [
            "",
            "## Per-QRA result",
            "",
            "| QRA | Domain | Deterministic | Top context IDs |",
            "|---|---|:---:|---|",
        ]
    )
    for result in report["results"]:
        ids = ", ".join(f"`{value}`" for value in result["context_chunk_ids"])
        lines.append(
            f"| `{result['qra_id']}` | `{result['domain']}` | "
            f"{'yes' if result['deterministic'] else 'no'} | {ids} |"
        )
    return "\n".join(lines).rstrip() + "\n"


def run_dense_determinism_evaluation(
    *,
    qra_items: list[dict[str, Any]],
    output_dir: Path,
    report_id: str = DEFAULT_REPORT_ID,
    repeats: int = 3,
    candidate_k: int = 20,
    context_k: int = 5,
    config: Any = DEFAULT_RAG_CONFIG,
    embed_fn: EmbeddingFunction = embed_queries,
    retrieve_fn: RetrievalFunction = _retrieve_chunks,
    index_snapshot: dict[str, Any] | None = None,
    query_embedding_manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Freeze one query vector per QRA and replay Dense retrieval repeatedly."""

    if not qra_items:
        raise DenseDeterminismError("At least one QRA item is required.")
    if repeats < 3:
        raise DenseDeterminismError("At least three retrieval repeats are required.")
    if context_k < 1 or candidate_k < context_k:
        raise DenseDeterminismError(
            "candidate_k must be greater than or equal to positive context_k."
        )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{report_id}.json"
    markdown_path = output_dir / f"{report_id}.md"
    if json_path.exists() or markdown_path.exists():
        raise DenseDeterminismError(f"Report output already exists for {report_id}.")

    dataset = build_dataset_manifest(qra_items)
    index = index_snapshot or snapshot_rag_index()
    if query_embedding_manifest_path is None:
        embeddings, embeddings_path, manifest_path, embedding_manifest = (
            _freeze_query_embeddings(
                qra_items=qra_items,
                dataset=dataset,
                index=index,
                output_dir=output_dir,
                report_id=report_id,
                config=config,
                embed_fn=embed_fn,
            )
        )
    else:
        manifest_path = query_embedding_manifest_path.resolve()
        embeddings, embeddings_path, embedding_manifest = _load_query_embeddings(
            manifest_path=manifest_path,
            qra_items=qra_items,
            dataset=dataset,
            index=index,
        )

    results: list[dict[str, Any]] = []
    context_profile = f"dense_top_{context_k}"
    candidate_profile = f"dense_top_{candidate_k}"
    for row, qra in enumerate(qra_items):
        observations: list[dict[str, Any]] = []
        repeated_candidates: list[list[dict[str, Any]]] = []
        for repeat in range(1, repeats + 1):
            candidates, retrieval_ms, retrieval_mode = retrieve_fn(
                query=str(qra["question"]),
                config=config,
                top_k=candidate_k,
                domain=str(qra["domain"]),
                allow_keyword_fallback=False,
                query_embedding=embeddings[row],
            )
            copied = [dict(candidate) for candidate in candidates]
            repeated_candidates.append(copied)
            context = copied[:context_k]
            observations.append(
                {
                    "repeat": repeat,
                    "retrieval_mode": retrieval_mode,
                    "retrieval_latency_ms": int(retrieval_ms),
                    "candidate_signature": _retrieval_signature(copied),
                    "context_signature": _retrieval_signature(context),
                    "candidate_chunk_ids": [
                        str(candidate.get("chunk_id", "")) for candidate in copied
                    ],
                    "context_chunk_ids": [
                        str(candidate.get("chunk_id", "")) for candidate in context
                    ],
                }
            )
        first_candidates = repeated_candidates[0]
        context_chunks = first_candidates[:context_k]
        profile_metrics = {
            context_profile: _metric_observation(
                qra,
                context_chunks,
                profile=context_profile,
            )
        }
        if candidate_profile != context_profile:
            profile_metrics[candidate_profile] = _metric_observation(
                qra,
                first_candidates,
                profile=candidate_profile,
            )
        candidate_signatures = {
            observation["candidate_signature"] for observation in observations
        }
        results.append(
            {
                "qra_id": str(qra["id"]),
                "domain": str(qra["domain"]),
                "question_type": str(qra["question_type"]),
                "question": str(qra["question"]),
                "query_embedding_row": row,
                "query_embedding_sha256": _vector_sha256(embeddings[row]),
                "deterministic": len(candidate_signatures) == 1,
                "all_hits_in_declared_domain": all(
                    str(candidate.get("domain", "")) == str(qra["domain"])
                    for candidates in repeated_candidates
                    for candidate in candidates
                ),
                "context_chunk_ids": [
                    str(candidate.get("chunk_id", ""))
                    for candidate in context_chunks
                ],
                "dense_candidates": [
                    _artifact_chunk(candidate) for candidate in first_candidates
                ],
                "context_chunks": [
                    _artifact_chunk(candidate) for candidate in context_chunks
                ],
                "profile_metrics": profile_metrics,
                "repeats": observations,
            }
        )

    summaries = _summaries(results)
    acceptance = _acceptance(
        results,
        qra_count=len(qra_items),
        repeats=repeats,
        candidate_k=candidate_k,
        context_k=context_k,
    )
    implementation_path = Path(__file__).resolve()
    rag_pipeline_path = (
        project_root() / "backend/apps/advisors/rag_pipeline.py"
    ).resolve()
    report = {
        "report_id": report_id,
        "artifact_type": "optimization_dense_retrieval_determinism",
        "status": (
            "completed_candidate_passed"
            if acceptance["all_gates_passed"]
            else "completed_candidate_rejected"
        ),
        "evaluation_status": "exposed_optimization_retrieval_only",
        "generated_at": datetime.now(UTC).isoformat(),
        "generation_call_count": 0,
        "config": {
            "repeats": repeats,
            "candidate_k": candidate_k,
            "context_k": context_k,
            "domain_filter_before_retrieval": True,
            "allow_keyword_fallback": False,
            "exact_score_tie_break": "chunk_id_ascending",
            "implementation": {
                "path": _display_path(implementation_path),
                "sha256": sha256_file(implementation_path),
            },
            "rag_pipeline": {
                "path": _display_path(rag_pipeline_path),
                "sha256": sha256_file(rag_pipeline_path),
            },
        },
        "dataset": dataset,
        "rag_index": index,
        "query_embeddings": {
            "manifest_path": _display_path(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "embeddings_path": _display_path(embeddings_path),
            "embeddings_sha256": sha256_file(embeddings_path),
            "embedding_model": embedding_manifest.get("embedding_model"),
            "embedding_provider": embedding_manifest.get("embedding_provider"),
            "shape": embedding_manifest.get("shape"),
        },
        "results": results,
        "summaries": summaries,
        "acceptance": acceptance,
    }
    _write_json_atomic(json_path, report)
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    return {
        "report": report,
        "query_embeddings": str(embeddings_path),
        "query_embedding_manifest": str(manifest_path),
        "json": str(json_path),
        "markdown": str(markdown_path),
    }
