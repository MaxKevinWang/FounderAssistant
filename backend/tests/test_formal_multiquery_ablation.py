"""Tests for the formal multi-query retrieval ablation."""

from __future__ import annotations

import json
from pathlib import Path

from apps.advisors.multiquery_retrieval import MultiQueryRetrieval
from apps.advisors.query_decomposition import QueryDecomposition
from apps.evaluations.formal_multiquery_ablation import (
    run_formal_multiquery_ablation,
)
from apps.evaluations.formal_query_decomposition_manifest import (
    run_query_decomposition_freeze,
)


def _qra(qra_id: str, domain: str) -> dict[str, object]:
    return {
        "id": qra_id,
        "version": "1.0",
        "status": "frozen",
        "split": "test",
        "review_status": "single_checked",
        "domain": domain,
        "question_type": "Information Integration",
        "language": "de",
        "question": f"Question {qra_id}",
        "answerable": True,
        "answer": "Answer",
        "reference_answer": "Answer",
        "keypoints": [{"id": "K1", "reference_ids": ["R1"]}],
        "references": [
            {
                "reference_id": "R1",
                "raw_source_path": f"data/{domain}/source.txt",
                "quote": "exact evidence",
            }
        ],
        "configuration_path": "config.json",
        "configuration_sha256": "config-hash",
        "_artifact": {"path": f"{qra_id}.json", "sha256": f"hash-{qra_id}"},
    }


def _decompose(question: str, **kwargs) -> QueryDecomposition:  # noqa: ANN003
    return QueryDecomposition(
        original_query=question,
        needs_decomposition=True,
        subqueries=["first aspect", "second aspect"],
        provider="test",
        model=kwargs["model"],
        raw_response="{}",
        latency_ms=12,
        finish_reason="stop",
        usage={},
        prompt_version="test-v1",
        prompt_sha256="prompt-hash",
    )


def _retrieve(**kwargs) -> MultiQueryRetrieval:  # noqa: ANN003
    domain = kwargs["domain"]
    irrelevant = {
        "rank": 1,
        "dense_rank": 1,
        "score": 1.0,
        "chunk_id": f"{domain}/irrelevant",
        "source_path": f"{domain}/other.txt",
        "text": "unrelated",
        "query_ranks": [1, None, None],
        "rrf_score": 0.1,
    }
    exact = {
        "rank": 2,
        "dense_rank": 7,
        "score": 0.9,
        "chunk_id": f"{domain}/exact",
        "source_path": f"{domain}/source.txt",
        "text": "exact evidence",
        "query_ranks": [7, 1, None],
        "rrf_score": 0.09,
        "coverage_gain": 1.0,
    }
    distractors = [
        {
            **irrelevant,
            "rank": rank,
            "dense_rank": rank,
            "chunk_id": f"{domain}/distractor-{rank}",
            "query_ranks": [rank, None, rank],
            "rrf_score": 0.08 - rank / 1000,
        }
        for rank in range(3, 21)
    ]
    candidates = [irrelevant, exact, *distractors]
    selected = [exact, irrelevant, *distractors[:3]]
    original_hits = [irrelevant, *distractors[:5], exact, *distractors[5:18]]
    return MultiQueryRetrieval(
        hits=selected,
        candidates=candidates,
        query_results=[
            {
                "query": kwargs["queries"][0],
                "retrieval_mode": "dense_embedding",
                "latency_ms": 10,
                "hits": original_hits,
            },
            {
                "query": kwargs["queries"][1],
                "retrieval_mode": "dense_embedding",
                "latency_ms": 10,
                "hits": [exact],
            },
            {
                "query": kwargs["queries"][2],
                "retrieval_mode": "dense_embedding",
                "latency_ms": 10,
                "hits": distractors,
            },
        ],
        retrieval_modes=["dense_embedding"] * 3,
        retrieval_mode="dense_embedding_multiquery_rrf_coverage",
        per_query_k=20,
        candidate_k=20,
        context_k=5,
        candidate_retrieval_latency_ms=30,
        fusion_latency_ms=1,
        selection_latency_ms=1,
        estimated_context_tokens=100,
        token_estimator="test",
    )


def test_formal_multiquery_ablation_compares_original_and_coverage_selection(
    tmp_path: Path,
    monkeypatch,
) -> None:
    qras = [
        _qra("Q1", "market_validation"),
        _qra("Q2", "funding_preparation"),
        _qra("Q3", "legal_form_selection"),
    ]
    monkeypatch.setattr(
        "apps.evaluations.formal_multiquery_ablation.snapshot_rag_index",
        lambda: {
            "metadata_sha256": "metadata",
            "embeddings_sha256": "embeddings",
        },
    )

    paths = run_formal_multiquery_ablation(
        qra_items=qras,
        output_dir=tmp_path,
        report_id="MULTIQUERY-TEST",
        decomposer_model="test-model",
        decompose_fn=_decompose,
        retrieval_fn=_retrieve,
    )

    report = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert report["status"] == "completed"
    assert report["summaries"]["original_dense_top_5"]["overall"][
        "macro_reference_recall_at_k"
    ] == 0.0
    assert report["summaries"]["multiquery_coverage_top_5"]["overall"][
        "macro_reference_recall_at_k"
    ] == 1.0
    assert report["acceptance"]["all_gates_passed"] is True
    assert report["decomposition"]["decomposed_question_count"] == 3
    assert not (tmp_path / "MULTIQUERY-TEST.partial.json").exists()
    assert "Multi-Query Retrieval" in paths["markdown"].read_text(encoding="utf-8")


def test_formal_multiquery_ablation_replays_frozen_decompositions(
    tmp_path: Path,
    monkeypatch,
) -> None:
    qras = [
        _qra("Q1", "market_validation"),
        _qra("Q2", "funding_preparation"),
        _qra("Q3", "legal_form_selection"),
    ]
    manifest_path = run_query_decomposition_freeze(
        qra_items=qras,
        output_dir=tmp_path,
        manifest_id="DECOMP-TEST",
        model="test-model",
        decompose_fn=_decompose,
    )
    monkeypatch.setattr(
        "apps.evaluations.formal_multiquery_ablation.snapshot_rag_index",
        lambda: {
            "metadata_sha256": "metadata",
            "embeddings_sha256": "embeddings",
        },
    )

    def fail_if_called(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("live decomposer should not be called")

    paths = run_formal_multiquery_ablation(
        qra_items=qras,
        output_dir=tmp_path,
        report_id="MULTIQUERY-FROZEN-TEST",
        decomposer_model="test-model",
        decomposition_manifest_path=manifest_path,
        decompose_fn=fail_if_called,
        retrieval_fn=_retrieve,
    )

    report = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert report["config"]["decomposition_source"] == "frozen_manifest"
    assert report["latency"]["decomposition_runtime_mean_ms"] == 0.0
    assert report["config"]["decomposition_manifest"]["sha256"]
