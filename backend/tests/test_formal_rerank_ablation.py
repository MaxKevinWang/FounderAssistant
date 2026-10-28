"""Tests for the formal retrieval-only reranker ablation."""

from __future__ import annotations

import json
from pathlib import Path

from apps.advisors.rag_pipeline import RetrievalSelection
from apps.evaluations.formal_rerank_ablation import run_formal_rerank_ablation


def _qra(qra_id: str, domain: str) -> dict[str, object]:
    return {
        "id": qra_id,
        "version": "1.0",
        "status": "frozen",
        "split": "test",
        "review_status": "single_checked",
        "domain": domain,
        "question_type": "Factual",
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


def _retrieval(**kwargs) -> RetrievalSelection:  # noqa: ANN003
    domain = kwargs["domain"]
    candidates = []
    for dense_rank in range(1, 21):
        exact = dense_rank == 7
        candidates.append(
            {
                "rank": 1 if exact else dense_rank + 1,
                "dense_rank": dense_rank,
                "score": 3.0 if exact else -float(dense_rank),
                "score_type": "cross_encoder_logit",
                "dense_score": 1.0 - dense_rank / 100,
                "reranker_score": 3.0 if exact else -float(dense_rank),
                "reranker_model": kwargs["reranker_model"],
                "reranker_revision": kwargs["reranker_revision"],
                "chunk_id": f"{domain}/chunk-{dense_rank}",
                "domain": domain,
                "source_path": (
                    f"{domain}/source.txt"
                    if exact
                    else f"{domain}/other-{dense_rank}.txt"
                ),
                "source_name": "source",
                "chunk_index": dense_rank,
                "text": "exact evidence" if exact else "unrelated",
            }
        )
    candidates.sort(key=lambda item: int(item["rank"]))
    for rank, item in enumerate(candidates, start=1):
        item["rank"] = rank
    selected = [dict(item) for item in candidates[:5]]
    return RetrievalSelection(
        hits=selected,
        candidates=candidates,
        retrieval_mode="dense_embedding_rerank",
        candidate_k=20,
        context_k=5,
        candidate_retrieval_latency_ms=10,
        reranker_latency_ms=20,
        selection_latency_ms=0,
        estimated_context_tokens=100,
        token_estimator="test",
        reranker_model=kwargs["reranker_model"],
        reranker_revision=kwargs["reranker_revision"],
    )


def test_formal_rerank_ablation_compares_same_candidate_pool(
    tmp_path: Path,
    monkeypatch,
) -> None:
    qras = [
        _qra("Q1", "market_validation"),
        _qra("Q2", "funding_preparation"),
        _qra("Q3", "legal_form_selection"),
    ]
    monkeypatch.setattr(
        "apps.evaluations.formal_rerank_ablation.snapshot_rag_index",
        lambda: {
            "metadata_sha256": "metadata",
            "embeddings_sha256": "embeddings",
        },
    )

    paths = run_formal_rerank_ablation(
        qra_items=qras,
        output_dir=tmp_path,
        report_id="RERANK-TEST",
        retrieval_fn=_retrieval,
    )

    report = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert report["status"] == "completed"
    assert report["summaries"]["dense_top_5"]["overall"][
        "macro_reference_recall_at_k"
    ] == 0.0
    assert report["summaries"]["dense_top_8"]["overall"][
        "macro_reference_recall_at_k"
    ] == 1.0
    assert report["summaries"]["rerank_top_5"]["overall"][
        "macro_reference_recall_at_k"
    ] == 1.0
    assert report["acceptance"]["all_gates_passed"] is True
    assert not (tmp_path / "RERANK-TEST.partial.json").exists()
    assert "Cross-Encoder Reranking" in paths["markdown"].read_text(
        encoding="utf-8"
    )
