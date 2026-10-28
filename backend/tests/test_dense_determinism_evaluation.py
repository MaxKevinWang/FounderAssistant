"""Tests for frozen-query Dense@5 determinism evaluation."""

from pathlib import Path

import numpy as np

from apps.evaluations.dense_determinism import run_dense_determinism_evaluation


def _qra() -> dict:
    return {
        "id": "FTMV-TEST-001",
        "domain": "market_validation",
        "question_type": "Factual",
        "language": "de",
        "question": "Welche Evidenz ist relevant?",
        "answerable": True,
        "answer": "Target evidence.",
        "references": [
            {
                "reference_id": "R1",
                "raw_source_path": "data/market_validation/target.html",
                "quote": "Target evidence.",
            }
        ],
        "keypoints": [{"id": "K1", "reference_ids": ["R1"]}],
        "configuration_path": "config.json",
        "configuration_sha256": "config-hash",
        "split": "test",
        "review_status": "single_checked",
        "_artifact": {"path": "qra.json", "sha256": "qra-hash"},
    }


def _hit(chunk_id: str, rank: int, text: str) -> dict:
    return {
        "rank": rank,
        "score": 1.0 - rank / 10,
        "chunk_id": chunk_id,
        "domain": "market_validation",
        "source_path": "market_validation/target.html",
        "source_name": "target",
        "chunk_index": rank,
        "text": text,
    }


def test_freezes_one_query_embedding_and_reuses_it_for_three_repeats(
    tmp_path: Path,
) -> None:
    embed_calls: list[list[str]] = []
    retrieval_vectors: list[np.ndarray] = []

    def fake_embed(queries, config):  # noqa: ANN001, ANN202
        embed_calls.append(list(queries))
        return np.asarray([[3.0, 4.0]], dtype=np.float32)

    def fake_retrieve(**kwargs):  # noqa: ANN003, ANN202
        retrieval_vectors.append(np.asarray(kwargs["query_embedding"]).copy())
        return [
            _hit("target::1", 1, "Target evidence."),
            _hit("target::2", 2, "Background."),
        ], 2, "dense_embedding_frozen_query"

    result = run_dense_determinism_evaluation(
        qra_items=[_qra()],
        output_dir=tmp_path,
        report_id="TEST-DENSE-DETERMINISM-001",
        repeats=3,
        candidate_k=2,
        context_k=1,
        config=None,
        embed_fn=fake_embed,
        retrieve_fn=fake_retrieve,
        index_snapshot={
            "metadata_sha256": "metadata-hash",
            "embeddings_sha256": "index-embeddings-hash",
            "embedding_model": "test-embedding",
            "embedding_provider": "test-provider",
        },
    )

    report = result["report"]
    assert embed_calls == [["Welche Evidenz ist relevant?"]]
    assert len(retrieval_vectors) == 3
    assert all(np.array_equal(vector, retrieval_vectors[0]) for vector in retrieval_vectors)
    assert report["acceptance"]["all_gates_passed"] is True
    assert report["results"][0]["deterministic"] is True
    assert report["results"][0]["context_chunk_ids"] == ["target::1"]
    assert report["summaries"]["dense_top_1"]["overall"][
        "evidence_weighted_reference_recall_at_k"
    ] == 1.0
    assert Path(result["query_embeddings"]).exists()
    assert Path(result["query_embedding_manifest"]).exists()
    assert Path(result["json"]).exists()
    assert Path(result["markdown"]).exists()


def test_fails_determinism_gate_when_repeated_candidate_order_changes(
    tmp_path: Path,
) -> None:
    call_count = 0

    def fake_embed(queries, config):  # noqa: ANN001, ANN202
        return np.asarray([[1.0, 0.0]], dtype=np.float32)

    def fake_retrieve(**kwargs):  # noqa: ANN003, ANN202
        nonlocal call_count
        call_count += 1
        hits = [
            _hit("target::1", 1, "Target evidence."),
            _hit("target::2", 2, "Background."),
        ]
        if call_count == 2:
            hits.reverse()
            for rank, hit in enumerate(hits, start=1):
                hit["rank"] = rank
        return hits, 2, "dense_embedding_frozen_query"

    result = run_dense_determinism_evaluation(
        qra_items=[_qra()],
        output_dir=tmp_path,
        report_id="TEST-DENSE-DETERMINISM-FAIL-001",
        repeats=3,
        candidate_k=2,
        context_k=1,
        config=None,
        embed_fn=fake_embed,
        retrieve_fn=fake_retrieve,
        index_snapshot={
            "metadata_sha256": "metadata-hash",
            "embeddings_sha256": "index-embeddings-hash",
            "embedding_model": "test-embedding",
            "embedding_provider": "test-provider",
        },
    )

    assert result["report"]["acceptance"]["all_gates_passed"] is False
    assert result["report"]["acceptance"]["gates"][
        "all_candidate_rankings_identical"
    ] is False
    assert result["report"]["results"][0]["deterministic"] is False


def test_replays_frozen_query_embedding_without_live_embedding_call(
    tmp_path: Path,
) -> None:
    def fake_embed(queries, config):  # noqa: ANN001, ANN202
        return np.asarray([[3.0, 4.0]], dtype=np.float32)

    def fake_retrieve(**kwargs):  # noqa: ANN003, ANN202
        return [
            _hit("target::1", 1, "Target evidence."),
            _hit("target::2", 2, "Background."),
        ], 2, "dense_embedding_frozen_query"

    index = {
        "metadata_sha256": "metadata-hash",
        "embeddings_sha256": "index-embeddings-hash",
        "embedding_model": "test-embedding",
        "embedding_provider": "test-provider",
    }
    frozen = run_dense_determinism_evaluation(
        qra_items=[_qra()],
        output_dir=tmp_path,
        report_id="TEST-DENSE-FREEZE-001",
        repeats=3,
        candidate_k=2,
        context_k=1,
        config=None,
        embed_fn=fake_embed,
        retrieve_fn=fake_retrieve,
        index_snapshot=index,
    )

    replay = run_dense_determinism_evaluation(
        qra_items=[_qra()],
        output_dir=tmp_path,
        report_id="TEST-DENSE-REPLAY-001",
        repeats=3,
        candidate_k=2,
        context_k=1,
        config=None,
        embed_fn=lambda queries, config: (_ for _ in ()).throw(
            AssertionError("live embedding service must not be called")
        ),
        retrieve_fn=fake_retrieve,
        index_snapshot=index,
        query_embedding_manifest_path=Path(frozen["query_embedding_manifest"]),
    )

    assert replay["report"]["acceptance"]["all_gates_passed"] is True
    assert replay["report"]["query_embeddings"]["manifest_sha256"] == frozen[
        "report"
    ]["query_embeddings"]["manifest_sha256"]
