"""Tests for candidate retrieval, cross-encoder reranking, and context selection."""

from __future__ import annotations

import numpy as np
import pytest

from apps.advisors import rag_pipeline


def _hit(position: int, *, text: str | None = None) -> dict[str, object]:
    return {
        "rank": position,
        "score": 1.0 - (position / 100),
        "chunk_id": f"chunk-{position}",
        "domain": "market_validation",
        "source_path": f"market/source-{position}.html",
        "source_name": f"source-{position}",
        "chunk_index": position,
        "text": text or f"Evidence candidate {position}",
        "text_preview": text or f"Evidence candidate {position}",
    }


def test_rerank_candidates_preserves_dense_provenance_and_reorders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = [_hit(1), _hit(2), _hit(3)]
    monkeypatch.setattr(
        rag_pipeline,
        "_score_reranker_pairs",
        lambda **kwargs: [0.2, 2.4, -0.1],
    )

    reranked, latency_ms = rag_pipeline._rerank_candidates(
        query="Which evidence is most relevant?",
        candidates=candidates,
        model_name="test/reranker",
        batch_size=2,
        max_length=128,
    )

    assert [hit["chunk_id"] for hit in reranked] == [
        "chunk-2",
        "chunk-1",
        "chunk-3",
    ]
    assert [hit["rank"] for hit in reranked] == [1, 2, 3]
    assert [hit["dense_rank"] for hit in reranked] == [2, 1, 3]
    assert reranked[0]["dense_score"] == candidates[1]["score"]
    assert reranked[0]["reranker_score"] == 2.4
    assert reranked[0]["score_type"] == "cross_encoder_logit"
    assert latency_ms >= 0


def test_select_context_respects_count_and_token_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = [_hit(position) for position in range(1, 7)]
    token_counts = {
        "Evidence candidate 1": 700,
        "Evidence candidate 2": 700,
        "Evidence candidate 3": 700,
        "Evidence candidate 4": 200,
        "Evidence candidate 5": 200,
        "Evidence candidate 6": 200,
    }
    monkeypatch.setattr(
        rag_pipeline,
        "_estimate_text_tokens",
        lambda text: token_counts[str(text)],
    )

    selected, estimated_tokens, latency_ms = rag_pipeline._select_context_hits(
        candidates,
        context_k=5,
        max_context_tokens=1700,
        metadata_token_allowance=0,
    )

    assert [hit["chunk_id"] for hit in selected] == [
        "chunk-1",
        "chunk-2",
        "chunk-4",
    ]
    assert [hit["rank"] for hit in selected] == [1, 2, 3]
    assert estimated_tokens == 1600
    assert latency_ms >= 0


def test_retrieve_context_uses_twenty_candidates_but_selects_five(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = [_hit(position) for position in range(1, 21)]
    calls: list[dict[str, object]] = []

    def fake_retrieve(**kwargs):  # noqa: ANN003, ANN202
        calls.append(kwargs)
        return candidates, 12, "dense_embedding"

    def fake_rerank(**kwargs):  # noqa: ANN003, ANN202
        reranked = [dict(hit) for hit in reversed(kwargs["candidates"])]
        for rank, hit in enumerate(reranked, start=1):
            hit["rank"] = rank
            hit["dense_rank"] = 21 - rank
            hit["dense_score"] = hit["score"]
            hit["reranker_score"] = float(21 - rank)
            hit["score_type"] = "cross_encoder_logit"
        return reranked, 34

    monkeypatch.setattr(rag_pipeline, "_retrieve_chunks", fake_retrieve)
    monkeypatch.setattr(rag_pipeline, "_rerank_candidates", fake_rerank)

    result = rag_pipeline._retrieve_context(
        query="question",
        config=None,
        context_k=5,
        candidate_k=20,
        domain="market_validation",
        reranker_model="test/reranker",
        max_context_tokens=3000,
        allow_keyword_fallback=False,
    )

    assert calls[0]["top_k"] == 20
    assert calls[0]["allow_keyword_fallback"] is False
    assert len(result.candidates) == 20
    assert len(result.hits) == 5
    assert result.hits[0]["chunk_id"] == "chunk-20"
    assert result.retrieval_mode == "dense_embedding_rerank"
    assert result.candidate_retrieval_latency_ms == 12
    assert result.reranker_latency_ms == 34
    assert result.context_k == 5
    assert result.candidate_k == 20


def test_retrieve_chunks_can_forbid_keyword_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks = [
        {
            "chunk_id": "market-1",
            "domain": "market_validation",
            "source_path": "market/source.html",
            "source_name": "source",
            "chunk_index": 0,
            "text": "Evidence",
        }
    ]
    monkeypatch.setattr(
        rag_pipeline,
        "_load_index",
        lambda config: (
            np.asarray([[1.0, 0.0]], dtype=np.float32),
            {"chunks": chunks},
        ),
    )
    monkeypatch.setattr(
        rag_pipeline,
        "_embed_texts",
        lambda texts, config: (_ for _ in ()).throw(
            rag_pipeline.RAGError("embedding unavailable")
        ),
    )

    with pytest.raises(rag_pipeline.RAGError, match="embedding unavailable"):
        rag_pipeline._retrieve_chunks(
            query="question",
            config=None,
            top_k=5,
            domain="market_validation",
            allow_keyword_fallback=False,
        )


def test_retrieve_chunks_reuses_provided_query_embedding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks = [
        {
            "chunk_id": "market-1",
            "domain": "market_validation",
            "source_path": "market/one.html",
            "source_name": "one",
            "chunk_index": 0,
            "text": "First evidence",
        },
        {
            "chunk_id": "market-2",
            "domain": "market_validation",
            "source_path": "market/two.html",
            "source_name": "two",
            "chunk_index": 0,
            "text": "Second evidence",
        },
    ]
    monkeypatch.setattr(
        rag_pipeline,
        "_load_index",
        lambda config: (
            np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            {"chunks": chunks},
        ),
    )
    monkeypatch.setattr(
        rag_pipeline,
        "_embed_texts",
        lambda texts, config: (_ for _ in ()).throw(
            AssertionError("live embedding service must not be called")
        ),
    )

    first, _, first_mode = rag_pipeline._retrieve_chunks(
        query="question",
        config=None,
        top_k=2,
        domain="market_validation",
        allow_keyword_fallback=False,
        query_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
    )
    second, _, second_mode = rag_pipeline._retrieve_chunks(
        query="question",
        config=None,
        top_k=2,
        domain="market_validation",
        allow_keyword_fallback=False,
        query_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
    )

    assert [row["chunk_id"] for row in first] == ["market-1", "market-2"]
    assert first == second
    assert first_mode == second_mode == "dense_embedding_frozen_query"


def test_retrieve_chunks_rejects_invalid_provided_query_embedding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        rag_pipeline,
        "_load_index",
        lambda config: (
            np.asarray([[1.0, 0.0]], dtype=np.float32),
            {
                "chunks": [
                    {
                        "chunk_id": "market-1",
                        "domain": "market_validation",
                        "source_path": "market/one.html",
                        "source_name": "one",
                        "chunk_index": 0,
                        "text": "Evidence",
                    }
                ]
            },
        ),
    )

    with pytest.raises(rag_pipeline.RAGError, match="finite one-dimensional"):
        rag_pipeline._retrieve_chunks(
            query="question",
            config=None,
            top_k=1,
            domain="market_validation",
            allow_keyword_fallback=False,
            query_embedding=np.asarray([np.nan, 0.0], dtype=np.float32),
        )


def test_retrieve_chunks_breaks_exact_score_ties_by_chunk_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks = [
        {
            "chunk_id": "market-b",
            "domain": "market_validation",
            "source_path": "market/b.html",
            "source_name": "b",
            "chunk_index": 0,
            "text": "Evidence B",
        },
        {
            "chunk_id": "market-a",
            "domain": "market_validation",
            "source_path": "market/a.html",
            "source_name": "a",
            "chunk_index": 0,
            "text": "Evidence A",
        },
    ]
    monkeypatch.setattr(
        rag_pipeline,
        "_load_index",
        lambda config: (
            np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
            {"chunks": chunks},
        ),
    )

    hits, _, _ = rag_pipeline._retrieve_chunks(
        query="question",
        config=None,
        top_k=2,
        domain="market_validation",
        allow_keyword_fallback=False,
        query_embedding=np.asarray([1.0, 0.0], dtype=np.float32),
    )

    assert [row["chunk_id"] for row in hits] == ["market-a", "market-b"]


def test_neighbor_expansion_merges_only_adjacent_chunks_from_same_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks = [
        {
            "chunk_id": "source::0",
            "source_path": "market/source.html",
            "chunk_index": 0,
            "text": "previous evidence",
            "page_start": "1",
            "page_end": "1",
        },
        {
            "chunk_id": "source::1",
            "source_path": "market/source.html",
            "chunk_index": 1,
            "text": "selected evidence",
            "page_start": "2",
            "page_end": "2",
        },
        {
            "chunk_id": "source::2",
            "source_path": "market/source.html",
            "chunk_index": 2,
            "text": "next evidence",
            "page_start": "3",
            "page_end": "3",
        },
        {
            "chunk_id": "other::0",
            "source_path": "market/other.html",
            "chunk_index": 0,
            "text": "other source",
        },
    ]
    monkeypatch.setattr(
        rag_pipeline,
        "_load_index",
        lambda config: (np.empty((0, 0)), {"chunks": chunks}),
    )

    expanded, estimated_tokens, latency_ms = (
        rag_pipeline._expand_context_hits_with_neighbors(
            [
                {
                    **chunks[1],
                    "rank": 1,
                    "score": 1.0,
                }
            ],
            config=None,
            max_context_tokens=3000,
            max_neighbor_distance=1,
        )
    )

    assert expanded[0]["expanded_neighbor_chunk_ids"] == [
        "source::0",
        "source::2",
    ]
    assert "previous evidence" in expanded[0]["text"]
    assert "next evidence" in expanded[0]["text"]
    assert "other source" not in expanded[0]["text"]
    assert expanded[0]["expanded_neighbor_count"] == 2
    assert estimated_tokens <= 3000
    assert latency_ms >= 0


def test_neighbor_expansion_skips_neighbors_that_exceed_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks = [
        {
            "chunk_id": "source::0",
            "source_path": "market/source.html",
            "chunk_index": 0,
            "text": "neighbor",
        },
        {
            "chunk_id": "source::1",
            "source_path": "market/source.html",
            "chunk_index": 1,
            "text": "selected",
        },
    ]
    monkeypatch.setattr(
        rag_pipeline,
        "_load_index",
        lambda config: (np.empty((0, 0)), {"chunks": chunks}),
    )
    monkeypatch.setattr(
        rag_pipeline,
        "_estimate_text_tokens",
        lambda text: 100 if text == "selected" else 500,
    )

    expanded, estimated_tokens, _ = (
        rag_pipeline._expand_context_hits_with_neighbors(
            [{**chunks[1], "rank": 1, "score": 1.0}],
            config=None,
            max_context_tokens=200,
            max_neighbor_distance=1,
            metadata_token_allowance=0,
        )
    )

    assert expanded[0]["text"] == "selected"
    assert expanded[0]["expanded_neighbor_chunk_ids"] == []
    assert estimated_tokens == 100


@pytest.mark.parametrize(
    ("scores", "expected_level"),
    [
        ([-1.0, -2.0, -3.0], "insufficient"),
        ([0.2, -1.0, -2.0], "limited"),
        ([0.2, 0.1, -2.0], "adequate"),
    ],
)
def test_retrieval_confidence_uses_nonnegative_reranker_count(
    scores: list[float],
    expected_level: str,
) -> None:
    hits = [
        {**_hit(index + 1), "reranker_score": score}
        for index, score in enumerate(scores)
    ]

    confidence = rag_pipeline._assess_retrieval_confidence(
        hits,
        reranker_model="test/reranker",
    )

    assert confidence["level"] == expected_level
    assert confidence["nonnegative_reranker_hit_count"] == sum(
        score >= 0 for score in scores
    )
    assert confidence["policy_version"] == "reranker_nonnegative_count_v1"
