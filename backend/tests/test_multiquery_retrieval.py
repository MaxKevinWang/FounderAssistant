from __future__ import annotations

from pathlib import Path
from typing import Any

from apps.advisors.multiquery_retrieval import (
    fuse_query_rankings,
    rerank_candidates_for_queries,
    retrieve_multiquery_context,
    select_query_coverage_hits,
)
from apps.advisors.rag_pipeline import RAGConfig


def _config() -> RAGConfig:
    return RAGConfig(
        data_dir=Path("knowledge_base"),
        index_dir=Path("index"),
        embedding_model="test-embedding",
        llm_provider="ollama",
        llm_base_url="http://localhost",
        llm_api_key="",
        verify_ssl=True,
        timeout_seconds=30,
        chunk_size=1200,
        chunk_overlap=180,
        min_chunk_chars=180,
    )


def _hit(chunk_id: str, rank: int, text: str | None = None) -> dict[str, Any]:
    return {
        "rank": rank,
        "score": 1.0 / rank,
        "chunk_id": chunk_id,
        "source_path": f"{chunk_id}.md",
        "text": text or (chunk_id * 30),
    }


def test_fuse_query_rankings_uses_rrf_and_preserves_per_query_provenance() -> None:
    fused = fuse_query_rankings(
        [
            ("original", [_hit("a", 1), _hit("b", 2)]),
            ("subquery", [_hit("c", 1), _hit("a", 2)]),
        ],
        candidate_k=3,
        rrf_k=60,
    )

    assert [item["chunk_id"] for item in fused] == ["a", "c", "b"]
    assert fused[0]["query_ranks"] == [1, 2]
    assert fused[0]["query_hit_count"] == 2
    assert fused[1]["query_ranks"] == [None, 1]
    assert fused[0]["score_type"] == "reciprocal_rank_fusion"


def test_query_coverage_selection_prefers_complementary_chunks() -> None:
    candidates = [
        {**_hit("broad", 1), "query_ranks": [1, 4], "rrf_score": 0.9},
        {**_hit("second-original", 2), "query_ranks": [2, None], "rrf_score": 0.8},
        {**_hit("subquery-best", 3), "query_ranks": [None, 1], "rrf_score": 0.7},
    ]

    selected, _, _ = select_query_coverage_hits(
        candidates,
        query_count=2,
        context_k=2,
        max_context_tokens=3000,
    )

    assert [item["chunk_id"] for item in selected] == [
        "broad",
        "subquery-best",
    ]
    assert selected[1]["coverage_gain"] > 0


def test_rerank_candidates_for_queries_records_cross_query_ranks(
    monkeypatch,
) -> None:
    candidates = [
        {**_hit("a", 1), "query_ranks": [1, 2], "rrf_score": 0.9},
        {**_hit("b", 2), "query_ranks": [2, 1], "rrf_score": 0.8},
    ]
    scores = {"original": [0.9, 0.2], "subquery": [0.1, 0.8]}

    def fake_score(**kwargs) -> list[float]:  # noqa: ANN003
        return scores[kwargs["query"]]

    monkeypatch.setattr(
        "apps.advisors.multiquery_retrieval._score_reranker_pairs",
        fake_score,
    )

    reranked, latency_ms = rerank_candidates_for_queries(
        queries=["original", "subquery"],
        candidates=candidates,
        model_name="test-reranker",
        revision="test-revision",
    )

    assert latency_ms >= 0
    assert reranked[0]["chunk_id"] == "a"
    assert reranked[0]["query_reranker_scores"] == [0.9, 0.1]
    assert reranked[0]["query_reranker_ranks"] == [1, 2]
    assert reranked[1]["query_reranker_ranks"] == [2, 1]
    assert reranked[0]["fusion_rank"] == 1
    assert reranked[0]["reranker_model"] == "test-reranker"


def test_query_coverage_selection_can_use_cross_encoder_rank_field() -> None:
    candidates = [
        {
            **_hit("original-best", 1),
            "query_ranks": [1, 1],
            "query_reranker_ranks": [1, 3],
            "rrf_score": 0.9,
        },
        {
            **_hit("subquery-best", 2),
            "query_ranks": [2, 2],
            "query_reranker_ranks": [3, 1],
            "rrf_score": 0.8,
        },
    ]

    selected, _, _ = select_query_coverage_hits(
        candidates,
        query_count=2,
        context_k=2,
        max_context_tokens=3000,
        rank_field="query_reranker_ranks",
    )

    assert [item["chunk_id"] for item in selected] == [
        "original-best",
        "subquery-best",
    ]


def test_retrieve_multiquery_context_fuses_one_domain_filtered_run_per_query() -> None:
    calls: list[tuple[str, int, str, bool]] = []
    answers = {
        "original": [_hit("a", 1), _hit("b", 2)],
        "subquery": [_hit("c", 1), _hit("a", 2)],
    }

    def fake_retrieve(
        query: str,
        config: RAGConfig,
        top_k: int,
        domain: str,
        allow_keyword_fallback: bool,
    ) -> tuple[list[dict[str, Any]], int, str]:
        assert config == _config()
        calls.append((query, top_k, domain, allow_keyword_fallback))
        return answers[query], 10, "dense_embedding"

    result = retrieve_multiquery_context(
        queries=["original", "subquery"],
        config=_config(),
        per_query_k=20,
        candidate_k=3,
        context_k=2,
        domain="market_validation",
        max_context_tokens=3000,
        allow_keyword_fallback=False,
        retrieve_fn=fake_retrieve,
    )

    assert calls == [
        ("original", 20, "market_validation", False),
        ("subquery", 20, "market_validation", False),
    ]
    assert len(result.candidates) == 3
    assert len(result.hits) == 2
    assert result.retrieval_modes == ["dense_embedding", "dense_embedding"]
    assert result.candidate_retrieval_latency_ms == 20
    assert result.retrieval_mode == "dense_embedding_multiquery_rrf_coverage"
