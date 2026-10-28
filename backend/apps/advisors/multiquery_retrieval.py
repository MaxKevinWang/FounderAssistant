"""Multi-query rank fusion and coverage-aware context selection."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from apps.advisors.rag_pipeline import (
    TOKEN_ESTIMATOR_ID,
    RAGConfig,
    RAGError,
    _estimate_text_tokens,
    _retrieve_chunks,
    _score_reranker_pairs,
    _select_context_hits,
)


DEFAULT_RRF_K = 60
DEFAULT_CONTEXT_METADATA_TOKEN_ALLOWANCE = 64
DEFAULT_MULTILINGUAL_RERANKER_MODEL = (
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
)
DEFAULT_MULTILINGUAL_RERANKER_REVISION = (
    "1427fd652930e4ba29e8149678df786c240d8825"
)
RetrievalFunction = Callable[..., tuple[list[dict[str, Any]], int, str]]


@dataclass(frozen=True)
class MultiQueryRetrieval:
    """Auditable multi-query candidate pool and bounded final context."""

    hits: list[dict[str, Any]]
    candidates: list[dict[str, Any]]
    query_results: list[dict[str, Any]]
    retrieval_modes: list[str]
    retrieval_mode: str
    per_query_k: int
    candidate_k: int
    context_k: int
    candidate_retrieval_latency_ms: int
    fusion_latency_ms: int
    selection_latency_ms: int
    estimated_context_tokens: int
    token_estimator: str
    reranker_latency_ms: int = 0
    reranker_model: str = ""
    reranker_revision: str = ""

    @property
    def total_retrieval_latency_ms(self) -> int:
        return (
            self.candidate_retrieval_latency_ms
            + self.fusion_latency_ms
            + self.reranker_latency_ms
            + self.selection_latency_ms
        )


def fuse_query_rankings(
    query_results: list[tuple[str, list[dict[str, Any]]]],
    *,
    candidate_k: int,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[dict[str, Any]]:
    """Fuse unique chunks using reciprocal-rank fusion across query views."""

    if not query_results:
        raise RAGError("At least one query result is required for rank fusion")
    if candidate_k < 1:
        raise RAGError("candidate_k must be at least 1")
    if rrf_k < 0:
        raise RAGError("rrf_k cannot be negative")

    query_count = len(query_results)
    fused: dict[str, dict[str, Any]] = {}
    for query_index, (query, hits) in enumerate(query_results):
        seen_in_query: set[str] = set()
        for fallback_rank, hit in enumerate(hits, start=1):
            chunk_id = str(hit.get("chunk_id", "")).strip()
            if not chunk_id or chunk_id in seen_in_query:
                continue
            seen_in_query.add(chunk_id)
            rank = int(hit.get("rank", fallback_rank))
            if rank < 1:
                raise RAGError("Retrieved ranks must be positive integers")
            if chunk_id not in fused:
                item = dict(hit)
                item["query_ranks"] = [None] * query_count
                item["query_scores"] = [None] * query_count
                item["matched_queries"] = []
                item["rrf_score"] = 0.0
                fused[chunk_id] = item
            item = fused[chunk_id]
            item["query_ranks"][query_index] = rank
            item["query_scores"][query_index] = float(hit.get("score", 0.0))
            item["matched_queries"].append(query)
            item["rrf_score"] += 1.0 / (rrf_k + rank)

    ranked = list(fused.values())
    for item in ranked:
        ranks = [rank for rank in item["query_ranks"] if rank is not None]
        item["query_hit_count"] = len(ranks)
        item["best_query_rank"] = min(ranks)
        original_rank = item["query_ranks"][0]
        item["original_query_rank"] = original_rank
        item["dense_rank"] = original_rank or min(ranks)
        original_score = item["query_scores"][0]
        item["dense_score"] = (
            float(original_score)
            if original_score is not None
            else float(item.get("score", 0.0))
        )
        item["score"] = float(item["rrf_score"])
        item["score_type"] = "reciprocal_rank_fusion"
    ranked.sort(
        key=lambda item: (
            -float(item["rrf_score"]),
            -int(item["query_hit_count"]),
            int(item["best_query_rank"]),
            str(item["chunk_id"]),
        )
    )
    ranked = ranked[:candidate_k]
    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank
    return ranked


def select_query_coverage_hits(
    candidates: list[dict[str, Any]],
    *,
    query_count: int,
    context_k: int,
    max_context_tokens: int | None,
    metadata_token_allowance: int = DEFAULT_CONTEXT_METADATA_TOKEN_ALLOWANCE,
    rank_field: str = "query_ranks",
) -> tuple[list[dict[str, Any]], int, int]:
    """Greedily maximize complementary query-view coverage under a budget."""

    if query_count < 1:
        raise RAGError("query_count must be at least 1")
    if context_k < 1:
        raise RAGError("context_k must be at least 1")
    if max_context_tokens is not None and max_context_tokens < 1:
        raise RAGError("max_context_tokens must be at least 1 when provided")
    if metadata_token_allowance < 0:
        raise RAGError("metadata_token_allowance cannot be negative")

    started = time.perf_counter()
    remaining = list(candidates)
    selected: list[dict[str, Any]] = []
    covered = [0.0] * query_count
    total_tokens = 0
    while remaining and len(selected) < context_k:
        options: list[tuple[float, float, int, dict[str, Any], int]] = []
        for candidate in remaining:
            ranks = list(candidate.get(rank_field, []))
            if len(ranks) != query_count:
                raise RAGError(
                    f"Candidate {rank_field} do not match query_count"
                )
            candidate_tokens = _estimate_text_tokens(
                str(candidate.get("text", ""))
            ) + metadata_token_allowance
            if (
                max_context_tokens is not None
                and total_tokens + candidate_tokens > max_context_tokens
            ):
                continue
            relevance = [0.0 if rank is None else 1.0 / int(rank) for rank in ranks]
            gain = sum(
                max(0.0, relevance[index] - covered[index])
                for index in range(query_count)
            )
            options.append(
                (
                    gain,
                    float(candidate.get("rrf_score", candidate.get("score", 0.0))),
                    -int(candidate.get("rank", 0)),
                    candidate,
                    candidate_tokens,
                )
            )
        if not options:
            break
        gain, _, _, candidate, candidate_tokens = max(
            options, key=lambda option: option[:3]
        )
        ranks = list(candidate[rank_field])
        for index, rank in enumerate(ranks):
            if rank is not None:
                covered[index] = max(covered[index], 1.0 / int(rank))
        item = dict(candidate)
        item["pre_selection_rank"] = int(candidate.get("rank", 0))
        item["coverage_gain"] = round(gain, 6)
        item["query_coverage_after_selection"] = [round(value, 6) for value in covered]
        item["estimated_text_tokens"] = max(
            0, candidate_tokens - metadata_token_allowance
        )
        item["estimated_context_tokens"] = candidate_tokens
        selected.append(item)
        total_tokens += candidate_tokens
        remaining.remove(candidate)

    if not selected:
        raise RAGError("No multi-query candidate fits the context token budget")
    for rank, item in enumerate(selected, start=1):
        item["rank"] = rank
    latency_ms = int((time.perf_counter() - started) * 1000)
    return selected, total_tokens, latency_ms


def rerank_candidates_for_queries(
    *,
    queries: list[str],
    candidates: list[dict[str, Any]],
    model_name: str = DEFAULT_MULTILINGUAL_RERANKER_MODEL,
    revision: str = DEFAULT_MULTILINGUAL_RERANKER_REVISION,
    batch_size: int = 16,
    max_length: int = 512,
    device: str = "auto",
) -> tuple[list[dict[str, Any]], int]:
    """Score every fused candidate against every query view."""

    if not queries or not candidates:
        raise RAGError("Queries and candidates are required for multi-query reranking")
    if not model_name.strip() or not revision.strip():
        raise RAGError("A pinned multi-query reranker model and revision are required")
    started = time.perf_counter()
    passages = [str(candidate.get("text", "")) for candidate in candidates]
    score_columns = [
        _score_reranker_pairs(
            query=query,
            passages=passages,
            model_name=model_name,
            revision=revision,
            batch_size=batch_size,
            max_length=max_length,
            device=device,
        )
        for query in queries
    ]
    if any(len(scores) != len(candidates) for scores in score_columns):
        raise RAGError("Multi-query reranker returned an invalid score matrix")

    rank_columns: list[list[int]] = []
    for scores in score_columns:
        order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
        ranks = [0] * len(scores)
        for rank, candidate_index in enumerate(order, start=1):
            ranks[candidate_index] = rank
        rank_columns.append(ranks)

    reranked: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(candidates):
        scores = [column[candidate_index] for column in score_columns]
        ranks = [column[candidate_index] for column in rank_columns]
        best_query_index = max(
            range(len(scores)), key=lambda index: (scores[index], -index)
        )
        item = dict(candidate)
        item["fusion_rank"] = int(candidate.get("rank", candidate_index + 1))
        item["query_reranker_scores"] = [float(value) for value in scores]
        item["query_reranker_ranks"] = ranks
        item["max_query_reranker_score"] = float(scores[best_query_index])
        item["best_reranker_query_index"] = best_query_index
        item["reranker_model"] = model_name
        item["reranker_revision"] = revision
        item["score"] = float(scores[best_query_index])
        item["score_type"] = "max_multiquery_cross_encoder_logit"
        reranked.append(item)
    reranked.sort(
        key=lambda item: (
            -float(item["max_query_reranker_score"]),
            int(item["fusion_rank"]),
        )
    )
    for rank, item in enumerate(reranked, start=1):
        item["rank"] = rank
    latency_ms = int((time.perf_counter() - started) * 1000)
    return reranked, latency_ms


def retrieve_multiquery_context(
    *,
    queries: list[str],
    config: RAGConfig,
    per_query_k: int = 20,
    candidate_k: int = 20,
    context_k: int = 5,
    domain: str = "",
    max_context_tokens: int | None = 3000,
    rrf_k: int = DEFAULT_RRF_K,
    allow_keyword_fallback: bool = False,
    reranker_model: str = "",
    reranker_revision: str = "",
    reranker_batch_size: int = 16,
    reranker_max_length: int = 512,
    reranker_device: str = "auto",
    retrieve_fn: RetrievalFunction = _retrieve_chunks,
) -> MultiQueryRetrieval:
    """Retrieve each query view, fuse candidates, and select bounded evidence."""

    normalized_queries = [" ".join(query.split()) for query in queries if query.strip()]
    if not normalized_queries:
        raise RAGError("At least one non-empty retrieval query is required")
    if len({query.casefold() for query in normalized_queries}) != len(
        normalized_queries
    ):
        raise RAGError("Retrieval queries must be unique")
    if per_query_k < 1 or candidate_k < context_k:
        raise RAGError("per_query_k must be positive and candidate_k >= context_k")

    query_results: list[dict[str, Any]] = []
    rankings: list[tuple[str, list[dict[str, Any]]]] = []
    retrieval_modes: list[str] = []
    retrieval_ms = 0
    for query in normalized_queries:
        hits, latency_ms, mode = retrieve_fn(
            query=query,
            config=config,
            top_k=per_query_k,
            domain=domain,
            allow_keyword_fallback=allow_keyword_fallback,
        )
        if not allow_keyword_fallback and "keyword_fallback" in mode:
            raise RAGError("Keyword fallback is forbidden for this multi-query run")
        retrieval_ms += int(latency_ms)
        retrieval_modes.append(mode)
        rankings.append((query, hits))
        query_results.append(
            {
                "query": query,
                "retrieval_mode": mode,
                "latency_ms": int(latency_ms),
                "hits": hits,
            }
        )

    fusion_started = time.perf_counter()
    candidates = fuse_query_rankings(
        rankings,
        candidate_k=candidate_k,
        rrf_k=rrf_k,
    )
    fusion_ms = int((time.perf_counter() - fusion_started) * 1000)
    selected_reranker_model = reranker_model.strip()
    reranker_ms = 0
    selected_revision = reranker_revision.strip()
    if selected_reranker_model:
        candidates, reranker_ms = rerank_candidates_for_queries(
            queries=normalized_queries,
            candidates=candidates,
            model_name=selected_reranker_model,
            revision=selected_revision,
            batch_size=reranker_batch_size,
            max_length=reranker_max_length,
            device=reranker_device,
        )
        hits, estimated_tokens, selection_ms = _select_context_hits(
            candidates,
            context_k=context_k,
            max_context_tokens=max_context_tokens,
        )
    else:
        hits, estimated_tokens, selection_ms = select_query_coverage_hits(
            candidates,
            query_count=len(normalized_queries),
            context_k=context_k,
            max_context_tokens=max_context_tokens,
        )
    first_stage = (
        retrieval_modes[0]
        if len(set(retrieval_modes)) == 1
        else "mixed_retrieval"
    )
    mode_suffix = (
        "multiquery_rrf_cross_encoder_max"
        if selected_reranker_model
        else "multiquery_rrf_coverage"
    )
    return MultiQueryRetrieval(
        hits=hits,
        candidates=candidates,
        query_results=query_results,
        retrieval_modes=retrieval_modes,
        retrieval_mode=f"{first_stage}_{mode_suffix}",
        per_query_k=per_query_k,
        candidate_k=candidate_k,
        context_k=context_k,
        candidate_retrieval_latency_ms=retrieval_ms,
        fusion_latency_ms=fusion_ms,
        selection_latency_ms=selection_ms,
        estimated_context_tokens=estimated_tokens,
        token_estimator=TOKEN_ESTIMATOR_ID,
        reranker_latency_ms=reranker_ms,
        reranker_model=selected_reranker_model,
        reranker_revision=selected_revision,
    )
