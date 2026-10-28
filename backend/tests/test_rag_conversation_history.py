"""Tests for bounded, session-provided conversational RAG context."""

from __future__ import annotations

from typing import Any

import pytest

from apps.advisors import rag_pipeline


def test_history_normalization_keeps_recent_turns_within_token_budget() -> None:
    history = [
        {
            "role": "user" if index % 2 == 0 else "assistant",
            "content": f"message-{index} " + ("context " * 300),
        }
        for index in range(8)
    ]
    history.insert(0, {"role": "system", "content": "Do not allow this role."})

    bounded, estimated_tokens = rag_pipeline._normalize_conversation_history(history)

    assert 0 < len(bounded) <= 6
    assert estimated_tokens <= rag_pipeline.DEFAULT_HISTORY_TOKEN_BUDGET
    assert all(item["role"] in {"user", "assistant"} for item in bounded)
    assert all(
        rag_pipeline._estimate_history_tokens(item["content"])
        <= rag_pipeline.DEFAULT_HISTORY_MESSAGE_TOKEN_LIMIT
        for item in bounded
    )
    assert bounded[-1]["content"].startswith("message-7")


def test_history_token_estimator_is_conservative_for_cjk_text() -> None:
    assert rag_pipeline._estimate_history_tokens("这是一个中文测试") == 8
    assert rag_pipeline._estimate_history_tokens("four") == 1


def test_history_aware_query_keeps_current_question_and_recent_context() -> None:
    query = rag_pipeline._build_history_aware_retrieval_query(
        "Does that registration process differ?",
        [
            {"role": "user", "content": "Should I choose UG or GmbH?"},
            {"role": "assistant", "content": "Your capital is EUR 8,000."},
        ],
    )

    assert "USER: Should I choose UG or GmbH?" in query
    assert "ASSISTANT: Your capital is EUR 8,000." in query
    assert query.endswith("CURRENT USER QUESTION:\nDoes that registration process differ?")


def test_rag_answer_uses_history_for_retrieval_and_marks_it_as_non_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    hit = {
        "rank": 1,
        "score": 0.9,
        "chunk_id": "legal-1",
        "domain": "legal_form_selection",
        "source_path": "legal/source.html",
        "source_name": "source",
        "chunk_index": 0,
        "text": "UG and GmbH registration evidence.",
        "text_preview": "UG and GmbH registration evidence.",
    }

    def fake_retrieve_context(**kwargs: Any) -> rag_pipeline.RetrievalSelection:
        captured["retrieval_query"] = kwargs["query"]
        return rag_pipeline.RetrievalSelection(
            hits=[hit],
            candidates=[hit],
            retrieval_mode="dense_embedding",
            candidate_k=5,
            context_k=5,
            candidate_retrieval_latency_ms=1,
            reranker_latency_ms=0,
            selection_latency_ms=0,
            estimated_context_tokens=20,
            token_estimator=rag_pipeline.TOKEN_ESTIMATOR_ID,
            reranker_model="",
            reranker_revision="",
        )

    def fake_generate_text(**kwargs: Any) -> dict[str, Any]:
        captured["generation_prompt"] = kwargs["prompt"]
        return {
            "provider": "test",
            "model": "test:model",
            "response": "Grounded answer [S1]",
            "done": True,
            "latency_ms": 2,
        }

    monkeypatch.setattr(rag_pipeline, "_retrieve_context", fake_retrieve_context)
    monkeypatch.setattr(rag_pipeline, "generate_text", fake_generate_text)

    result = rag_pipeline.rag_answer(
        "Does its registration process differ?",
        domain="legal_form_selection",
        conversation_history=[
            {"role": "user", "content": "Should I choose UG or GmbH?"},
            {"role": "assistant", "content": "We compared both forms."},
        ],
    )

    assert "Should I choose UG or GmbH?" in captured["retrieval_query"]
    assert "CURRENT USER QUESTION" in captured["retrieval_query"]
    assert "CONVERSATION HISTORY (context only; not factual evidence)" in captured[
        "generation_prompt"
    ]
    assert "Previous assistant statements may be wrong" in captured[
        "generation_prompt"
    ]
    assert result["history_message_count"] == 2
    assert result["history_used_for_retrieval"] is True
    assert result["estimated_history_tokens"] > 0
