from __future__ import annotations

import pytest

from apps.advisors.query_decomposition import (
    QueryDecompositionError,
    decompose_query,
    parse_query_decomposition,
)


def test_parse_query_decomposition_accepts_fenced_json_and_deduplicates() -> None:
    parsed = parse_query_decomposition(
        """```json
        {
          "needs_decomposition": true,
          "queries": [
            "Haftung bei GbR und OHG",
            " Haftung bei GbR und OHG ",
            "Kapital und Registerbezug bei GmbH und UG"
          ]
        }
        ```""",
        original_query="Vergleiche die Rechtsformen.",
        max_subqueries=4,
    )

    assert parsed == (
        True,
        [
            "Haftung bei GbR und OHG",
            "Kapital und Registerbezug bei GmbH und UG",
        ],
    )


def test_parse_query_decomposition_rejects_inconsistent_atomic_output() -> None:
    with pytest.raises(QueryDecompositionError, match="must not contain"):
        parse_query_decomposition(
            '{"needs_decomposition": false, "queries": ["extra query"]}',
            original_query="Was ist ein MVP?",
        )


def test_parse_query_decomposition_ignores_extra_audit_fields() -> None:
    parsed = parse_query_decomposition(
        (
            '{"needs_decomposition": true, '
            '"queries": ["erste Sicht", "zweite Sicht"], '
            '"reason": "compound question"}'
        ),
        original_query="Eine zusammengesetzte Frage",
    )

    assert parsed == (True, ["erste Sicht", "zweite Sicht"])


def test_decompose_query_records_auditable_generation_metadata() -> None:
    calls: list[dict[str, object]] = []

    def fake_generator(prompt: str, **kwargs: object) -> dict[str, object]:
        calls.append({"prompt": prompt, **kwargs})
        return {
            "provider": "test-provider",
            "model": "test-model",
            "response": (
                '{"needs_decomposition": true, "queries": '
                '["Umsatz und Kosten", "Risiko und Rentabilit\u00e4t"]}'
            ),
            "done": True,
            "latency_ms": 123,
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 10},
        }

    result = decompose_query(
        "Wie führen Umsatz und Kosten zu Risiko und Rentabilität?",
        model="test-model",
        generator=fake_generator,
    )

    assert result.needs_decomposition is True
    assert result.subqueries == ["Umsatz und Kosten", "Risiko und Rentabilität"]
    assert result.retrieval_queries == [
        "Wie führen Umsatz und Kosten zu Risiko und Rentabilität?",
        "Umsatz und Kosten",
        "Risiko und Rentabilität",
    ]
    assert result.model == "test-model"
    assert result.latency_ms == 123
    assert result.prompt_sha256
    assert calls[0]["temperature"] == 0.0
    assert calls[0]["max_tokens"] == 320
    assert calls[0]["seed"] == 0
