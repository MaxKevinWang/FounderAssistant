"""Tests for structured request assessment and application-controlled actions."""

from __future__ import annotations

import json
from typing import Any

from rest_framework.test import APIRequestFactory

from apps.advisors import request_guard, views


def test_explicit_evasion_request_is_refused_without_llm_call(monkeypatch) -> None:
    def fail_if_called(**kwargs: Any) -> dict[str, Any]:
        raise AssertionError("The LLM must not be called for an explicit evasion pattern")

    monkeypatch.setattr(request_guard, "generate_text", fail_if_called)

    assessment = request_guard.assess_advisor_request(
        "How can I hide revenue without reporting it for tax?"
    )

    assert assessment.action == "refuse_and_redirect"
    assert assessment.questionable_shortcut is True
    assert assessment.source == "deterministic_rule"
    assert "cannot help evade" in assessment.api_payload()["message"]


def test_debt_concealment_request_is_refused_without_llm_call(monkeypatch) -> None:
    def fail_if_called(**kwargs: Any) -> dict[str, Any]:
        raise AssertionError("The LLM must not be called for debt concealment")

    monkeypatch.setattr(request_guard, "generate_text", fail_if_called)

    assessment = request_guard.assess_advisor_request(
        "Wie kann ich Investoren bestehende Schulden verschweigen?"
    )

    assert assessment.action == "refuse_and_redirect"
    assert assessment.questionable_shortcut is True
    assert assessment.source == "deterministic_rule"


def test_structured_classifier_requests_one_clarification(monkeypatch) -> None:
    monkeypatch.setattr(
        request_guard,
        "generate_text",
        lambda **kwargs: {
            "response": """```json
{"language":"en","risk_level":"low","risk_categories":[],"questionable_shortcut":false,"sensitive_categories":[],"needs_clarification":true,"clarification_question":"How much capital can you invest initially?"}
```"""
        },
    )

    assessment = request_guard.assess_advisor_request(
        "Which legal form is best for me?",
        domain="legal_form_selection",
    )

    assert assessment.action == "clarify"
    assert assessment.needs_clarification is True
    assert assessment.clarification_question == "How much capital can you invest initially?"


def test_guard_uses_small_history_budget_and_reports_telemetry(monkeypatch) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.delenv("ADVISOR_GUARD_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("ADVISOR_GUARD_MAX_TOKENS", raising=False)
    monkeypatch.delenv("ADVISOR_GUARD_REASONING_EFFORT", raising=False)

    def fake_generation(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {
            "response": '{"language":"en","risk_level":"low","risk_categories":[],"questionable_shortcut":false,"sensitive_categories":[],"needs_clarification":false,"clarification_question":""}',
            "model": "ministral-3:14b",
            "latency_ms": 321,
            "usage": {"prompt_tokens": 410, "completion_tokens": 55, "ignored": "x"},
        }

    monkeypatch.setattr(request_guard, "generate_text", fake_generation)
    history = [
        {"role": "user" if index % 2 == 0 else "assistant", "content": str(index) * 900}
        for index in range(6)
    ]

    assessment = request_guard.assess_advisor_request(
        "What is a customer interview?",
        conversation_history=history,
        model="ministral-3:14b",
    )
    classifier_payload = json.loads(captured["prompt"])

    assert captured["max_tokens"] == request_guard.GUARD_MAX_OUTPUT_TOKENS == 200
    assert captured["timeout_seconds"] == 30
    assert len(classifier_payload["history"]) <= 4
    assert sum(len(item["content"]) for item in classifier_payload["history"]) <= 2400
    assert classifier_payload["history"][-1]["content"].startswith("5")
    assert assessment.guard_model == "ministral-3:14b"
    assert assessment.guard_latency_ms == 321
    assert assessment.guard_usage == {"prompt_tokens": 410, "completion_tokens": 55}


def test_guard_accepts_model_specific_output_and_reasoning_settings(monkeypatch) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setenv("ADVISOR_GUARD_MAX_TOKENS", "800")
    monkeypatch.setenv("ADVISOR_GUARD_REASONING_EFFORT", "low")

    def fake_generation(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {
            "response": '{"language":"de","risk_level":"low","risk_categories":[],"questionable_shortcut":false,"sensitive_categories":[],"needs_clarification":false,"clarification_question":""}',
        }

    monkeypatch.setattr(request_guard, "generate_text", fake_generation)

    assessment = request_guard.assess_advisor_request(
        "Welche Unterlagen brauche ich vor einem Investorengespräch?"
    )

    assert assessment.source == "llm_structured"
    assert captured["max_tokens"] == 800
    assert captured["reasoning_effort"] == "low"


def test_high_risk_sensitive_result_adds_notice_and_generation_instruction(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        request_guard,
        "generate_text",
        lambda **kwargs: {
            "response": """{"language":"de","risk_level":"high","risk_categories":["tax"],"questionable_shortcut":false,"sensitive_categories":["internal_financial_information"],"needs_clarification":false,"clarification_question":""}"""
        },
    )

    payload = request_guard.assess_advisor_request(
        "Wie behandle ich diese konkrete Steuererklärung?"
    ).api_payload()

    assert payload["action"] == "answer_with_warning"
    assert "risikoreichen Entscheidung" in payload["notice"]
    assert "sensiblen Informationen" in payload["notice"]
    assert "preliminary orientation" in payload["generation_instruction"]
    assert "Do not repeat" in payload["generation_instruction"]


def test_invalid_classifier_output_uses_deterministic_fallback(monkeypatch) -> None:
    monkeypatch.setattr(
        request_guard,
        "generate_text",
        lambda **kwargs: {"response": "not json"},
    )

    assessment = request_guard.assess_advisor_request(
        "What general tax topics should a founder know?"
    )

    assert assessment.action == "answer"
    assert assessment.risk_level == "medium"
    assert assessment.risk_categories == ("tax",)
    assert assessment.source == "fallback_rule"


def test_request_assessment_endpoint_returns_validated_control_payload(
    monkeypatch,
) -> None:
    assessment = request_guard.RequestAssessment(
        action="clarify",
        language="en",
        needs_clarification=True,
        clarification_question="Who are your target customers?",
    )
    captured: dict[str, Any] = {}

    def fake_assessment(question: str, **kwargs: Any):
        captured.update({"question": question, **kwargs})
        return assessment

    monkeypatch.setattr(views, "assess_advisor_request", fake_assessment)
    request = APIRequestFactory().post(
        "/api/advisors/request-assessment/",
        {
            "prompt": "Give me a specific validation plan.",
            "domain": "market_validation",
            "history": [{"role": "user", "content": "I have a new product."}],
            "model": "test:model",
        },
        format="json",
    )

    response = views.AdvisorRequestAssessmentView.as_view()(request)

    assert response.status_code == 200
    assert response.data["action"] == "clarify"
    assert response.data["message"] == "Who are your target customers?"
    assert captured == {
        "question": "Give me a specific validation plan.",
        "domain": "market_validation",
        "conversation_history": [
            {"role": "user", "content": "I have a new product."}
        ],
        "model": "test:model",
    }
