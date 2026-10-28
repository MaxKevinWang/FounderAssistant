"""Tests for normalized generation metadata from local LLM providers."""

from __future__ import annotations

from typing import Any

import pytest

from apps.advisors import local_llm


class _FakeResponse:
    def __init__(self, payload: dict[str, Any], *, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = "response body"

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeStreamingResponse:
    def __init__(self) -> None:
        self.encoding = "ISO-8859-1"

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    def iter_lines(self, *, decode_unicode: bool):
        assert decode_unicode is True
        assert self.encoding == "utf-8"
        yield "data: {\"choices\":[{\"delta\":{\"content\":\"Geschäftsidee\"}}]}"


def test_get_json_uses_requests_transport_with_runtime_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_get(url, **kwargs):
        captured.update(url=url, **kwargs)
        return _FakeResponse({"data": [{"id": "model-a"}]})

    monkeypatch.setattr(local_llm.requests, "get", fake_get)
    config = local_llm.LocalLLMConfig(
        provider="openwebui",
        base_url="https://example.invalid",
        model="model-a",
        timeout_seconds=17,
        api_key="test-key",
        verify_ssl=False,
    )

    payload = local_llm._get_json("https://example.invalid/api/models", config)

    assert payload["data"][0]["id"] == "model-a"
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["timeout"] == 17
    assert captured["verify"] is False


def test_stream_transport_forces_utf8_for_sse_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        local_llm.requests,
        "post",
        lambda *args, **kwargs: _FakeStreamingResponse(),
    )
    config = local_llm.LocalLLMConfig(
        provider="openwebui",
        base_url="https://example.invalid",
        model="model-a",
        timeout_seconds=17,
        api_key="test-key",
        verify_ssl=True,
    )

    lines = list(
        local_llm._post_stream_json(
            "https://example.invalid/api/chat/completions",
            {"stream": True},
            config,
            accept="text/event-stream",
        )
    )

    assert lines == [
        'data: {"choices":[{"delta":{"content":"Geschäftsidee"}}]}'
    ]


def test_openwebui_forwards_reasoning_effort_and_audits_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_post_json(url, payload, config):
        captured.update(url=url, payload=payload, config=config)
        return {
            "model": "kimi-k3",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": "Final answer",
                        "reasoning_content": "private reasoning",
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 30,
                "completion_tokens_details": {"reasoning_tokens": 10},
            },
        }

    monkeypatch.setattr(local_llm, "_post_json", fake_post_json)
    config = local_llm.LocalLLMConfig(
        provider="openwebui",
        base_url="https://example.invalid",
        model="cloud.moonshot/kimi-k3",
        timeout_seconds=30,
        api_key="test-key",
        verify_ssl=True,
    )

    result = local_llm.generate_text(
        "Question",
        temperature=1,
        max_tokens=4096,
        reasoning_effort="low",
        seed=42,
        config=config,
    )

    assert captured["payload"]["reasoning_effort"] == "low"
    assert captured["payload"]["seed"] == 42
    assert result["response"] == "Final answer"
    assert result["model"] == "kimi-k3"
    assert result["finish_reason"] == "stop"
    assert result["usage"]["completion_tokens"] == 30
    assert result["reasoning_content_present"] is True
    assert result["reasoning_content_char_count"] == len("private reasoning")
    assert "reasoning_content" not in result


def test_openwebui_none_content_normalizes_to_empty_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        local_llm,
        "_post_json",
        lambda *args, **kwargs: {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {"content": None, "reasoning_content": "thinking"},
                }
            ],
            "usage": {"completion_tokens": 4096},
        },
    )
    config = local_llm.LocalLLMConfig(
        provider="openwebui",
        base_url="https://example.invalid",
        model="cloud.moonshot/kimi-k3",
        timeout_seconds=30,
        api_key="test-key",
        verify_ssl=True,
    )

    result = local_llm.generate_text("Question", config=config)

    assert result["response"] == ""
    assert result["finish_reason"] == "length"
    assert result["reasoning_content_present"] is True


def test_generate_text_applies_per_request_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_post_json(url, payload, config):
        captured.update(url=url, payload=payload, config=config)
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": "OK"},
                }
            ]
        }

    monkeypatch.setattr(local_llm, "_post_json", fake_post_json)
    config = local_llm.LocalLLMConfig(
        provider="openwebui",
        base_url="https://example.invalid",
        model="test-model",
        timeout_seconds=360,
        api_key="test-key",
        verify_ssl=True,
    )

    result = local_llm.generate_text(
        "Question",
        timeout_seconds=30,
        config=config,
    )

    assert result["response"] == "OK"
    assert captured["config"].timeout_seconds == 30


def test_openwebui_stream_preserves_length_finish_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lines = [
        'data: {"model":"test-model","choices":[{"delta":{"content":"Partial"},"finish_reason":null}]}',
        'data: {"model":"test-model","choices":[{"delta":{},"finish_reason":"length"}]}',
        "data: [DONE]",
    ]
    monkeypatch.setattr(local_llm, "_post_stream_json", lambda *args, **kwargs: iter(lines))
    config = local_llm.LocalLLMConfig(
        provider="openwebui",
        base_url="https://example.invalid",
        model="test-model",
        timeout_seconds=30,
        api_key="test-key",
        verify_ssl=True,
    )

    events = list(local_llm.stream_generate_text("Question", config=config))

    assert events[0] == {"type": "token", "content": "Partial"}
    assert events[-1]["done"] is True
    assert events[-1]["finish_reason"] == "length"


def test_openwebui_stream_reports_missing_completion_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lines = [
        'data: {"model":"test-model","choices":[{"delta":{"content":"Partial"},"finish_reason":null}]}',
    ]
    monkeypatch.setattr(local_llm, "_post_stream_json", lambda *args, **kwargs: iter(lines))
    config = local_llm.LocalLLMConfig(
        provider="openwebui",
        base_url="https://example.invalid",
        model="test-model",
        timeout_seconds=30,
        api_key="test-key",
        verify_ssl=True,
    )

    events = list(local_llm.stream_generate_text("Question", config=config))

    assert events[-1]["done"] is False
    assert events[-1]["finish_reason"] == ""
