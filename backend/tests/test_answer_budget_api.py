"""RAG endpoints share a default budget and retain explicit overrides."""

import json

import pytest
from rest_framework.test import APIRequestFactory

from apps.advisors import views


@pytest.mark.parametrize("model", ["gemma4:31b", "gpt-oss:20b"])
@pytest.mark.parametrize("budget", [None, 2048])
@pytest.mark.parametrize("stream", [False, True])
def test_rag_answer_budget(monkeypatch, model, budget, stream) -> None:
    captured = {}

    def fake_answer(**kwargs):
        captured.update(kwargs)
        return {"model": model, "response": "Answer", "done": True}

    def fake_stream(**kwargs):
        yield {"type": "done", **fake_answer(**kwargs)}

    monkeypatch.setattr(views, "rag_answer", fake_answer)
    monkeypatch.setattr(views, "rag_answer_stream", fake_stream)
    payload = {"prompt": "How can I validate demand?", "model": model}
    if budget is not None:
        payload["max_tokens"] = budget
    request = APIRequestFactory().post("/", payload, format="json")
    view = views.RAGChatStreamView if stream else views.RAGChatView
    response = view.as_view()(request)
    assert response.status_code == 200
    if stream:
        body = b"".join(response.streaming_content).decode("utf-8")
        profile = json.loads(body.split("data: ", 1)[1].split("\n", 1)[0])
    else:
        profile = response.data["runtime_profile"]
    expected = 900 if budget is None else budget
    assert captured["max_tokens"] == expected
    assert profile["max_tokens"] == expected
    assert profile["auto_applied"] is (budget is None)
