"""Tests for frozen-QRA development generation and checkpoint artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from apps.advisors import rag_pipeline
from apps.evaluations import formal_development_benchmark as formal


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _frozen_qra_fixture(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "project"
    qra_dir = root / "docs" / "evaluation" / "formal_dataset" / "qra"
    config_path = (
        root
        / "docs"
        / "evaluation"
        / "formal_dataset"
        / "configurations"
        / "C-TEST-F-001_v1.0.json"
    )
    _write_json(
        config_path,
        {
            "configuration_id": "C-TEST-F-001",
            "version": "1.0",
            "status": "frozen",
        },
    )
    qra_path = qra_dir / "TEST-F-001_v1.0.json"
    _write_json(
        qra_path,
        {
            "id": "TEST-F-001",
            "version": "1.0",
            "status": "frozen",
            "split": "development",
            "review_status": "single_checked",
            "domain": "market_validation",
            "question_type": "Factual",
            "language": "de",
            "question": "Welche Tatsache nennt die Quelle?",
            "answerable": True,
            "answer": "Die Quelle nennt eine überprüfbare Tatsache.",
            "keypoints": [
                {"id": "K1", "text": "Punkt eins", "required": True},
                {"id": "K2", "text": "Punkt zwei", "required": True},
                {"id": "K3", "text": "Punkt drei", "required": True},
            ],
            "references": [],
            "configuration_path": str(config_path),
            "configuration_sha256": formal.sha256_file(config_path),
        },
    )
    freeze_path = qra_dir / "TEST-F-001_v1.0_freeze_record.json"
    _write_json(
        freeze_path,
        {
            "qra_id": "TEST-F-001",
            "artifacts": {
                "qra": {
                    "path": str(qra_path),
                    "sha256": formal.sha256_file(qra_path),
                }
            },
        },
    )
    return root, qra_dir


def _formal_test_manifest_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "project"
    release_dir = (
        root
        / "docs"
        / "evaluation"
        / "formal_dataset"
        / "qra"
        / "formal_test"
    )
    qra_dir = release_dir / "market_validation"
    config_path = (
        root
        / "docs"
        / "evaluation"
        / "formal_dataset"
        / "configurations"
        / "formal_test"
        / "C-FT-MV-F-001_v1.0.json"
    )
    _write_json(
        config_path,
        {
            "configuration_id": "C-FT-MV-F-001",
            "version": "1.0",
            "status": "frozen",
        },
    )
    qra_path = qra_dir / "FTMV-F-001_v1.0.json"
    _write_json(
        qra_path,
        {
            "id": "FTMV-F-001",
            "version": "1.0",
            "status": "frozen",
            "split": "test",
            "review_status": "single_checked",
            "domain": "market_validation",
            "question_type": "Factual",
            "language": "de",
            "question": "Welche Tatsache nennt die formale Quelle?",
            "answerable": True,
            "answer": "Die Quelle nennt eine formale Tatsache.",
            "keypoints": [
                {"id": "K1", "text": "Punkt eins", "required": True},
                {"id": "K2", "text": "Punkt zwei", "required": True},
                {"id": "K3", "text": "Punkt drei", "required": True},
            ],
            "references": [],
            "configuration_path": str(config_path),
            "configuration_sha256": formal.sha256_file(config_path),
        },
    )
    freeze_path = qra_path.with_name("FTMV-F-001_v1.0_freeze_record.json")
    _write_json(
        freeze_path,
        {
            "qra_id": "FTMV-F-001",
            "split": "test",
            "artifacts": {
                "frozen_qra": {
                    "path": str(qra_path),
                    "sha256": formal.sha256_file(qra_path),
                }
            },
        },
    )
    manifest_path = release_dir / "formal_test_qra_manifest_v1.0.json"
    _write_json(
        manifest_path,
        {
            "manifest_id": "formal_test_qra_manifest_v1.0",
            "version": "1.0",
            "status": "frozen",
            "split": "test",
            "review_status": "single_checked",
            "qra_count": 1,
            "configuration_manifest": {
                "path": str(config_path),
                "sha256": formal.sha256_file(config_path),
            },
            "qras": [
                {
                    "qra_id": "FTMV-F-001",
                    "version": "1.0",
                    "domain": "market_validation",
                    "question_type": "Factual",
                    "review_status": "single_checked",
                    "path": str(qra_path),
                    "file_sha256": formal.sha256_file(qra_path),
                    "freeze_record_path": str(freeze_path),
                    "freeze_record_sha256": formal.sha256_file(freeze_path),
                }
            ],
        },
    )
    manifest_freeze_path = release_dir / "formal_test_qra_manifest_v1.0_freeze_record.json"
    _write_json(
        manifest_freeze_path,
        {
            "manifest_id": "formal_test_qra_manifest_v1.0",
            "status": "frozen",
            "file": str(manifest_path),
            "file_sha256": formal.sha256_file(manifest_path),
        },
    )
    return root, release_dir, manifest_path


def _prompt_fixture(tmp_path: Path) -> Path:
    path = tmp_path / "prompt.json"
    _write_json(
        path,
        {
            "prompt_id": "test_prompt_v0.1",
            "version": "0.1",
            "status": "development",
            "system_prompt": "Answer accurately.",
            "baseline_prompt_template": "QUESTION:\n{question}",
        },
    )
    return path


def test_loader_accepts_verified_frozen_qra(tmp_path: Path) -> None:
    root, qra_dir = _frozen_qra_fixture(tmp_path)

    items = formal.load_frozen_development_qras(qra_dir, root=root)

    assert [item["id"] for item in items] == ["TEST-F-001"]
    assert items[0]["_artifact"]["sha256"] == formal.sha256_file(
        qra_dir / "TEST-F-001_v1.0.json"
    )


def test_loader_rejects_qra_changed_after_freeze(tmp_path: Path) -> None:
    root, qra_dir = _frozen_qra_fixture(tmp_path)
    qra_path = qra_dir / "TEST-F-001_v1.0.json"
    qra_path.write_text(
        qra_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(formal.FormalBenchmarkError, match="hash mismatch"):
        formal.load_frozen_development_qras(qra_dir, root=root)


def test_loader_accepts_verified_formal_test_manifest(tmp_path: Path) -> None:
    root, release_dir, manifest_path = _formal_test_manifest_fixture(tmp_path)

    items = formal.load_frozen_development_qras(release_dir, root=root)
    dataset = formal.build_dataset_manifest(items)

    assert [item["id"] for item in items] == ["FTMV-F-001"]
    assert dataset["dataset_id"] == "formal_test_qra_manifest_v1.0"
    assert dataset["split"] == "test"
    assert dataset["release_manifest"]["sha256"] == formal.sha256_file(
        manifest_path
    )


def test_loader_rejects_formal_manifest_changed_after_freeze(tmp_path: Path) -> None:
    root, release_dir, manifest_path = _formal_test_manifest_fixture(tmp_path)
    manifest_path.write_text(
        manifest_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(formal.FormalBenchmarkError, match="manifest hash mismatch"):
        formal.load_frozen_development_qras(release_dir, root=root)


def test_validate_models_rejects_unavailable_or_duplicate_ids() -> None:
    with pytest.raises(formal.FormalBenchmarkError, match="unavailable"):
        formal.validate_models(["missing:model"], ["available:model"])
    with pytest.raises(formal.FormalBenchmarkError, match="Duplicate"):
        formal.validate_models(
            ["available:model", "available:model"],
            ["available:model"],
        )


def test_run_exports_paired_generation_and_removes_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, qra_dir = _frozen_qra_fixture(tmp_path)
    items = formal.load_frozen_development_qras(qra_dir, root=root)
    prompt_path = _prompt_fixture(tmp_path)
    baseline_calls: list[dict] = []
    rag_calls: list[dict] = []

    def fake_baseline(**kwargs):
        baseline_calls.append(kwargs)
        return {
            "provider": "test",
            "model": kwargs["model"],
            "response": "Baseline answer",
            "done": True,
            "latency_ms": 1200,
            "finish_reason": "stop",
            "usage": {"completion_tokens": 12},
        }

    def fake_rag(**kwargs):
        rag_calls.append(kwargs)
        return {
            "provider": "test",
            "model": kwargs["model"],
            "response": "RAG answer [S1]",
            "done": True,
            "latency_ms": 1500,
            "retrieval_latency_ms": 200,
            "retrieval_mode": "dense_embedding",
            "citations": [{"chunk_id": "chunk-1"}],
            "retrieved_chunks": [
                {"chunk_id": "chunk-1", "text": "Complete retrieved text"}
            ],
        }

    monkeypatch.setattr(
        formal,
        "snapshot_rag_index",
        lambda: {
            "embedding_model": "test-embed",
            "document_count": 1,
            "chunk_count": 1,
        },
    )
    output_dir = tmp_path / "runs"
    events: list[dict] = []

    result = formal.run_formal_development_benchmark(
        models=["available:model"],
        qra_items=items,
        output_dir=output_dir,
        modes=["baseline", "rag"],
        prompt_path=prompt_path,
        baseline_fn=fake_baseline,
        rag_fn=fake_rag,
        reasoning_effort="low",
        rag_options={
            "candidate_k": 20,
            "reranker_model": "test/reranker",
            "reranker_revision": "revision",
            "max_context_tokens": 3000,
            "neighbor_expansion_distance": 2,
            "allow_keyword_fallback": False,
        },
        service_status={
            "reachable": True,
            "provider": "test",
            "generation_models": ["available:model"],
            "embedding_models": ["test-embed"],
        },
        progress_callback=events.append,
    )

    assert len(baseline_calls) == 1
    assert len(rag_calls) == 1
    assert baseline_calls[0]["max_tokens"] == 900
    assert rag_calls[0]["max_tokens"] == 900
    assert result["report"]["config"]["max_tokens"] == 900
    assert baseline_calls[0]["reasoning_effort"] == "low"
    assert rag_calls[0]["reasoning_effort"] == "low"
    assert rag_calls[0]["domain"] == "market_validation"
    assert rag_calls[0]["top_k"] == 5
    assert rag_calls[0]["include_retrieved_text"] is True
    assert rag_calls[0]["candidate_k"] == 20
    assert rag_calls[0]["neighbor_expansion_distance"] == 2
    assert rag_calls[0]["allow_keyword_fallback"] is False
    assert len(result["report"]["results"]) == 2
    rag_result = result["report"]["results"][1]["generation"]
    assert rag_result["retrieved_chunks"][0]["text"] == "Complete retrieved text"
    baseline_result = result["report"]["results"][0]["generation"]
    assert baseline_result["finish_reason"] == "stop"
    assert baseline_result["usage"] == {"completion_tokens": 12}
    assert result["report"]["config"]["reasoning_effort"] == "low"
    assert result["report"]["config"]["rag_options"]["candidate_k"] == 20
    assert result["report"]["status"] == "completed"
    assert Path(result["json_path"]).exists()
    assert Path(result["markdown_path"]).exists()
    assert not list(output_dir.glob("*.partial.json"))
    assert [event["type"] for event in events] == [
        "start",
        "complete",
        "start",
        "complete",
    ]


def test_run_resumes_only_failed_repeated_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, qra_dir = _frozen_qra_fixture(tmp_path)
    items = formal.load_frozen_development_qras(qra_dir, root=root)
    prompt_path = _prompt_fixture(tmp_path)
    rag_attempts = 0

    def fake_baseline(**kwargs):
        return {
            "provider": "test",
            "model": kwargs["model"],
            "response": "Baseline answer",
            "done": True,
            "latency_ms": 10,
            "finish_reason": "stop",
        }

    def fake_rag(**kwargs):
        nonlocal rag_attempts
        rag_attempts += 1
        if rag_attempts == 1:
            raise TimeoutError("temporary timeout")
        return {
            "provider": "test",
            "model": kwargs["model"],
            "response": "RAG answer [S1]",
            "done": True,
            "latency_ms": 12,
            "retrieval_latency_ms": 2,
            "retrieval_mode": "dense_embedding",
            "citations": [],
            "retrieved_chunks": [],
            "finish_reason": "stop",
        }

    monkeypatch.setattr(
        formal,
        "snapshot_rag_index",
        lambda: {
            "metadata_sha256": "metadata-hash",
            "embeddings_sha256": "embeddings-hash",
            "embedding_model": "test-embed",
            "document_count": 1,
            "chunk_count": 1,
        },
    )
    output_dir = tmp_path / "runs"
    common = {
        "models": ["available:model"],
        "qra_items": items,
        "output_dir": output_dir,
        "modes": ["baseline", "rag"],
        "prompt_path": prompt_path,
        "baseline_fn": fake_baseline,
        "rag_fn": fake_rag,
        "service_status": {
            "reachable": True,
            "provider": "test",
            "generation_models": ["available:model"],
            "embedding_models": ["test-embed"],
        },
        "repeats": 2,
        "schedule_seed": 17,
        "run_id": "FORMAL-RESUME-TEST",
    }

    first = formal.run_formal_development_benchmark(**common)

    assert first["report"]["status"] == "completed_with_errors"
    assert len(first["report"]["results"]) == 4
    assert {item["repeat"] for item in first["report"]["results"]} == {1, 2}
    assert (output_dir / "FORMAL-RESUME-TEST.partial.json").exists()

    resumed = formal.run_formal_development_benchmark(**common, resume=True)

    assert resumed["report"]["status"] == "completed"
    assert len(resumed["report"]["results"]) == 4
    assert len(resumed["report"]["prior_attempts"]) == 1
    assert resumed["report"]["resume"] == {
        "resumed": True,
        "resume_count": 1,
        "preexisting_successful_answers": 3,
    }
    assert rag_attempts == 3
    assert not (output_dir / "FORMAL-RESUME-TEST.partial.json").exists()


def test_formal_run_lock_rejects_live_owner_and_reclaims_stale_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_path = tmp_path / "FORMAL-LOCK-TEST.lock"

    with formal.formal_run_lock(tmp_path, "FORMAL-LOCK-TEST"):
        assert lock_path.exists()
        with pytest.raises(formal.FormalBenchmarkError, match="already active"):
            with formal.formal_run_lock(tmp_path, "FORMAL-LOCK-TEST"):
                pass

    assert not lock_path.exists()
    lock_path.write_text(
        json.dumps({"pid": 999_999_999, "token": "stale"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(formal, "_process_is_running", lambda pid: False)

    with formal.formal_run_lock(tmp_path, "FORMAL-LOCK-TEST"):
        owner = json.loads(lock_path.read_text(encoding="utf-8"))
        assert owner["pid"] > 0
        assert owner["token"] != "stale"

    assert not lock_path.exists()


def test_rag_answer_only_returns_full_chunks_when_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hits = [
        {
            "rank": 1,
            "score": 0.9,
            "chunk_id": "chunk-1",
            "domain": "market_validation",
            "source_path": "data/source.html",
            "source_name": "source",
            "chunk_index": 0,
            "text": "Complete retrieved evidence.",
            "text_preview": "Complete retrieved evidence.",
        }
    ]
    retrieval_calls: list[dict] = []

    def fake_retrieve(*, query, config, top_k, domain, allow_keyword_fallback):
        retrieval_calls.append(
            {
                "query": query,
                "config": config,
                "top_k": top_k,
                "domain": domain,
                "allow_keyword_fallback": allow_keyword_fallback,
            }
        )
        return hits, 12, "dense_embedding"

    monkeypatch.setattr(
        rag_pipeline,
        "_retrieve_chunks",
        fake_retrieve,
    )
    monkeypatch.setattr(
        rag_pipeline,
        "generate_text",
        lambda **kwargs: {
            "provider": "test",
            "model": "test:model",
            "response": "Answer [S1]",
            "done": True,
            "latency_ms": 20,
        },
    )

    default_result = rag_pipeline.rag_answer("Question")
    evaluation_result = rag_pipeline.rag_answer(
        "Question",
        domain="market_validation",
        include_retrieved_text=True,
    )

    assert "retrieved_chunks" not in default_result
    assert default_result["citations"][0]["chunk_id"] == "chunk-1"
    assert evaluation_result["retrieved_chunks"][0]["chunk_id"] == "chunk-1"
    assert evaluation_result["retrieved_chunks"][0]["text"] == hits[0]["text"]
    assert evaluation_result["retrieved_chunks"][0]["dense_rank"] == 1
    assert evaluation_result["candidate_count"] == 1
    assert evaluation_result["selected_chunk_count"] == 1
    assert retrieval_calls[0]["top_k"] == 5
    assert retrieval_calls[0]["domain"] == ""
    assert retrieval_calls[1]["domain"] == "market_validation"


def test_retrieve_chunks_filters_dense_ranking_to_requested_domain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks = [
        {
            "chunk_id": "cross-domain-best",
            "domain": "funding_preparation",
            "source_path": "funding.html",
            "source_name": "funding",
            "chunk_index": 0,
            "text": "Funding evidence",
        },
        {
            "chunk_id": "market-second",
            "domain": "market_validation",
            "source_path": "market-2.html",
            "source_name": "market-2",
            "chunk_index": 0,
            "text": "Market evidence two",
        },
        {
            "chunk_id": "market-third",
            "domain": "market_validation",
            "source_path": "market-3.html",
            "source_name": "market-3",
            "chunk_index": 0,
            "text": "Market evidence three",
        },
    ]
    monkeypatch.setattr(
        rag_pipeline,
        "_load_index",
        lambda config: (
            np.asarray([[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]], dtype=np.float32),
            {"chunks": chunks},
        ),
    )
    monkeypatch.setattr(
        rag_pipeline,
        "_embed_texts",
        lambda texts, config: np.asarray([[1.0, 0.0]], dtype=np.float32),
    )

    hits, _, mode = rag_pipeline._retrieve_chunks(
        query="market question",
        config=None,
        top_k=5,
        domain="market_validation",
    )

    assert mode == "dense_embedding"
    assert [hit["chunk_id"] for hit in hits] == ["market-second", "market-third"]
    assert {hit["domain"] for hit in hits} == {"market_validation"}


def test_retrieve_chunks_rejects_domain_without_indexed_chunks(
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
                        "chunk_id": "market-only",
                        "domain": "market_validation",
                        "source_path": "market.html",
                        "source_name": "market",
                        "chunk_index": 0,
                        "text": "Market evidence",
                    }
                ]
            },
        ),
    )

    with pytest.raises(rag_pipeline.RAGError, match="No RAG chunks found"):
        rag_pipeline._retrieve_chunks(
            query="question",
            config=None,
            domain="funding_preparation",
        )


def test_rag_prompt_rejects_cross_business_model_analogies() -> None:
    prompt = rag_pipeline._build_rag_prompt(
        question="Does a rule for online courses also apply to design templates?",
        hits=[],
    )

    assert "Do not transfer a source's business-model-specific conclusion" in prompt
    assert "instead of treating the analogy as evidence" in prompt


def test_rag_prompt_requires_an_evidence_checklist_and_explicit_calculations() -> None:
    prompt = rag_pipeline._build_rag_prompt(
        question="Compare two values and explain every requested criterion.",
        hits=[],
    )

    assert "silently build an evidence checklist" in prompt
    assert "Map every requested entity, dimension, stage, and comparison" in prompt
    assert "show the formula or arithmetic" in prompt
    assert "Distinguish source-supported facts from your own inference" in prompt


def test_rag_prompt_can_replay_the_legacy_formal_prompt() -> None:
    prompt = rag_pipeline._build_rag_prompt(
        question="Compare two values.",
        hits=[],
        prompt_version=rag_pipeline.LEGACY_RAG_PROMPT_VERSION,
    )

    assert "silently build an evidence checklist" not in prompt
    assert "show the formula or arithmetic" not in prompt
    assert "Do not transfer a source's business-model-specific conclusion" in prompt


def test_rag_prompt_applies_explicit_low_confidence_gating() -> None:
    prompt = rag_pipeline._build_rag_prompt(
        question="Answer every part.",
        hits=[],
        retrieval_confidence={
            "level": "insufficient",
            "reason": "No selected chunk reached the relevance threshold.",
        },
    )

    assert "EVIDENCE SUFFICIENCY: insufficient" in prompt
    assert "Do not fill missing items from general knowledge" in prompt
    assert "state that the retrieved sources do not establish it" in prompt


def test_empty_model_response_is_recorded_as_error() -> None:
    result = formal._safe_generation_call(
        mode="baseline",
        question="Question",
        model="test:model",
        system_prompt="System",
        baseline_prompt_template="{question}",
        temperature=0.1,
        max_tokens=100,
        top_k=3,
        baseline_fn=lambda **kwargs: {
            "response": "",
            "model": kwargs["model"],
        },
        rag_fn=lambda **kwargs: {},
    )

    assert result["status"] == "error"
    assert result["error_type"] == "FormalBenchmarkError"
    assert "empty response" in result["error"]


def test_none_model_response_is_not_stringified_as_success() -> None:
    result = formal._safe_generation_call(
        mode="baseline",
        question="Question",
        model="test:model",
        system_prompt="System",
        baseline_prompt_template="{question}",
        temperature=0.1,
        max_tokens=100,
        top_k=3,
        baseline_fn=lambda **kwargs: {
            "response": None,
            "model": kwargs["model"],
        },
        rag_fn=lambda **kwargs: {},
    )

    assert result["status"] == "error"
    assert result["response"] == ""
    assert "empty response" in result["error"]


def test_token_limit_finish_reason_is_recorded_as_error_with_usage() -> None:
    result = formal._safe_generation_call(
        mode="baseline",
        question="Question",
        model="test:model",
        system_prompt="System",
        baseline_prompt_template="{question}",
        temperature=1,
        max_tokens=4096,
        top_k=3,
        reasoning_effort="low",
        baseline_fn=lambda **kwargs: {
            "provider": "test",
            "response": "Partial answer",
            "model": kwargs["model"],
            "finish_reason": "length",
            "usage": {"completion_tokens": 4096},
            "reasoning_content_present": True,
            "reasoning_content_char_count": 8000,
        },
        rag_fn=lambda **kwargs: {},
    )

    assert result["status"] == "error"
    assert result["response"] == "Partial answer"
    assert result["finish_reason"] == "length"
    assert result["usage"] == {"completion_tokens": 4096}
    assert result["reasoning_content_char_count"] == 8000
    assert "token limit" in result["error"]


def test_reported_output_token_limit_overrides_misleading_stop_reason() -> None:
    result = formal._safe_generation_call(
        mode="baseline",
        question="Question",
        model="test:model",
        system_prompt="System",
        baseline_prompt_template="{question}",
        temperature=1,
        max_tokens=8192,
        top_k=3,
        baseline_fn=lambda **kwargs: {
            "provider": "test",
            "response": "Apparently complete answer",
            "model": kwargs["model"],
            "finish_reason": "stop",
            "usage": {"output_tokens": 8192},
        },
        rag_fn=lambda **kwargs: {},
    )

    assert result["status"] == "error"
    assert result["finish_reason"] == "stop"
    assert "8192" in result["error"]
    assert "may be truncated" in result["error"]


def test_output_tokens_above_requested_limit_can_include_hidden_reasoning() -> None:
    result = formal._safe_generation_call(
        mode="baseline",
        question="Question",
        model="test:model",
        system_prompt="System",
        baseline_prompt_template="{question}",
        temperature=0.1,
        max_tokens=900,
        top_k=5,
        baseline_fn=lambda **kwargs: {
            "provider": "test",
            "response": "Complete visible answer.",
            "model": kwargs["model"],
            "finish_reason": "stop",
            "usage": {"output_tokens": 1400},
        },
        rag_fn=lambda **kwargs: {},
    )

    assert result["status"] == "success"
    assert result["finish_reason"] == "stop"
    assert result["usage"] == {"output_tokens": 1400}
