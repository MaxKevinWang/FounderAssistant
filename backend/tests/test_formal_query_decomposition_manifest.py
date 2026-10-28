"""Tests for freezing query decompositions before retrieval ablations."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from apps.advisors.query_decomposition import QueryDecomposition
from apps.evaluations.formal_query_decomposition_manifest import (
    FormalQueryDecompositionManifestError,
    load_query_decomposition_manifest,
    run_query_decomposition_freeze,
)


def _qra(qra_id: str) -> dict[str, object]:
    return {
        "id": qra_id,
        "version": "1.0",
        "status": "frozen",
        "split": "test",
        "review_status": "single_checked",
        "domain": "market_validation",
        "question_type": "Information Integration",
        "language": "de",
        "question": f"Question {qra_id}",
        "answerable": True,
        "answer": "Answer",
        "reference_answer": "Answer",
        "keypoints": [],
        "references": [],
        "configuration_path": "config.json",
        "configuration_sha256": "config-hash",
        "_artifact": {"path": f"{qra_id}.json", "sha256": f"hash-{qra_id}"},
    }


def _decompose(question: str, **kwargs) -> QueryDecomposition:  # noqa: ANN003
    return QueryDecomposition(
        original_query=question,
        needs_decomposition=True,
        subqueries=["first aspect", "second aspect"],
        provider="test",
        model=kwargs["model"],
        raw_response="{}",
        latency_ms=12,
        finish_reason="stop",
        usage={},
        prompt_version="test-v1",
        prompt_sha256="prompt-hash",
        seed=kwargs["seed"],
    )


def test_freeze_and_load_query_decomposition_manifest(tmp_path: Path) -> None:
    qras = [_qra("Q1"), _qra("Q2")]

    path = run_query_decomposition_freeze(
        qra_items=qras,
        output_dir=tmp_path,
        manifest_id="DECOMP-TEST",
        model="test-model",
        decompose_fn=_decompose,
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
    assert payload["summary"]["decomposed_question_count"] == 2
    assert not (tmp_path / "DECOMP-TEST.partial.json").exists()
    loaded = load_query_decomposition_manifest(path, qra_items=qras)
    assert loaded["Q1"].retrieval_queries == [
        "Question Q1",
        "first aspect",
        "second aspect",
    ]


def test_manifest_loader_rejects_changed_question(tmp_path: Path) -> None:
    qras = [_qra("Q1")]
    path = run_query_decomposition_freeze(
        qra_items=qras,
        output_dir=tmp_path,
        manifest_id="DECOMP-TEST",
        model="test-model",
        decompose_fn=_decompose,
    )
    changed = [_qra("Q1")]
    changed[0]["question"] = "Changed question"

    with pytest.raises(FormalQueryDecompositionManifestError, match="question"):
        load_query_decomposition_manifest(path, qra_items=changed)
