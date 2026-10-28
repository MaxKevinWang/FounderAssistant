"""Tests for formal retrieval diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

from apps.evaluations.formal_retrieval_diagnostics import (
    build_formal_retrieval_diagnostics,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _qra(qra_id: str, domain: str) -> dict[str, object]:
    return {
        "id": qra_id,
        "domain": domain,
        "question_type": "Factual",
        "question": f"Question {qra_id}",
        "references": [
            {
                "reference_id": "R1",
                "raw_source_path": f"data/{domain}/source-a.txt",
                "quote": "alpha beta",
            },
            {
                "reference_id": "R2",
                "raw_source_path": f"data/{domain}/source-b.txt",
                "quote": "gamma delta",
            },
        ],
        "keypoints": [
            {"id": "K1", "reference_ids": ["R1"]},
            {"id": "K2", "reference_ids": ["R1", "R2"]},
        ],
    }


def _result(
    *, model: str, qra_id: str, domain: str, variant: int
) -> dict[str, object]:
    if qra_id == "Q1":
        chunks = [
            {
                "chunk_id": f"{domain}/source-a.txt::0",
                "source_path": f"{domain}/source-a.txt",
                "text": "intro alpha beta outro",
            },
            {
                "chunk_id": f"{domain}/source-b.txt::{variant}",
                "source_path": f"{domain}/source-b.txt",
                "text": "wrong section only",
            },
        ]
    else:
        chunks = [
            {
                "chunk_id": f"{domain}/unrelated.txt::{variant}",
                "source_path": f"{domain}/unrelated.txt",
                "text": "unrelated words",
            },
            {
                "chunk_id": f"{domain}/also-unrelated.txt::0",
                "source_path": f"{domain}/also-unrelated.txt",
                "text": "more unrelated words",
            },
        ]
    return {
        "qra_id": qra_id,
        "domain": domain,
        "question_type": "Factual",
        "model": model,
        "model_type": "dense",
        "mode": "rag",
        "repeat": 1,
        "generation": {
            "status": "success",
            "finish_reason": "stop",
            "retrieval_mode": "dense_embedding",
            "retrieved_chunks": chunks,
        },
    }


def _score(model: str, qra_id: str, mode: str) -> dict[str, object]:
    rag = mode == "rag"
    labels = (
        ["covered", "neither"]
        if qra_id == "Q1" and rag
        else ["neither", "neither"]
    )
    if mode == "baseline" and qra_id == "Q1":
        labels = ["neither", "neither"]
    return {
        "candidate_id": f"{mode}-{model}-{qra_id}",
        "model": model,
        "qra_id": qra_id,
        "mode": mode,
        "keypoint_labels": [
            {"keypoint_id": keypoint_id, "label": label}
            for keypoint_id, label in zip(("K1", "K2"), labels, strict=True)
        ],
    }


def test_build_formal_retrieval_diagnostics(tmp_path: Path) -> None:
    qras = [_qra("Q1", "market_validation"), _qra("Q2", "funding_preparation")]
    models = ("model-a", "model-b")
    results = [
        _result(
            model=model,
            qra_id=str(qra["id"]),
            domain=str(qra["domain"]),
            variant=model_index,
        )
        for model_index, model in enumerate(models)
        for qra in qras
    ]
    source = tmp_path / "rag.json"
    _write_json(
        source,
        {
            "run_id": "RAG-TEST",
            "status": "completed",
            "config": {"top_k": 2},
            "dataset": {"manifest_sha256": "dataset-hash", "items": qras},
            "results": results,
        },
    )
    score_paths = {}
    for mode in ("rag", "baseline"):
        path = tmp_path / f"{mode}-scores.json"
        score_results = [
            _score(model, str(qra["id"]), mode)
            for model in models
            for qra in qras
        ]
        _write_json(
            path,
            {
                "judge_run_id": f"JUDGE-{mode}",
                "status": "completed",
                "result_count": len(score_results),
                "results": score_results,
            },
        )
        score_paths[mode] = path

    outputs = build_formal_retrieval_diagnostics(
        source_paths=[source],
        rag_scores_path=score_paths["rag"],
        baseline_scores_path=score_paths["baseline"],
        output_dir=tmp_path / "output",
        report_id="RETRIEVAL-TEST",
    )

    report = json.loads(outputs["json"].read_text(encoding="utf-8"))
    overall = report["retrieval"]["overall"]
    assert report["panel"]["answer_count"] == 4
    assert overall["macro_reference_recall_at_k"] == 0.25
    assert overall["evidence_weighted_reference_recall_at_k"] == 0.25
    assert overall["reference_failure_layers"] == {
        "exact": {"count": 2, "rate": 0.25},
        "right_document_wrong_chunk": {"count": 2, "rate": 0.25},
        "right_document_absent": {"count": 4, "rate": 0.5},
    }
    assert overall["keypoint_evidence_availability"] == {
        "keypoint_count": 8,
        "all": {"count": 2, "rate": 0.25},
        "partial": {"count": 2, "rate": 0.25},
        "none": {"count": 4, "rate": 0.5},
    }
    assert overall["exact_evidence_word_rate_proxy"]["matched_word_count"] == 4
    assert report["retrieval_agreement"]["divergent_qra_count"] == 2
    support = {
        row["support"]: row
        for row in report["baseline_to_rag_conditioned_on_evidence"][
            "by_evidence_support"
        ]
    }
    assert support["all"]["coverage_delta"] == 1.0
    assert support["partial"]["coverage_delta"] == 0.0
    assert "Formal Retrieval Diagnostics" in outputs["markdown"].read_text(
        encoding="utf-8"
    )
