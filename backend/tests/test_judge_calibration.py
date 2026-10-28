"""Tests for corrected Judge comparison and human calibration artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from apps.evaluations.judge_calibration import prepare_calibration_artifacts


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _judge(status: str, completeness: float) -> dict:
    return {
        "status": status,
        "metrics": (
            {
                "completeness": completeness,
                "hallucination": 0.0,
                "irrelevancy": round(1 - completeness, 4),
            }
            if status == "success"
            else None
        ),
        "keypoint_evaluations": (
            [{"keypoint_id": "K1", "label": "covered"}] if status == "success" else []
        ),
    }


def _result(candidate_id: str, suffix: str, status: str, score: float) -> dict:
    return {
        "candidate_id": candidate_id,
        "tested_model": f"model-a::{suffix}",
        "qra_id": "Q1",
        "mode": "rag",
        "domain": "market_validation",
        "question_type": "Factual",
        "judge": _judge(status, score),
    }


def test_prepare_calibration_merges_failure_and_hides_conditions(
    tmp_path: Path,
) -> None:
    primary_path = tmp_path / "primary.json"
    correction_path = tmp_path / "correction.json"
    blinded_path = tmp_path / "blinded.json"
    _write(
        primary_path,
        {
            "judge_run_id": "primary-run",
            "results": [
                _result("C0001", "top3_global", "error", 0.0),
                _result("C0002", "top5_domain", "success", 1.0),
            ],
        },
    )
    corrected = _result("C0001", "top3_global", "success", 0.0)
    _write(
        correction_path,
        {"judge_run_id": "correction-run", "results": [corrected]},
    )
    judge_input = {
        "qra_id": "Q1",
        "language": "de",
        "question": "Frage?",
        "answerable": True,
        "keypoints": [{"id": "K1", "text": "Punkt"}],
        "candidate_answer": "Antwort",
        "reference_answer": "Hidden reference",
        "reference_evidence": [],
    }
    _write(
        blinded_path,
        {
            "candidates": [
                {"candidate_id": "C0001", "judge_input": judge_input},
                {"candidate_id": "C0002", "judge_input": judge_input},
            ]
        },
    )

    paths = prepare_calibration_artifacts(
        primary_scores_path=primary_path,
        correction_scores_paths=[correction_path],
        blinded_inputs_path=blinded_path,
        output_dir=tmp_path / "out",
        analysis_id="CAL-001",
        sample_size=2,
        seed=7,
    )

    corrected_scores = json.loads(paths["corrected_scores"].read_text())
    assert corrected_scores["result_count"] == 2
    assert corrected_scores["corrections"][0]["judge_run_id"] == "correction-run"
    comparison = json.loads(paths["comparison"].read_text())
    assert comparison["overall"]["top5_minus_top3"]["completeness"] == 1.0
    packet = json.loads(paths["calibration_packet"].read_text())
    serialized_packet = json.dumps(packet)
    assert "model-a" not in serialized_packet
    assert "top3_global" not in serialized_packet
    assert "automated_keypoint_labels" not in serialized_packet
    assert "Hidden reference" not in serialized_packet
    mapping = json.loads(paths["calibration_mapping"].read_text())
    assert {item["pipeline"] for item in mapping["mapping"]} == {
        "top3_global",
        "top5_domain",
    }
