"""Tests for the formal three-arm reranker comparison."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from apps.evaluations.formal_rerank_comparison import (
    build_formal_rerank_comparison,
)


def _write(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _score_item(
    model: str,
    qra_id: str,
    mode: str,
    completeness: float,
) -> dict[str, Any]:
    covered = round(completeness * 4)
    return {
        "candidate_id": f"{model}-{qra_id}-{mode}",
        "qra_id": qra_id,
        "domain": "market_validation" if qra_id == "Q1" else "funding_preparation",
        "question_type": "Factual",
        "model": model,
        "model_type": "dense",
        "mode": mode,
        "keypoint_count": 4,
        "covered": covered,
        "contradicted": 0,
        "neither": 4 - covered,
        "completeness": completeness,
        "hallucination": 0.0,
        "irrelevancy": 1.0 - completeness,
        "keypoint_labels": [],
        "judge_note": "",
    }


def _generation(
    path: Path,
    *,
    mode: str,
    max_tokens: int,
    include_retry: bool = False,
) -> Path:
    results = []
    for model in ("model-a", "model-b"):
        for qra_id in ("Q1", "Q2"):
            results.append(
                {
                    "model": model,
                    "qra_id": qra_id,
                    "mode": mode,
                    "generation": {
                        "status": "success",
                        "latency_ms": 1000,
                        "retrieval_mode": "dense_embedding_rerank"
                        if mode == "rag"
                        else "",
                    },
                }
            )
    prior_attempts = []
    if include_retry:
        prior_attempts.append(
            {
                "model": "model-a",
                "qra_id": "Q1",
                "generation": {"status": "error", "error": "token limit"},
            }
        )
    return _write(
        path,
        {
            "run_id": path.stem,
            "status": "completed",
            "config": {
                "temperature": 0.1,
                "max_tokens": max_tokens,
            },
            "results": results,
            "prior_attempts": prior_attempts,
            "resume": {"resume_count": int(include_retry)},
        },
    )


def test_build_formal_rerank_comparison_pairs_three_arms(tmp_path: Path) -> None:
    values = {
        "baseline": (0.25, 0.0, 0.5, 0.25),
        "dense5": (0.5, 0.25, 0.25, 0.5),
        "rerank": (0.75, 0.25, 0.75, 0.25),
    }
    score_paths = {}
    manifest_paths = {}
    judge = {
        "requested_model": "gpt-5.6-sol",
        "reasoning_effort": "medium",
        "prompt_version": "test-v1",
    }
    for arm in ("baseline", "dense5", "rerank"):
        mode = "baseline" if arm == "baseline" else "rag"
        generation = _generation(
            tmp_path / f"{arm}-generation.json",
            mode=mode,
            max_tokens=200 if arm == "baseline" else 100,
            include_retry=arm == "rerank",
        )
        results = [
            _score_item(model, qra_id, mode, completeness)
            for (model, qra_id), completeness in zip(
                (
                    ("model-a", "Q1"),
                    ("model-a", "Q2"),
                    ("model-b", "Q1"),
                    ("model-b", "Q2"),
                ),
                values[arm],
                strict=True,
            )
        ]
        score_paths[arm] = _write(
            tmp_path / f"{arm}-scores.json",
            {
                "status": "completed",
                "result_count": len(results),
                "results": results,
            },
        )
        manifest_paths[arm] = _write(
            tmp_path / f"{arm}-manifest.json",
            {
                "status": "completed",
                "answer_count": len(results),
                "judge": judge,
                "source_runs": [{"path": str(generation)}],
            },
        )

    outputs = build_formal_rerank_comparison(
        baseline_scores_path=score_paths["baseline"],
        baseline_manifest_path=manifest_paths["baseline"],
        dense5_scores_path=score_paths["dense5"],
        dense5_manifest_path=manifest_paths["dense5"],
        rerank_scores_path=score_paths["rerank"],
        rerank_manifest_path=manifest_paths["rerank"],
        output_dir=tmp_path / "output",
        report_id="TEST-RERANK",
        bootstrap_iterations=100,
        seed=7,
    )

    report = json.loads(outputs["json"].read_text(encoding="utf-8"))
    assert report["panel"] == {
        "answer_count": 4,
        "model_count": 2,
        "qra_count": 2,
        "models": ["model-a", "model-b"],
        "qra_ids": ["Q1", "Q2"],
    }
    assert report["overall"]["arms"]["rerank"]["macro"]["completeness"] == 0.5
    contrast = report["overall"]["contrasts"]["rerank_minus_dense5"]
    assert contrast["metric_deltas"]["completeness"]["mean"] == 0.125
    assert contrast["completeness_win_tie_loss"] == {
        "win": 2,
        "tie": 1,
        "loss": 1,
    }
    assert report["generation_parameter_check"]["dense5_mismatches"] == []
    assert len(report["generation_parameter_check"]["baseline_mismatches"]) == 4
    assert report["generation_parameter_check"]["matched_sensitivity"][
        "rerank_minus_dense5"
    ]["retained_pair_count"] == 4
    assert report["generation_parameter_check"]["matched_sensitivity"][
        "rerank_minus_baseline"
    ]["retained_pair_count"] == 0
    assert report["generation_health"][0]["prior_attempt_count"] == 1
    markdown = outputs["markdown"].read_text(encoding="utf-8")
    assert "七模型 rerank 三臂配对比较" in markdown
    assert "Rerank − Dense@5" in markdown
    assert report["overall"]["arms"]["baseline"]["macro"][
        "completeness"
    ] == pytest.approx(0.25)
