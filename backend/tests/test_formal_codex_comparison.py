"""Tests for formal paired Codex Judge comparison reports."""

from __future__ import annotations

import json
from pathlib import Path

from apps.evaluations.formal_codex_comparison import build_formal_codex_comparison


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _result(
    *,
    model: str,
    qra_id: str,
    mode: str,
    covered: int,
    contradicted: int,
) -> dict[str, object]:
    keypoint_count = 2
    neither = keypoint_count - covered - contradicted
    return {
        "candidate_id": f"{mode}-{model}-{qra_id}",
        "qra_id": qra_id,
        "domain": "market_validation" if qra_id == "Q1" else "funding_preparation",
        "question_type": "Factual" if qra_id == "Q1" else "Summarization",
        "model": model,
        "model_type": "dense" if model == "model-a" else "expert",
        "mode": mode,
        "keypoint_count": keypoint_count,
        "covered": covered,
        "contradicted": contradicted,
        "neither": neither,
        "completeness": covered / keypoint_count,
        "hallucination": contradicted / keypoint_count,
        "irrelevancy": neither / keypoint_count,
        "keypoint_labels": [],
        "judge_note": "Synthetic.",
    }


def test_build_formal_codex_comparison_pairs_and_aggregates(tmp_path: Path) -> None:
    judge = {
        "surface": "codex_cli_exec",
        "requested_model": "gpt-5.6-sol",
        "reasoning_effort": "medium",
        "prompt_version": "formal_codex_keypoint_judge_v1.0",
        "reference_isolated": True,
        "unsupported_claims_evaluated": False,
    }
    baseline_results = [
        _result(
            model=model,
            qra_id=qra_id,
            mode="baseline",
            covered=0 if qra_id == "Q1" else 1,
            contradicted=0,
        )
        for model in ("model-a", "model-b")
        for qra_id in ("Q1", "Q2")
    ]
    rag_results = [
        _result(
            model=model,
            qra_id=qra_id,
            mode="rag",
            covered=2 if qra_id == "Q1" else 1,
            contradicted=0,
        )
        for model in ("model-a", "model-b")
        for qra_id in ("Q1", "Q2")
    ]
    paths = {}
    for mode, results in (("baseline", baseline_results), ("rag", rag_results)):
        score_path = tmp_path / f"{mode}_scores.json"
        manifest_path = tmp_path / f"{mode}_manifest.json"
        _write_json(
            score_path,
            {
                "judge_run_id": f"JUDGE-{mode}",
                "status": "completed",
                "result_count": len(results),
                "results": results,
            },
        )
        _write_json(
            manifest_path,
            {
                "status": "completed",
                "answer_count": len(results),
                "dataset_manifest_sha256": "dataset-hash",
                "judge": judge,
            },
        )
        paths[mode] = (score_path, manifest_path)

    outputs = build_formal_codex_comparison(
        baseline_scores_path=paths["baseline"][0],
        baseline_manifest_path=paths["baseline"][1],
        rag_scores_path=paths["rag"][0],
        rag_manifest_path=paths["rag"][1],
        output_dir=tmp_path / "comparison",
        report_id="FORMAL-COMP-TEST",
        bootstrap_iterations=100,
        seed=9,
    )

    report = json.loads(outputs["json"].read_text(encoding="utf-8"))
    assert report["paired_answer_count"] == 4
    assert report["model_count"] == 2
    assert report["qra_count"] == 2
    assert len(report["domain_summary"]) == 4
    assert len(report["question_type_summary"]) == 4
    assert report["model_summary"][0]["rag_minus_baseline"]["completeness"][
        "mean"
    ] == 0.5
    assert report["model_summary"][0]["domain_macro"]["rag_minus_baseline"][
        "completeness"
    ] == 0.5
    assert report["pooled_descriptive_summary"]["paired_answer_count"] == 4
    assert report["model_summary"][0]["completeness_win_tie_loss"] == {
        "rag_win": 1,
        "tie": 1,
        "rag_loss": 0,
    }
    assert "Baseline 与 RAG 配对对比" in outputs["markdown"].read_text(
        encoding="utf-8"
    )
