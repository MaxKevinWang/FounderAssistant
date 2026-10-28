"""Regression coverage for merged-generation token provenance."""

from __future__ import annotations

import json
from pathlib import Path

from apps.evaluations.formal_codex_comparison import build_formal_codex_comparison


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _score(mode: str, qra_id: str) -> dict[str, object]:
    return {
        "candidate_id": f"{mode}-{qra_id}",
        "qra_id": qra_id,
        "domain": "market_validation",
        "question_type": "Factual",
        "model": "model-a",
        "model_type": "dense",
        "mode": mode,
        "keypoint_count": 2,
        "covered": 1,
        "contradicted": 0,
        "neither": 1,
        "completeness": 0.5,
        "hallucination": 0.0,
        "irrelevancy": 0.5,
        "keypoint_labels": [],
        "judge_note": "Synthetic.",
    }


def test_comparison_uses_per_result_generation_provenance(tmp_path: Path) -> None:
    judge = {
        "surface": "codex_cli_exec",
        "requested_model": "gpt-5.6-sol",
        "reasoning_effort": "medium",
        "prompt_version": "formal_codex_keypoint_judge_v1.0",
        "reference_isolated": True,
        "unsupported_claims_evaluated": False,
    }
    paths = {}
    for mode in ("baseline", "rag"):
        scores = [_score(mode, qra_id) for qra_id in ("Q1", "Q2")]
        generation_path = tmp_path / f"{mode}_generation.json"
        _write_json(
            generation_path,
            {
                "run_id": f"GEN-{mode}",
                "config": {},
                "results": [
                    {
                        "model": "model-a",
                        "qra_id": qra_id,
                        "generation": {"status": "success"},
                        "source_provenance": {
                            "source_run_id": f"SOURCE-{mode}-{qra_id}",
                            "source_path": f"source/{mode}/{qra_id}.json",
                            "configured_max_tokens": (
                                16384
                                if mode == "baseline" and qra_id == "Q2"
                                else 8192
                            ),
                        },
                    }
                    for qra_id in ("Q1", "Q2")
                ],
            },
        )
        score_path = tmp_path / f"{mode}_scores.json"
        manifest_path = tmp_path / f"{mode}_manifest.json"
        _write_json(
            score_path,
            {
                "judge_run_id": f"JUDGE-{mode}",
                "status": "completed",
                "result_count": len(scores),
                "results": scores,
            },
        )
        _write_json(
            manifest_path,
            {
                "status": "completed",
                "answer_count": len(scores),
                "dataset_manifest_sha256": "dataset-hash",
                "judge": judge,
                "source_runs": [{"path": generation_path.as_posix()}],
            },
        )
        paths[mode] = (score_path, manifest_path)

    outputs = build_formal_codex_comparison(
        baseline_scores_path=paths["baseline"][0],
        baseline_manifest_path=paths["baseline"][1],
        rag_scores_path=paths["rag"][0],
        rag_manifest_path=paths["rag"][1],
        output_dir=tmp_path / "comparison",
        report_id="FORMAL-PROVENANCE-TEST",
        bootstrap_iterations=10,
        seed=11,
    )

    report = json.loads(outputs["json"].read_text(encoding="utf-8"))
    sensitivity = report["generation_parameter_sensitivity"]
    assert sensitivity["max_tokens_mismatch_count"] == 1
    mismatch = sensitivity["max_tokens_mismatches"][0]
    assert mismatch["qra_id"] == "Q2"
    assert mismatch["baseline"]["max_tokens"] == 16384
    assert mismatch["rag"]["max_tokens"] == 8192
