"""Tests for the resumable formal Codex Judge artifact workflow."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from apps.evaluations.formal_codex_judge import (
    FormalCodexJudgeError,
    prepare_formal_codex_judge_run,
    run_formal_codex_judge_phase,
)


def _write_source(
    path: Path,
    *,
    model: str,
    qra_ids: tuple[str, ...],
    mode: str = "rag",
    failed_qra_ids: tuple[str, ...] = (),
    manifest_sha256: str = "dataset-hash",
) -> None:
    items = [
        {
            "id": qra_id,
            "domain": "market_validation",
            "question_type": "factual",
            "language": "en",
            "question": f"Question {qra_id}?",
            "answerable": True,
            "reference_answer": "Reference",
            "keypoints": [
                {"id": "K1", "text": "The required fact is explicit."},
                {"id": "K2", "text": "The required number is 10."},
            ],
            "references": [],
        }
        for qra_id in qra_ids
    ]
    payload = {
        "run_id": f"RUN-{model}",
        "dataset": {
            "manifest_sha256": manifest_sha256,
            "items": items,
        },
        "results": [
            {
                "sequence": position,
                "qra_id": qra_id,
                "domain": "market_validation",
                "question_type": "factual",
                "model": model,
                "model_type": "dense",
                "mode": mode,
                "repeat": 1,
                "generation": {
                    "status": "error" if qra_id in failed_qra_ids else "success",
                    "response": (
                        ""
                        if qra_id in failed_qra_ids
                        else "The required fact is explicit.\nThe required number is 10."
                    ),
                },
            }
            for position, qra_id in enumerate(qra_ids, start=1)
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _cover_everything(batch):  # noqa: ANN001
    decisions = []
    for packet in batch:
        decisions.append(
            {
                "candidate_id": packet["candidate_id"],
                "keypoint_labels": [
                    {
                        "keypoint_id": keypoint["id"],
                        "label": "covered",
                        "evidence_segment_ids": ["A0001"],
                        "note": "Explicit.",
                    }
                    for keypoint in packet["keypoints"]
                ],
                "overall_note": "Complete.",
            }
        )
    return {"decisions": decisions}, {
        "batch_name": "fake",
        "elapsed_seconds": 1.0,
        "candidate_count": len(batch),
    }


def test_formal_codex_judge_pauses_and_resumes_across_model_phases(
    tmp_path: Path,
) -> None:
    sources = []
    for model in ("model-a", "model-b", "model-c"):
        path = tmp_path / f"{model}.json"
        _write_source(path, model=model, qra_ids=("Q1", "Q2"))
        sources.append(path)
    run_dir = tmp_path / "judge" / "RUN-JUDGE"
    prepare_formal_codex_judge_run(
        sources,
        run_dir=run_dir,
        run_id="RUN-JUDGE",
        seed=7,
        first_phase_model_count=2,
        judge_model="gpt-5.6-sol",
        reasoning_effort="medium",
    )

    paused = run_formal_codex_judge_phase(
        run_dir=run_dir,
        run_id="RUN-JUDGE",
        phase="first",
        batch_size=2,
        batch_judge=_cover_everything,
        resume_command="run --phase first --resume",
        max_batches=1,
    )
    assert paused["status"] == "paused"
    assert paused["phase_completed_candidate_count"] == 2
    assert paused["next_candidate_id"]

    first_done = run_formal_codex_judge_phase(
        run_dir=run_dir,
        run_id="RUN-JUDGE",
        phase="first",
        batch_size=2,
        batch_judge=_cover_everything,
        resume_command="run --phase first --resume",
    )
    assert first_done["status"] == "first_phase_completed"
    assert len(first_done["decisions"]) == 4
    assert "--phase remaining" in first_done["resume_command"]
    progress = json.loads(
        (run_dir / "RUN-JUDGE_progress.json").read_text(encoding="utf-8")
    )
    assert progress["remaining_models"] == ["model-c"]
    assert progress["estimated_reserved_phase_seconds"] == 1
    assert progress["usage"]["input_tokens"] == 0

    all_done = run_formal_codex_judge_phase(
        run_dir=run_dir,
        run_id="RUN-JUDGE",
        phase="remaining",
        batch_size=2,
        batch_judge=_cover_everything,
        resume_command="run --phase remaining --resume",
    )
    assert all_done["status"] == "completed"
    assert len(all_done["decisions"]) == 6
    assert all_done["resume_command"] is None
    progress = json.loads(
        (run_dir / "RUN-JUDGE_progress.json").read_text(encoding="utf-8")
    )
    assert progress["remaining_models"] == []
    assert progress["estimated_reserved_phase_seconds"] is None
    manifest = json.loads(
        (run_dir / "RUN-JUDGE_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "completed"
    scores = json.loads(
        (run_dir / "RUN-JUDGE_scores.json").read_text(encoding="utf-8")
    )
    assert scores["result_count"] == 6
    assert {item["model"] for item in scores["summary"]} == {
        "model-a",
        "model-b",
        "model-c",
    }
    assert all(item["mean_completeness"] == 1.0 for item in scores["summary"])


def test_formal_codex_judge_completes_single_model_in_first_phase(
    tmp_path: Path,
) -> None:
    source = tmp_path / "model-a.json"
    _write_source(source, model="model-a", qra_ids=("Q1", "Q2"))
    run_dir = tmp_path / "judge" / "RUN-SINGLE-JUDGE"
    prepare_formal_codex_judge_run(
        [source],
        run_dir=run_dir,
        run_id="RUN-SINGLE-JUDGE",
        seed=7,
        first_phase_model_count=1,
        judge_model="gpt-5.6-sol",
        reasoning_effort="medium",
    )

    completed = run_formal_codex_judge_phase(
        run_dir=run_dir,
        run_id="RUN-SINGLE-JUDGE",
        phase="first",
        batch_size=2,
        batch_judge=_cover_everything,
        resume_command="run --phase first --resume",
    )

    assert completed["status"] == "completed"
    assert completed["resume_command"] is None
    manifest = json.loads(
        (run_dir / "RUN-SINGLE-JUDGE_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["remaining_phase_models"] == []
    assert manifest["status"] == "completed"


def test_formal_codex_judge_accepts_baseline_retry_subsets(tmp_path: Path) -> None:
    primary = tmp_path / "primary.json"
    supplement = tmp_path / "supplement.json"
    second_model = tmp_path / "second-model.json"
    _write_source(
        primary,
        model="model-a",
        qra_ids=("Q1", "Q2"),
        mode="baseline",
        failed_qra_ids=("Q2",),
    )
    _write_source(
        supplement,
        model="model-a",
        qra_ids=("Q2",),
        mode="baseline",
        manifest_sha256="subset-hash",
    )
    _write_source(
        second_model,
        model="model-b",
        qra_ids=("Q1", "Q2"),
        mode="baseline",
    )

    run_dir = tmp_path / "judge" / "RUN-BASELINE-JUDGE"
    prepare_formal_codex_judge_run(
        [primary, supplement, second_model],
        run_dir=run_dir,
        run_id="RUN-BASELINE-JUDGE",
        seed=7,
        first_phase_model_count=1,
        judge_model="gpt-5.6-sol",
        reasoning_effort="medium",
        condition_mode="baseline",
    )

    manifest = json.loads(
        (run_dir / "RUN-BASELINE-JUDGE_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["condition_mode"] == "baseline"
    assert manifest["answer_count"] == 4
    assert manifest["dataset_manifest_sha256"] == "dataset-hash"
    assert [item["skipped_result_count"] for item in manifest["source_runs"]] == [
        1,
        0,
        0,
    ]


def test_formal_codex_judge_rejects_covered_label_without_evidence(
    tmp_path: Path,
) -> None:
    sources = []
    for model in ("model-a", "model-b"):
        path = tmp_path / f"{model}.json"
        _write_source(path, model=model, qra_ids=("Q1",))
        sources.append(path)
    run_dir = tmp_path / "judge" / "RUN-JUDGE"
    prepare_formal_codex_judge_run(
        sources,
        run_dir=run_dir,
        run_id="RUN-JUDGE",
        seed=7,
        first_phase_model_count=1,
        judge_model="gpt-5.6-sol",
        reasoning_effort="medium",
    )

    def invalid(batch):  # noqa: ANN001
        packet = batch[0]
        return {
            "decisions": [
                {
                    "candidate_id": packet["candidate_id"],
                    "keypoint_labels": [
                        {
                            "keypoint_id": keypoint["id"],
                            "label": "covered",
                            "evidence_segment_ids": [],
                            "note": "Missing evidence.",
                        }
                        for keypoint in packet["keypoints"]
                    ],
                    "overall_note": "Invalid.",
                }
            ]
        }, {"batch_name": "fake", "elapsed_seconds": 1.0, "candidate_count": 1}

    with pytest.raises(FormalCodexJudgeError, match="requires evidence"):
        run_formal_codex_judge_phase(
            run_dir=run_dir,
            run_id="RUN-JUDGE",
            phase="first",
            batch_size=1,
            batch_judge=invalid,
            resume_command="run --phase first --resume",
            max_batch_retries=0,
        )
    checkpoint = json.loads(
        (run_dir / "RUN-JUDGE_checkpoint.json").read_text(encoding="utf-8")
    )
    assert checkpoint["status"] == "interrupted"
    assert checkpoint["failed_batches"][0]["candidate_ids"]
