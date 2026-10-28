"""Consolidate corrected Judge results and prepare blinded human calibration."""

from __future__ import annotations

import hashlib
import json
import random
import statistics
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable


class JudgeCalibrationError(RuntimeError):
    """Raised when Judge correction or calibration artifacts are inconsistent."""


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise JudgeCalibrationError(f"Expected a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    repository_root = Path(__file__).resolve().parents[3]
    try:
        return resolved.relative_to(repository_root).as_posix()
    except ValueError:
        return resolved.as_posix()


def _condition_key(result: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(result["tested_model"]),
        str(result["qra_id"]),
        str(result["mode"]),
    )


def merge_corrected_results(
    primary_results: Iterable[dict[str, Any]],
    correction_reports: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Replace only failed primary conditions with successful corrections."""

    results = [dict(item) for item in primary_results]
    by_key = {_condition_key(item): item for item in results}
    if len(by_key) != len(results):
        raise JudgeCalibrationError(
            "Primary Judge results contain duplicate conditions"
        )

    corrections: list[dict[str, Any]] = []
    for report in correction_reports:
        correction_run_id = str(report.get("judge_run_id", ""))
        for corrected in report.get("results", []):
            key = _condition_key(corrected)
            primary = by_key.get(key)
            if primary is None:
                raise JudgeCalibrationError(
                    f"Correction condition is absent from primary results: {key}"
                )
            if primary["judge"]["status"] == "success":
                raise JudgeCalibrationError(
                    f"Correction would overwrite a successful primary result: {key}"
                )
            if corrected["judge"]["status"] != "success":
                raise JudgeCalibrationError(
                    f"Correction result is not successful: {key}"
                )
            primary["judge"] = corrected["judge"]
            primary["correction"] = {
                "judge_run_id": correction_run_id,
                "correction_candidate_id": corrected["candidate_id"],
            }
            corrections.append(
                {
                    "condition": {
                        "tested_model": key[0],
                        "qra_id": key[1],
                        "mode": key[2],
                    },
                    "primary_candidate_id": primary["candidate_id"],
                    **primary["correction"],
                }
            )

    remaining = [item for item in results if item["judge"]["status"] != "success"]
    if remaining:
        raise JudgeCalibrationError(
            f"Uncorrected primary Judge failures remain: {len(remaining)}"
        )
    return results, corrections


def _mean(values: list[float]) -> float:
    return round(statistics.mean(values), 4)


def _split_tested_model(tested_model: str) -> tuple[str, str]:
    if "::" not in tested_model:
        raise JudgeCalibrationError(
            f"Tested model has no pipeline suffix: {tested_model}"
        )
    base_model, pipeline = tested_model.rsplit("::", 1)
    return base_model, pipeline


def build_pipeline_comparison(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare Top5-domain with Top3-global on matched model/QRA answers."""

    metrics = ("completeness", "hallucination", "irrelevancy")
    indexed: dict[tuple[str, str, str], dict[str, Any]] = {}
    for result in results:
        base_model, pipeline = _split_tested_model(str(result["tested_model"]))
        indexed[(base_model, str(result["qra_id"]), pipeline)] = result

    pairs: list[dict[str, Any]] = []
    pair_keys = sorted({(model, qra_id) for model, qra_id, _ in indexed})
    for model, qra_id in pair_keys:
        top3 = indexed.get((model, qra_id, "top3_global"))
        top5 = indexed.get((model, qra_id, "top5_domain"))
        if top3 is None or top5 is None:
            raise JudgeCalibrationError(f"Incomplete pipeline pair: {model}/{qra_id}")
        top3_metrics = top3["judge"]["metrics"]
        top5_metrics = top5["judge"]["metrics"]
        pairs.append(
            {
                "model": model,
                "qra_id": qra_id,
                "domain": top5["domain"],
                "question_type": top5["question_type"],
                "top3_global": {key: top3_metrics[key] for key in metrics},
                "top5_domain": {key: top5_metrics[key] for key in metrics},
                "top5_minus_top3": {
                    key: round(top5_metrics[key] - top3_metrics[key], 4)
                    for key in metrics
                },
            }
        )

    def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
        summary: dict[str, Any] = {"paired_qra_count": len(items)}
        for pipeline in ("top3_global", "top5_domain"):
            summary[pipeline] = {
                key: _mean([float(item[pipeline][key]) for item in items])
                for key in metrics
            }
        summary["top5_minus_top3"] = {
            key: round(
                summary["top5_domain"][key] - summary["top3_global"][key],
                4,
            )
            for key in metrics
        }
        deltas = [item["top5_minus_top3"]["completeness"] for item in items]
        summary["completeness_direction"] = {
            "improved": sum(value > 0 for value in deltas),
            "unchanged": sum(value == 0 for value in deltas),
            "decreased": sum(value < 0 for value in deltas),
        }
        return summary

    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        by_model[pair["model"]].append(pair)
        by_domain[pair["domain"]].append(pair)
    return {
        "overall": summarize(pairs),
        "by_model": [
            {"model": model, **summarize(items)}
            for model, items in sorted(by_model.items())
        ],
        "by_domain": [
            {"domain": domain, **summarize(items)}
            for domain, items in sorted(by_domain.items())
        ],
        "pairs": pairs,
    }


def _build_calibration_selection(
    results: list[dict[str, Any]],
    *,
    sample_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        _, pipeline = _split_tested_model(str(result["tested_model"]))
        strata[(str(result["domain"]), pipeline)].append(result)
    if not strata or sample_size % len(strata):
        raise JudgeCalibrationError(
            "Calibration sample size must divide evenly across domain/pipeline strata"
        )
    per_stratum = sample_size // len(strata)
    rng = random.Random(seed)
    models = sorted(
        {_split_tested_model(str(result["tested_model"]))[0] for result in results}
    )
    shuffled_models = list(models)
    rng.shuffle(shuffled_models)
    minimum_per_model, remainder = divmod(sample_size, len(models))
    remaining_quota = {
        model: minimum_per_model + (position < remainder)
        for position, model in enumerate(shuffled_models)
    }
    selected: list[dict[str, Any]] = []
    for stratum, candidates in sorted(strata.items()):
        if len(candidates) < per_stratum:
            raise JudgeCalibrationError(
                f"Calibration stratum {stratum} has too few candidates"
            )
        by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for candidate in candidates:
            model, _ = _split_tested_model(str(candidate["tested_model"]))
            by_model[model].append(candidate)
        random_tiebreak = {model: rng.random() for model in models}
        chosen_models = sorted(
            (
                model
                for model in models
                if remaining_quota[model] > 0 and by_model[model]
            ),
            key=lambda model: (-remaining_quota[model], random_tiebreak[model]),
        )[:per_stratum]
        if len(chosen_models) != per_stratum:
            raise JudgeCalibrationError(
                f"Cannot satisfy model-balanced calibration stratum {stratum}"
            )
        for model in chosen_models:
            selected.append(rng.choice(by_model[model]))
            remaining_quota[model] -= 1
    if any(remaining_quota.values()):
        raise JudgeCalibrationError(
            f"Calibration model quotas were not exhausted: {remaining_quota}"
        )
    rng.shuffle(selected)
    return selected


def prepare_calibration_artifacts(
    *,
    primary_scores_path: Path,
    correction_scores_paths: Iterable[Path],
    blinded_inputs_path: Path,
    output_dir: Path,
    analysis_id: str,
    sample_size: int = 24,
    seed: int = 20260807,
) -> dict[str, Path]:
    """Write corrected comparison and a condition/Judge-blinded human packet."""

    primary = _load_json(primary_scores_path)
    correction_paths = list(correction_scores_paths)
    correction_reports = [_load_json(path) for path in correction_paths]
    results, corrections = merge_corrected_results(
        primary.get("results", []),
        correction_reports,
    )
    comparison = build_pipeline_comparison(results)
    blinded = _load_json(blinded_inputs_path)
    input_by_candidate = {
        str(item["candidate_id"]): item["judge_input"]
        for item in blinded.get("candidates", [])
    }
    selected = _build_calibration_selection(
        results,
        sample_size=sample_size,
        seed=seed,
    )

    packets: list[dict[str, Any]] = []
    mappings: list[dict[str, Any]] = []
    templates: list[dict[str, Any]] = []
    for position, result in enumerate(selected, start=1):
        calibration_id = f"H{position:03d}"
        candidate_id = str(result["candidate_id"])
        judge_input = input_by_candidate.get(candidate_id)
        if judge_input is None:
            raise JudgeCalibrationError(
                f"Primary blinded input is missing candidate {candidate_id}"
            )
        base_model, pipeline = _split_tested_model(str(result["tested_model"]))
        packets.append(
            {
                "calibration_id": calibration_id,
                "qra_id": judge_input["qra_id"],
                "domain": result["domain"],
                "question_type": result["question_type"],
                "language": judge_input["language"],
                "question": judge_input["question"],
                "answerable": judge_input["answerable"],
                "keypoints": judge_input["keypoints"],
                "candidate_answer": judge_input["candidate_answer"],
            }
        )
        mappings.append(
            {
                "calibration_id": calibration_id,
                "primary_candidate_id": candidate_id,
                "model": base_model,
                "pipeline": pipeline,
                "qra_id": result["qra_id"],
                "automated_keypoint_labels": [
                    {
                        "keypoint_id": item["keypoint_id"],
                        "label": item["label"],
                    }
                    for item in result["judge"]["keypoint_evaluations"]
                ],
            }
        )
        templates.append(
            {
                "calibration_id": calibration_id,
                "keypoint_labels": [
                    {
                        "keypoint_id": item["id"],
                        "label": "",
                        "reviewer_note": "",
                    }
                    for item in judge_input["keypoints"]
                ],
                "reviewer_note": "",
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    corrected_path = output_dir / f"{analysis_id}_corrected_scores.json"
    comparison_path = output_dir / f"{analysis_id}_pipeline_comparison.json"
    packet_path = output_dir / f"{analysis_id}_human_calibration_blinded.json"
    mapping_path = output_dir / f"{analysis_id}_human_calibration_mapping.json"
    template_path = output_dir / f"{analysis_id}_human_decision_template.json"
    manifest_path = output_dir / f"{analysis_id}_manifest.json"
    _write_json(
        corrected_path,
        {
            "analysis_id": analysis_id,
            "status": "completed",
            "evaluation_status": "development_api_judge_human_validation_pending",
            "primary_judge_run_id": primary.get("judge_run_id", ""),
            "corrections": corrections,
            "result_count": len(results),
            "results": results,
        },
    )
    _write_json(comparison_path, {"analysis_id": analysis_id, **comparison})
    _write_json(
        packet_path,
        {
            "analysis_id": analysis_id,
            "status": "prepared",
            "blinding": "model_pipeline_and_automated_labels_hidden",
            "sample_seed": seed,
            "sample_count": len(packets),
            "instructions": {
                "labels": ["covered", "contradicted", "neither"],
                "covered": "Every required element is explicit in the candidate answer.",
                "contradicted": "The candidate explicitly makes an incompatible claim.",
                "neither": "Any required element is absent, vague, or only partially present.",
            },
            "packets": packets,
        },
    )
    _write_json(
        mapping_path,
        {
            "analysis_id": analysis_id,
            "status": "sealed_until_human_labels_complete",
            "mapping": mappings,
        },
    )
    _write_json(
        template_path,
        {
            "analysis_id": analysis_id,
            "status": "awaiting_human_labels",
            "allowed_labels": ["covered", "contradicted", "neither"],
            "decisions": templates,
        },
    )
    sources = [primary_scores_path, blinded_inputs_path, *correction_paths]
    artifacts = [
        corrected_path,
        comparison_path,
        packet_path,
        mapping_path,
        template_path,
    ]
    _write_json(
        manifest_path,
        {
            "analysis_id": analysis_id,
            "created_at": datetime.now(UTC).isoformat(),
            "status": "human_validation_pending",
            "sources": [
                {"path": _display_path(path), "sha256": _sha256(path)}
                for path in sources
            ],
            "artifacts": [
                {"path": path.name, "sha256": _sha256(path)} for path in artifacts
            ],
            "calibration": {
                "sample_seed": seed,
                "sample_count": len(packets),
                "stratification": (
                    "domain_x_pipeline_equal_allocation_with_balanced_models"
                ),
                "human_labels_complete": False,
            },
        },
    )
    return {
        "corrected_scores": corrected_path,
        "comparison": comparison_path,
        "calibration_packet": packet_path,
        "calibration_mapping": mapping_path,
        "decision_template": template_path,
        "manifest": manifest_path,
    }
