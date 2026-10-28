"""Validate, merge, and re-blind partitioned DeepSeek Judge results."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

import django  # noqa: E402

django.setup()

from apps.evaluations.openai_judge import (  # noqa: E402
    _usage_totals,
    aggregate_scores,
    build_markdown_report,
    build_paired_comparison,
    load_generation_run,
    prepare_blinded_candidates,
    sha256_file,
)


RUN_ROOT = REPO_ROOT / "docs/evaluation/runs/optimization_v1"
SOURCE_PATH = RUN_ROOT / "generation/OPT-V1-ALL-AUDITED-001.json"
OUTPUT_ROOT = RUN_ROOT / "judge/deepseek_api"
PLAN_PATH = OUTPUT_ROOT / "OPT-V1-DEEPSEEKV4-FULL-001_partitions.json"
FINAL_RUN_ID = "OPT-V1-DEEPSEEKV4-FULL-001"
CALIBRATION_RUN_ID = "OPT-V1-DEEPSEEKV4-CALIBRATION12-001"


class MergeError(RuntimeError):
    """Raised when partitioned results cannot be safely consolidated."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MergeError(f"Could not read {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise MergeError(f"Expected a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _display_path(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _condition(item: dict[str, Any]) -> tuple[str, str, int, str]:
    return (
        str(item["tested_model"]),
        str(item["qra_id"]),
        int(item.get("repeat", 1)),
        str(item["mode"]),
    )


def _run_paths(run_id: str) -> dict[str, Path]:
    run_dir = OUTPUT_ROOT / run_id
    return {
        "dir": run_dir,
        "scores": run_dir / f"{run_id}_scores.json",
        "manifest": run_dir / f"{run_id}_manifest.json",
    }


def _component_specs(plan: dict[str, Any]) -> list[dict[str, Any]]:
    calibration_paths = _run_paths(CALIBRATION_RUN_ID)
    calibration = _read_json(calibration_paths["scores"])
    calibration_qras = sorted(
        {str(item["qra_id"]) for item in calibration.get("results", [])}
    )
    specs = [
        {
            "run_id": CALIBRATION_RUN_ID,
            "role": "reused_calibration",
            "qra_ids": calibration_qras,
        }
    ]
    for part in plan.get("parts", []):
        specs.append(
            {
                "run_id": str(part["run_id"]),
                "role": f"remaining_partition_{int(part['part']):02d}",
                "qra_ids": sorted(str(value) for value in part["qra_ids"]),
            }
        )
    return specs


def validate_plan() -> dict[str, Any]:
    """Prove that calibration plus four parts cover the source exactly once."""

    plan = _read_json(PLAN_PATH)
    source = load_generation_run(SOURCE_PATH)
    source_qras = {str(item["id"]) for item in source["dataset"]["items"]}
    source_domains = {
        str(item["id"]): str(item["domain"])
        for item in source["dataset"]["items"]
    }
    if plan.get("plan_id") != FINAL_RUN_ID:
        raise MergeError("Partition plan ID does not match the final run ID")
    if int(plan.get("seed", 0)) != 20260825:
        raise MergeError("Partition plan does not use the frozen blind seed")
    parts = plan.get("parts")
    if not isinstance(parts, list) or len(parts) != 4:
        raise MergeError("Partition plan must contain exactly four parts")

    specs = _component_specs(plan)
    assigned: list[str] = []
    for spec in specs:
        qra_ids = spec["qra_ids"]
        expected_count = 12
        if len(qra_ids) != expected_count or len(set(qra_ids)) != expected_count:
            raise MergeError(
                f"{spec['run_id']} must select {expected_count} unique QRA IDs"
            )
        counts = Counter(source_domains.get(qra_id) for qra_id in qra_ids)
        if None in counts or sorted(counts.values()) != [4, 4, 4]:
            raise MergeError(
                f"{spec['run_id']} must contain four QRA per source domain: {counts}"
            )
        assigned.extend(qra_ids)

    duplicates = sorted(qra_id for qra_id, count in Counter(assigned).items() if count > 1)
    if duplicates:
        raise MergeError(f"QRA IDs occur in multiple components: {duplicates}")
    if set(assigned) != source_qras:
        raise MergeError(
            "Component QRA union differs from the 60-QRA source set: "
            f"missing={sorted(source_qras - set(assigned))}, "
            f"extra={sorted(set(assigned) - source_qras)}"
        )

    full_selection = prepare_blinded_candidates(
        source,
        modes=("baseline", "rag"),
        seed=int(plan["seed"]),
    )
    if full_selection["candidate_count"] != 840:
        raise MergeError(
            f"Expected 840 canonical candidates, found {full_selection['candidate_count']}"
        )
    return {
        "plan": plan,
        "source": source,
        "specs": specs,
        "selection": full_selection,
    }


def _validate_component(
    spec: dict[str, Any],
    *,
    source_sha256: str,
    expected_judge: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = _run_paths(spec["run_id"])
    scores = _read_json(paths["scores"])
    manifest = _read_json(paths["manifest"])
    if scores.get("status") != "completed":
        raise MergeError(f"{spec['run_id']} is not completed")
    if scores.get("source_run", {}).get("sha256") != source_sha256:
        raise MergeError(f"{spec['run_id']} uses a different source artifact")
    results = scores.get("results", [])
    if len(results) != 168 or any(
        item.get("judge", {}).get("status") != "success" for item in results
    ):
        raise MergeError(f"{spec['run_id']} must contain 168 successful results")
    actual_qras = {str(item["qra_id"]) for item in results}
    if actual_qras != set(spec["qra_ids"]):
        raise MergeError(f"{spec['run_id']} result QRA set differs from its plan")
    if manifest.get("artifacts", {}).get("scores", {}).get("sha256") != sha256_file(
        paths["scores"]
    ):
        raise MergeError(f"{spec['run_id']} score hash fails manifest validation")

    judge = scores.get("judge", {})
    frozen_fields = (
        "provider",
        "api",
        "base_url",
        "requested_model",
        "reasoning_effort",
        "max_output_tokens",
        "timeout_seconds",
        "max_retries",
        "structured_outputs",
        "store",
        "input_policy",
        "model_visible_input_fields",
        "answer_evidence_format",
        "unsupported_claims_evaluation",
        "prompt",
    )
    judge_fingerprint = {field: judge.get(field) for field in frozen_fields}
    if expected_judge is not None and judge_fingerprint != expected_judge:
        raise MergeError(f"{spec['run_id']} Judge configuration is not frozen-identical")
    return scores, judge_fingerprint


def merge() -> Path:
    validated = validate_plan()
    source = validated["source"]
    specs = validated["specs"]
    selection = validated["selection"]
    final_dir = OUTPUT_ROOT / FINAL_RUN_ID
    if final_dir.exists():
        final_scores = final_dir / f"{FINAL_RUN_ID}_scores.json"
        final_manifest = final_dir / f"{FINAL_RUN_ID}_manifest.json"
        if final_scores.exists() and final_manifest.exists():
            print(f"Final merged run already exists: {final_dir}")
            return final_dir
        raise MergeError(f"Refusing to overwrite incomplete final directory: {final_dir}")

    components: list[tuple[dict[str, Any], dict[str, Any]]] = []
    expected_judge: dict[str, Any] | None = None
    for spec in specs:
        scores, judge_fingerprint = _validate_component(
            spec,
            source_sha256=source["_artifact"]["sha256"],
            expected_judge=expected_judge,
        )
        expected_judge = expected_judge or judge_fingerprint
        components.append((spec, scores))

    canonical_by_condition = {
        _condition(candidate): candidate for candidate in selection["candidates"]
    }
    component_results: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    component_owners: dict[tuple[str, str, int, str], str] = {}
    prior_attempts: list[dict[str, Any]] = []
    component_manifest_rows = []
    for spec, scores in components:
        for item in scores["results"]:
            key = _condition(item)
            if key in component_results:
                raise MergeError(f"Duplicate scored condition across components: {key}")
            component_results[key] = item
            component_owners[key] = spec["run_id"]
        for item in scores.get("prior_attempts", []):
            remapped = copy.deepcopy(item)
            candidate = canonical_by_condition.get(_condition(remapped))
            if candidate is None:
                raise MergeError("Prior attempt has no canonical full-run candidate")
            component_candidate_id = remapped["candidate_id"]
            remapped["candidate_id"] = candidate["candidate_id"]
            remapped["merge_provenance"] = {
                "component_run_id": spec["run_id"],
                "component_candidate_id": component_candidate_id,
            }
            prior_attempts.append(remapped)
        paths = _run_paths(spec["run_id"])
        component_manifest_rows.append(
            {
                "run_id": spec["run_id"],
                "role": spec["role"],
                "qra_ids": spec["qra_ids"],
                "scores": {
                    "path": _display_path(paths["scores"]),
                    "sha256": sha256_file(paths["scores"]),
                },
                "manifest": {
                    "path": _display_path(paths["manifest"]),
                    "sha256": sha256_file(paths["manifest"]),
                },
            }
        )

    if set(component_results) != set(canonical_by_condition):
        raise MergeError("Merged result conditions differ from canonical 840 candidates")

    results = []
    for candidate in selection["candidates"]:
        source_result = copy.deepcopy(component_results[_condition(candidate)])
        component_candidate_id = source_result["candidate_id"]
        source_result["candidate_id"] = candidate["candidate_id"]
        source_result["source_sequence"] = candidate["source_sequence"]
        source_result["merge_provenance"] = {
            "component_run_id": component_owners[_condition(source_result)],
            "component_candidate_id": component_candidate_id,
        }
        results.append(source_result)

    first_report = components[0][1]
    started_values = [str(scores["started_at"]) for _, scores in components]
    completed_values = [str(scores["completed_at"]) for _, scores in components]
    judge = copy.deepcopy(first_report["judge"])
    judge["response_models"] = sorted(
        {
            str(item["judge"]["response_model"])
            for item in results
            if item["judge"].get("response_model")
        }
    )
    report = {
        "judge_run_id": FINAL_RUN_ID,
        "status": "completed",
        "evaluation_status": "development_api_judge_unvalidated",
        "started_at": min(started_values),
        "completed_at": max(completed_values),
        "source_run": copy.deepcopy(first_report["source_run"]),
        "judge": judge,
        "selection": {
            "seed": selection["seed"],
            "candidate_count": selection["candidate_count"],
            "skipped_generation_errors": selection["skipped_generation_errors"],
            "selection_sha256": selection["selection_sha256"],
            "filters": selection["filters"],
        },
        "summary": aggregate_scores(
            results,
            group_fields=("tested_model", "tested_model_type", "mode"),
        ),
        "domain_summary": aggregate_scores(
            results,
            group_fields=("tested_model", "mode", "domain"),
        ),
        "paired_comparison": build_paired_comparison(results),
        "usage": _usage_totals([*prior_attempts, *results]),
        "results": results,
        "prior_attempts": prior_attempts,
        "resume": {
            "resumed": any(
                bool(scores.get("resume", {}).get("resumed"))
                for _, scores in components
            ),
            "resume_count": sum(
                int(scores.get("resume", {}).get("resume_count", 0))
                for _, scores in components
            ),
            "preexisting_successful_answers": sum(
                int(
                    scores.get("resume", {}).get(
                        "preexisting_successful_answers",
                        0,
                    )
                )
                for _, scores in components
            ),
        },
        "merge": {
            "strategy": "disjoint_qra_partitions_reblinded_to_full_selection",
            "merged_at": datetime.now(UTC).isoformat(),
            "plan": {
                "path": _display_path(PLAN_PATH),
                "sha256": sha256_file(PLAN_PATH),
            },
            "components": component_manifest_rows,
        },
    }

    blinded = {
        "judge_run_id": FINAL_RUN_ID,
        "seed": selection["seed"],
        "candidate_count": selection["candidate_count"],
        "selection_sha256": selection["selection_sha256"],
        "candidates": [
            {
                "candidate_id": item["candidate_id"],
                "judge_input": item["judge_input"],
            }
            for item in selection["candidates"]
        ],
    }
    mapping = {
        "judge_run_id": FINAL_RUN_ID,
        "candidate_count": selection["candidate_count"],
        "selection_sha256": selection["selection_sha256"],
        "candidates": [
            {
                key: item[key]
                for key in (
                    "candidate_id",
                    "source_sequence",
                    "qra_id",
                    "domain",
                    "question_type",
                    "tested_model",
                    "tested_model_type",
                    "mode",
                    "repeat",
                )
            }
            for item in selection["candidates"]
        ],
    }

    temporary_dir = OUTPUT_ROOT / f".{FINAL_RUN_ID}.assembling"
    if temporary_dir.exists():
        raise MergeError(f"Refusing to overwrite temporary directory: {temporary_dir}")
    temporary_dir.mkdir(parents=True)
    blinded_path = temporary_dir / f"{FINAL_RUN_ID}_blinded_inputs.json"
    mapping_path = temporary_dir / f"{FINAL_RUN_ID}_candidate_mapping.json"
    scores_path = temporary_dir / f"{FINAL_RUN_ID}_scores.json"
    report_path = temporary_dir / f"{FINAL_RUN_ID}_report.md"
    manifest_path = temporary_dir / f"{FINAL_RUN_ID}_manifest.json"
    _write_json(blinded_path, blinded)
    _write_json(mapping_path, mapping)
    _write_json(scores_path, report)
    report_path.write_text(build_markdown_report(report), encoding="utf-8")

    artifacts = {}
    for name, path in (
        ("blinded_inputs", blinded_path),
        ("candidate_mapping", mapping_path),
        ("scores", scores_path),
        ("markdown_report", report_path),
    ):
        artifacts[name] = {
            "path": path.name,
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    manifest = {
        "judge_run_id": FINAL_RUN_ID,
        "status": report["status"],
        "evaluation_status": report["evaluation_status"],
        "source_run": report["source_run"],
        "judge": report["judge"],
        "selection": report["selection"],
        "merge": report["merge"],
        "implementation": {
            "judge": {
                "path": "backend/apps/evaluations/openai_judge.py",
                "sha256": sha256_file(
                    REPO_ROOT / "backend/apps/evaluations/openai_judge.py"
                ),
            },
            "merge": {
                "path": _display_path(Path(__file__)),
                "sha256": sha256_file(Path(__file__)),
            },
        },
        "artifacts": artifacts,
        "limitations": [
            "Automated development-set Judge results are not formal thesis results.",
            "Human agreement validation and adjudication have not yet been completed.",
            "The requested model alias may move; freeze a dated model snapshot before formal evaluation.",
            "Unsupported-claim extraction was not evaluated because the keypoint Judge input is reference-isolated; the corresponding counts are null/N/A, not zero.",
            "API calls were executed in five disjoint 12-QRA components; canonical full-run candidate IDs were assigned after scoring without changing any Judge decision.",
        ],
    }
    _write_json(manifest_path, manifest)
    temporary_dir.replace(final_dir)
    print(f"Merged {len(results)} successful scores into {final_dir}")
    print(f"Preserved prior attempts: {len(prior_attempts)}")
    print(f"Full selection SHA-256: {selection['selection_sha256']}")
    return final_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--validate-plan",
        action="store_true",
        help="Validate the 60-QRA partition plan without requiring part results.",
    )
    args = parser.parse_args()
    if args.validate_plan:
        validated = validate_plan()
        print(
            "Partition validation passed: "
            f"{len(validated['specs'])} components, "
            f"{validated['selection']['candidate_count']} canonical candidates"
        )
        return
    merge()


if __name__ == "__main__":
    main()
