"""Resumable, condition-blinded Codex Judge for formal RAG generations."""

from __future__ import annotations

import hashlib
import json
import random
import shutil
import subprocess
import tempfile
import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from django.conf import settings


ALLOWED_LABELS = {"covered", "contradicted", "neither"}
ALLOWED_PHASES = {"first", "remaining"}
PROMPT_VERSION = "formal_codex_keypoint_judge_v1.0"
DEFAULT_FORMAL_SOURCE_NAMES = (
    "FORMAL-CORE-REMAINING-RAG-001.json",
    "FORMAL-GEMMA4-31B-RAG-001.json",
    "FORMAL-KIMI-K3-RAG-001.json",
)


class FormalCodexJudgeError(RuntimeError):
    """Raised when formal Codex Judge inputs or checkpoints are inconsistent."""


BatchJudge = Callable[[list[dict[str, Any]]], tuple[dict[str, Any], dict[str, Any]]]


def project_root() -> Path:
    return Path(settings.BASE_DIR).parent


def default_formal_source_paths() -> list[Path]:
    root = project_root() / "docs/evaluation/runs/formal"
    return [root / name for name in DEFAULT_FORMAL_SOURCE_NAMES]


def default_codex_judge_output_root() -> Path:
    return project_root() / "docs/evaluation/runs/formal/judge/codex"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FormalCodexJudgeError(f"Could not read JSON artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise FormalCodexJudgeError(f"Expected a JSON object: {path}")
    return payload


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as target:
        target.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _summarize_codex_usage(raw_batch_dir: Path) -> dict[str, int]:
    usage = {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "cache_write_input_tokens": 0,
        "output_tokens": 0,
        "reasoning_output_tokens": 0,
    }
    for path in sorted(raw_batch_dir.glob("batch_*_events.jsonl")):
        try:
            lines = path.read_text(encoding="utf-8-sig").splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") != "turn.completed":
                continue
            event_usage = event.get("usage", {})
            for key in usage:
                usage[key] += int(event_usage.get(key, 0) or 0)
    usage["total_non_cached_tokens"] = usage["input_tokens"] + usage["output_tokens"]
    return usage


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(project_root().resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _segment_answer(answer: str) -> list[dict[str, str]]:
    segments = [line.strip() for line in answer.splitlines() if line.strip()]
    if not segments:
        raise FormalCodexJudgeError("Candidate answer has no non-empty segments")
    return [
        {"id": f"A{position:04d}", "text": text}
        for position, text in enumerate(segments, start=1)
    ]


def _collect_formal_candidates(
    source_paths: Iterable[Path],
    *,
    seed: int,
    condition_mode: str,
) -> dict[str, Any]:
    resolved = [path.resolve() for path in source_paths]
    if not resolved:
        raise FormalCodexJudgeError("At least one formal source run is required")
    missing = [str(path) for path in resolved if not path.is_file()]
    if missing:
        raise FormalCodexJudgeError(f"Formal source runs are missing: {missing}")

    runs = [(path, _load_json(path)) for path in resolved]
    if condition_mode not in {"baseline", "rag"}:
        raise FormalCodexJudgeError(f"Unsupported formal condition: {condition_mode}")

    # A generation that exhausted its original token cap can be completed in a
    # later run containing only the failed QRA subset.  Use the largest dataset
    # as the canonical benchmark and verify every smaller source is an exact
    # subset of it.
    canonical_path, canonical_run = max(
        runs,
        key=lambda item: len(item[1].get("dataset", {}).get("items", [])),
    )
    dataset_items = canonical_run.get("dataset", {}).get("items", [])
    if not dataset_items:
        raise FormalCodexJudgeError("Formal source run contains no dataset items")
    qra_by_id = {str(item["id"]): item for item in dataset_items}
    if len(qra_by_id) != len(dataset_items):
        raise FormalCodexJudgeError("Canonical formal dataset contains duplicate QRA IDs")
    dataset_fingerprint = _canonical_sha256(dataset_items)
    dataset_manifest_sha256 = str(
        canonical_run.get("dataset", {}).get("manifest_sha256", "")
    )
    if not dataset_manifest_sha256:
        raise FormalCodexJudgeError(
            f"Canonical formal dataset manifest is missing in {canonical_path}"
        )
    for path, run in runs:
        other_items = run.get("dataset", {}).get("items", [])
        other_by_id = {str(item["id"]): item for item in other_items}
        if len(other_by_id) != len(other_items):
            raise FormalCodexJudgeError(f"Dataset contains duplicate QRA IDs in {path}")
        for qra_id, item in other_by_id.items():
            if qra_id not in qra_by_id or _canonical_sha256(item) != _canonical_sha256(
                qra_by_id[qra_id]
            ):
                raise FormalCodexJudgeError(
                    f"Dataset item {qra_id} differs from the canonical dataset in {path}"
                )
        if len(other_items) == len(dataset_items):
            other_manifest = str(run.get("dataset", {}).get("manifest_sha256", ""))
            if (
                _canonical_sha256(other_items) != dataset_fingerprint
                or other_manifest != dataset_manifest_sha256
            ):
                raise FormalCodexJudgeError(f"Full dataset differs in {path}")

    conditions: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, int]] = set()
    source_models: set[str] = set()
    source_manifest: list[dict[str, Any]] = []
    for path, run in runs:
        source_dataset_ids = {
            str(item["id"]): item
            for item in run.get("dataset", {}).get("items", [])
        }
        successful_result_count = 0
        skipped_result_count = 0
        for result in run.get("results", []):
            generation = result.get("generation", {})
            model = str(result.get("model", ""))
            qra_id = str(result.get("qra_id", ""))
            mode = str(result.get("mode", ""))
            repeat = int(result.get("repeat", 1))
            source_models.add(model)
            key = (model, qra_id, mode, repeat)
            if mode != condition_mode or repeat != 1:
                raise FormalCodexJudgeError(
                    f"Expected one {condition_mode} repeat per formal condition: {key}"
                )
            qra = qra_by_id.get(qra_id)
            if qra is None:
                raise FormalCodexJudgeError(f"Unknown formal QRA: {qra_id}")
            if qra_id not in source_dataset_ids:
                raise FormalCodexJudgeError(
                    f"Generation {model}/{qra_id} is absent from its source dataset"
                )
            if generation.get("status") != "success":
                skipped_result_count += 1
                continue
            if key in seen:
                raise FormalCodexJudgeError(f"Duplicate successful formal condition: {key}")
            seen.add(key)
            successful_result_count += 1
            answer = str(generation.get("response", "")).strip()
            if not answer:
                raise FormalCodexJudgeError(f"Empty successful answer: {model}/{qra_id}")
            conditions.append(
                {
                    "model": model,
                    "model_type": str(result.get("model_type", "")),
                    "qra_id": qra_id,
                    "mode": mode,
                    "domain": str(result.get("domain", qra.get("domain", ""))),
                    "question_type": str(
                        result.get("question_type", qra.get("question_type", ""))
                    ),
                    "source_sequence": int(result.get("sequence", 0)),
                    "candidate_answer": answer,
                }
            )
        source_manifest.append(
            {
                "path": _display_path(path),
                "sha256": _file_sha256(path),
                "run_id": str(run.get("run_id", "")),
                "dataset_item_count": len(source_dataset_ids),
                "dataset_manifest_sha256": str(
                    run.get("dataset", {}).get("manifest_sha256", "")
                ),
                "result_count": len(run.get("results", [])),
                "successful_result_count": successful_result_count,
                "skipped_result_count": skipped_result_count,
            }
        )

    models = sorted(source_models)
    expected = len(qra_by_id) * len(models)
    if len(conditions) != expected:
        raise FormalCodexJudgeError(
            f"Expected {expected} formal conditions, found {len(conditions)}"
        )

    random.Random(seed).shuffle(conditions)
    packets: list[dict[str, Any]] = []
    mappings: list[dict[str, Any]] = []
    for position, condition in enumerate(conditions, start=1):
        candidate_id = f"C{position:04d}"
        qra = qra_by_id[condition["qra_id"]]
        answer_segments = _segment_answer(condition["candidate_answer"])
        packets.append(
            {
                "candidate_id": candidate_id,
                "qra_id": condition["qra_id"],
                "language": str(qra["language"]),
                "question": str(qra["question"]),
                "answerable": bool(qra["answerable"]),
                "keypoints": [
                    {"id": str(item["id"]), "text": str(item["text"])}
                    for item in qra["keypoints"]
                ],
                "candidate_answer_segments": answer_segments,
            }
        )
        mappings.append(
            {
                "candidate_id": candidate_id,
                **{
                    key: value
                    for key, value in condition.items()
                    if key != "candidate_answer"
                },
                "answer_sha256": _canonical_sha256(condition["candidate_answer"]),
            }
        )
    return {
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "dataset_item_count": len(qra_by_id),
        "models": models,
        "packets": packets,
        "mappings": mappings,
        "source_manifest": source_manifest,
    }


def prepare_formal_codex_judge_run(
    source_paths: Iterable[Path],
    *,
    run_dir: Path,
    run_id: str,
    seed: int,
    first_phase_model_count: int,
    judge_model: str,
    reasoning_effort: str,
    condition_mode: str = "rag",
) -> dict[str, Path]:
    """Create immutable blinded packets, sealed mappings, and run metadata."""

    if run_dir.exists() and any(run_dir.iterdir()):
        raise FormalCodexJudgeError(
            f"Judge run directory already contains files; use --resume: {run_dir}"
        )
    collected = _collect_formal_candidates(
        source_paths,
        seed=seed,
        condition_mode=condition_mode,
    )
    models = collected["models"]
    if first_phase_model_count < 1 or first_phase_model_count > len(models):
        raise FormalCodexJudgeError(
            "first_phase_model_count must be between 1 and the model count"
        )
    first_models = models[:first_phase_model_count]
    remaining_models = models[first_phase_model_count:]
    run_dir.mkdir(parents=True, exist_ok=True)
    packet_path = run_dir / f"{run_id}_blinded_packets.json"
    mapping_path = run_dir / f"{run_id}_candidate_mapping.json"
    manifest_path = run_dir / f"{run_id}_manifest.json"
    checkpoint_path = run_dir / f"{run_id}_checkpoint.json"
    _write_json_atomic(
        packet_path,
        {
            "judge_run_id": run_id,
            "status": "prepared",
            "judge_type": "codex_cli_condition_blinded",
            "prompt_version": PROMPT_VERSION,
            "shuffle_seed": seed,
            "dataset_manifest_sha256": collected["dataset_manifest_sha256"],
            "packet_count": len(collected["packets"]),
            "packets": collected["packets"],
        },
    )
    _write_json_atomic(
        mapping_path,
        {
            "judge_run_id": run_id,
            "status": "sealed_until_scoring_complete",
            "candidate_count": len(collected["mappings"]),
            "mapping": collected["mappings"],
        },
    )
    manifest = {
        "judge_run_id": run_id,
        "status": "prepared",
        "evaluation_status": "formal_multi_judge_sensitivity_pending_validation",
        "created_at": _utc_now(),
        "judge": {
            "surface": "codex_cli_exec",
            "requested_model": judge_model,
            "reasoning_effort": reasoning_effort,
            "prompt_version": PROMPT_VERSION,
            "reference_isolated": True,
            "unsupported_claims_evaluated": False,
        },
        "condition_mode": condition_mode,
        "dataset_manifest_sha256": collected["dataset_manifest_sha256"],
        "qra_count": collected["dataset_item_count"],
        "model_order": models,
        "first_phase_models": first_models,
        "remaining_phase_models": remaining_models,
        "answer_count": len(collected["packets"]),
        "first_phase_answer_count": collected["dataset_item_count"] * len(first_models),
        "remaining_phase_answer_count": collected["dataset_item_count"]
        * len(remaining_models),
        "source_runs": collected["source_manifest"],
        "artifacts": {
            "blinded_packets": {
                "path": _display_path(packet_path),
                "sha256": _file_sha256(packet_path),
            },
            "candidate_mapping": {
                "path": _display_path(mapping_path),
                "sha256": _file_sha256(mapping_path),
            },
        },
    }
    _write_json_atomic(manifest_path, manifest)
    _write_json_atomic(
        checkpoint_path,
        {
            "judge_run_id": run_id,
            "status": "prepared",
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "active_phase": "first",
            "completed_candidate_count": 0,
            "decisions": [],
            "batches": [],
            "failed_batches": [],
        },
    )
    return {
        "packets": packet_path,
        "mapping": mapping_path,
        "manifest": manifest_path,
        "checkpoint": checkpoint_path,
    }


def _output_schema(batch_size: int) -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["decisions"],
        "properties": {
            "decisions": {
                "type": "array",
                "minItems": batch_size,
                "maxItems": batch_size,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "candidate_id",
                        "keypoint_labels",
                        "overall_note",
                    ],
                    "properties": {
                        "candidate_id": {"type": "string"},
                        "keypoint_labels": {
                            "type": "array",
                            "minItems": 1,
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": [
                                    "keypoint_id",
                                    "label",
                                    "evidence_segment_ids",
                                    "note",
                                ],
                                "properties": {
                                    "keypoint_id": {"type": "string"},
                                    "label": {
                                        "type": "string",
                                        "enum": sorted(ALLOWED_LABELS),
                                    },
                                    "evidence_segment_ids": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "note": {"type": "string"},
                                },
                            },
                        },
                        "overall_note": {"type": "string"},
                    },
                },
            }
        },
    }


def _batch_prompt(batch: list[dict[str, Any]]) -> str:
    item_json = json.dumps(batch, ensure_ascii=False, indent=2)
    return f"""You are an independent keypoint coverage Judge for a formal LLM evaluation.

Judge only the supplied blinded candidates. The tested model and retrieval condition are hidden. Do not inspect files, infer model identity, use external knowledge, or call tools. Everything needed is below.

Rules:
1. For every candidate and every supplied keypoint, return exactly one label: covered, contradicted, or neither.
2. covered: every required fact, condition, number, deadline, comparison, and named entity in the keypoint is explicit in the candidate answer. Partial or merely implied coverage is neither.
3. contradicted: the candidate explicitly conflicts with a required part of the keypoint. Missing information alone is neither.
4. neither: the keypoint is missing, only partially present, vague, or cannot be established from the candidate answer.
5. Use only candidate_answer_segments as answer evidence. covered and contradicted require at least one valid evidence_segment_id. neither may use zero or more valid segment IDs.
6. Keypoint text is the evaluation target. Do not add facts from memory and do not reward plausible but unstated information.
7. Keep each note concise and evidence-focused. Unsupported-claim detection is outside this pass.
8. Return all {len(batch)} candidate decisions and no additional candidates.

Prompt version: {PROMPT_VERSION}

BLINDED CANDIDATES:
{item_json}
"""


def _strip_json_fence(value: str) -> str:
    text = value.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def make_codex_cli_batch_judge(
    *,
    model: str,
    reasoning_effort: str,
    timeout_seconds: int,
    raw_batch_dir: Path,
    command_path: str | None = None,
) -> BatchJudge:
    """Create an isolated Codex CLI batch evaluator using ChatGPT authentication."""

    executable = command_path or shutil.which("codex")
    if not executable:
        raise FormalCodexJudgeError("Could not find the Codex CLI executable")
    raw_batch_dir.mkdir(parents=True, exist_ok=True)
    call_index = len(list(raw_batch_dir.glob("batch_*_response.json")))

    def judge(batch: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
        nonlocal call_index
        call_index += 1
        batch_name = f"batch_{call_index:04d}"
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="formal-codex-judge-") as temp_name:
            temp_dir = Path(temp_name)
            schema_path = temp_dir / "output_schema.json"
            response_path = temp_dir / "response.json"
            schema_path.write_text(
                json.dumps(_output_schema(len(batch)), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            command = [
                executable,
                "exec",
                "-",
                "--model",
                model,
                "-c",
                f'model_reasoning_effort="{reasoning_effort}"',
                "--ephemeral",
                "--ignore-user-config",
                "--ignore-rules",
                "--skip-git-repo-check",
                "--sandbox",
                "read-only",
                "--cd",
                str(temp_dir),
                "--output-schema",
                str(schema_path),
                "--output-last-message",
                str(response_path),
                "--json",
                "--color",
                "never",
            ]
            try:
                completed = subprocess.run(
                    command,
                    input=_batch_prompt(batch),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise FormalCodexJudgeError(
                    f"Codex Judge batch timed out after {timeout_seconds}s"
                ) from exc
            elapsed = round(time.perf_counter() - started, 3)
            events_path = raw_batch_dir / f"{batch_name}_events.jsonl"
            events_path.write_text(completed.stdout, encoding="utf-8")
            stderr_path = raw_batch_dir / f"{batch_name}_stderr.log"
            stderr_path.write_text(completed.stderr, encoding="utf-8")
            if completed.returncode != 0:
                raise FormalCodexJudgeError(
                    f"Codex Judge exited {completed.returncode}; see {stderr_path}"
                )
            if not response_path.is_file():
                raise FormalCodexJudgeError(
                    f"Codex Judge produced no final response; see {events_path}"
                )
            raw_response = response_path.read_text(encoding="utf-8-sig")
            response_copy = raw_batch_dir / f"{batch_name}_response.json"
            response_copy.write_text(raw_response, encoding="utf-8")
            try:
                payload = json.loads(_strip_json_fence(raw_response))
            except json.JSONDecodeError as exc:
                raise FormalCodexJudgeError(
                    f"Codex Judge returned invalid JSON in {response_copy}: {exc}"
                ) from exc
            metadata = {
                "batch_name": batch_name,
                "elapsed_seconds": elapsed,
                "candidate_count": len(batch),
                "response_path": _display_path(response_copy),
                "events_path": _display_path(events_path),
                "stderr_path": _display_path(stderr_path),
            }
            return payload, metadata

    return judge


def _validate_batch_decisions(
    batch: list[dict[str, Any]],
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    expected = {str(item["candidate_id"]): item for item in batch}
    raw_decisions = payload.get("decisions")
    if not isinstance(raw_decisions, list):
        raise FormalCodexJudgeError("Codex response has no decisions array")
    decisions = {str(item.get("candidate_id", "")): item for item in raw_decisions}
    if len(decisions) != len(raw_decisions) or set(decisions) != set(expected):
        raise FormalCodexJudgeError(
            "Codex response candidate IDs differ from the requested batch"
        )

    validated: list[dict[str, Any]] = []
    for candidate_id, packet in expected.items():
        decision = decisions[candidate_id]
        expected_keypoints = {
            str(keypoint["id"]): keypoint for keypoint in packet["keypoints"]
        }
        expected_segments = {
            str(segment["id"]) for segment in packet["candidate_answer_segments"]
        }
        raw_labels = decision.get("keypoint_labels")
        if not isinstance(raw_labels, list):
            raise FormalCodexJudgeError(f"Missing labels for {candidate_id}")
        labels = {str(item.get("keypoint_id", "")): item for item in raw_labels}
        if len(labels) != len(raw_labels) or set(labels) != set(expected_keypoints):
            raise FormalCodexJudgeError(f"Invalid keypoint IDs for {candidate_id}")
        ordered = []
        for keypoint in packet["keypoints"]:
            keypoint_id = str(keypoint["id"])
            item = labels[keypoint_id]
            label = str(item.get("label", ""))
            if label not in ALLOWED_LABELS:
                raise FormalCodexJudgeError(
                    f"Invalid label for {candidate_id}/{keypoint_id}: {label}"
                )
            evidence = [str(value) for value in item.get("evidence_segment_ids", [])]
            if any(value not in expected_segments for value in evidence):
                raise FormalCodexJudgeError(
                    f"Invalid evidence segment for {candidate_id}/{keypoint_id}"
                )
            if label in {"covered", "contradicted"} and not evidence:
                raise FormalCodexJudgeError(
                    f"{label} requires evidence for {candidate_id}/{keypoint_id}"
                )
            ordered.append(
                {
                    "keypoint_id": keypoint_id,
                    "label": label,
                    "evidence_segment_ids": evidence,
                    "note": str(item.get("note", "")),
                }
            )
        validated.append(
            {
                "candidate_id": candidate_id,
                "keypoint_labels": ordered,
                "overall_note": str(decision.get("overall_note", "")),
            }
        )
    return validated


def _write_progress(
    *,
    run_dir: Path,
    run_id: str,
    checkpoint: dict[str, Any],
    phase: str,
    phase_total: int,
    total_candidates: int,
    phase_models: list[str],
    remaining_models: list[str],
) -> None:
    completed = len(checkpoint["decisions"])
    phase_completed = int(checkpoint.get("phase_completed_candidate_count", 0))
    durations = [
        float(item.get("elapsed_seconds", 0))
        for item in checkpoint.get("batches", [])
        if item.get("elapsed_seconds")
    ]
    mean_batch_seconds = round(sum(durations) / len(durations), 2) if durations else None
    last_batch_size = (
        int(checkpoint["batches"][-1].get("candidate_count", 0))
        if checkpoint.get("batches")
        else 0
    )
    remaining_count = max(0, phase_total - phase_completed)
    eta_seconds = None
    if mean_batch_seconds and last_batch_size:
        eta_seconds = round(mean_batch_seconds * remaining_count / last_batch_size)
    reserved_candidate_count = (
        max(0, total_candidates - phase_total) if phase == "first" else 0
    )
    reserved_eta_seconds = None
    if mean_batch_seconds and last_batch_size and reserved_candidate_count:
        reserved_eta_seconds = round(
            mean_batch_seconds * reserved_candidate_count / last_batch_size
        )
    usage = _summarize_codex_usage(run_dir / "raw_batches")
    payload = {
        "judge_run_id": run_id,
        "status": checkpoint["status"],
        "updated_at": checkpoint["updated_at"],
        "active_phase": phase,
        "phase_models": phase_models,
        "remaining_models": remaining_models,
        "total_candidate_count": total_candidates,
        "completed_candidate_count": completed,
        "phase_candidate_count": phase_total,
        "phase_completed_candidate_count": phase_completed,
        "phase_remaining_candidate_count": remaining_count,
        "completed_batch_count": len(checkpoint.get("batches", [])),
        "failed_batch_count": len(checkpoint.get("failed_batches", [])),
        "mean_batch_seconds": mean_batch_seconds,
        "estimated_phase_remaining_seconds": eta_seconds,
        "estimated_reserved_phase_seconds": reserved_eta_seconds,
        "usage": usage,
        "next_candidate_id": checkpoint.get("next_candidate_id"),
        "resume_command": checkpoint.get("resume_command"),
    }
    _write_json_atomic(run_dir / f"{run_id}_progress.json", payload)
    eta_text = (
        f"{eta_seconds // 60}m {eta_seconds % 60}s" if eta_seconds is not None else "pending pilot"
    )
    reserved_eta_text = (
        f"{reserved_eta_seconds // 60}m {reserved_eta_seconds % 60}s"
        if reserved_eta_seconds is not None
        else "not applicable"
    )
    lines = [
        f"# {run_id} progress",
        "",
        f"- Status: `{payload['status']}`",
        f"- Active phase: `{phase}`",
        f"- Phase models: `{', '.join(phase_models)}`",
        f"- Models reserved for later: `{', '.join(remaining_models) or 'none'}`",
        f"- Overall completed: `{completed}/{total_candidates}`",
        f"- Active phase completed: `{phase_completed}/{phase_total}`",
        f"- Completed batches: `{payload['completed_batch_count']}`",
        f"- Failed batches: `{payload['failed_batch_count']}`",
        f"- Mean completed batch time: `{mean_batch_seconds or 'pending pilot'} s`",
        f"- Estimated active-phase time remaining: `{eta_text}`",
        f"- Estimated reserved-phase runtime: `{reserved_eta_text}`",
        f"- Codex input tokens recorded: `{usage['input_tokens']}`",
        f"- Codex cached input tokens recorded: `{usage['cached_input_tokens']}`",
        f"- Codex output tokens recorded: `{usage['output_tokens']}`",
        f"- Codex reasoning output tokens recorded: `{usage['reasoning_output_tokens']}`",
        f"- Next candidate: `{payload['next_candidate_id'] or 'none'}`",
        f"- Updated at: `{payload['updated_at']}`",
        "",
        "## Resume",
        "",
        "```powershell",
        str(payload["resume_command"] or "No resume command recorded."),
        "```",
        "",
    ]
    (run_dir / f"{run_id}_progress.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def _write_scores(
    *,
    run_dir: Path,
    run_id: str,
    packets: dict[str, dict[str, Any]],
    mappings: dict[str, dict[str, Any]],
    decisions: list[dict[str, Any]],
    status: str,
) -> None:
    scored: list[dict[str, Any]] = []
    for decision in decisions:
        candidate_id = str(decision["candidate_id"])
        packet = packets[candidate_id]
        mapping = mappings[candidate_id]
        counts = {
            label: sum(
                item["label"] == label for item in decision["keypoint_labels"]
            )
            for label in ALLOWED_LABELS
        }
        total = len(packet["keypoints"])
        scored.append(
            {
                "candidate_id": candidate_id,
                "qra_id": mapping["qra_id"],
                "domain": mapping["domain"],
                "question_type": mapping["question_type"],
                "model": mapping["model"],
                "model_type": mapping["model_type"],
                "mode": mapping["mode"],
                "keypoint_count": total,
                "covered": counts["covered"],
                "contradicted": counts["contradicted"],
                "neither": counts["neither"],
                "completeness": round(counts["covered"] / total, 4),
                "hallucination": round(counts["contradicted"] / total, 4),
                "irrelevancy": round(counts["neither"] / total, 4),
                "keypoint_labels": decision["keypoint_labels"],
                "judge_note": decision["overall_note"],
            }
        )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in scored:
        grouped[item["model"]].append(item)
    summary = []
    for model, items in sorted(grouped.items()):
        summary.append(
            {
                "model": model,
                "answer_count": len(items),
                "mean_completeness": round(
                    sum(item["completeness"] for item in items) / len(items), 4
                ),
                "mean_hallucination": round(
                    sum(item["hallucination"] for item in items) / len(items), 4
                ),
                "mean_irrelevancy": round(
                    sum(item["irrelevancy"] for item in items) / len(items), 4
                ),
            }
        )
    _write_json_atomic(
        run_dir / f"{run_id}_scores.json",
        {
            "judge_run_id": run_id,
            "status": status,
            "evaluation_status": "formal_multi_judge_sensitivity_pending_validation",
            "updated_at": _utc_now(),
            "summary": summary,
            "result_count": len(scored),
            "results": scored,
        },
    )


def run_formal_codex_judge_phase(
    *,
    run_dir: Path,
    run_id: str,
    phase: str,
    batch_size: int,
    batch_judge: BatchJudge,
    resume_command: str,
    max_batches: int | None = None,
    max_batch_retries: int = 1,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Judge one configured model phase, persisting after every successful batch."""

    if phase not in ALLOWED_PHASES:
        raise FormalCodexJudgeError(f"phase must be one of {sorted(ALLOWED_PHASES)}")
    if batch_size < 1:
        raise FormalCodexJudgeError("batch_size must be positive")
    manifest_path = run_dir / f"{run_id}_manifest.json"
    packet_path = run_dir / f"{run_id}_blinded_packets.json"
    mapping_path = run_dir / f"{run_id}_candidate_mapping.json"
    checkpoint_path = run_dir / f"{run_id}_checkpoint.json"
    manifest = _load_json(manifest_path)
    packet_payload = _load_json(packet_path)
    mapping_payload = _load_json(mapping_path)
    checkpoint = _load_json(checkpoint_path)
    if manifest.get("judge_run_id") != run_id or checkpoint.get("judge_run_id") != run_id:
        raise FormalCodexJudgeError("Run ID differs across formal Judge artifacts")
    artifacts = manifest.get("artifacts", {})
    if _file_sha256(packet_path) != artifacts.get("blinded_packets", {}).get("sha256"):
        raise FormalCodexJudgeError("Blinded packet hash differs from the manifest")
    if _file_sha256(mapping_path) != artifacts.get("candidate_mapping", {}).get("sha256"):
        raise FormalCodexJudgeError("Candidate mapping hash differs from the manifest")

    packets = {
        str(item["candidate_id"]): item for item in packet_payload["packets"]
    }
    mappings = {
        str(item["candidate_id"]): item for item in mapping_payload["mapping"]
    }
    if set(packets) != set(mappings):
        raise FormalCodexJudgeError("Blinded packet and mapping candidate IDs differ")
    phase_models = list(
        manifest[
            "first_phase_models" if phase == "first" else "remaining_phase_models"
        ]
    )
    models_reserved_for_later = (
        list(manifest["remaining_phase_models"]) if phase == "first" else []
    )
    phase_candidate_ids = sorted(
        candidate_id
        for candidate_id, mapping in mappings.items()
        if mapping["model"] in phase_models
    )
    completed = {
        str(decision["candidate_id"]) for decision in checkpoint.get("decisions", [])
    }
    pending = [value for value in phase_candidate_ids if value not in completed]
    checkpoint.update(
        {
            "status": "running",
            "active_phase": phase,
            "updated_at": _utc_now(),
            "phase_candidate_count": len(phase_candidate_ids),
            "phase_completed_candidate_count": len(phase_candidate_ids) - len(pending),
            "next_candidate_id": pending[0] if pending else None,
            "resume_command": resume_command,
        }
    )
    _write_json_atomic(checkpoint_path, checkpoint)
    log_path = run_dir / f"{run_id}_events.jsonl"
    _append_jsonl(
        log_path,
        {
            "event": "phase_started",
            "at": _utc_now(),
            "phase": phase,
            "phase_models": phase_models,
            "pending_candidate_count": len(pending),
        },
    )

    batches_run = 0
    while pending and (max_batches is None or batches_run < max_batches):
        candidate_ids = pending[:batch_size]
        batch = [packets[value] for value in candidate_ids]
        error: Exception | None = None
        for attempt in range(1, max_batch_retries + 2):
            try:
                payload, metadata = batch_judge(batch)
                validated = _validate_batch_decisions(batch, payload)
                break
            except Exception as exc:  # noqa: BLE001 - preserve resumable audit trail
                error = exc
                failure = {
                    "at": _utc_now(),
                    "phase": phase,
                    "candidate_ids": candidate_ids,
                    "attempt": attempt,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                checkpoint.setdefault("failed_batches", []).append(failure)
                checkpoint["updated_at"] = _utc_now()
                _write_json_atomic(checkpoint_path, checkpoint)
                _append_jsonl(log_path, {"event": "batch_failed", **failure})
        else:
            checkpoint["status"] = "interrupted"
            checkpoint["next_candidate_id"] = candidate_ids[0]
            checkpoint["updated_at"] = _utc_now()
            _write_json_atomic(checkpoint_path, checkpoint)
            _write_progress(
                run_dir=run_dir,
                run_id=run_id,
                checkpoint=checkpoint,
                phase=phase,
                phase_total=len(phase_candidate_ids),
                total_candidates=len(packets),
                phase_models=phase_models,
                remaining_models=models_reserved_for_later,
            )
            assert error is not None
            raise FormalCodexJudgeError(
                f"Batch failed after retries; resume from {candidate_ids[0]}: {error}"
            ) from error

        checkpoint.setdefault("decisions", []).extend(validated)
        checkpoint.setdefault("batches", []).append(
            {
                **metadata,
                "phase": phase,
                "attempt": attempt,
                "candidate_ids": candidate_ids,
                "completed_at": _utc_now(),
            }
        )
        completed.update(candidate_ids)
        pending = [value for value in pending if value not in completed]
        checkpoint.update(
            {
                "updated_at": _utc_now(),
                "completed_candidate_count": len(completed),
                "phase_completed_candidate_count": len(phase_candidate_ids)
                - len(pending),
                "next_candidate_id": pending[0] if pending else None,
            }
        )
        _write_json_atomic(checkpoint_path, checkpoint)
        _append_jsonl(
            log_path,
            {
                "event": "batch_completed",
                "at": _utc_now(),
                "phase": phase,
                "candidate_ids": candidate_ids,
                **metadata,
            },
        )
        _write_scores(
            run_dir=run_dir,
            run_id=run_id,
            packets=packets,
            mappings=mappings,
            decisions=checkpoint["decisions"],
            status="running",
        )
        _write_progress(
            run_dir=run_dir,
            run_id=run_id,
            checkpoint=checkpoint,
            phase=phase,
            phase_total=len(phase_candidate_ids),
            total_candidates=len(packets),
            phase_models=phase_models,
            remaining_models=models_reserved_for_later,
        )
        batches_run += 1
        if progress_callback:
            progress_callback(
                {
                    "phase": phase,
                    "completed": checkpoint["phase_completed_candidate_count"],
                    "total": len(phase_candidate_ids),
                    "last_batch_seconds": metadata.get("elapsed_seconds"),
                    "next_candidate_id": checkpoint["next_candidate_id"],
                }
            )

    if pending:
        checkpoint["status"] = "paused"
    else:
        if phase == "first" and models_reserved_for_later:
            checkpoint["status"] = "first_phase_completed"
            checkpoint["resume_command"] = resume_command.replace(
                "--phase first", "--phase remaining"
            )
        else:
            checkpoint["status"] = "completed"
            checkpoint["resume_command"] = None
    checkpoint["updated_at"] = _utc_now()
    checkpoint["next_candidate_id"] = pending[0] if pending else None
    _write_json_atomic(checkpoint_path, checkpoint)
    _write_scores(
        run_dir=run_dir,
        run_id=run_id,
        packets=packets,
        mappings=mappings,
        decisions=checkpoint["decisions"],
        status=checkpoint["status"],
    )
    _write_progress(
        run_dir=run_dir,
        run_id=run_id,
        checkpoint=checkpoint,
        phase=phase,
        phase_total=len(phase_candidate_ids),
        total_candidates=len(packets),
        phase_models=phase_models,
        remaining_models=models_reserved_for_later,
    )
    _append_jsonl(
        log_path,
        {
            "event": "phase_stopped",
            "at": _utc_now(),
            "phase": phase,
            "status": checkpoint["status"],
            "phase_completed_candidate_count": checkpoint[
                "phase_completed_candidate_count"
            ],
            "phase_candidate_count": len(phase_candidate_ids),
            "next_candidate_id": checkpoint["next_candidate_id"],
        },
    )
    manifest["status"] = checkpoint["status"]
    manifest["updated_at"] = checkpoint["updated_at"]
    _write_json_atomic(manifest_path, manifest)
    return checkpoint
