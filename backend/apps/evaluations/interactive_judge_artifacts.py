"""Build and validate reproducible artifacts for an interactive blind judge.

The interactive judge is intentionally separate from ``openai_judge.py``.  It
does not call an API.  Instead, it prepares condition-blinded review packets,
validates labels supplied by a human or interactive assistant, and computes
the same keypoint metrics used by the automated judge pipeline.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import statistics
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable


ALLOWED_LABELS = {"covered", "contradicted", "neither"}


class InteractiveJudgeArtifactError(RuntimeError):
    """Raised when source runs or supplied judge decisions are inconsistent."""


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise InteractiveJudgeArtifactError(f"Expected a JSON object: {path}")
    return payload


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def prepare_blinded_review(
    source_paths: Iterable[Path],
    *,
    output_dir: Path,
    experiment_id: str,
    judge_run_id: str,
    seed: int,
) -> dict[str, Path]:
    """Merge compatible generation runs and create separately mapped packets."""

    resolved_paths = [path.resolve() for path in source_paths]
    if not resolved_paths:
        raise InteractiveJudgeArtifactError("At least one source run is required")

    runs = [(path, _load_json(path)) for path in resolved_paths]
    dataset_hashes = {str(run.get("dataset", {}).get("manifest_sha256", "")) for _, run in runs}
    if len(dataset_hashes) != 1 or "" in dataset_hashes:
        raise InteractiveJudgeArtifactError(
            f"Source runs do not share one dataset manifest: {sorted(dataset_hashes)}"
        )

    dataset = runs[0][1]["dataset"]
    qra_by_id = {str(item["id"]): item for item in dataset["items"]}
    conditions: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    source_manifest: list[dict[str, Any]] = []
    for source_path, run in runs:
        source_manifest.append(
            {
                "path": source_path.as_posix(),
                "sha256": _file_hash(source_path),
                "run_id": str(run.get("run_id", "")),
                "result_count": len(run.get("results", [])),
            }
        )
        for result in run.get("results", []):
            generation = result.get("generation", {})
            if generation.get("status") != "success":
                raise InteractiveJudgeArtifactError(
                    "Interactive review requires successful generations only: "
                    f"{result.get('model')}/{result.get('qra_id')}/{result.get('mode')}"
                )
            key = (
                str(result.get("model", "")),
                str(result.get("qra_id", "")),
                str(result.get("mode", "")),
            )
            if key in seen:
                raise InteractiveJudgeArtifactError(f"Duplicate condition: {key}")
            seen.add(key)
            qra = qra_by_id.get(key[1])
            if qra is None:
                raise InteractiveJudgeArtifactError(f"Unknown QRA: {key[1]}")
            conditions.append(
                {
                    "model": key[0],
                    "model_type": str(result.get("model_type", "")),
                    "qra_id": key[1],
                    "mode": key[2],
                    "domain": str(result.get("domain", qra.get("domain", ""))),
                    "question_type": str(
                        result.get("question_type", qra.get("question_type", ""))
                    ),
                    "source_sequence": int(result.get("sequence", 0)),
                    "candidate_answer": str(generation.get("response", "")).strip(),
                }
            )

    expected = len(qra_by_id) * len({item["model"] for item in conditions}) * 2
    if len(conditions) != expected:
        raise InteractiveJudgeArtifactError(
            f"Expected {expected} paired conditions, found {len(conditions)}"
        )

    random.Random(seed).shuffle(conditions)
    packets: list[dict[str, Any]] = []
    mappings: list[dict[str, Any]] = []
    for position, condition in enumerate(conditions, start=1):
        candidate_id = f"C{position:04d}"
        qra = qra_by_id[condition["qra_id"]]
        packets.append(
            {
                "candidate_id": candidate_id,
                "qra_id": condition["qra_id"],
                "domain": condition["domain"],
                "question_type": condition["question_type"],
                "language": str(qra["language"]),
                "question": str(qra["question"]),
                "answerable": bool(qra["answerable"]),
                "reference_answer": str(qra["reference_answer"]),
                "keypoints": [
                    {"id": str(item["id"]), "text": str(item["text"])}
                    for item in qra["keypoints"]
                ],
                "reference_evidence": [
                    {
                        "reference_id": str(item["reference_id"]),
                        "quote": str(item["quote"]),
                    }
                    for item in qra["references"]
                ],
                "candidate_answer": condition["candidate_answer"],
            }
        )
        mappings.append(
            {
                "candidate_id": candidate_id,
                **{key: value for key, value in condition.items() if key != "candidate_answer"},
                "answer_sha256": _canonical_hash(condition["candidate_answer"]),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    packet_path = output_dir / f"{judge_run_id}_blinded_packets.json"
    mapping_path = output_dir / f"{judge_run_id}_candidate_mapping.json"
    source_path = output_dir.parents[2] / f"{experiment_id}_source_manifest.json"
    _write_json(
        packet_path,
        {
            "judge_run_id": judge_run_id,
            "status": "prepared",
            "judge_type": "interactive_condition_blinded",
            "shuffle_seed": seed,
            "dataset_manifest_sha256": next(iter(dataset_hashes)),
            "packet_count": len(packets),
            "packets": packets,
        },
    )
    _write_json(
        mapping_path,
        {
            "judge_run_id": judge_run_id,
            "status": "sealed_until_scoring_complete",
            "candidate_count": len(mappings),
            "mapping": mappings,
        },
    )
    _write_json(
        source_path,
        {
            "experiment_id": experiment_id,
            "created_at": datetime.now(UTC).isoformat(),
            "dataset_manifest_sha256": next(iter(dataset_hashes)),
            "qra_count": len(qra_by_id),
            "model_count": len({item["model"] for item in conditions}),
            "answer_count": len(conditions),
            "source_runs": source_manifest,
            "blinding": {
                "seed": seed,
                "packet_sha256": _file_hash(packet_path),
                "mapping_sha256": _file_hash(mapping_path),
            },
        },
    )
    return {"packets": packet_path, "mapping": mapping_path, "manifest": source_path}


def score_interactive_review(
    *,
    packet_path: Path,
    mapping_path: Path,
    decision_path: Path,
    output_dir: Path,
    judge_run_id: str,
) -> dict[str, Path]:
    """Validate manual labels, restore conditions, and aggregate paired scores."""

    packet_payload = _load_json(packet_path)
    mapping_payload = _load_json(mapping_path)
    decision_payload = _load_json(decision_path)
    packets = {
        str(item["candidate_id"]): item for item in packet_payload["packets"]
    }
    mappings = {
        str(item["candidate_id"]): item for item in mapping_payload["mapping"]
    }
    decisions = {
        str(item["candidate_id"]): item for item in decision_payload["decisions"]
    }
    if set(packets) != set(mappings) or set(packets) != set(decisions):
        raise InteractiveJudgeArtifactError(
            "Packet, mapping, and decision candidate IDs must match exactly"
        )

    scored: list[dict[str, Any]] = []
    for candidate_id in sorted(packets):
        packet = packets[candidate_id]
        mapping = mappings[candidate_id]
        decision = decisions[candidate_id]
        expected_ids = [str(item["id"]) for item in packet["keypoints"]]
        labels = decision.get("keypoint_labels", [])
        by_id = {str(item["keypoint_id"]): item for item in labels}
        if set(by_id) != set(expected_ids) or len(labels) != len(expected_ids):
            raise InteractiveJudgeArtifactError(
                f"Invalid keypoint IDs for {candidate_id}: {sorted(by_id)}"
            )
        ordered = []
        for keypoint in packet["keypoints"]:
            item = by_id[str(keypoint["id"])]
            label = str(item.get("label", ""))
            if label not in ALLOWED_LABELS:
                raise InteractiveJudgeArtifactError(
                    f"Invalid label for {candidate_id}/{keypoint['id']}: {label}"
                )
            ordered.append(
                {
                    "keypoint_id": str(keypoint["id"]),
                    "keypoint_text": str(keypoint["text"]),
                    "label": label,
                    "judge_note": str(item.get("judge_note", "")),
                }
            )
        counts = {
            label: sum(item["label"] == label for item in ordered)
            for label in ALLOWED_LABELS
        }
        total = len(ordered)
        unsupported_review_status = str(
            decision.get("unsupported_claim_review_status", "not_assessed")
        )
        if unsupported_review_status not in {"assessed", "not_assessed"}:
            raise InteractiveJudgeArtifactError(
                f"Invalid unsupported-claim review status for {candidate_id}: "
                f"{unsupported_review_status}"
            )
        unsupported_claim_count = (
            int(decision.get("unsupported_claim_count", 0))
            if unsupported_review_status == "assessed"
            else None
        )
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
                "unsupported_claim_review_status": unsupported_review_status,
                "unsupported_claim_count": unsupported_claim_count,
                "keypoint_labels": ordered,
                "judge_note": str(decision.get("judge_note", "")),
            }
        )

    def mean(values: list[float]) -> float:
        return round(statistics.mean(values), 4) if values else 0.0

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in scored:
        groups[(item["model"], item["model_type"], item["mode"])].append(item)
    aggregate = []
    for (model, model_type, mode), items in sorted(groups.items()):
        reviewed_claim_items = [
            item
            for item in items
            if item["unsupported_claim_review_status"] == "assessed"
        ]
        aggregate.append(
            {
                "model": model,
                "model_type": model_type,
                "mode": mode,
                "answer_count": len(items),
                "mean_completeness": mean([item["completeness"] for item in items]),
                "mean_hallucination": mean([item["hallucination"] for item in items]),
                "mean_irrelevancy": mean([item["irrelevancy"] for item in items]),
                "unsupported_claim_answers_reviewed": len(reviewed_claim_items),
                "unsupported_claim_count": (
                    sum(item["unsupported_claim_count"] for item in reviewed_claim_items)
                    if reviewed_claim_items
                    else None
                ),
            }
        )

    pairs = []
    paired: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for item in scored:
        paired[(item["model"], item["qra_id"])][item["mode"]] = item
    for (model, qra_id), modes in sorted(paired.items()):
        if set(modes) != {"baseline", "rag"}:
            raise InteractiveJudgeArtifactError(f"Incomplete pair: {model}/{qra_id}")
        baseline, rag = modes["baseline"], modes["rag"]
        pairs.append(
            {
                "model": model,
                "model_type": rag["model_type"],
                "qra_id": qra_id,
                "domain": rag["domain"],
                "baseline_completeness": baseline["completeness"],
                "rag_completeness": rag["completeness"],
                "completeness_lift": round(
                    rag["completeness"] - baseline["completeness"], 4
                ),
                "hallucination_lift": round(
                    rag["hallucination"] - baseline["hallucination"], 4
                ),
                "irrelevancy_lift": round(
                    rag["irrelevancy"] - baseline["irrelevancy"], 4
                ),
            }
        )

    score_path = output_dir / f"{judge_run_id}_scores.json"
    csv_path = output_dir / f"{judge_run_id}_scores.csv"
    _write_json(
        score_path,
        {
            "judge_run_id": judge_run_id,
            "status": "completed",
            "evaluation_status": "exploratory_unvalidated_nonformal",
            "created_at": datetime.now(UTC).isoformat(),
            "summary": aggregate,
            "paired_results": pairs,
            "results": scored,
        },
    )
    with csv_path.open("w", encoding="utf-8", newline="") as target:
        fields = [
            "candidate_id",
            "qra_id",
            "domain",
            "question_type",
            "model",
            "model_type",
            "mode",
            "covered",
            "contradicted",
            "neither",
            "completeness",
            "hallucination",
            "irrelevancy",
            "unsupported_claim_review_status",
            "unsupported_claim_count",
        ]
        writer = csv.DictWriter(target, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(scored)
    return {"scores": score_path, "csv": csv_path}
