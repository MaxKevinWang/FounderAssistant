"""Freeze the audited 60-item formal QRA set as immutable version 1.0."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

from apps.evaluations.formal_configuration_generation import DOMAIN_SCHEMA_PATHS
from apps.evaluations.formal_qra_semantic_review import (
    AGENT_REVIEW_PATH,
    CANDIDATE_MANIFEST_PATH,
    OWNER_CONFIRMATION_PATH,
    REVISION_MANIFEST_PATH,
)
from apps.evaluations.rageval_artifacts import (
    ArtifactValidationError,
    sha256_file,
    validate_qra_record,
    write_json_artifact,
)


OUTPUT_ROOT = Path("docs/evaluation/formal_dataset/qra/formal_test")
FROZEN_MANIFEST_PATH = OUTPUT_ROOT / "formal_test_qra_manifest_v1.0.json"
FROZEN_MANIFEST_RECORD_PATH = (
    OUTPUT_ROOT / "formal_test_qra_manifest_v1.0_freeze_record.json"
)

EXPECTED_TYPE_COUNTS = {
    "Factual": 5,
    "Summarization": 3,
    "Multi-hop Reasoning": 3,
    "Information Integration": 3,
    "Numerical Comparison": 2,
    "Temporal Sequence": 2,
    "Unanswerable": 2,
}


def _read_json_object(path: Path) -> dict[str, Any]:
    """Read one UTF-8 JSON object."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(f"Cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ArtifactValidationError(f"Expected a JSON object: {path}")
    return payload


def _write_or_verify(path: Path, payload: dict[str, Any]) -> str:
    """Create an immutable artifact or verify an identical existing artifact."""

    if path.exists():
        if _read_json_object(path) != payload:
            raise ArtifactValidationError(
                f"Frozen artifact already exists with different content: {path}"
            )
        return "unchanged"
    write_json_artifact(path, payload)
    return "created"


def _schema_errors(
    payload: dict[str, Any],
    record_schema: dict[str, Any],
) -> list[str]:
    """Return compact JSON Schema errors."""

    errors = sorted(
        Draft202012Validator(
            record_schema,
            format_checker=FormatChecker(),
        ).iter_errors(payload),
        key=lambda error: list(error.absolute_path),
    )
    return [
        f"{'/'.join(str(part) for part in error.absolute_path)}: {error.message}"
        for error in errors
    ]


def _load_hash_bound_records(
    *,
    repo_root: Path,
    declarations: list[dict[str, Any]],
    id_field: str,
) -> dict[str, dict[str, Any]]:
    """Load records whose file hashes are declared by a frozen manifest."""

    records: dict[str, dict[str, Any]] = {}
    for declaration in declarations:
        path = repo_root / str(declaration["path"])
        if sha256_file(path) != declaration["file_sha256"]:
            raise ArtifactValidationError(f"Frozen dependency changed: {path}")
        record = _read_json_object(path)
        record_id = str(record[id_field])
        if record_id in records:
            raise ArtifactValidationError(f"Duplicate dependency ID: {record_id}")
        records[record_id] = record
    return records


def _assert_balance(records: list[dict[str, Any]]) -> None:
    """Protect the frozen benchmark's domain and question-type allocation."""

    if len(records) != 60 or len({record["id"] for record in records}) != 60:
        raise ArtifactValidationError("Frozen QRA set must contain 60 unique items")
    if Counter(record["answerable"] for record in records) != {True: 54, False: 6}:
        raise ArtifactValidationError("Frozen QRA answerability allocation changed")
    for domain in DOMAIN_SCHEMA_PATHS:
        domain_records = [record for record in records if record["domain"] == domain]
        if len(domain_records) != 20:
            raise ArtifactValidationError(f"Domain {domain} must contain 20 QRA items")
        if Counter(record["question_type"] for record in domain_records) != (
            EXPECTED_TYPE_COUNTS
        ):
            raise ArtifactValidationError(
                f"Question-type allocation changed for {domain}"
            )


def freeze_formal_qras(
    *,
    repo_root: Path,
    frozen_at: str = "2026-08-03",
) -> dict[str, Any]:
    """Freeze audited candidate v0.4 as the immutable formal-test QRA v1.0 set."""

    repo_root = repo_root.resolve()
    candidate_manifest_path = repo_root / CANDIDATE_MANIFEST_PATH
    agent_review_path = repo_root / AGENT_REVIEW_PATH
    owner_confirmation_path = repo_root / OWNER_CONFIRMATION_PATH
    revision_manifest_path = repo_root / REVISION_MANIFEST_PATH
    candidate_manifest = _read_json_object(candidate_manifest_path)
    agent_review = _read_json_object(agent_review_path)
    owner_confirmation = _read_json_object(owner_confirmation_path)
    revision_manifest = _read_json_object(revision_manifest_path)

    candidate_hash = sha256_file(candidate_manifest_path)
    if (
        candidate_manifest.get("status") != "review_candidate"
        or candidate_manifest.get("version") != "0.4"
        or candidate_manifest.get("qra_count") != 60
    ):
        raise ArtifactValidationError("Expected audited formal QRA candidate v0.4")
    if (
        agent_review.get("status") != "agent_semantic_review_passed"
        or agent_review.get("qra_count") != 60
        or agent_review.get("unresolved_blocker_count") != 0
        or agent_review.get("human_review_claimed") is not False
        or agent_review.get("agent_review_is_human_review") is not False
        or agent_review.get("candidate_manifest", {}).get("sha256")
        != candidate_hash
    ):
        raise ArtifactValidationError("Agent semantic review does not pass freeze gate")
    if (
        owner_confirmation.get("status")
        != "freeze_authorized_after_agent_review"
        or owner_confirmation.get("authorized_by_role") != "project_owner"
        or owner_confirmation.get("freeze_condition_satisfied") is not True
        or owner_confirmation.get("independent_second_human_review_claimed")
        is not False
        or owner_confirmation.get("freeze_target_after_agent_corrections", {}).get(
            "sha256"
        )
        != candidate_hash
        or owner_confirmation.get("agent_review", {}).get("sha256")
        != sha256_file(agent_review_path)
    ):
        raise ArtifactValidationError("Project-owner freeze confirmation is invalid")
    if revision_manifest.get("revised_qra_count") != 17:
        raise ArtifactValidationError("Expected 17 revised QRA dependencies")

    configuration_manifest_path = repo_root / str(
        candidate_manifest["configuration_manifest"]["path"]
    )
    document_manifest_path = repo_root / str(
        candidate_manifest["document_manifest"]["path"]
    )
    record_schema_path = repo_root / str(candidate_manifest["record_schema"]["path"])
    if (
        sha256_file(configuration_manifest_path)
        != candidate_manifest["configuration_manifest"]["sha256"]
        or sha256_file(document_manifest_path)
        != candidate_manifest["document_manifest"]["sha256"]
        or sha256_file(record_schema_path)
        != candidate_manifest["record_schema"]["sha256"]
    ):
        raise ArtifactValidationError("A formal QRA dependency hash changed")
    configuration_manifest = _read_json_object(configuration_manifest_path)
    document_manifest = _read_json_object(document_manifest_path)
    record_schema = _read_json_object(record_schema_path)
    if (
        configuration_manifest.get("status") != "frozen"
        or configuration_manifest.get("version") != "1.3"
        or document_manifest.get("status") != "frozen"
        or document_manifest.get("version") != "1.2"
    ):
        raise ArtifactValidationError("QRA dependencies are not the expected frozen set")
    configurations = _load_hash_bound_records(
        repo_root=repo_root,
        declarations=configuration_manifest["configurations"],
        id_field="configuration_id",
    )
    documents = _load_hash_bound_records(
        repo_root=repo_root,
        declarations=document_manifest["documents"],
        id_field="document_id",
    )

    revised_configuration_ids = {
        str(item["configuration_id"]) for item in revision_manifest["revisions"]
    }
    write_states: list[str] = []
    frozen_records: list[dict[str, Any]] = []
    declarations: list[dict[str, Any]] = []
    for candidate_declaration in candidate_manifest["qras"]:
        candidate_path = repo_root / str(candidate_declaration["path"])
        if sha256_file(candidate_path) != candidate_declaration["file_sha256"]:
            raise ArtifactValidationError(f"QRA candidate changed: {candidate_path}")
        candidate = _read_json_object(candidate_path)
        if (
            candidate.get("version") != "0.4"
            or candidate.get("status") != "development_candidate"
            or candidate.get("split") != "test"
        ):
            raise ArtifactValidationError(
                f"Unexpected candidate lifecycle: {candidate_path}"
            )
        frozen = deepcopy(candidate)
        frozen["version"] = "1.0"
        frozen["status"] = "frozen"
        frozen["review_status"] = "single_checked"
        frozen["generation_record"]["reproducibility_note"] = (
            "Immutable formal-test QRA v1.0 derived from audited candidate v0.4. "
            "A second independent human review is not claimed."
        )
        frozen["generation_record"]["mechanical_validation"] = (
            "passed: JSON Schema, frozen Configuration and Document identity, "
            "exact Reference location, answer alignment, Keypoint coverage, set "
            "balance, and pre-freeze semantic audit"
        )
        errors = _schema_errors(frozen, record_schema)
        if errors:
            raise ArtifactValidationError(
                f"Invalid frozen QRA {frozen['id']}: {'; '.join(errors[:5])}"
            )
        validate_qra_record(
            qra=frozen,
            configuration=configurations[frozen["configuration_id"]],
            documents=documents,
        )

        frozen_path = (
            repo_root
            / OUTPUT_ROOT
            / str(frozen["domain"])
            / f"{frozen['id']}_v1.0.json"
        )
        write_states.append(_write_or_verify(frozen_path, frozen))
        resolution = (
            "primary_law_revised"
            if frozen["configuration_id"] in revised_configuration_ids
            else "semantic_carry_forward"
        )
        freeze_record = {
            "artifact_type": "Formal RAGEval Question-Reference-Answer item",
            "qra_id": frozen["id"],
            "version": "1.0",
            "status": "frozen",
            "frozen_at": frozen_at,
            "split": "test",
            "review_status": "single_checked",
            "resolution": resolution,
            "artifacts": {
                "frozen_qra": {
                    "path": frozen_path.relative_to(repo_root).as_posix(),
                    "sha256": sha256_file(frozen_path),
                },
                "candidate_qra": {
                    "path": candidate_declaration["path"],
                    "sha256": candidate_declaration["file_sha256"],
                },
                "record_schema": candidate_manifest["record_schema"],
                "configuration": {
                    "path": frozen["configuration_path"],
                    "sha256": frozen["configuration_sha256"],
                },
            },
            "review_chain": {
                "project_owner_confirmation": {
                    "path": OWNER_CONFIRMATION_PATH.as_posix(),
                    "sha256": sha256_file(owner_confirmation_path),
                    "manual_review_reported": True,
                },
                "agent_semantic_review": {
                    "path": AGENT_REVIEW_PATH.as_posix(),
                    "sha256": sha256_file(agent_review_path),
                    "result": "pass",
                    "unresolved_blocker_count": 0,
                },
                "independent_second_human_review_claimed": False,
                "post_correction_human_line_by_line_review_claimed": False,
                "legal_expert_review_claimed": False,
            },
            "lifecycle_note": (
                "single_checked records one project-owner manual review in the "
                "artifact lineage plus explicit conditional freeze authorization. "
                "The agent audit is not counted as a second human review."
            ),
            "change_policy": (
                "Do not edit this QRA or freeze record in place. Publish a new "
                "version for any later content or review-status change."
            ),
        }
        freeze_record_path = frozen_path.with_name(
            f"{frozen_path.stem}_freeze_record.json"
        )
        write_states.append(_write_or_verify(freeze_record_path, freeze_record))
        declarations.append(
            {
                "qra_id": frozen["id"],
                "version": "1.0",
                "resolution": resolution,
                "domain": frozen["domain"],
                "question_type": frozen["question_type"],
                "answerable": frozen["answerable"],
                "review_status": frozen["review_status"],
                "path": frozen_path.relative_to(repo_root).as_posix(),
                "file_sha256": sha256_file(frozen_path),
                "freeze_record_path": freeze_record_path.relative_to(
                    repo_root
                ).as_posix(),
                "freeze_record_sha256": sha256_file(freeze_record_path),
                "configuration_id": frozen["configuration_id"],
                "configuration_version": frozen["configuration_version"],
                "keypoint_count": len(frozen["keypoints"]),
                "reference_count": len(frozen["references"]),
            }
        )
        frozen_records.append(frozen)

    _assert_balance(frozen_records)
    if Counter(item["resolution"] for item in declarations) != {
        "primary_law_revised": 17,
        "semantic_carry_forward": 43,
    }:
        raise ArtifactValidationError("Frozen QRA revision allocation changed")

    frozen_manifest = {
        "manifest_id": "formal_test_qra_manifest_v1.0",
        "version": "1.0",
        "status": "frozen",
        "frozen_at": frozen_at,
        "split": "test",
        "review_status": "single_checked",
        "double_checked": False,
        "candidate_manifest": {
            "path": CANDIDATE_MANIFEST_PATH.as_posix(),
            "sha256": candidate_hash,
        },
        "configuration_manifest": candidate_manifest["configuration_manifest"],
        "document_manifest": candidate_manifest["document_manifest"],
        "record_schema": candidate_manifest["record_schema"],
        "agent_semantic_review": {
            "path": AGENT_REVIEW_PATH.as_posix(),
            "sha256": sha256_file(agent_review_path),
            "status": agent_review["status"],
        },
        "project_owner_confirmation": {
            "path": OWNER_CONFIRMATION_PATH.as_posix(),
            "sha256": sha256_file(owner_confirmation_path),
            "status": owner_confirmation["status"],
        },
        "review_limitations": {
            "documented_human_reviewers": 1,
            "independent_second_human_review_claimed": False,
            "post_correction_human_line_by_line_review_claimed": False,
            "legal_expert_review_claimed": False,
        },
        "qra_count": 60,
        "primary_law_revised_count": 17,
        "semantic_carry_forward_count": 43,
        "answerable_count": 54,
        "unanswerable_count": 6,
        "qras": declarations,
        "change_policy": (
            "This manifest and every declared QRA are immutable. Any later "
            "correction or review-status upgrade requires a new version."
        ),
    }
    frozen_manifest_path = repo_root / FROZEN_MANIFEST_PATH
    write_states.append(_write_or_verify(frozen_manifest_path, frozen_manifest))
    manifest_freeze_record = {
        "artifact_type": "Formal test QRA manifest",
        "manifest_id": frozen_manifest["manifest_id"],
        "version": "1.0",
        "status": "frozen",
        "frozen_at": frozen_at,
        "file": FROZEN_MANIFEST_PATH.as_posix(),
        "file_sha256": sha256_file(frozen_manifest_path),
        "candidate_manifest": frozen_manifest["candidate_manifest"],
        "agent_semantic_review": frozen_manifest["agent_semantic_review"],
        "project_owner_confirmation": frozen_manifest[
            "project_owner_confirmation"
        ],
        "qra_count": 60,
        "review_status": "single_checked",
        "independent_second_human_review_claimed": False,
        "change_policy": "Do not edit manifest v1.0 in place.",
    }
    manifest_record_path = repo_root / FROZEN_MANIFEST_RECORD_PATH
    write_states.append(_write_or_verify(manifest_record_path, manifest_freeze_record))

    return {
        "qra_count": 60,
        "primary_law_revised_count": 17,
        "semantic_carry_forward_count": 43,
        "manifest_path": FROZEN_MANIFEST_PATH.as_posix(),
        "manifest_freeze_record_path": FROZEN_MANIFEST_RECORD_PATH.as_posix(),
        "created_count": write_states.count("created"),
        "unchanged_count": write_states.count("unchanged"),
    }
