"""Freeze the authorized 60-item optimization QRA candidate set."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

from apps.evaluations.formal_configuration_generation import DOMAIN_SCHEMA_PATHS
from apps.evaluations.rageval_artifacts import (
    ArtifactValidationError,
    sha256_file,
    validate_qra_record,
    write_json_artifact,
)


DEFAULT_CANDIDATE_MANIFEST = Path(
    "docs/evaluation/optimization_dataset/qra/candidates/generation_manifest_v0.1.json"
)
DEFAULT_AUTHORIZATION_PATH = Path(
    "docs/evaluation/optimization_dataset/qra/"
    "optimization_qra_freeze_authorization_v1.0.json"
)
DEFAULT_OUTPUT_ROOT = Path("docs/evaluation/optimization_dataset/qra/frozen")
DEFAULT_MANIFEST_PATH = DEFAULT_OUTPUT_ROOT / "optimization_qra_manifest_v1.0.json"

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
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(f"Cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ArtifactValidationError(f"Expected a JSON object: {path}")
    return payload


def _write_or_verify(path: Path, payload: dict[str, Any]) -> str:
    if path.exists():
        if _read_json_object(path) != payload:
            raise ArtifactValidationError(
                f"Frozen artifact already exists with different content: {path}"
            )
        return "unchanged"
    write_json_artifact(path, payload)
    return "created"


def _schema_errors(payload: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    errors = sorted(
        Draft202012Validator(
            schema,
            format_checker=FormatChecker(),
        ).iter_errors(payload),
        key=lambda error: list(error.absolute_path),
    )
    return [
        f"{'/'.join(str(part) for part in error.absolute_path)}: {error.message}"
        for error in errors
    ]


def _load_declared_records(
    *,
    repo_root: Path,
    declarations: list[dict[str, Any]],
    id_field: str,
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for declaration in declarations:
        path = repo_root / declaration["path"]
        if sha256_file(path) != declaration["file_sha256"]:
            raise ArtifactValidationError(f"Frozen dependency changed: {path}")
        record = _read_json_object(path)
        record_id = str(record[id_field])
        if record_id in records:
            raise ArtifactValidationError(f"Duplicate dependency ID: {record_id}")
        records[record_id] = record
    return records


def _assert_balance(records: list[dict[str, Any]]) -> None:
    if len(records) != 60 or len({record["id"] for record in records}) != 60:
        raise ArtifactValidationError("Optimization QRA set must contain 60 unique items")
    if Counter(record["answerable"] for record in records) != {True: 54, False: 6}:
        raise ArtifactValidationError("Optimization QRA answerability changed")
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


def freeze_optimization_qras(
    *,
    repo_root: Path,
    frozen_at: str,
    candidate_manifest_path: Path = DEFAULT_CANDIDATE_MANIFEST,
    authorization_path: Path = DEFAULT_AUTHORIZATION_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
) -> dict[str, Any]:
    """Freeze authorized optimization candidates without changing their semantics."""

    repo_root = repo_root.resolve()
    paths = [
        candidate_manifest_path,
        authorization_path,
        output_root,
        manifest_path,
    ]
    resolved = [path if path.is_absolute() else repo_root / path for path in paths]
    (
        candidate_manifest_path,
        authorization_path,
        output_root,
        manifest_path,
    ) = resolved
    if manifest_path.parent != output_root:
        raise ArtifactValidationError("Optimization QRA manifest must be in output root")

    candidate_manifest = _read_json_object(candidate_manifest_path)
    candidate_hash = sha256_file(candidate_manifest_path)
    authorization = _read_json_object(authorization_path)
    if (
        candidate_manifest.get("manifest_id")
        != "optimization_qra_generation_manifest_v0.1"
        or candidate_manifest.get("version") != "0.1"
        or candidate_manifest.get("status") != "review_candidate"
        or candidate_manifest.get("qra_count") != 60
    ):
        raise ArtifactValidationError("Expected the 60-item optimization QRA candidate set")
    authorized_target = authorization.get("scope", {}).get("candidate_manifest", {})
    if (
        authorization.get("status") != "authorized"
        or authorized_target.get("path")
        != candidate_manifest_path.relative_to(repo_root).as_posix()
        or authorized_target.get("sha256") != candidate_hash
        or authorization.get("scope", {}).get("qra_count") != 60
    ):
        raise ArtifactValidationError("Optimization QRA freeze authorization is invalid")
    authorization_declaration = {
        "path": authorization_path.relative_to(repo_root).as_posix(),
        "sha256": sha256_file(authorization_path),
    }

    configuration_manifest_path = (
        repo_root / candidate_manifest["configuration_manifest"]["path"]
    )
    document_manifest_path = repo_root / candidate_manifest["document_manifest"]["path"]
    record_schema_path = repo_root / candidate_manifest["record_schema"]["path"]
    for label, path, declaration in (
        (
            "Configuration manifest",
            configuration_manifest_path,
            candidate_manifest["configuration_manifest"],
        ),
        (
            "Document manifest",
            document_manifest_path,
            candidate_manifest["document_manifest"],
        ),
        ("QRA schema", record_schema_path, candidate_manifest["record_schema"]),
    ):
        if sha256_file(path) != declaration["sha256"]:
            raise ArtifactValidationError(f"{label} hash changed: {path}")

    configuration_manifest = _read_json_object(configuration_manifest_path)
    document_manifest = _read_json_object(document_manifest_path)
    record_schema = _read_json_object(record_schema_path)
    if (
        configuration_manifest.get("status") != "frozen"
        or configuration_manifest.get("version") != "1.0"
        or document_manifest.get("status") != "frozen"
    ):
        raise ArtifactValidationError("Optimization QRA dependencies are not frozen")
    configurations = _load_declared_records(
        repo_root=repo_root,
        declarations=configuration_manifest["configurations"],
        id_field="configuration_id",
    )
    documents = _load_declared_records(
        repo_root=repo_root,
        declarations=document_manifest["documents"],
        id_field="document_id",
    )

    declarations: list[dict[str, Any]] = []
    frozen_records: list[dict[str, Any]] = []
    write_states: list[str] = []
    for candidate_declaration in candidate_manifest["qras"]:
        candidate_path = repo_root / candidate_declaration["path"]
        if sha256_file(candidate_path) != candidate_declaration["file_sha256"]:
            raise ArtifactValidationError(f"QRA candidate changed: {candidate_path}")
        candidate = _read_json_object(candidate_path)
        if (
            candidate.get("version") != "0.1"
            or candidate.get("status") != "development_candidate"
            or candidate.get("review_status") != "pending"
            or candidate.get("split") != "test"
        ):
            raise ArtifactValidationError(
                f"Unexpected optimization QRA lifecycle: {candidate_path}"
            )

        frozen = deepcopy(candidate)
        frozen["version"] = "1.0"
        frozen["status"] = "frozen"
        frozen["review_status"] = "single_checked"
        frozen["generation_record"]["reproducibility_note"] = (
            "Immutable optimization-test QRA v1.0 frozen under explicit "
            "project-owner authorization. Independent second review and expert "
            "fact certification are not claimed."
        )
        frozen["generation_record"]["mechanical_validation"] = (
            "passed: JSON Schema, frozen Configuration and Document identity, exact "
            "Reference location, answer alignment, Keypoint coverage, set balance, "
            "candidate uniqueness, and project-owner freeze authorization"
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

        frozen_path = output_root / frozen["domain"] / f"{frozen['id']}_v1.0.json"
        write_states.append(_write_or_verify(frozen_path, frozen))
        freeze_record = {
            "artifact_type": "Optimization RAGEval Question-Reference-Answer item",
            "qra_id": frozen["id"],
            "version": "1.0",
            "status": "frozen",
            "frozen_at": frozen_at,
            "split": "test",
            "review_status": "single_checked",
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
            "freeze_authorization": authorization_declaration,
            "review_chain": {
                "project_owner_freeze_authorization": True,
                "independent_second_human_review_claimed": False,
                "human_line_by_line_review_claimed": False,
                "expert_fact_review_claimed": False,
            },
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
    frozen_manifest = {
        "manifest_id": "optimization_qra_manifest_v1.0",
        "version": "1.0",
        "status": "frozen",
        "frozen_at": frozen_at,
        "split": "test",
        "review_status": "single_checked",
        "double_checked": False,
        "candidate_manifest": {
            "path": candidate_manifest_path.relative_to(repo_root).as_posix(),
            "sha256": candidate_hash,
        },
        "configuration_manifest": candidate_manifest["configuration_manifest"],
        "document_manifest": candidate_manifest["document_manifest"],
        "record_schema": candidate_manifest["record_schema"],
        "freeze_authorization": authorization_declaration,
        "review_limitations": {
            "project_owner_freeze_authorization": True,
            "independent_second_human_review_claimed": False,
            "human_line_by_line_review_claimed": False,
            "expert_fact_review_claimed": False,
        },
        "qra_count": 60,
        "answerable_count": 54,
        "unanswerable_count": 6,
        "qras": declarations,
        "change_policy": (
            "This manifest and every declared QRA are immutable. Any correction "
            "requires a new version."
        ),
    }
    write_states.append(_write_or_verify(manifest_path, frozen_manifest))
    manifest_record_path = manifest_path.with_name(
        f"{manifest_path.stem}_freeze_record.json"
    )
    manifest_freeze_record = {
        "artifact_type": "Optimization test QRA manifest",
        "manifest_id": frozen_manifest["manifest_id"],
        "version": "1.0",
        "status": "frozen",
        "frozen_at": frozen_at,
        "file": manifest_path.relative_to(repo_root).as_posix(),
        "file_sha256": sha256_file(manifest_path),
        "candidate_manifest": frozen_manifest["candidate_manifest"],
        "freeze_authorization": authorization_declaration,
        "qra_count": 60,
        "review_status": "single_checked",
        "independent_second_human_review_claimed": False,
        "change_policy": "Do not edit manifest v1.0 in place.",
    }
    write_states.append(_write_or_verify(manifest_record_path, manifest_freeze_record))

    return {
        "qra_count": 60,
        "manifest_path": manifest_path.relative_to(repo_root).as_posix(),
        "manifest_freeze_record_path": manifest_record_path.relative_to(
            repo_root
        ).as_posix(),
        "created_count": write_states.count("created"),
        "unchanged_count": write_states.count("unchanged"),
    }
