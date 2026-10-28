"""Freeze reviewed formal Configuration C candidates as immutable v1.0 files."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

from apps.evaluations.corpus_paths import resolve_corpus_path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

from apps.evaluations.formal_configuration_generation import DOMAIN_SCHEMA_PATHS
from apps.evaluations.rageval_artifacts import (
    ArtifactValidationError,
    sha256_file,
    validate_configuration_record,
    validate_document_record,
    write_json_artifact,
)


FORMAL_CONFIGURATION_OUTPUT_DIR = (
    "docs/evaluation/formal_dataset/configurations/formal_test"
)
FORMAL_CONFIGURATION_MANIFEST_PATH = (
    f"{FORMAL_CONFIGURATION_OUTPUT_DIR}/"
    "formal_test_configuration_manifest_v1.0.json"
)
DEFAULT_CONFIGURATION_MANIFEST_ID = "formal_test_configuration_manifest_v1.0"

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
    """Return compact JSON Schema errors for one Configuration."""

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


def _assert_expected_balance(configurations: list[dict[str, Any]]) -> None:
    """Protect the approved 60-item allocation while freezing it."""

    if len(configurations) != 60:
        raise ArtifactValidationError(
            f"Formal Configuration set must contain 60 items, got {len(configurations)}"
        )
    identifiers = [item["configuration_id"] for item in configurations]
    if len(set(identifiers)) != len(identifiers):
        raise ArtifactValidationError("Formal Configuration IDs are not unique")
    if Counter(item["answerability"] for item in configurations) != {
        "answerable": 54,
        "unanswerable": 6,
    }:
        raise ArtifactValidationError("Formal answerability allocation has changed")

    for domain in DOMAIN_SCHEMA_PATHS:
        domain_items = [item for item in configurations if item["domain"] == domain]
        if len(domain_items) != 20:
            raise ArtifactValidationError(
                f"Domain {domain} must contain 20 Configurations"
            )
        if Counter(
            item["target_question_type"] for item in domain_items
        ) != EXPECTED_TYPE_COUNTS:
            raise ArtifactValidationError(
                f"Question-type allocation changed for {domain}"
            )


def freeze_formal_configurations(
    *,
    repo_root: Path,
    candidate_manifest_path: Path,
    frozen_at: str,
    output_dir: Path = Path(FORMAL_CONFIGURATION_OUTPUT_DIR),
    manifest_path: Path = Path(FORMAL_CONFIGURATION_MANIFEST_PATH),
    manifest_id: str = DEFAULT_CONFIGURATION_MANIFEST_ID,
    dataset_label: str = "Formal test",
    authorization_path: Path | None = None,
) -> dict[str, Any]:
    """Freeze the approved 60 Configuration candidates and write a manifest."""

    repo_root = repo_root.resolve()
    if not candidate_manifest_path.is_absolute():
        candidate_manifest_path = repo_root / candidate_manifest_path
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    if not manifest_path.is_absolute():
        manifest_path = repo_root / manifest_path
    if manifest_path.parent != output_dir:
        raise ArtifactValidationError(
            "Frozen Configuration manifest must be inside the configured output "
            f"directory: {manifest_path}"
        )

    authorization: dict[str, str] | None = None
    if authorization_path is not None:
        if not authorization_path.is_absolute():
            authorization_path = repo_root / authorization_path
        authorization_payload = _read_json_object(authorization_path)
        if authorization_payload.get("status") != "authorized":
            raise ArtifactValidationError(
                f"Expected an authorized freeze record: {authorization_path}"
            )
        authorization = {
            "path": authorization_path.relative_to(repo_root).as_posix(),
            "sha256": sha256_file(authorization_path),
        }
    candidate_manifest = _read_json_object(candidate_manifest_path)
    if (
        candidate_manifest.get("status") != "review_candidate"
        or candidate_manifest.get("configuration_count") != 60
    ):
        raise ArtifactValidationError(
            f"Expected a 60-item review-candidate manifest: {candidate_manifest_path}"
        )

    record_schema_path = repo_root / candidate_manifest["record_schema"]["path"]
    if sha256_file(record_schema_path) != candidate_manifest["record_schema"]["sha256"]:
        raise ArtifactValidationError("Configuration record schema hash mismatch")
    record_schema = _read_json_object(record_schema_path)

    document_manifest_path = (
        repo_root / candidate_manifest["document_manifest"]["path"]
    )
    if (
        sha256_file(document_manifest_path)
        != candidate_manifest["document_manifest"]["sha256"]
    ):
        raise ArtifactValidationError("Frozen Document manifest hash mismatch")
    document_manifest = _read_json_object(document_manifest_path)
    if document_manifest.get("status") != "frozen":
        raise ArtifactValidationError("Formal Document manifest is not frozen")

    documents: dict[str, dict[str, Any]] = {}
    document_declarations: dict[str, dict[str, Any]] = {}
    for declaration in document_manifest["documents"]:
        document_path = repo_root / declaration["path"]
        if sha256_file(document_path) != declaration["file_sha256"]:
            raise ArtifactValidationError(
                f"Frozen Document hash mismatch: {document_path}"
            )
        document = _read_json_object(document_path)
        validate_document_record(
            document=document,
            source_path=resolve_corpus_path(repo_root, document["source"]["path"]),
        )
        documents[document["document_id"]] = document
        document_declarations[document["document_id"]] = declaration

    domain_schemas = {
        domain: _read_json_object(repo_root / path)
        for domain, path in DOMAIN_SCHEMA_PATHS.items()
    }
    domain_schema_hashes = {
        domain: sha256_file(repo_root / path)
        for domain, path in DOMAIN_SCHEMA_PATHS.items()
    }

    candidate_configurations: list[dict[str, Any]] = []
    for declaration in candidate_manifest["configurations"]:
        candidate_path = repo_root / declaration["path"]
        if sha256_file(candidate_path) != declaration["file_sha256"]:
            raise ArtifactValidationError(
                f"Configuration candidate hash mismatch: {candidate_path}"
            )
        candidate = _read_json_object(candidate_path)
        if (
            candidate.get("version") != "0.1"
            or candidate.get("status") != "development_draft"
            or candidate.get("generation_record", {}).get("human_review_status")
            != "pending"
        ):
            raise ArtifactValidationError(
                f"Unexpected Configuration candidate lifecycle: {candidate_path}"
            )
        errors = _schema_errors(candidate, record_schema)
        if errors:
            raise ArtifactValidationError(
                f"Invalid Configuration candidate {candidate_path}: {'; '.join(errors[:5])}"
            )
        validate_configuration_record(
            configuration=candidate,
            schema=domain_schemas[candidate["domain"]],
            documents=documents,
        )
        candidate_configurations.append(candidate)

    _assert_expected_balance(candidate_configurations)

    frozen_declarations: list[dict[str, Any]] = []
    write_states: list[str] = []
    output_root = output_dir
    for declaration, candidate in zip(
        candidate_manifest["configurations"],
        candidate_configurations,
        strict=True,
    ):
        configuration_id = candidate["configuration_id"]
        frozen_configuration = deepcopy(candidate)
        frozen_configuration["version"] = "1.0"
        frozen_configuration["status"] = "frozen"
        frozen_configuration["supersedes"] = f"{configuration_id} v0.1"
        frozen_configuration["generation_record"]["human_review_status"] = "approved"

        errors = _schema_errors(frozen_configuration, record_schema)
        if errors:
            raise ArtifactValidationError(
                f"Invalid frozen Configuration {configuration_id}: "
                f"{'; '.join(errors[:5])}"
            )
        validate_configuration_record(
            configuration=frozen_configuration,
            schema=domain_schemas[frozen_configuration["domain"]],
            documents=documents,
        )

        frozen_path = (
            output_root
            / frozen_configuration["domain"]
            / f"{configuration_id}_v1.0.json"
        )
        write_states.append(_write_or_verify(frozen_path, frozen_configuration))

        source_document_records = []
        for source in frozen_configuration["source_documents"]:
            document_id = source["document_id"]
            document_declaration = document_declarations[document_id]
            source_document_records.append(
                {
                    "document_id": document_id,
                    "path": document_declaration["path"],
                    "file_sha256": document_declaration["file_sha256"],
                    "content_sha256": document_declaration["content_sha256"],
                    "freeze_record_path": document_declaration["freeze_record_path"],
                    "freeze_record_sha256": document_declaration[
                        "freeze_record_sha256"
                    ],
                }
            )

        freeze_record = {
            "artifact_type": "RAGEval Configuration C",
            "configuration_id": configuration_id,
            "version": "1.0",
            "status": "frozen",
            "frozen_at": frozen_at,
            "file": frozen_path.relative_to(repo_root).as_posix(),
            "file_sha256": sha256_file(frozen_path),
            "candidate_file": declaration["path"],
            "candidate_file_sha256": declaration["file_sha256"],
            "record_schema": {
                "path": record_schema_path.relative_to(repo_root).as_posix(),
                "sha256": sha256_file(record_schema_path),
            },
            "domain_schema": {
                "path": DOMAIN_SCHEMA_PATHS[frozen_configuration["domain"]],
                "sha256": domain_schema_hashes[frozen_configuration["domain"]],
            },
            "source_documents": source_document_records,
            "approval": {
                "basis": "explicit_project_owner_freeze_instruction",
                "approved_at": frozen_at,
                "approved_draft": f"{configuration_id} v0.1",
                "scope": (
                    "configured values, source scope, provenance, question type, "
                    "and answerability"
                ),
            },
            "change_policy": (
                "Do not edit this frozen Configuration or freeze record in place; "
                "create and review a new version for every later change."
            ),
        }
        if authorization is not None:
            freeze_record["freeze_authorization"] = authorization
        freeze_record_path = frozen_path.with_name(
            f"{frozen_path.stem}_freeze_record.json"
        )
        write_states.append(_write_or_verify(freeze_record_path, freeze_record))
        frozen_declarations.append(
            {
                "configuration_id": configuration_id,
                "domain": frozen_configuration["domain"],
                "target_question_type": frozen_configuration[
                    "target_question_type"
                ],
                "answerability": frozen_configuration["answerability"],
                "path": frozen_path.relative_to(repo_root).as_posix(),
                "file_sha256": sha256_file(frozen_path),
                "freeze_record_path": freeze_record_path.relative_to(
                    repo_root
                ).as_posix(),
                "freeze_record_sha256": sha256_file(freeze_record_path),
                "source_document_ids": [
                    source["document_id"]
                    for source in frozen_configuration["source_documents"]
                ],
                "provenance_count": len(frozen_configuration["provenance"]),
            }
        )

    frozen_manifest = {
        "manifest_id": manifest_id,
        "version": "1.0",
        "status": "frozen",
        "frozen_at": frozen_at,
        "candidate_manifest": {
            "path": candidate_manifest_path.relative_to(repo_root).as_posix(),
            "sha256": sha256_file(candidate_manifest_path),
        },
        "document_manifest": {
            "path": document_manifest_path.relative_to(repo_root).as_posix(),
            "sha256": sha256_file(document_manifest_path),
        },
        "record_schema": {
            "path": record_schema_path.relative_to(repo_root).as_posix(),
            "sha256": sha256_file(record_schema_path),
        },
        "configuration_count": len(frozen_declarations),
        "answerable_count": 54,
        "unanswerable_count": 6,
        "configurations": frozen_declarations,
    }
    if authorization is not None:
        frozen_manifest["freeze_authorization"] = authorization
    frozen_manifest_path = manifest_path
    write_states.append(_write_or_verify(frozen_manifest_path, frozen_manifest))

    manifest_freeze_record = {
        "artifact_type": f"{dataset_label} Configuration C manifest",
        "manifest_id": frozen_manifest["manifest_id"],
        "version": "1.0",
        "status": "frozen",
        "frozen_at": frozen_at,
        "file": frozen_manifest_path.relative_to(repo_root).as_posix(),
        "file_sha256": sha256_file(frozen_manifest_path),
        "candidate_manifest": candidate_manifest_path.relative_to(
            repo_root
        ).as_posix(),
        "candidate_manifest_sha256": sha256_file(candidate_manifest_path),
        "configuration_count": 60,
        "approval_basis": "explicit_project_owner_freeze_instruction",
    }
    if authorization is not None:
        manifest_freeze_record["freeze_authorization"] = authorization
    manifest_freeze_path = frozen_manifest_path.with_name(
        f"{frozen_manifest_path.stem}_freeze_record.json"
    )
    write_states.append(_write_or_verify(manifest_freeze_path, manifest_freeze_record))

    return {
        "configuration_count": len(frozen_declarations),
        "manifest_path": frozen_manifest_path.relative_to(repo_root).as_posix(),
        "manifest_freeze_record_path": manifest_freeze_path.relative_to(
            repo_root
        ).as_posix(),
        "created_count": write_states.count("created"),
        "unchanged_count": write_states.count("unchanged"),
    }
