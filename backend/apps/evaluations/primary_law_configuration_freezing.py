"""Freeze semantically curated primary-law Configurations as manifest v1.3."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

from apps.evaluations.corpus_paths import resolve_corpus_path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

from apps.evaluations.formal_configuration_generation import DOMAIN_SCHEMA_PATHS
from apps.evaluations.primary_law_configuration_curation import (
    AGENT_REVIEW_PATH,
    FULL_CANDIDATE_MANIFEST_PATH,
)
from apps.evaluations.rageval_artifacts import (
    ArtifactValidationError,
    sha256_file,
    validate_configuration_record,
    validate_document_record,
    write_json_artifact,
)


OUTPUT_ROOT = Path("docs/evaluation/formal_dataset/configurations/formal_test")
FROZEN_MANIFEST_PATH = OUTPUT_ROOT / "formal_test_configuration_manifest_v1.3.json"
AUTHORIZATION_PATH = Path(
    "docs/evaluation/formal_dataset/configurations/primary_law_candidates/"
    "reviewed_v0.2/primary_law_configuration_freeze_authorization_v1.0.json"
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


def _assert_balance(configurations: list[dict[str, Any]]) -> None:
    """Protect the approved 60-item domain and question-type allocation."""

    if len(configurations) != 60:
        raise ArtifactValidationError("Configuration manifest must contain 60 items")
    ids = [str(item["configuration_id"]) for item in configurations]
    if len(set(ids)) != 60:
        raise ArtifactValidationError("Configuration IDs are not unique")
    if Counter(item["answerability"] for item in configurations) != {
        "answerable": 54,
        "unanswerable": 6,
    }:
        raise ArtifactValidationError("Answerability allocation changed")
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


def _load_documents(
    *,
    repo_root: Path,
    manifest: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Load frozen Documents and retain their manifest declarations."""

    if manifest.get("status") != "frozen" or manifest.get("version") != "1.2":
        raise ArtifactValidationError("Expected frozen Document manifest v1.2")
    documents: dict[str, dict[str, Any]] = {}
    declarations: dict[str, dict[str, Any]] = {}
    for declaration in manifest["documents"]:
        path = repo_root / str(declaration["path"])
        if sha256_file(path) != declaration["file_sha256"]:
            raise ArtifactValidationError(f"Frozen Document changed: {path}")
        document = _read_json_object(path)
        validate_document_record(
            document=document,
            source_path=resolve_corpus_path(repo_root, str(document["source"]["path"])),
        )
        document_id = str(document["document_id"])
        documents[document_id] = document
        declarations[document_id] = declaration
    return documents, declarations


def _source_document_records(
    *,
    configuration: dict[str, Any],
    declarations: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Bind a Configuration freeze record to immutable Document files."""

    records: list[dict[str, Any]] = []
    for source in configuration["source_documents"]:
        declaration = declarations[str(source["document_id"])]
        records.append(
            {
                "document_id": declaration["document_id"],
                "path": declaration["path"],
                "file_sha256": declaration["file_sha256"],
                "content_sha256": declaration["content_sha256"],
                "freeze_record_path": declaration["freeze_record_path"],
                "freeze_record_sha256": declaration["freeze_record_sha256"],
            }
        )
    return records


def freeze_primary_law_configurations(
    *,
    repo_root: Path,
    frozen_at: str = "2026-08-03",
) -> dict[str, Any]:
    """Freeze 17 curated revisions and carry forward 43 immutable records."""

    repo_root = repo_root.resolve()
    candidate_manifest_path = repo_root / FULL_CANDIDATE_MANIFEST_PATH
    authorization_path = repo_root / AUTHORIZATION_PATH
    candidate_manifest = _read_json_object(candidate_manifest_path)
    authorization = _read_json_object(authorization_path)

    if (
        candidate_manifest.get("status") != "review_candidate"
        or candidate_manifest.get("configuration_count") != 60
        or candidate_manifest.get("revised_candidate_count") != 17
    ):
        raise ArtifactValidationError(
            "Expected the 60-item semantic candidate manifest with 17 revisions"
        )
    if (
        authorization.get("status") != "authorized"
        or authorization.get("authorized_by_role") != "project_owner"
        or not str(authorization.get("source_instruction") or "").strip()
        or authorization.get("human_line_by_line_review_claimed") is not False
        or authorization.get("legal_expert_review_claimed") is not False
    ):
        raise ArtifactValidationError("Freeze authorization is missing or misleading")

    review_path = repo_root / str(
        candidate_manifest["agent_semantic_review"]["path"]
    )
    if (
        sha256_file(review_path)
        != candidate_manifest["agent_semantic_review"]["sha256"]
    ):
        raise ArtifactValidationError("Agent semantic review hash mismatch")
    review = _read_json_object(review_path)
    if (
        review.get("status") != "agent_semantic_review_passed"
        or review.get("unresolved_blocker_count") != 0
        or review.get("human_line_by_line_review_claimed") is not False
    ):
        raise ArtifactValidationError("Agent semantic review has not passed its gate")
    authorized_review = authorization["agent_semantic_review"]
    if (
        authorized_review.get("path") != review_path.relative_to(repo_root).as_posix()
        or authorized_review.get("sha256") != sha256_file(review_path)
        or authorized_review.get("unresolved_blocker_count") != 0
    ):
        raise ArtifactValidationError("Authorization does not bind the current review")

    predecessor_manifest_path = repo_root / str(
        candidate_manifest["predecessor_manifest"]["path"]
    )
    if (
        sha256_file(predecessor_manifest_path)
        != candidate_manifest["predecessor_manifest"]["sha256"]
    ):
        raise ArtifactValidationError("Predecessor Configuration manifest changed")
    predecessor_manifest = _read_json_object(predecessor_manifest_path)
    if predecessor_manifest.get("status") != "frozen":
        raise ArtifactValidationError("Predecessor Configuration manifest is not frozen")

    document_manifest_path = repo_root / str(
        candidate_manifest["document_manifest"]["path"]
    )
    if (
        sha256_file(document_manifest_path)
        != candidate_manifest["document_manifest"]["sha256"]
    ):
        raise ArtifactValidationError("Document manifest hash mismatch")
    document_manifest = _read_json_object(document_manifest_path)
    documents, document_declarations = _load_documents(
        repo_root=repo_root,
        manifest=document_manifest,
    )

    record_schema_path = repo_root / str(candidate_manifest["record_schema"]["path"])
    if sha256_file(record_schema_path) != candidate_manifest["record_schema"]["sha256"]:
        raise ArtifactValidationError("Configuration record schema changed")
    record_schema = _read_json_object(record_schema_path)
    domain_schemas = {
        domain: _read_json_object(repo_root / path)
        for domain, path in DOMAIN_SCHEMA_PATHS.items()
    }
    domain_schema_hashes = {
        domain: sha256_file(repo_root / path)
        for domain, path in DOMAIN_SCHEMA_PATHS.items()
    }

    loaded: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for declaration in candidate_manifest["configurations"]:
        path = repo_root / str(declaration["path"])
        if sha256_file(path) != declaration["file_sha256"]:
            raise ArtifactValidationError(f"Configuration changed: {path}")
        configuration = _read_json_object(path)
        resolution = declaration["resolution"]
        if resolution == "carried_forward":
            if configuration.get("status") != "frozen":
                raise ArtifactValidationError(f"Carry-forward is not frozen: {path}")
            freeze_record_path = repo_root / str(declaration["freeze_record_path"])
            if sha256_file(freeze_record_path) != declaration["freeze_record_sha256"]:
                raise ArtifactValidationError(
                    f"Carry-forward freeze record changed: {freeze_record_path}"
                )
        elif resolution == "primary_law_semantic_candidate":
            if (
                configuration.get("status") != "development_draft"
                or configuration.get("generation_record", {}).get(
                    "human_review_status"
                )
                != "pending"
            ):
                raise ArtifactValidationError(
                    f"Unexpected curated candidate lifecycle: {path}"
                )
        else:
            raise ArtifactValidationError(f"Unknown resolution: {resolution}")

        errors = _schema_errors(configuration, record_schema)
        if errors:
            raise ArtifactValidationError(
                f"Invalid Configuration {path}: {'; '.join(errors[:5])}"
            )
        validate_configuration_record(
            configuration=configuration,
            schema=domain_schemas[configuration["domain"]],
            documents=documents,
        )
        loaded.append((declaration, configuration))

    _assert_balance([configuration for _, configuration in loaded])

    write_states: list[str] = []
    frozen_declarations: list[dict[str, Any]] = []
    revised_count = 0
    for declaration, configuration in loaded:
        if declaration["resolution"] == "carried_forward":
            frozen_declarations.append(deepcopy(declaration))
            continue

        revised_count += 1
        frozen_configuration = deepcopy(configuration)
        frozen_configuration["status"] = "frozen"
        # Pending is intentional: the owner authorized an experimental freeze,
        # not a human line-by-line legal approval.
        frozen_configuration["generation_record"]["human_review_status"] = "pending"
        errors = _schema_errors(frozen_configuration, record_schema)
        if errors:
            raise ArtifactValidationError(
                f"Invalid frozen Configuration {configuration['configuration_id']}: "
                f"{'; '.join(errors[:5])}"
            )
        validate_configuration_record(
            configuration=frozen_configuration,
            schema=domain_schemas[frozen_configuration["domain"]],
            documents=documents,
        )

        frozen_path = (
            repo_root
            / OUTPUT_ROOT
            / str(frozen_configuration["domain"])
            / (
                f"{frozen_configuration['configuration_id']}_"
                f"v{frozen_configuration['version']}.json"
            )
        )
        write_states.append(_write_or_verify(frozen_path, frozen_configuration))
        freeze_record = {
            "artifact_type": "RAGEval Configuration C",
            "configuration_id": frozen_configuration["configuration_id"],
            "version": frozen_configuration["version"],
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
            "source_documents": _source_document_records(
                configuration=frozen_configuration,
                declarations=document_declarations,
            ),
            "agent_semantic_review": {
                "path": review_path.relative_to(repo_root).as_posix(),
                "sha256": sha256_file(review_path),
                "status": review["status"],
                "unresolved_blocker_count": 0,
            },
            "freeze_authorization": {
                "path": AUTHORIZATION_PATH.as_posix(),
                "sha256": sha256_file(authorization_path),
                "basis": "explicit_project_owner_continuation_instruction",
                "source_instruction": authorization["source_instruction"],
                "human_line_by_line_review_claimed": False,
                "legal_expert_review_claimed": False,
            },
            "lifecycle_note": (
                "Frozen as a reproducible experimental dependency after agent "
                "semantic review; this does not claim human or legal-expert approval."
            ),
            "change_policy": (
                "Do not edit this frozen Configuration or freeze record in place; "
                "create a new reviewed version for every later correction."
            ),
        }
        freeze_record_path = frozen_path.with_name(
            f"{frozen_path.stem}_freeze_record.json"
        )
        write_states.append(_write_or_verify(freeze_record_path, freeze_record))
        frozen_declarations.append(
            {
                "configuration_id": frozen_configuration["configuration_id"],
                "version": frozen_configuration["version"],
                "resolution": "primary_law_revision",
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

    if revised_count != 17:
        raise ArtifactValidationError(f"Expected 17 revisions, got {revised_count}")
    frozen_manifest = {
        "manifest_id": "formal_test_configuration_manifest_v1.3",
        "version": "1.3",
        "status": "frozen",
        "frozen_at": frozen_at,
        "supersedes": predecessor_manifest["manifest_id"],
        "predecessor_manifest": {
            "path": predecessor_manifest_path.relative_to(repo_root).as_posix(),
            "sha256": sha256_file(predecessor_manifest_path),
        },
        "semantic_candidate_manifest": {
            "path": FULL_CANDIDATE_MANIFEST_PATH.as_posix(),
            "sha256": sha256_file(candidate_manifest_path),
        },
        "freeze_authorization": {
            "path": AUTHORIZATION_PATH.as_posix(),
            "sha256": sha256_file(authorization_path),
            "human_line_by_line_review_claimed": False,
            "legal_expert_review_claimed": False,
        },
        "agent_semantic_review": {
            "path": AGENT_REVIEW_PATH.as_posix(),
            "sha256": sha256_file(review_path),
            "status": review["status"],
        },
        "document_manifest": {
            "path": document_manifest_path.relative_to(repo_root).as_posix(),
            "sha256": sha256_file(document_manifest_path),
        },
        "record_schema": candidate_manifest["record_schema"],
        "configuration_count": 60,
        "revised_configuration_count": 17,
        "carried_forward_count": 43,
        "answerable_count": 54,
        "unanswerable_count": 6,
        "configurations": frozen_declarations,
    }
    frozen_manifest_path = repo_root / FROZEN_MANIFEST_PATH
    write_states.append(_write_or_verify(frozen_manifest_path, frozen_manifest))

    manifest_freeze_record = {
        "artifact_type": "Formal test Configuration C manifest",
        "manifest_id": frozen_manifest["manifest_id"],
        "version": "1.3",
        "status": "frozen",
        "frozen_at": frozen_at,
        "file": FROZEN_MANIFEST_PATH.as_posix(),
        "file_sha256": sha256_file(frozen_manifest_path),
        "semantic_candidate_manifest": {
            "path": FULL_CANDIDATE_MANIFEST_PATH.as_posix(),
            "sha256": sha256_file(candidate_manifest_path),
        },
        "freeze_authorization": {
            "path": AUTHORIZATION_PATH.as_posix(),
            "sha256": sha256_file(authorization_path),
        },
        "configuration_count": 60,
        "revised_configuration_count": 17,
        "human_line_by_line_review_claimed": False,
        "legal_expert_review_claimed": False,
        "change_policy": (
            "Do not edit manifest v1.3 in place; publish a new immutable version."
        ),
    }
    manifest_freeze_record_path = frozen_manifest_path.with_name(
        "formal_test_configuration_manifest_v1.3_freeze_record.json"
    )
    write_states.append(
        _write_or_verify(manifest_freeze_record_path, manifest_freeze_record)
    )

    return {
        "configuration_count": 60,
        "revised_configuration_count": 17,
        "carried_forward_count": 43,
        "manifest_path": FROZEN_MANIFEST_PATH.as_posix(),
        "manifest_freeze_record_path": manifest_freeze_record_path.relative_to(
            repo_root
        ).as_posix(),
        "created_count": write_states.count("created"),
        "unchanged_count": write_states.count("unchanged"),
    }
