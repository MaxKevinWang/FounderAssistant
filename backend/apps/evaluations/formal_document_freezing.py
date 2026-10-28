"""Freeze reviewed formal Document D candidates without changing their blocks."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from apps.evaluations.corpus_paths import resolve_corpus_path
from typing import Any

from apps.evaluations.rageval_artifacts import (
    ArtifactValidationError,
    sha256_file,
    validate_document_record,
    write_json_artifact,
)


DEFAULT_DOCUMENT_MANIFEST_PATH = Path(
    "docs/evaluation/formal_dataset/documents/"
    "formal_test_document_manifest_v1.0.json"
)
DEFAULT_FROZEN_SELECTION_PATH = Path(
    "docs/evaluation/formal_dataset/source_selection/"
    "formal_test_source_selection_v1.0.json"
)


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
    """Write a new immutable artifact or verify an identical existing artifact."""

    if path.exists():
        if _read_json_object(path) != payload:
            raise ArtifactValidationError(
                f"Frozen artifact already exists with different content: {path}"
            )
        return "unchanged"
    write_json_artifact(path, payload)
    return "created"


def _frozen_path_for_candidate(candidate_path: Path) -> Path:
    """Map ``formal_candidates/*_v0.1.json`` to ``documents/*_v1.0.json``."""

    suffix = "_v0.1.json"
    if not candidate_path.name.endswith(suffix):
        raise ArtifactValidationError(
            f"Unexpected formal candidate filename: {candidate_path}"
        )
    return candidate_path.parent.parent / (
        candidate_path.name[: -len(suffix)] + "_v1.0.json"
    )


def freeze_formal_documents(
    *,
    repo_root: Path,
    selection_path: Path,
    frozen_at: str,
    document_manifest_id: str = "formal_test_document_manifest_v1.0",
    document_manifest_path: Path = DEFAULT_DOCUMENT_MANIFEST_PATH,
    frozen_selection_id: str = "formal_test_source_selection_v1.0",
    frozen_selection_path: Path = DEFAULT_FROZEN_SELECTION_PATH,
    authorization_path: Path | None = None,
) -> dict[str, Any]:
    """Freeze reviewed candidates and resolve one 24-document benchmark set."""

    repo_root = repo_root.resolve()
    if not selection_path.is_absolute():
        selection_path = repo_root / selection_path
    if not document_manifest_path.is_absolute():
        document_manifest_path = repo_root / document_manifest_path
    if not frozen_selection_path.is_absolute():
        frozen_selection_path = repo_root / frozen_selection_path

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

    selection = _read_json_object(selection_path)
    if selection.get("status") != "review_candidate":
        raise ArtifactValidationError(
            f"Expected a review-candidate source selection: {selection_path}"
        )

    resolved_documents: list[dict[str, Any]] = []
    for domain_name, domain_payload in selection["domains"].items():
        for selected in domain_payload["selected_documents"]:
            document_id = str(selected["document_id"])
            source_path = resolve_corpus_path(repo_root, str(selected["path"]))
            action = selected["document_action"]

            if action == "generate_candidate":
                candidate_path = repo_root / str(selected["document_output_path"])
                candidate = _read_json_object(candidate_path)
                validate_document_record(document=candidate, source_path=source_path)
                if (
                    candidate.get("document_id") != document_id
                    or candidate.get("version") != "0.1"
                    or candidate.get("status") != "review_candidate"
                ):
                    raise ArtifactValidationError(
                        f"Candidate identity/status mismatch: {candidate_path}"
                    )

                frozen_document = deepcopy(candidate)
                frozen_document["version"] = "1.0"
                frozen_document["status"] = "frozen"
                frozen_document["supersedes"] = f"{document_id} v0.1"
                frozen_document["frozen_at"] = frozen_at
                validate_document_record(
                    document=frozen_document,
                    source_path=source_path,
                )

                frozen_path = _frozen_path_for_candidate(candidate_path)
                _write_or_verify(frozen_path, frozen_document)
                freeze_record = {
                    "artifact_type": "RAGEval Document D",
                    "document_id": document_id,
                    "version": "1.0",
                    "status": "frozen",
                    "frozen_at": frozen_at,
                    "file": frozen_path.relative_to(repo_root).as_posix(),
                    "file_sha256": sha256_file(frozen_path),
                    "content_sha256": frozen_document["content_sha256"],
                    "candidate_file": candidate_path.relative_to(repo_root).as_posix(),
                    "candidate_file_sha256": sha256_file(candidate_path),
                    "source_file": frozen_document["source"]["path"],
                    "source_sha256": frozen_document["source"]["sha256"],
                    "block_count": frozen_document["block_count"],
                    "selection_candidate_file": selection_path.relative_to(
                        repo_root
                    ).as_posix(),
                    "selection_candidate_sha256": sha256_file(selection_path),
                    "approval": {
                        "basis": "explicit_project_owner_freeze_instruction",
                        "approved_at": frozen_at,
                        "scope": (
                            "source representation, extraction order, block identity, "
                            "and hashes"
                        ),
                        "temporal_fact_validation": (
                            "deferred to Configuration and QRA review"
                        ),
                    },
                    "note": (
                        "Freezing Document D fixes the reviewed source representation; "
                        "it does not independently certify current legal or programme "
                        "validity."
                    ),
                }
                if authorization is not None:
                    freeze_record["freeze_authorization"] = authorization
                freeze_record_path = frozen_path.with_name(
                    f"{frozen_path.stem}_freeze_record.json"
                )
                _write_or_verify(freeze_record_path, freeze_record)
                resolved_documents.append(
                    {
                        "document_id": document_id,
                        "domain": domain_name,
                        "resolution": "newly_frozen",
                        "path": frozen_path.relative_to(repo_root).as_posix(),
                        "file_sha256": sha256_file(frozen_path),
                        "freeze_record_path": freeze_record_path.relative_to(
                            repo_root
                        ).as_posix(),
                        "freeze_record_sha256": sha256_file(freeze_record_path),
                        "content_sha256": frozen_document["content_sha256"],
                        "source_path": frozen_document["source"]["path"],
                        "source_sha256": frozen_document["source"]["sha256"],
                        "block_count": frozen_document["block_count"],
                    }
                )
                continue

            if action != "reuse_existing":
                raise ArtifactValidationError(
                    f"Unknown document_action {action!r} for {document_id}"
                )
            frozen_path = repo_root / str(selected["existing_document_path"])
            frozen_document = _read_json_object(frozen_path)
            validate_document_record(
                document=frozen_document,
                source_path=source_path,
            )
            if (
                frozen_document.get("document_id") != document_id
                or frozen_document.get("version") != "1.0"
                or frozen_document.get("status") != "frozen"
            ):
                raise ArtifactValidationError(
                    f"Reused frozen document identity mismatch: {frozen_path}"
                )
            if sha256_file(frozen_path) != selected["existing_document_sha256"]:
                raise ArtifactValidationError(
                    f"Reused frozen document hash changed: {frozen_path}"
                )
            freeze_record_path = frozen_path.with_name(
                f"{frozen_path.stem}_freeze_record.json"
            )
            if not freeze_record_path.exists():
                raise ArtifactValidationError(
                    f"Reused document freeze record is missing: {freeze_record_path}"
                )
            resolved_documents.append(
                {
                    "document_id": document_id,
                    "domain": domain_name,
                    "resolution": "reused_frozen",
                    "path": frozen_path.relative_to(repo_root).as_posix(),
                    "file_sha256": sha256_file(frozen_path),
                    "freeze_record_path": freeze_record_path.relative_to(
                        repo_root
                    ).as_posix(),
                    "freeze_record_sha256": sha256_file(freeze_record_path),
                    "content_sha256": frozen_document["content_sha256"],
                    "source_path": frozen_document["source"]["path"],
                    "source_sha256": frozen_document["source"]["sha256"],
                    "block_count": frozen_document["block_count"],
                }
            )

    if len(resolved_documents) != 24:
        raise ArtifactValidationError(
            f"Formal document set must resolve to 24 documents, got "
            f"{len(resolved_documents)}"
        )
    if len({item["document_id"] for item in resolved_documents}) != 24:
        raise ArtifactValidationError("Formal document IDs are not unique")

    document_manifest = {
        "manifest_id": document_manifest_id,
        "version": "1.0",
        "status": "frozen",
        "frozen_at": frozen_at,
        "source_selection_candidate": {
            "path": selection_path.relative_to(repo_root).as_posix(),
            "sha256": sha256_file(selection_path),
        },
        "document_count": len(resolved_documents),
        "newly_frozen_count": sum(
            item["resolution"] == "newly_frozen" for item in resolved_documents
        ),
        "reused_frozen_count": sum(
            item["resolution"] == "reused_frozen" for item in resolved_documents
        ),
        "documents": resolved_documents,
    }
    if authorization is not None:
        document_manifest["freeze_authorization"] = authorization
    _write_or_verify(document_manifest_path, document_manifest)

    frozen_selection = deepcopy(selection)
    frozen_selection["selection_id"] = frozen_selection_id
    frozen_selection["version"] = "1.0"
    frozen_selection["status"] = "frozen"
    frozen_selection["frozen_at"] = frozen_at
    frozen_selection["supersedes"] = selection["selection_id"]
    frozen_selection["candidate_source"] = {
        "path": selection_path.relative_to(repo_root).as_posix(),
        "sha256": sha256_file(selection_path),
    }
    resolved_by_id = {
        item["document_id"]: item for item in resolved_documents
    }
    for domain_payload in frozen_selection["domains"].values():
        for selected in domain_payload["selected_documents"]:
            resolved = resolved_by_id[selected["document_id"]]
            selected["resolved_document_path"] = resolved["path"]
            selected["resolved_document_sha256"] = resolved["file_sha256"]
            selected["resolved_content_sha256"] = resolved["content_sha256"]
            selected["resolved_freeze_record_path"] = resolved["freeze_record_path"]
            selected["resolution"] = resolved["resolution"]
    frozen_selection["document_manifest"] = {
        "path": document_manifest_path.relative_to(repo_root).as_posix(),
        "sha256": sha256_file(document_manifest_path),
    }
    if authorization is not None:
        frozen_selection["freeze_authorization"] = authorization
    _write_or_verify(frozen_selection_path, frozen_selection)

    selection_freeze_record = {
        "artifact_type": "Formal test source selection",
        "selection_id": frozen_selection["selection_id"],
        "version": "1.0",
        "status": "frozen",
        "frozen_at": frozen_at,
        "file": frozen_selection_path.relative_to(repo_root).as_posix(),
        "file_sha256": sha256_file(frozen_selection_path),
        "candidate_file": selection_path.relative_to(repo_root).as_posix(),
        "candidate_file_sha256": sha256_file(selection_path),
        "document_manifest": document_manifest_path.relative_to(repo_root).as_posix(),
        "document_manifest_sha256": sha256_file(document_manifest_path),
        "document_count": 24,
        "note": (
            "The source set and Document D identities are frozen before formal "
            "Configuration and QRA authoring."
        ),
    }
    if authorization is not None:
        selection_freeze_record["freeze_authorization"] = authorization
    selection_freeze_record_path = frozen_selection_path.with_name(
        f"{frozen_selection_path.stem}_freeze_record.json"
    )
    _write_or_verify(selection_freeze_record_path, selection_freeze_record)

    return {
        "document_manifest_path": document_manifest_path.relative_to(
            repo_root
        ).as_posix(),
        "source_selection_path": frozen_selection_path.relative_to(
            repo_root
        ).as_posix(),
        "newly_frozen_count": document_manifest["newly_frozen_count"],
        "reused_frozen_count": document_manifest["reused_frozen_count"],
        "documents": resolved_documents,
    }
