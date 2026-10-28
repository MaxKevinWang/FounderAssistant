"""Freeze reviewed primary-law Document D candidates as an additive revision.

This module deliberately does not treat an automated review as a human legal
review.  A freeze invocation records the project owner's continuation
instruction, the completed agent checks, and the fact that no line-by-line
human review is being claimed.  Freezing makes the source representation
immutable; it does not certify that the selected provisions are legally
complete or current beyond their recorded retrieval date.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from apps.evaluations.rageval_artifacts import (
    ArtifactValidationError,
    sha256_file,
    validate_document_record,
    write_json_artifact,
)


CANDIDATE_MANIFEST_PATH = Path(
    "docs/evaluation/formal_dataset/documents/"
    "formal_test_document_manifest_v1.2_candidate.json"
)
AGENT_REVIEW_PATH = Path(
    "docs/evaluation/formal_dataset/documents/primary_law_candidates/"
    "primary_law_agent_review_v0.1.json"
)
AUTHORIZATION_PATH = Path(
    "docs/evaluation/formal_dataset/documents/primary_law_candidates/"
    "primary_law_freeze_authorization_v1.0.json"
)
FROZEN_MANIFEST_PATH = Path(
    "docs/evaluation/formal_dataset/documents/"
    "formal_test_document_manifest_v1.2.json"
)
FROZEN_MANIFEST_RECORD_PATH = Path(
    "docs/evaluation/formal_dataset/documents/"
    "formal_test_document_manifest_v1.2_freeze_record.json"
)


def _read_json_object(path: Path) -> dict[str, Any]:
    """Read one UTF-8 JSON object and report artifact-oriented errors."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(f"Cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ArtifactValidationError(f"Expected a JSON object: {path}")
    return payload


def _resolve(repo_root: Path, path: Path | str) -> Path:
    """Resolve a repository-relative path without changing absolute paths."""

    candidate = Path(path)
    return candidate if candidate.is_absolute() else repo_root / candidate


def _write_or_verify(path: Path, payload: dict[str, Any]) -> str:
    """Create an immutable JSON artifact or verify an identical rerun."""

    if path.exists():
        if _read_json_object(path) != payload:
            raise ArtifactValidationError(
                f"Frozen artifact already exists with different content: {path}"
            )
        return "unchanged"
    write_json_artifact(path, payload)
    return "created"


def _frozen_path(candidate_path: Path, document_root: Path) -> Path:
    """Map a primary-law ``*_v0.1`` candidate to a root ``*_v1.0`` file."""

    suffix = "_v0.1.json"
    if not candidate_path.name.endswith(suffix):
        raise ArtifactValidationError(
            f"Unexpected primary-law candidate filename: {candidate_path}"
        )
    return document_root / (candidate_path.name[: -len(suffix)] + "_v1.0.json")


def _validate_review_chain(
    *,
    repo_root: Path,
    candidate_manifest_path: Path,
    candidate_manifest: dict[str, Any],
    agent_review_path: Path,
    agent_review: dict[str, Any],
) -> list[dict[str, Any]]:
    """Validate the candidate manifest and automated review before freezing."""

    if candidate_manifest.get("status") != "review_candidate":
        raise ArtifactValidationError("Primary-law Document manifest is not a candidate")
    if candidate_manifest.get("document_count") != 38:
        raise ArtifactValidationError("Expected 38 documents in manifest v1.2 candidate")
    if candidate_manifest.get("carried_forward_frozen_count") != 24:
        raise ArtifactValidationError("Expected 24 carried-forward frozen documents")
    if candidate_manifest.get("added_review_candidate_count") != 14:
        raise ArtifactValidationError("Expected 14 primary-law candidates")

    predecessor = candidate_manifest.get("predecessor_manifest", {})
    predecessor_path = _resolve(repo_root, str(predecessor.get("path", "")))
    if not predecessor_path.exists() or sha256_file(predecessor_path) != predecessor.get(
        "sha256"
    ):
        raise ArtifactValidationError("Frozen predecessor Document manifest changed")
    predecessor_manifest = _read_json_object(predecessor_path)
    if predecessor_manifest.get("status") != "frozen":
        raise ArtifactValidationError("Predecessor Document manifest is not frozen")

    if agent_review.get("status") != "agent_checks_passed_human_review_pending":
        raise ArtifactValidationError("Primary-law agent review has not passed")
    if agent_review.get("blocking_failure_count") != 0:
        raise ArtifactValidationError("Primary-law agent review contains blocking failures")
    if agent_review.get("candidate_count") != 14:
        raise ArtifactValidationError("Primary-law agent review does not cover 14 candidates")
    if agent_review.get("key_fact_check_count") != 46:
        raise ArtifactValidationError("Primary-law agent review does not contain 46 checks")
    reviewed_manifest = agent_review.get("candidate_document_manifest", {})
    if reviewed_manifest.get("path") != candidate_manifest_path.relative_to(
        repo_root
    ).as_posix() or reviewed_manifest.get("sha256") != sha256_file(
        candidate_manifest_path
    ):
        raise ArtifactValidationError("Agent review does not bind the current candidate manifest")

    added = [
        item
        for item in candidate_manifest["documents"]
        if item.get("resolution") == "added_review_candidate"
    ]
    reviewed = {str(item["document_id"]): item for item in agent_review["documents"]}
    if {str(item["document_id"]) for item in added} != set(reviewed):
        raise ArtifactValidationError("Candidate manifest and agent review document IDs differ")
    for declaration in added:
        review = reviewed[str(declaration["document_id"])]
        if review.get("agent_result") != "pass_human_review_pending":
            raise ArtifactValidationError(
                f"Agent review did not pass for {declaration['document_id']}"
            )
        if review.get("failures"):
            raise ArtifactValidationError(
                f"Agent review contains failures for {declaration['document_id']}"
            )
        candidate_path = _resolve(repo_root, str(declaration["path"]))
        if sha256_file(candidate_path) != declaration.get("file_sha256"):
            raise ArtifactValidationError(f"Candidate changed: {candidate_path}")
        if sha256_file(candidate_path) != review.get("artifact_sha256"):
            raise ArtifactValidationError(
                f"Agent review hash changed for {declaration['document_id']}"
            )

    # Verify that the carried-forward prefix is byte-for-byte represented by
    # the frozen predecessor declaration list.
    carried = candidate_manifest["documents"][: len(predecessor_manifest["documents"])]
    if carried != predecessor_manifest["documents"]:
        raise ArtifactValidationError(
            "Candidate manifest changed a carried-forward Document declaration"
        )
    return added


def freeze_primary_law_documents(
    *,
    repo_root: Path,
    frozen_at: str,
    authorization_instruction: str,
    candidate_manifest_path: Path = CANDIDATE_MANIFEST_PATH,
    agent_review_path: Path = AGENT_REVIEW_PATH,
) -> dict[str, Any]:
    """Freeze 14 primary-law candidates and publish Document manifest v1.2."""

    repo_root = repo_root.resolve()
    instruction = authorization_instruction.strip()
    if not instruction:
        raise ArtifactValidationError("An explicit continuation instruction is required")
    candidate_manifest_path = _resolve(repo_root, candidate_manifest_path)
    agent_review_path = _resolve(repo_root, agent_review_path)
    candidate_manifest = _read_json_object(candidate_manifest_path)
    agent_review = _read_json_object(agent_review_path)
    added = _validate_review_chain(
        repo_root=repo_root,
        candidate_manifest_path=candidate_manifest_path,
        candidate_manifest=candidate_manifest,
        agent_review_path=agent_review_path,
        agent_review=agent_review,
    )

    authorization = {
        "artifact_type": "Primary-law Document D freeze authorization",
        "authorization_id": "primary_law_freeze_authorization_v1.0",
        "version": "1.0",
        "status": "authorized_to_freeze_after_agent_review",
        "authorized_at": frozen_at,
        "project_owner_instruction": instruction,
        "authorization_interpretation": (
            "The project owner instructed the workflow to continue after receiving "
            "the agent-review result and next-step explanation."
        ),
        "human_line_by_line_review_claimed": False,
        "agent_review": {
            "path": agent_review_path.relative_to(repo_root).as_posix(),
            "sha256": sha256_file(agent_review_path),
            "status": agent_review["status"],
            "blocking_failure_count": agent_review["blocking_failure_count"],
            "key_fact_check_count": agent_review["key_fact_check_count"],
        },
        "candidate_manifest": {
            "path": candidate_manifest_path.relative_to(repo_root).as_posix(),
            "sha256": sha256_file(candidate_manifest_path),
        },
        "authorized_document_ids": [str(item["document_id"]) for item in added],
        "freeze_scope": (
            "source representation, selected statutory units, block identity, order, "
            "and hashes"
        ),
        "known_limitations_carried_forward": [
            "No line-by-line human legal review is claimed.",
            "effective_from remains unresolved where the candidate records only "
            "current_at_retrieval.",
            "Freezing does not certify legal completeness, interpretation, or future "
            "validity.",
            "Configuration-level legal claims remain subject to separate review.",
        ],
    }
    authorization_path = repo_root / AUTHORIZATION_PATH
    _write_or_verify(authorization_path, authorization)

    document_root = (repo_root / FROZEN_MANIFEST_PATH).parent
    agent_by_id = {
        str(item["document_id"]): item for item in agent_review["documents"]
    }
    frozen_declarations: list[dict[str, Any]] = []
    created_count = 0
    for declaration in added:
        document_id = str(declaration["document_id"])
        candidate_path = _resolve(repo_root, str(declaration["path"]))
        candidate = _read_json_object(candidate_path)
        source_path = _resolve(repo_root, str(candidate["source"]["path"]))
        validate_document_record(document=candidate, source_path=source_path)
        if (
            candidate.get("document_id") != document_id
            or candidate.get("version") != "0.1"
            or candidate.get("status") != "review_candidate"
        ):
            raise ArtifactValidationError(f"Candidate identity mismatch: {candidate_path}")

        frozen_document = deepcopy(candidate)
        frozen_document["version"] = "1.0"
        frozen_document["status"] = "frozen"
        frozen_document["supersedes"] = f"{document_id} v0.1"
        frozen_document["frozen_at"] = frozen_at
        validate_document_record(document=frozen_document, source_path=source_path)

        frozen_path = _frozen_path(candidate_path, document_root)
        if _write_or_verify(frozen_path, frozen_document) == "created":
            created_count += 1
        review = agent_by_id[document_id]
        freeze_record = {
            "artifact_type": "RAGEval primary-law Document D",
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
            "primary_law_selection_file": candidate_manifest[
                "primary_law_source_selection"
            ]["path"],
            "primary_law_selection_sha256": candidate_manifest[
                "primary_law_source_selection"
            ]["sha256"],
            "agent_review": {
                "path": agent_review_path.relative_to(repo_root).as_posix(),
                "sha256": sha256_file(agent_review_path),
                "result": review["agent_result"],
                "key_fact_check_count": review["key_fact_check_count"],
                "advisories": review["advisories"],
            },
            "authorization": {
                "path": AUTHORIZATION_PATH.as_posix(),
                "sha256": sha256_file(authorization_path),
                "basis": "explicit_project_owner_continuation_after_agent_review",
                "human_line_by_line_review_claimed": False,
            },
            "note": (
                "This freeze fixes the selected legal-source representation. It does "
                "not independently certify legal completeness, interpretation, or "
                "validity after the recorded retrieval date."
            ),
        }
        freeze_record_path = frozen_path.with_name(
            f"{frozen_path.stem}_freeze_record.json"
        )
        _write_or_verify(freeze_record_path, freeze_record)
        frozen_declarations.append(
            {
                "document_id": document_id,
                "domain": frozen_document["domain"],
                "resolution": "added_frozen_primary_law",
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
                "review_status": "frozen_after_agent_review_and_owner_authorization",
            }
        )

    predecessor_path = _resolve(
        repo_root, str(candidate_manifest["predecessor_manifest"]["path"])
    )
    predecessor = _read_json_object(predecessor_path)
    documents = [deepcopy(item) for item in predecessor["documents"]]
    documents.extend(frozen_declarations)
    domain_counts = Counter(str(item["domain"]) for item in documents)
    frozen_manifest = {
        "manifest_id": "formal_test_document_manifest_v1.2",
        "version": "1.2",
        "status": "frozen",
        "frozen_at": frozen_at,
        "supersedes": predecessor["manifest_id"],
        "predecessor_manifest": {
            "path": predecessor_path.relative_to(repo_root).as_posix(),
            "sha256": sha256_file(predecessor_path),
        },
        "candidate_manifest": {
            "path": candidate_manifest_path.relative_to(repo_root).as_posix(),
            "sha256": sha256_file(candidate_manifest_path),
        },
        "primary_law_source_selection": candidate_manifest[
            "primary_law_source_selection"
        ],
        "configuration_impact_plan": candidate_manifest[
            "configuration_impact_plan"
        ],
        "agent_review": {
            "path": agent_review_path.relative_to(repo_root).as_posix(),
            "sha256": sha256_file(agent_review_path),
        },
        "authorization": {
            "path": AUTHORIZATION_PATH.as_posix(),
            "sha256": sha256_file(authorization_path),
            "human_line_by_line_review_claimed": False,
        },
        "document_count": len(documents),
        "carried_forward_frozen_count": len(predecessor["documents"]),
        "added_primary_law_count": len(frozen_declarations),
        "domain_document_counts": dict(sorted(domain_counts.items())),
        "sampling_balance_note": candidate_manifest["sampling_balance_note"],
        "documents": documents,
    }
    frozen_manifest_path = repo_root / FROZEN_MANIFEST_PATH
    _write_or_verify(frozen_manifest_path, frozen_manifest)
    manifest_freeze_record = {
        "artifact_type": "Formal test Document D manifest",
        "manifest_id": frozen_manifest["manifest_id"],
        "version": frozen_manifest["version"],
        "status": "frozen",
        "frozen_at": frozen_at,
        "file": FROZEN_MANIFEST_PATH.as_posix(),
        "file_sha256": sha256_file(frozen_manifest_path),
        "predecessor_manifest": predecessor_path.relative_to(repo_root).as_posix(),
        "predecessor_manifest_sha256": sha256_file(predecessor_path),
        "candidate_manifest": candidate_manifest_path.relative_to(repo_root).as_posix(),
        "candidate_manifest_sha256": sha256_file(candidate_manifest_path),
        "authorization": AUTHORIZATION_PATH.as_posix(),
        "authorization_sha256": sha256_file(authorization_path),
        "agent_review": agent_review_path.relative_to(repo_root).as_posix(),
        "agent_review_sha256": sha256_file(agent_review_path),
        "document_count": len(documents),
        "added_document_ids": [item["document_id"] for item in frozen_declarations],
        "approval_basis": "explicit_project_owner_continuation_after_agent_review",
        "human_line_by_line_review_claimed": False,
    }
    manifest_record_path = repo_root / FROZEN_MANIFEST_RECORD_PATH
    _write_or_verify(manifest_record_path, manifest_freeze_record)

    return {
        "document_manifest_path": FROZEN_MANIFEST_PATH.as_posix(),
        "document_manifest_sha256": sha256_file(frozen_manifest_path),
        "authorization_path": AUTHORIZATION_PATH.as_posix(),
        "created_document_count": created_count,
        "frozen_document_count": len(frozen_declarations),
        "carried_forward_count": len(predecessor["documents"]),
        "documents": frozen_declarations,
    }
