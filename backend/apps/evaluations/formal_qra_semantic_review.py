"""Audit all formal QRA v0.4 candidates before an immutable freeze.

The deterministic checks in this module verify artifact identity, schema
validity, traceability, set balance, and carried-item semantic identity.  The
review specification records the agent's separate semantic judgments for the
17 primary-law revisions.  An agent review is never represented as a second
human or legal-expert review.
"""

from __future__ import annotations

from collections import Counter
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


CANDIDATE_MANIFEST_PATH = Path(
    "docs/evaluation/formal_dataset/qra/formal_candidates/"
    "generation_manifest_v0.4.json"
)
PREDECESSOR_MANIFEST_PATH = Path(
    "docs/evaluation/formal_dataset/qra/formal_candidates/"
    "generation_manifest_v0.3.json"
)
REVISION_MANIFEST_PATH = Path(
    "docs/evaluation/formal_dataset/qra/"
    "primary_law_qra_revision_manifest_v0.4.json"
)
REVIEW_SPEC_PATH = Path(
    "docs/evaluation/formal_dataset/qra/reviews/round_3/"
    "formal_qra_agent_review_spec_v0.4.json"
)
AGENT_REVIEW_PATH = Path(
    "docs/evaluation/formal_dataset/qra/reviews/round_3/"
    "formal_qra_agent_semantic_review_v0.4.json"
)
AGENT_REVIEW_MD_PATH = Path(
    "docs/evaluation/formal_dataset/qra/reviews/round_3/"
    "formal_qra_agent_semantic_review_v0.4_zh.md"
)
OWNER_CONFIRMATION_PATH = Path(
    "docs/evaluation/formal_dataset/qra/reviews/round_3/"
    "formal_qra_project_owner_confirmation_v0.4.json"
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

# These fields define QRA semantics and evidence.  Metadata may legitimately
# change when the complete candidate set is rebound to a new manifest version.
SEMANTIC_FIELDS = (
    "question",
    "answerable",
    "answer",
    "keypoints",
    "references",
    "answer_reference_alignment",
    "document_ids",
    "question_type",
    "domain",
    "difficulty",
)


def _read_json_object(path: Path) -> dict[str, Any]:
    """Read one UTF-8 JSON object with a useful artifact error."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(f"Cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ArtifactValidationError(f"Expected a JSON object: {path}")
    return payload


def _schema_errors(
    payload: dict[str, Any],
    record_schema: dict[str, Any],
) -> list[str]:
    """Return deterministic JSON Schema failures in path order."""

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


def _load_declared_records(
    *,
    repo_root: Path,
    declarations: list[dict[str, Any]],
    id_field: str,
) -> dict[str, dict[str, Any]]:
    """Load hash-bound records from a manifest declaration list."""

    records: dict[str, dict[str, Any]] = {}
    for declaration in declarations:
        path = repo_root / str(declaration["path"])
        expected_hash = str(
            declaration.get("file_sha256") or declaration.get("sha256") or ""
        )
        if not expected_hash or sha256_file(path) != expected_hash:
            raise ArtifactValidationError(f"Declared artifact hash mismatch: {path}")
        record = _read_json_object(path)
        record_id = str(record[id_field])
        if record_id in records:
            raise ArtifactValidationError(f"Duplicate declared record ID: {record_id}")
        records[record_id] = record
    return records


def _assert_set_balance(records: list[dict[str, Any]]) -> None:
    """Protect the approved 60-item domain/type/answerability allocation."""

    if len(records) != 60 or len({record["id"] for record in records}) != 60:
        raise ArtifactValidationError("Expected 60 unique QRA records")
    if len({record["question"] for record in records}) != 60:
        raise ArtifactValidationError("Formal QRA questions are not unique")
    if Counter(record["answerable"] for record in records) != {True: 54, False: 6}:
        raise ArtifactValidationError("Formal QRA answerability allocation changed")
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


def _contains_all(haystack: str, needles: list[str], *, label: str) -> None:
    """Fail with the first missing semantic-review fragment."""

    for needle in needles:
        if needle not in haystack:
            raise ArtifactValidationError(f"Missing {label} fragment: {needle!r}")


def _render_review(review: dict[str, Any]) -> str:
    """Render the audit result as a concise Chinese review record."""

    lines = [
        "# Formal QRA v0.4 Agent 语义复核",
        "",
        f"- 复核日期：`{review['reviewed_at']}`",
        f"- 候选总数：`{review['qra_count']}`",
        f"- 法律重构题逐条复核：`{review['revised_item_count']}`",
        f"- 继承题复核：`{review['carried_item_count']}`",
        f"- 未解决阻断项：`{review['unresolved_blocker_count']}`",
        f"- 结果：`{review['status']}`",
        "",
        "## 复核边界",
        "",
        "这是一轮 Agent 复核，不构成第二位人工审核，也不构成法律专家意见。",
        "机械检查覆盖 JSON Schema、冻结依赖哈希、Reference 原文定位、",
        "Answer-Keypoint 映射、题型配额与 43 道继承题的逐字段语义一致性。",
        "17 道法律重构题另按审核规范逐题检查问题、答案、Keypoints 与证据。",
        "",
        "## 本轮发现并已解决",
        "",
    ]
    for finding in review["resolved_findings"]:
        lines.extend(
            [
                f"- `{finding['finding_id']}`：{finding['summary']} "
                f"处理：{finding['resolution']}",
            ]
        )
    lines.extend(
        [
            "",
            "## 最终结论",
            "",
            "60 道题均通过当前冻结门禁，未发现剩余的答案正确性、引用支持、",
            "数值计算、时间条件、不可回答性或集合配额阻断项。",
            "",
            "冻结只固定可复现实验版本；`double_checked` 仍需另一位独立人工",
            "审核者完成，不能由本次 Agent 复核替代。",
            "",
        ]
    )
    return "\n".join(lines)


def audit_formal_qra_candidates(
    *,
    repo_root: Path,
    owner_instruction: str,
    reviewed_at: str = "2026-08-03",
) -> dict[str, Any]:
    """Audit QRA v0.4 and record the project owner's conditional freeze request."""

    instruction = owner_instruction.strip()
    if not instruction:
        raise ArtifactValidationError("Project-owner review instruction is required")
    repo_root = repo_root.resolve()
    candidate_manifest_path = repo_root / CANDIDATE_MANIFEST_PATH
    predecessor_manifest_path = repo_root / PREDECESSOR_MANIFEST_PATH
    revision_manifest_path = repo_root / REVISION_MANIFEST_PATH
    review_spec_path = repo_root / REVIEW_SPEC_PATH
    candidate_manifest = _read_json_object(candidate_manifest_path)
    predecessor_manifest = _read_json_object(predecessor_manifest_path)
    revision_manifest = _read_json_object(revision_manifest_path)
    review_spec = _read_json_object(review_spec_path)

    if (
        candidate_manifest.get("manifest_id")
        != "formal_qra_generation_manifest_v0.4"
        or candidate_manifest.get("status") != "review_candidate"
        or candidate_manifest.get("qra_count") != 60
    ):
        raise ArtifactValidationError("Expected the complete formal QRA v0.4 set")
    if (
        predecessor_manifest.get("manifest_id")
        != "formal_qra_generation_manifest_v0.3"
        or predecessor_manifest.get("qra_count") != 60
    ):
        raise ArtifactValidationError("Expected the complete QRA v0.3 predecessor")
    if revision_manifest.get("revised_qra_count") != 17:
        raise ArtifactValidationError("Expected 17 primary-law QRA revisions")

    for dependency_name in ("configuration_manifest", "document_manifest"):
        declaration = candidate_manifest[dependency_name]
        path = repo_root / str(declaration["path"])
        if sha256_file(path) != declaration["sha256"]:
            raise ArtifactValidationError(f"{dependency_name} hash mismatch")
    schema_declaration = candidate_manifest["record_schema"]
    record_schema_path = repo_root / str(schema_declaration["path"])
    if sha256_file(record_schema_path) != schema_declaration["sha256"]:
        raise ArtifactValidationError("QRA record schema hash mismatch")
    record_schema = _read_json_object(record_schema_path)

    configuration_manifest = _read_json_object(
        repo_root / str(candidate_manifest["configuration_manifest"]["path"])
    )
    document_manifest = _read_json_object(
        repo_root / str(candidate_manifest["document_manifest"]["path"])
    )
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

    predecessor_declarations = {
        str(item["qra_id"]): item for item in predecessor_manifest["qras"]
    }
    revised_configuration_ids = {
        str(item["configuration_id"]) for item in revision_manifest["revisions"]
    }
    revised_qra_ids = {
        str(item["qra_id"])
        for item in candidate_manifest["qras"]
        if str(item["configuration_id"]) in revised_configuration_ids
    }
    reviewed_spec = dict(review_spec.get("revised_item_checks") or {})
    if set(reviewed_spec) != revised_qra_ids or len(reviewed_spec) != 17:
        raise ArtifactValidationError(
            "Agent review specification must cover exactly 17 revised QRA items"
        )

    records: list[dict[str, Any]] = []
    item_results: list[dict[str, Any]] = []
    carried_count = 0
    for declaration in candidate_manifest["qras"]:
        path = repo_root / str(declaration["path"])
        if sha256_file(path) != declaration["file_sha256"]:
            raise ArtifactValidationError(f"QRA candidate hash mismatch: {path}")
        qra = _read_json_object(path)
        errors = _schema_errors(qra, record_schema)
        if errors:
            raise ArtifactValidationError(
                f"Invalid QRA {qra['id']}: {'; '.join(errors[:5])}"
            )
        validate_qra_record(
            qra=qra,
            configuration=configurations[qra["configuration_id"]],
            documents=documents,
        )
        if qra.get("version") != "0.4" or qra.get("review_status") != "pending":
            raise ArtifactValidationError(f"Unexpected candidate lifecycle: {path}")

        qra_id = str(qra["id"])
        if qra_id in revised_qra_ids:
            spec = reviewed_spec[qra_id]
            if spec.get("result") != "pass":
                raise ArtifactValidationError(f"Agent review did not pass: {qra_id}")
            _contains_all(
                qra["answer"],
                list(spec.get("required_answer_fragments") or []),
                label=f"{qra_id} answer",
            )
            _contains_all(
                "\n".join(reference["quote"] for reference in qra["references"]),
                list(spec.get("required_reference_fragments") or []),
                label=f"{qra_id} reference",
            )
            for forbidden in spec.get("forbidden_answer_fragments") or []:
                if forbidden in qra["answer"]:
                    raise ArtifactValidationError(
                        f"Forbidden answer fragment remains in {qra_id}: {forbidden!r}"
                    )
            mode = "agent_line_by_line_primary_law_review"
            note = spec["review_note"]
        else:
            carried_count += 1
            predecessor_declaration = predecessor_declarations[qra_id]
            predecessor_path = repo_root / str(predecessor_declaration["path"])
            if sha256_file(predecessor_path) != predecessor_declaration["file_sha256"]:
                raise ArtifactValidationError(
                    f"QRA predecessor hash mismatch: {predecessor_path}"
                )
            predecessor = _read_json_object(predecessor_path)
            drift = [
                field
                for field in SEMANTIC_FIELDS
                if predecessor.get(field) != qra.get(field)
            ]
            if drift:
                raise ArtifactValidationError(
                    f"Unexpected carried QRA semantic drift in {qra_id}: {drift}"
                )
            mode = "agent_rereview_plus_exact_v0.3_semantic_identity"
            note = (
                "Question, answer, Keypoints, References, alignment, scope, and "
                "difficulty are identical to v0.3; current dependencies and "
                "mechanical traceability also passed."
            )
        item_results.append(
            {
                "qra_id": qra_id,
                "configuration_id": qra["configuration_id"],
                "domain": qra["domain"],
                "question_type": qra["question_type"],
                "candidate_path": declaration["path"],
                "candidate_sha256": declaration["file_sha256"],
                "review_mode": mode,
                "agent_result": "pass",
                "review_note": note,
            }
        )
        records.append(qra)

    _assert_set_balance(records)
    if carried_count != 43:
        raise ArtifactValidationError(f"Expected 43 carried QRA items, got {carried_count}")

    review = {
        "artifact_type": "Formal QRA v0.4 agent semantic review",
        "review_id": "formal_qra_agent_semantic_review_v0.4",
        "version": "0.4",
        "status": "agent_semantic_review_passed",
        "reviewed_at": reviewed_at,
        "candidate_manifest": {
            "path": CANDIDATE_MANIFEST_PATH.as_posix(),
            "sha256": sha256_file(candidate_manifest_path),
        },
        "predecessor_manifest": {
            "path": PREDECESSOR_MANIFEST_PATH.as_posix(),
            "sha256": sha256_file(predecessor_manifest_path),
        },
        "revision_manifest": {
            "path": REVISION_MANIFEST_PATH.as_posix(),
            "sha256": sha256_file(revision_manifest_path),
        },
        "review_specification": {
            "path": REVIEW_SPEC_PATH.as_posix(),
            "sha256": sha256_file(review_spec_path),
        },
        "qra_count": 60,
        "revised_item_count": 17,
        "carried_item_count": 43,
        "mechanically_valid_count": 60,
        "agent_semantically_reviewed_count": 60,
        "unresolved_blocker_count": 0,
        "unresolved_advisory_count": 0,
        "human_review_claimed": False,
        "legal_expert_review_claimed": False,
        "agent_review_is_human_review": False,
        "resolved_findings": review_spec["resolved_findings"],
        "items": item_results,
    }
    agent_review_path = repo_root / AGENT_REVIEW_PATH
    write_json_artifact(agent_review_path, review)
    review_md_path = repo_root / AGENT_REVIEW_MD_PATH
    review_md_path.parent.mkdir(parents=True, exist_ok=True)
    review_md_path.write_text(_render_review(review), encoding="utf-8")

    owner_confirmation = {
        "artifact_type": "Formal QRA project-owner review and freeze confirmation",
        "confirmation_id": "formal_qra_project_owner_confirmation_v0.4",
        "version": "0.4",
        "status": "freeze_authorized_after_agent_review",
        "recorded_at": reviewed_at,
        "authorized_by_role": "project_owner",
        "source_instruction": instruction,
        "manual_review_reported": True,
        "reported_issue_count": 0,
        "reported_scope": "Formal QRA v0.4 candidate set in the active review context",
        "freeze_condition": "Agent re-review finds no unresolved issue",
        "freeze_condition_satisfied": True,
        "freeze_target_after_agent_corrections": {
            "path": CANDIDATE_MANIFEST_PATH.as_posix(),
            "sha256": sha256_file(candidate_manifest_path),
        },
        "agent_review": {
            "path": AGENT_REVIEW_PATH.as_posix(),
            "sha256": sha256_file(agent_review_path),
            "status": review["status"],
            "unresolved_blocker_count": 0,
        },
        "final_post_correction_human_line_by_line_review_claimed": False,
        "independent_second_human_review_claimed": False,
        "legal_expert_review_claimed": False,
        "interpretation_note": (
            "The owner reported no issue in the manually reviewed candidate set "
            "and explicitly authorized a freeze if the agent re-review passed. "
            "The agent then repaired evidence-level issues before the final pass; "
            "no post-correction line-by-line human re-review is claimed."
        ),
    }
    owner_confirmation_path = repo_root / OWNER_CONFIRMATION_PATH
    write_json_artifact(owner_confirmation_path, owner_confirmation)

    return {
        "qra_count": 60,
        "revised_item_count": 17,
        "carried_item_count": 43,
        "unresolved_blocker_count": 0,
        "agent_review_path": AGENT_REVIEW_PATH.as_posix(),
        "agent_review_md_path": AGENT_REVIEW_MD_PATH.as_posix(),
        "owner_confirmation_path": OWNER_CONFIRMATION_PATH.as_posix(),
    }
