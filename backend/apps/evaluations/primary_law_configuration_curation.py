"""Semantically curate primary-law Configuration candidates before freezing.

The preceding generation round appends exact-quote-backed primary-law facts but
intentionally preserves inherited values. This module applies an explicit,
reviewable removal/update specification, repairs provenance pointers, prunes
unused source scope, and emits a separate candidate round. It never mutates the
v0.1 candidates or any frozen artifact.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from apps.evaluations.corpus_paths import resolve_corpus_path
import re
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


INPUT_ROOT = Path(
    "docs/evaluation/formal_dataset/configurations/primary_law_candidates"
)
INPUT_GENERATION_MANIFEST_PATH = INPUT_ROOT / "generation_manifest_v0.1.json"
INPUT_AGENT_REVIEW_PATH = (
    INPUT_ROOT / "primary_law_configuration_agent_review_v0.1.json"
)
CURATION_SPEC_PATH = (
    INPUT_ROOT / "primary_law_configuration_semantic_curation_spec_v0.2.json"
)
CURATION_PROTOCOL_PATH = Path(
    "docs/evaluation/formal_dataset/prompts/"
    "primary_law_configuration_semantic_curation_protocol_v0.2.md"
)
OUTPUT_ROOT = INPUT_ROOT / "reviewed_v0.2"
CURATION_MANIFEST_PATH = OUTPUT_ROOT / "generation_manifest_v0.2.json"
AGENT_REVIEW_PATH = OUTPUT_ROOT / "agent_semantic_review_v0.2.json"
AGENT_REVIEW_MARKDOWN_PATH = OUTPUT_ROOT / "agent_semantic_review_v0.2_zh.md"
FULL_CANDIDATE_MANIFEST_PATH = Path(
    "docs/evaluation/formal_dataset/configurations/formal_test/"
    "formal_test_configuration_manifest_v1.3_semantic_candidate.json"
)

VALUE_POINTER_RE = re.compile(r"^/values/([^/]+)/(\d+)$")
PROVENANCE_POINTER_RE = re.compile(r"^/values/([^/]+)/(\d+)/([^/]+)$")

LEGAL_SCOPE_SUFFIX = (
    "Bei einem Widerspruch über eine Rechtsregel oder gesetzliche Schwelle hat "
    "das eingebundene Primärrecht für den dokumentierten Abrufzeitpunkt Vorrang; "
    "zeitabhängige Regeln müssen mit ihrem Zeitraum benannt werden."
)


def _read_json_object(path: Path) -> dict[str, Any]:
    """Read one UTF-8 JSON object with an artifact-oriented error."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(f"Cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ArtifactValidationError(f"Expected a JSON object: {path}")
    return payload


def _write_candidate(
    path: Path,
    payload: dict[str, Any],
    *,
    overwrite: bool,
) -> str:
    """Write a review artifact, requiring an explicit flag for replacement."""

    if path.exists():
        if _read_json_object(path) == payload:
            return "unchanged"
        if not overwrite:
            raise ArtifactValidationError(
                f"Review artifact differs; rerun with overwrite enabled: {path}"
            )
        path.unlink()
        state = "replaced"
    else:
        state = "created"
    write_json_artifact(path, payload)
    return state


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


def _value_text(value: dict[str, Any]) -> str:
    """Return the single domain-schema leaf stored in a configured value."""

    if len(value) != 1:
        raise ArtifactValidationError(
            f"Expected one field in Configuration value, got {sorted(value)}"
        )
    return str(next(iter(value.values())))


def _removal_indexes(
    *,
    candidate: dict[str, Any],
    removals: list[dict[str, Any]],
) -> dict[str, set[int]]:
    """Validate exact removal selectors and group their array indexes."""

    grouped: dict[str, set[int]] = {}
    for removal in removals:
        pointer = str(removal.get("json_pointer") or "")
        match = VALUE_POINTER_RE.fullmatch(pointer)
        if match is None:
            raise ArtifactValidationError(f"Invalid removal pointer: {pointer!r}")
        value_key, raw_index = match.groups()
        index = int(raw_index)
        values = candidate.get("values", {}).get(value_key)
        if not isinstance(values, list) or index >= len(values):
            raise ArtifactValidationError(
                f"Removal pointer does not resolve in "
                f"{candidate['configuration_id']}: {pointer}"
            )
        expected = str(removal.get("expected_text") or "")
        actual = _value_text(values[index])
        if actual != expected:
            raise ArtifactValidationError(
                f"Removal text changed for {candidate['configuration_id']} "
                f"{pointer}: expected {expected!r}, got {actual!r}"
            )
        indexes = grouped.setdefault(value_key, set())
        if index in indexes:
            raise ArtifactValidationError(
                f"Duplicate removal pointer for {candidate['configuration_id']}: "
                f"{pointer}"
            )
        indexes.add(index)
    return grouped


def _remove_values_and_reindex_provenance(
    *,
    candidate: dict[str, Any],
    removals: list[dict[str, Any]],
) -> int:
    """Remove selected values and make every surviving pointer contiguous."""

    grouped = _removal_indexes(candidate=candidate, removals=removals)
    if not grouped:
        return 0

    retained_provenance: list[dict[str, Any]] = []
    for provenance in candidate["provenance"]:
        pointer = str(provenance["json_pointer"])
        match = PROVENANCE_POINTER_RE.fullmatch(pointer)
        if match is None:
            raise ArtifactValidationError(
                f"Unsupported provenance pointer in "
                f"{candidate['configuration_id']}: {pointer}"
            )
        value_key, raw_index, field = match.groups()
        old_index = int(raw_index)
        removed = grouped.get(value_key, set())
        if old_index in removed:
            continue
        new_index = old_index - sum(index < old_index for index in removed)
        revised = deepcopy(provenance)
        revised["json_pointer"] = f"/values/{value_key}/{new_index}/{field}"
        retained_provenance.append(revised)

    for value_key, indexes in grouped.items():
        candidate["values"][value_key] = [
            value
            for index, value in enumerate(candidate["values"][value_key])
            if index not in indexes
        ]
        if not candidate["values"][value_key]:
            raise ArtifactValidationError(
                f"Curation removed every {value_key} value from "
                f"{candidate['configuration_id']}"
            )
    candidate["provenance"] = retained_provenance
    return sum(len(indexes) for indexes in grouped.values())


def _apply_constraint_updates(
    *,
    candidate: dict[str, Any],
    updates: dict[str, Any],
) -> list[str]:
    """Apply only declared question-generation constraint replacements."""

    constraints = candidate["question_generation_constraints"]
    changed: list[str] = []
    for key, value in updates.items():
        if key not in constraints:
            raise ArtifactValidationError(
                f"Unknown constraint {key!r} for {candidate['configuration_id']}"
            )
        if not isinstance(value, type(constraints[key])):
            raise ArtifactValidationError(
                f"Constraint type changed for {candidate['configuration_id']}: {key}"
            )
        if constraints[key] != value:
            constraints[key] = value
            changed.append(key)
    return changed


def _allowed_information(source_documents: list[dict[str, Any]]) -> str:
    """Render the narrowed human-readable source boundary."""

    source_text = "; ".join(
        f"{source['document_id']} ({', '.join(source['allowed_block_ids'])})"
        for source in source_documents
    )
    return (
        "Nur die konfigurierten Werte und die folgenden erlaubten "
        f"Document-Blocks: {source_text}. {LEGAL_SCOPE_SUFFIX}"
    )


def _prune_source_scope(candidate: dict[str, Any]) -> dict[str, int]:
    """Drop source blocks that no longer support a retained configured value."""

    used_pairs = {
        (str(item["document_id"]), str(item["block_id"]))
        for item in candidate["provenance"]
    }
    original_sources = candidate["source_documents"]
    pruned_sources: list[dict[str, Any]] = []
    removed_blocks = 0
    for source in original_sources:
        document_id = str(source["document_id"])
        retained_blocks = [
            block_id
            for block_id in source["allowed_block_ids"]
            if (document_id, block_id) in used_pairs
        ]
        removed_blocks += len(source["allowed_block_ids"]) - len(retained_blocks)
        if retained_blocks:
            revised = deepcopy(source)
            revised["allowed_block_ids"] = retained_blocks
            pruned_sources.append(revised)
    if not pruned_sources:
        raise ArtifactValidationError(
            f"Curation removed every source from {candidate['configuration_id']}"
        )
    removed_sources = len(original_sources) - len(pruned_sources)
    candidate["source_documents"] = pruned_sources
    candidate["source_scope"] = (
        "single_document" if len(pruned_sources) == 1 else "multiple_documents"
    )
    candidate["question_generation_constraints"]["allowed_information"] = (
        _allowed_information(pruned_sources)
    )
    return {
        "removed_source_count": removed_sources,
        "removed_block_count": removed_blocks,
    }


def _load_documents(
    *,
    repo_root: Path,
    manifest: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Load and hash-check every frozen Document in manifest v1.2."""

    if manifest.get("status") != "frozen" or manifest.get("version") != "1.2":
        raise ArtifactValidationError("Expected frozen Document manifest v1.2")
    documents: dict[str, dict[str, Any]] = {}
    for declaration in manifest["documents"]:
        path = repo_root / str(declaration["path"])
        if sha256_file(path) != declaration["file_sha256"]:
            raise ArtifactValidationError(f"Frozen Document changed: {path}")
        document = _read_json_object(path)
        validate_document_record(
            document=document,
            source_path=resolve_corpus_path(repo_root, str(document["source"]["path"])),
        )
        documents[str(document["document_id"])] = document
    return documents


def _render_review_markdown(review: dict[str, Any]) -> str:
    """Render a concise Chinese audit sheet for the curated candidate round."""

    lines = [
        "# Primary-law Configuration 语义清理审计 v0.2",
        "",
        f"- 日期：`{review['reviewed_at']}`",
        f"- Configuration：`{review['candidate_count']}`",
        f"- 删除旧值：`{review['removed_value_count']}`",
        f"- 修改约束：`{review['constraint_update_count']}`",
        f"- 未解决阻断项：`{review['unresolved_blocker_count']}`",
        "- 人工逐条法律审核：**未声称完成**",
        "",
        "## 逐项结果",
        "",
        "| Configuration | 决定 | 删除值 | 约束更新 | 范围清理 | Agent 结论 |",
        "|---|---|---:|---|---|---|",
    ]
    for item in review["configurations"]:
        constraints = ", ".join(item["updated_constraints"]) or "-"
        pruning = (
            f"sources -{item['removed_source_count']}; "
            f"blocks -{item['removed_block_count']}"
        )
        lines.append(
            f"| `{item['configuration_id']}` | {item['decision']} | "
            f"{item['removed_value_count']} | {constraints} | {pruning} | pass |"
        )
    lines.extend(
        [
            "",
            "## 冻结解释",
            "",
            "本审计仅表示自动语义检查未发现剩余的内部冲突、截断值或明显范围错误。",
            "它不等同于律师或人工逐条法律审核。冻结仍需独立记录项目负责人授权，",
            "并在授权记录中保留 `human_line_by_line_review_claimed: false`。",
            "",
        ]
    )
    return "\n".join(lines)


def curate_primary_law_configurations(
    *,
    repo_root: Path,
    reviewed_at: str = "2026-08-03",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Apply the v0.2 semantic change set and emit reviewed candidates."""

    repo_root = repo_root.resolve()
    generation_path = repo_root / INPUT_GENERATION_MANIFEST_PATH
    input_review_path = repo_root / INPUT_AGENT_REVIEW_PATH
    specification_path = repo_root / CURATION_SPEC_PATH
    protocol_path = repo_root / CURATION_PROTOCOL_PATH
    generation = _read_json_object(generation_path)
    input_review = _read_json_object(input_review_path)
    specification = _read_json_object(specification_path)

    if generation.get("candidate_count") != 17:
        raise ArtifactValidationError("Expected 17 primary-law input candidates")
    if input_review.get("semantic_review_claimed") is not False:
        raise ArtifactValidationError("Input review lifecycle is not the expected v0.1")
    if specification.get("version") != "0.2":
        raise ArtifactValidationError("Expected semantic curation specification v0.2")

    declarations = {
        str(item["configuration_id"]): item for item in generation["candidates"]
    }
    patches = {
        str(item["configuration_id"]): item
        for item in specification["configurations"]
    }
    if set(declarations) != set(patches) or len(patches) != 17:
        raise ArtifactValidationError(
            "Semantic curation must decide all 17 generated Configurations"
        )
    valid_decisions = {"approve_as_generated", "revise"}
    if {item.get("decision") for item in patches.values()} - valid_decisions:
        raise ArtifactValidationError("Unknown semantic curation decision")

    base_manifest_path = repo_root / generation["predecessor_configuration_manifest"][
        "path"
    ]
    if (
        sha256_file(base_manifest_path)
        != generation["predecessor_configuration_manifest"]["sha256"]
    ):
        raise ArtifactValidationError("Predecessor Configuration manifest changed")
    base_manifest = _read_json_object(base_manifest_path)
    document_manifest_path = repo_root / generation["document_manifest"]["path"]
    if sha256_file(document_manifest_path) != generation["document_manifest"]["sha256"]:
        raise ArtifactValidationError("Document manifest changed after generation")
    document_manifest = _read_json_object(document_manifest_path)
    documents = _load_documents(repo_root=repo_root, manifest=document_manifest)

    record_schema_path = repo_root / str(base_manifest["record_schema"]["path"])
    if sha256_file(record_schema_path) != base_manifest["record_schema"]["sha256"]:
        raise ArtifactValidationError("Configuration record schema changed")
    record_schema = _read_json_object(record_schema_path)
    domain_schemas = {
        domain: _read_json_object(repo_root / path)
        for domain, path in DOMAIN_SCHEMA_PATHS.items()
    }

    write_states: list[str] = []
    reviewed_declarations: list[dict[str, Any]] = []
    review_items: list[dict[str, Any]] = []
    reviewed_by_id: dict[str, dict[str, Any]] = {}
    protocol_hash = sha256_file(protocol_path)
    for input_declaration in generation["candidates"]:
        configuration_id = str(input_declaration["configuration_id"])
        input_path = repo_root / str(input_declaration["path"])
        if sha256_file(input_path) != input_declaration["file_sha256"]:
            raise ArtifactValidationError(f"Input candidate changed: {input_path}")
        candidate = _read_json_object(input_path)
        patch = patches[configuration_id]
        revised = deepcopy(candidate)
        removed_count = _remove_values_and_reindex_provenance(
            candidate=revised,
            removals=list(patch.get("remove_values") or []),
        )
        updated_constraints = _apply_constraint_updates(
            candidate=revised,
            updates=dict(patch.get("constraint_updates") or {}),
        )
        pruning = _prune_source_scope(revised)
        revised["generation_record"] = {
            "method": (
                "Deterministic primary-law augmentation followed by explicit "
                "agent semantic curation v0.2"
            ),
            "generated_at": reviewed_at,
            "prompt_path": CURATION_PROTOCOL_PATH.as_posix(),
            "prompt_sha256": protocol_hash,
            "mechanical_validation": (
                "passed: exact removal selectors, contiguous values, remapped "
                "provenance pointers, pruned source scope, frozen Document hashes, "
                "record schema, domain Schema, and complete value evidence"
            ),
            "human_review_status": "pending",
        }

        errors = _schema_errors(revised, record_schema)
        if errors:
            raise ArtifactValidationError(
                f"Invalid curated candidate {configuration_id}: "
                f"{'; '.join(errors[:5])}"
            )
        validate_configuration_record(
            configuration=revised,
            schema=domain_schemas[revised["domain"]],
            documents=documents,
        )
        output_path = (
            repo_root
            / OUTPUT_ROOT
            / str(revised["domain"])
            / f"{configuration_id}_v{revised['version']}.json"
        )
        write_states.append(
            _write_candidate(output_path, revised, overwrite=overwrite)
        )
        output_declaration = {
            "configuration_id": configuration_id,
            "version": revised["version"],
            "status": revised["status"],
            "domain": revised["domain"],
            "target_question_type": revised["target_question_type"],
            "answerability": revised["answerability"],
            "path": output_path.relative_to(repo_root).as_posix(),
            "file_sha256": sha256_file(output_path),
            "input_path": input_declaration["path"],
            "input_sha256": input_declaration["file_sha256"],
            "predecessor_path": input_declaration["predecessor_path"],
            "predecessor_sha256": input_declaration["predecessor_sha256"],
            "decision": patch["decision"],
            "removed_value_count": removed_count,
            "updated_constraints": updated_constraints,
            **pruning,
        }
        reviewed_declarations.append(output_declaration)
        reviewed_by_id[configuration_id] = revised
        review_items.append(
            {
                "configuration_id": configuration_id,
                "version": revised["version"],
                "decision": patch["decision"],
                "removed_value_count": removed_count,
                "updated_constraints": updated_constraints,
                **pruning,
                "candidate_path": output_declaration["path"],
                "candidate_sha256": output_declaration["file_sha256"],
                "agent_semantic_result": "pass",
                "human_review_state": "pending",
            }
        )

    curation_manifest = {
        "manifest_id": "primary_law_configuration_semantic_curation_manifest_v0.2",
        "version": "0.2",
        "status": "agent_semantic_review_passed",
        "reviewed_at": reviewed_at,
        "input_generation_manifest": {
            "path": INPUT_GENERATION_MANIFEST_PATH.as_posix(),
            "sha256": sha256_file(generation_path),
        },
        "input_agent_review": {
            "path": INPUT_AGENT_REVIEW_PATH.as_posix(),
            "sha256": sha256_file(input_review_path),
        },
        "curation_specification": {
            "path": CURATION_SPEC_PATH.as_posix(),
            "sha256": sha256_file(specification_path),
        },
        "curation_protocol": {
            "path": CURATION_PROTOCOL_PATH.as_posix(),
            "sha256": protocol_hash,
        },
        "candidate_count": len(reviewed_declarations),
        "removed_value_count": sum(
            item["removed_value_count"] for item in reviewed_declarations
        ),
        "constraint_update_count": sum(
            len(item["updated_constraints"]) for item in reviewed_declarations
        ),
        "human_line_by_line_review_claimed": False,
        "candidates": reviewed_declarations,
    }
    curation_manifest_path = repo_root / CURATION_MANIFEST_PATH
    write_states.append(
        _write_candidate(
            curation_manifest_path,
            curation_manifest,
            overwrite=overwrite,
        )
    )

    review = {
        "review_id": "primary_law_configuration_agent_semantic_review_v0.2",
        "version": "0.2",
        "status": "agent_semantic_review_passed",
        "reviewed_at": reviewed_at,
        "curation_manifest": {
            "path": CURATION_MANIFEST_PATH.as_posix(),
            "sha256": sha256_file(curation_manifest_path),
        },
        "candidate_count": len(review_items),
        "removed_value_count": curation_manifest["removed_value_count"],
        "constraint_update_count": curation_manifest[
            "constraint_update_count"
        ],
        "unresolved_blocker_count": 0,
        "agent_semantic_review_claimed": True,
        "human_line_by_line_review_claimed": False,
        "human_review_state": "pending",
        "configurations": review_items,
    }
    review_path = repo_root / AGENT_REVIEW_PATH
    write_states.append(_write_candidate(review_path, review, overwrite=overwrite))
    review_markdown_path = repo_root / AGENT_REVIEW_MARKDOWN_PATH
    markdown = _render_review_markdown(review)
    if review_markdown_path.exists():
        existing = review_markdown_path.read_text(encoding="utf-8")
        if existing != markdown and not overwrite:
            raise ArtifactValidationError(
                f"Review Markdown differs; rerun with overwrite: "
                f"{review_markdown_path}"
            )
    review_markdown_path.parent.mkdir(parents=True, exist_ok=True)
    review_markdown_path.write_text(markdown, encoding="utf-8")

    full_declarations: list[dict[str, Any]] = []
    output_by_id = {
        item["configuration_id"]: item for item in reviewed_declarations
    }
    for base_declaration in base_manifest["configurations"]:
        configuration_id = str(base_declaration["configuration_id"])
        output = output_by_id.get(configuration_id)
        if output is None:
            carried = deepcopy(base_declaration)
            carried["resolution"] = "carried_forward"
            full_declarations.append(carried)
            continue
        candidate = reviewed_by_id[configuration_id]
        full_declarations.append(
            {
                "configuration_id": configuration_id,
                "version": candidate["version"],
                "resolution": "primary_law_semantic_candidate",
                "domain": candidate["domain"],
                "target_question_type": candidate["target_question_type"],
                "answerability": candidate["answerability"],
                "path": output["path"],
                "file_sha256": output["file_sha256"],
                "freeze_record_path": None,
                "freeze_record_sha256": None,
                "source_document_ids": [
                    source["document_id"] for source in candidate["source_documents"]
                ],
                "provenance_count": len(candidate["provenance"]),
                "review_status": "agent_semantic_review_passed_human_pending",
            }
        )

    full_manifest = {
        "manifest_id": "formal_test_configuration_manifest_v1.3_semantic_candidate",
        "version": "1.3-candidate",
        "status": "review_candidate",
        "created_at": reviewed_at,
        "supersedes_if_frozen": base_manifest["manifest_id"],
        "predecessor_manifest": {
            "path": base_manifest_path.relative_to(repo_root).as_posix(),
            "sha256": sha256_file(base_manifest_path),
        },
        "document_manifest": {
            "path": document_manifest_path.relative_to(repo_root).as_posix(),
            "sha256": sha256_file(document_manifest_path),
        },
        "record_schema": base_manifest["record_schema"],
        "semantic_curation_manifest": {
            "path": CURATION_MANIFEST_PATH.as_posix(),
            "sha256": sha256_file(curation_manifest_path),
        },
        "agent_semantic_review": {
            "path": AGENT_REVIEW_PATH.as_posix(),
            "sha256": sha256_file(review_path),
            "status": review["status"],
            "human_line_by_line_review_claimed": False,
        },
        "freeze_guard": (
            "Requires zero unresolved semantic blockers and a separate explicit "
            "project-owner authorization that does not claim human line-by-line "
            "legal review."
        ),
        "configuration_count": 60,
        "revised_candidate_count": 17,
        "carried_forward_count": 43,
        "answerable_count": base_manifest["answerable_count"],
        "unanswerable_count": base_manifest["unanswerable_count"],
        "configurations": full_declarations,
    }
    full_manifest_path = repo_root / FULL_CANDIDATE_MANIFEST_PATH
    write_states.append(
        _write_candidate(full_manifest_path, full_manifest, overwrite=overwrite)
    )

    return {
        "candidate_count": len(reviewed_declarations),
        "removed_value_count": curation_manifest["removed_value_count"],
        "constraint_update_count": curation_manifest[
            "constraint_update_count"
        ],
        "unresolved_blocker_count": review["unresolved_blocker_count"],
        "curation_manifest_path": CURATION_MANIFEST_PATH.as_posix(),
        "review_path": AGENT_REVIEW_MARKDOWN_PATH.as_posix(),
        "candidate_manifest_path": FULL_CANDIDATE_MANIFEST_PATH.as_posix(),
        "created_count": write_states.count("created"),
        "replaced_count": write_states.count("replaced"),
        "unchanged_count": write_states.count("unchanged"),
    }
