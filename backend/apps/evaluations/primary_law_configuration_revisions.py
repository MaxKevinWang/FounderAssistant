"""Prepare authority-grounded Configuration revisions for human review.

The formal Configuration set is already frozen.  This module therefore writes
new versioned drafts into a separate candidate directory, validates every
source and exact quote against frozen Document manifest v1.2, and publishes a
full-set manifest candidate.  It never freezes a Configuration or regenerates
dependent QRAs.
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


BASE_CONFIGURATION_MANIFEST_PATH = Path(
    "docs/evaluation/formal_dataset/configurations/formal_test/"
    "formal_test_configuration_manifest_v1.2.json"
)
DOCUMENT_MANIFEST_PATH = Path(
    "docs/evaluation/formal_dataset/documents/"
    "formal_test_document_manifest_v1.2.json"
)
IMPACT_MATRIX_PATH = Path(
    "docs/evaluation/formal_dataset/configurations/formal_test/"
    "primary_law_configuration_impact_matrix_v0.1.json"
)
REVISION_SPEC_PATH = Path(
    "docs/evaluation/formal_dataset/configurations/primary_law_candidates/"
    "primary_law_configuration_revision_spec_v0.1.json"
)
REVISION_PROTOCOL_PATH = Path(
    "docs/evaluation/formal_dataset/prompts/"
    "primary_law_configuration_revision_protocol_v1.0.md"
)
OUTPUT_ROOT = Path(
    "docs/evaluation/formal_dataset/configurations/primary_law_candidates"
)
GENERATION_MANIFEST_PATH = OUTPUT_ROOT / "generation_manifest_v0.1.json"
AGENT_REVIEW_PATH = OUTPUT_ROOT / "primary_law_configuration_agent_review_v0.1.json"
AGENT_REVIEW_MARKDOWN_PATH = (
    OUTPUT_ROOT / "primary_law_configuration_agent_review_v0.1_zh.md"
)
CANDIDATE_MANIFEST_PATH = Path(
    "docs/evaluation/formal_dataset/configurations/formal_test/"
    "formal_test_configuration_manifest_v1.3_candidate.json"
)


def _read_json_object(path: Path) -> dict[str, Any]:
    """Read one UTF-8 JSON object with artifact-oriented errors."""

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
    """Write a review artifact, guarding accidental semantic replacement."""

    existed = path.exists()
    if existed:
        existing = _read_json_object(path)
        if existing == payload:
            return "unchanged"
        if not overwrite:
            raise ArtifactValidationError(
                f"Review artifact differs; rerun with --overwrite after review: {path}"
            )
    write_json_artifact(path, payload)
    return "updated" if existed else "created"


def _resolve_exact_quote(text: str, needle: str) -> str | None:
    """Resolve a quote while tolerating PDF-only whitespace differences.

    Legal PDFs frequently use non-breaking spaces between numbers and currency
    units.  The specification stays readable with ordinary spaces, while the
    stored provenance retains the exact substring found in the frozen block.
    """

    if needle in text:
        return needle
    parts = needle.split()
    if not parts:
        return None
    match = re.search(r"\s+".join(re.escape(part) for part in parts), text)
    return match.group(0) if match else None


def _schema_errors(
    payload: dict[str, Any],
    record_schema: dict[str, Any],
) -> list[str]:
    """Return compact JSON Schema failures for one Configuration."""

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


def _next_version(version: str) -> str:
    """Increment a two-component Configuration version."""

    try:
        major_text, minor_text = version.split(".", maxsplit=1)
        return f"{int(major_text)}.{int(minor_text) + 1}"
    except (ValueError, AttributeError) as exc:
        raise ArtifactValidationError(f"Invalid Configuration version: {version}") from exc


def _load_documents(
    *,
    repo_root: Path,
    manifest: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Load and validate every frozen Document declared by manifest v1.2."""

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


def _load_base_configurations(
    *,
    repo_root: Path,
    manifest: dict[str, Any],
) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    """Load the 60 immutable predecessor Configurations with hash checks."""

    if (
        manifest.get("status") != "frozen"
        or manifest.get("version") != "1.2"
        or manifest.get("configuration_count") != 60
    ):
        raise ArtifactValidationError("Expected frozen Configuration manifest v1.2")
    configurations: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for declaration in manifest["configurations"]:
        path = repo_root / str(declaration["path"])
        if sha256_file(path) != declaration["file_sha256"]:
            raise ArtifactValidationError(f"Frozen Configuration changed: {path}")
        configuration = _read_json_object(path)
        configurations[str(configuration["configuration_id"])] = (
            declaration,
            configuration,
        )
    if len(configurations) != 60:
        raise ArtifactValidationError("Configuration IDs are not unique")
    return configurations


def _append_legal_facts(
    *,
    candidate: dict[str, Any],
    specification: dict[str, Any],
    impact: dict[str, Any],
    documents: dict[str, dict[str, Any]],
) -> tuple[int, int, set[str]]:
    """Append configured legal facts and exact-quote provenance records."""

    allowed_by_document = {
        str(source["document_id"]): set(source["allowed_block_ids"])
        for source in impact["primary_law_sources"]
    }
    block_maps = {
        document_id: {
            str(block["block_id"]): block
            for block in documents[document_id]["blocks"]
        }
        for document_id in allowed_by_document
    }
    next_provenance_number = max(
        int(str(item["provenance_id"])[1:]) for item in candidate["provenance"]
    ) + 1
    evidence_count = 0
    used_documents: set[str] = set()

    for addition in specification["additions"]:
        value_key = str(addition["value_key"])
        field = str(addition["field"])
        values = candidate["values"].get(value_key)
        if not isinstance(values, list):
            raise ArtifactValidationError(
                f"{candidate['configuration_id']} has no list value {value_key}"
            )
        value_index = len(values)
        values.append({field: str(addition["text"])})
        pointer = f"/values/{value_key}/{value_index}/{field}"
        evidence_items = addition["evidence"]
        support_type = "derived" if len(evidence_items) > 1 else "faithful_paraphrase"
        derivation = (
            "The configured value synthesizes the cited statutory clauses without "
            "adding a rule outside their combined scope."
            if support_type == "derived"
            else ""
        )

        for evidence in evidence_items:
            document_id = str(evidence["document_id"])
            block_id = str(evidence["block_id"])
            needle = str(evidence["needle"])
            if document_id not in allowed_by_document:
                raise ArtifactValidationError(
                    f"{candidate['configuration_id']} uses undeclared law {document_id}"
                )
            if block_id not in allowed_by_document[document_id]:
                raise ArtifactValidationError(
                    f"{candidate['configuration_id']} uses out-of-scope block {block_id}"
                )
            block = block_maps[document_id].get(block_id)
            exact_quote = (
                _resolve_exact_quote(str(block["text"]), needle)
                if block is not None
                else None
            )
            if exact_quote is None:
                raise ArtifactValidationError(
                    f"Exact legal quote not found: {document_id}/{block_id}: {needle!r}"
                )
            candidate["provenance"].append(
                {
                    "provenance_id": f"P{next_provenance_number:03d}",
                    "json_pointer": pointer,
                    "document_id": document_id,
                    "block_id": block_id,
                    "heading_path": block.get("heading_path", ""),
                    "support_type": support_type,
                    "derivation": derivation,
                    "exact_quote": exact_quote,
                }
            )
            next_provenance_number += 1
            evidence_count += 1
            used_documents.add(document_id)

    expected_documents = set(allowed_by_document)
    if used_documents != expected_documents:
        raise ArtifactValidationError(
            f"{candidate['configuration_id']} does not use every planned law source: "
            f"missing {sorted(expected_documents - used_documents)}"
        )
    return len(specification["additions"]), evidence_count, used_documents


def _revise_configuration(
    *,
    base: dict[str, Any],
    specification: dict[str, Any],
    impact: dict[str, Any],
    documents: dict[str, dict[str, Any]],
    generated_at: str,
    protocol_hash: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create one new-version draft while preserving predecessor evidence."""

    candidate = deepcopy(base)
    current_version = str(base["version"])
    candidate["version"] = _next_version(current_version)
    candidate["status"] = "development_draft"
    candidate["supersedes"] = f"{base['configuration_id']} v{current_version}"

    existing_document_ids = {
        str(item["document_id"]) for item in candidate["source_documents"]
    }
    for source in impact["primary_law_sources"]:
        document_id = str(source["document_id"])
        if document_id in existing_document_ids:
            raise ArtifactValidationError(
                f"Primary-law source already exists in {base['configuration_id']}: "
                f"{document_id}"
            )
        document = documents[document_id]
        if document["content_sha256"] != source["document_content_sha256"]:
            raise ArtifactValidationError(
                f"Frozen and reviewed content hashes differ for {document_id}"
            )
        candidate["source_documents"].append(
            {
                "document_id": document_id,
                "content_sha256": document["content_sha256"],
                "allowed_block_ids": list(source["allowed_block_ids"]),
            }
        )
        existing_document_ids.add(document_id)
    candidate["source_scope"] = (
        "multiple_documents"
        if len(candidate["source_documents"]) > 1
        else "single_document"
    )

    addition_count, evidence_count, used_documents = _append_legal_facts(
        candidate=candidate,
        specification=specification,
        impact=impact,
        documents=documents,
    )
    source_descriptions = [
        f"{item['document_id']} ({', '.join(item['allowed_block_ids'])})"
        for item in candidate["source_documents"]
    ]
    constraints = candidate["question_generation_constraints"]
    constraints["allowed_information"] = (
        "Nur die konfigurierten Werte und die folgenden erlaubten Document-Blocks: "
        + "; ".join(source_descriptions)
        + ". Bei einem Widerspruch über eine Rechtsregel oder gesetzliche Schwelle "
        "hat das eingebundene Primärrecht für den dokumentierten Abrufzeitpunkt "
        "Vorrang; zeitabhängige Regeln müssen mit ihrem Zeitraum benannt werden."
    )
    constraints["forbidden_information"] = (
        "Keine externen Fakten, keine nicht konfigurierten Rechts- oder "
        "Programmdetails, keine erfundenen Zahlen, keine Verallgemeinerung über "
        "den gesetzlichen Wortlaut hinaus und keine Behauptung zeitloser Gültigkeit."
    )
    candidate["generation_record"] = {
        "method": (
            "Deterministic primary-law augmentation with manually authored legal "
            "fact mappings and exact-quote validation"
        ),
        "generated_at": generated_at,
        "prompt_path": REVISION_PROTOCOL_PATH.as_posix(),
        "prompt_sha256": protocol_hash,
        "mechanical_validation": (
            "pending final validation: predecessor hashes, Configuration record "
            "schema v1.1, domain Schema keys, frozen Document manifest v1.2, "
            "allowed-block scope, exact quotes, provenance pointers, and full-set "
            "allocation"
        ),
        "human_review_status": "pending",
    }
    return candidate, {
        "addition_count": addition_count,
        "evidence_count": evidence_count,
        "primary_law_document_ids": sorted(used_documents),
    }


def _candidate_path(
    *,
    repo_root: Path,
    candidate: dict[str, Any],
) -> Path:
    """Return a domain-grouped, versioned review-candidate path."""

    return (
        repo_root
        / OUTPUT_ROOT
        / str(candidate["domain"])
        / f"{candidate['configuration_id']}_v{candidate['version']}.json"
    )


def _render_review_markdown(review: dict[str, Any]) -> str:
    """Render a compact Chinese review guide for the 17 drafts."""

    lines = [
        "# Primary-law Configuration 修订候选审核",
        "",
        f"- 生成日期：`{review['created_at']}`",
        f"- 候选数量：`{review['candidate_count']}`",
        f"- 新增法律事实：`{review['legal_value_addition_count']}`",
        f"- 精确原文证据：`{review['legal_evidence_count']}`",
        f"- 机械阻断错误：`{review['blocking_failure_count']}`",
        f"- 需要重点人工判断的 Configuration：`{review['attention_required_count']}`",
        "- 状态：仅通过结构、哈希、block 和 exact-quote 校验；尚未声称人工语义审核通过。",
        "",
        "## 审核表",
        "",
        "| Configuration | 版本 | 新增事实 | 法律来源 | 审核类别 | 需要注意 | 决定 |",
        "|---|---:|---:|---|---|---|---|",
    ]
    for item in review["configurations"]:
        notes = "<br>".join(item["attention_notes"]) or "-"
        sources = ", ".join(item["primary_law_document_ids"])
        lines.append(
            f"| `{item['configuration_id']}` | `{item['predecessor_version']} -> "
            f"{item['candidate_version']}` | {item['addition_count']} | {sources} | "
            f"{item['review_class']} | {notes} | 待填写：approve / revise / reject |"
        )
    lines.extend(
        [
            "",
            "## 审核方法",
            "",
            "1. 打开表中 Configuration 对应的候选 JSON。",
            "2. 查看 `values` 末尾新增的法律事实，再按同一 JSON pointer 查找新增 `provenance`。",
            "3. 打开 provenance 指向的冻结 Document 和 block，确认原文是否真正支持该概括。",
            "4. 重点检查时间口径、例外条件、跨法律推导，以及旧二手资料与现行法冲突时是否明确让 primary law 优先。",
            "5. 在本表最后一列填 `approve`、`revise: 原因` 或 `reject: 原因`。在 17 条完成前，不冻结 Configuration，也不重生 QRA。",
            "",
        ]
    )
    return "\n".join(lines)


def prepare_primary_law_configuration_revisions(
    *,
    repo_root: Path,
    generated_at: str = "2026-08-03",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Generate, validate, and document all 17 Configuration drafts."""

    repo_root = repo_root.resolve()
    base_manifest_path = repo_root / BASE_CONFIGURATION_MANIFEST_PATH
    document_manifest_path = repo_root / DOCUMENT_MANIFEST_PATH
    impact_matrix_path = repo_root / IMPACT_MATRIX_PATH
    specification_path = repo_root / REVISION_SPEC_PATH
    protocol_path = repo_root / REVISION_PROTOCOL_PATH
    base_manifest = _read_json_object(base_manifest_path)
    document_manifest = _read_json_object(document_manifest_path)
    impact_matrix = _read_json_object(impact_matrix_path)
    specification = _read_json_object(specification_path)
    protocol_hash = sha256_file(protocol_path)

    if base_manifest["document_manifest"]["path"] == DOCUMENT_MANIFEST_PATH.as_posix():
        raise ArtifactValidationError(
            "The predecessor Configuration manifest must remain bound to its old "
            "Document manifest; revisions bind v1.2 separately."
        )
    base_configurations = _load_base_configurations(
        repo_root=repo_root,
        manifest=base_manifest,
    )
    documents, _ = _load_documents(
        repo_root=repo_root,
        manifest=document_manifest,
    )

    record_schema_path = repo_root / str(base_manifest["record_schema"]["path"])
    if sha256_file(record_schema_path) != base_manifest["record_schema"]["sha256"]:
        raise ArtifactValidationError("Configuration record schema changed")
    record_schema = _read_json_object(record_schema_path)
    domain_schemas = {
        domain: _read_json_object(repo_root / path)
        for domain, path in DOMAIN_SCHEMA_PATHS.items()
    }

    impacts = {
        str(item["configuration_id"]): item
        for item in impact_matrix["matrix"]
        if item["configuration_revision_required"]
    }
    specifications = {
        str(item["configuration_id"]): item
        for item in specification["configurations"]
    }
    if set(impacts) != set(specifications) or len(impacts) != 17:
        raise ArtifactValidationError(
            "Revision spec must cover exactly the 17 impacted Configurations"
        )

    write_states: list[str] = []
    generated: dict[str, dict[str, Any]] = {}
    generation_declarations: list[dict[str, Any]] = []
    review_items: list[dict[str, Any]] = []
    for configuration_id in [
        str(item["configuration_id"])
        for item in base_manifest["configurations"]
        if str(item["configuration_id"]) in impacts
    ]:
        base_declaration, base = base_configurations[configuration_id]
        candidate, stats = _revise_configuration(
            base=base,
            specification=specifications[configuration_id],
            impact=impacts[configuration_id],
            documents=documents,
            generated_at=generated_at,
            protocol_hash=protocol_hash,
        )
        candidate["generation_record"]["mechanical_validation"] = (
            "passed: predecessor hashes, Configuration record schema v1.1, domain "
            "Schema keys, frozen Document manifest v1.2, allowed-block scope, exact "
            "quotes, provenance pointers, and complete evidence for every new value"
        )
        errors = _schema_errors(candidate, record_schema)
        if errors:
            raise ArtifactValidationError(
                f"Invalid candidate {configuration_id}: {'; '.join(errors[:5])}"
            )
        validate_configuration_record(
            configuration=candidate,
            schema=domain_schemas[candidate["domain"]],
            documents=documents,
        )
        output_path = _candidate_path(repo_root=repo_root, candidate=candidate)
        write_states.append(
            _write_candidate(output_path, candidate, overwrite=overwrite)
        )
        generated[configuration_id] = candidate
        declaration = {
            "configuration_id": configuration_id,
            "predecessor_version": base["version"],
            "candidate_version": candidate["version"],
            "status": candidate["status"],
            "domain": candidate["domain"],
            "target_question_type": candidate["target_question_type"],
            "answerability": candidate["answerability"],
            "path": output_path.relative_to(repo_root).as_posix(),
            "file_sha256": sha256_file(output_path),
            "predecessor_path": base_declaration["path"],
            "predecessor_sha256": base_declaration["file_sha256"],
            "primary_law_document_ids": stats["primary_law_document_ids"],
            "legal_value_addition_count": stats["addition_count"],
            "legal_evidence_count": stats["evidence_count"],
        }
        generation_declarations.append(declaration)
        review_items.append(
            {
                "configuration_id": configuration_id,
                "predecessor_version": base["version"],
                "candidate_version": candidate["version"],
                "candidate_path": declaration["path"],
                "candidate_sha256": declaration["file_sha256"],
                "review_class": specifications[configuration_id]["review_class"],
                "attention_notes": specifications[configuration_id][
                    "attention_notes"
                ],
                "addition_count": stats["addition_count"],
                "evidence_count": stats["evidence_count"],
                "primary_law_document_ids": stats["primary_law_document_ids"],
                "mechanical_result": "pass",
                "human_decision": "pending",
            }
        )

    generation_manifest = {
        "manifest_id": "primary_law_configuration_generation_manifest_v0.1",
        "version": "0.1",
        "status": "review_candidate",
        "generated_at": generated_at,
        "predecessor_configuration_manifest": {
            "path": BASE_CONFIGURATION_MANIFEST_PATH.as_posix(),
            "sha256": sha256_file(base_manifest_path),
        },
        "document_manifest": {
            "path": DOCUMENT_MANIFEST_PATH.as_posix(),
            "sha256": sha256_file(document_manifest_path),
        },
        "impact_matrix": {
            "path": IMPACT_MATRIX_PATH.as_posix(),
            "sha256": sha256_file(impact_matrix_path),
        },
        "revision_spec": {
            "path": REVISION_SPEC_PATH.as_posix(),
            "sha256": sha256_file(specification_path),
        },
        "revision_protocol": {
            "path": REVISION_PROTOCOL_PATH.as_posix(),
            "sha256": protocol_hash,
        },
        "candidate_count": len(generation_declarations),
        "legal_value_addition_count": sum(
            item["legal_value_addition_count"]
            for item in generation_declarations
        ),
        "legal_evidence_count": sum(
            item["legal_evidence_count"] for item in generation_declarations
        ),
        "candidates": generation_declarations,
    }
    generation_manifest_path = repo_root / GENERATION_MANIFEST_PATH
    write_states.append(
        _write_candidate(
            generation_manifest_path,
            generation_manifest,
            overwrite=overwrite,
        )
    )

    review = {
        "review_id": "primary_law_configuration_agent_review_v0.1",
        "version": "0.1",
        "status": "mechanically_valid_semantic_review_pending",
        "created_at": generated_at,
        "generation_manifest": {
            "path": GENERATION_MANIFEST_PATH.as_posix(),
            "sha256": sha256_file(generation_manifest_path),
        },
        "candidate_count": len(review_items),
        "legal_value_addition_count": generation_manifest[
            "legal_value_addition_count"
        ],
        "legal_evidence_count": generation_manifest["legal_evidence_count"],
        "blocking_failure_count": 0,
        "attention_required_count": sum(
            bool(item["attention_notes"]) for item in review_items
        ),
        "human_review_state": "pending",
        "semantic_review_claimed": False,
        "configurations": review_items,
    }
    review_path = repo_root / AGENT_REVIEW_PATH
    write_states.append(
        _write_candidate(review_path, review, overwrite=overwrite)
    )
    markdown_path = repo_root / AGENT_REVIEW_MARKDOWN_PATH
    markdown = _render_review_markdown(review)
    if markdown_path.exists() and markdown_path.read_text(encoding="utf-8") != markdown:
        if not overwrite:
            raise ArtifactValidationError(
                f"Review Markdown differs; rerun with --overwrite: {markdown_path}"
            )
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(markdown, encoding="utf-8")

    full_set_declarations: list[dict[str, Any]] = []
    for base_declaration in base_manifest["configurations"]:
        configuration_id = str(base_declaration["configuration_id"])
        candidate = generated.get(configuration_id)
        if candidate is None:
            carried = deepcopy(base_declaration)
            carried["resolution"] = "carried_forward_if_frozen"
            full_set_declarations.append(carried)
            continue
        generated_declaration = next(
            item
            for item in generation_declarations
            if item["configuration_id"] == configuration_id
        )
        full_set_declarations.append(
            {
                "configuration_id": configuration_id,
                "version": candidate["version"],
                "resolution": "primary_law_review_candidate",
                "domain": candidate["domain"],
                "target_question_type": candidate["target_question_type"],
                "answerability": candidate["answerability"],
                "path": generated_declaration["path"],
                "file_sha256": generated_declaration["file_sha256"],
                "freeze_record_path": None,
                "freeze_record_sha256": None,
                "source_document_ids": [
                    source["document_id"] for source in candidate["source_documents"]
                ],
                "provenance_count": len(candidate["provenance"]),
                "review_status": "mechanically_valid_semantic_review_pending",
            }
        )
    candidate_manifest = {
        "manifest_id": "formal_test_configuration_manifest_v1.3_candidate",
        "version": "1.3-candidate",
        "status": "review_candidate",
        "created_at": generated_at,
        "supersedes_if_frozen": base_manifest["manifest_id"],
        "predecessor_manifest": {
            "path": BASE_CONFIGURATION_MANIFEST_PATH.as_posix(),
            "sha256": sha256_file(base_manifest_path),
        },
        "document_manifest": {
            "path": DOCUMENT_MANIFEST_PATH.as_posix(),
            "sha256": sha256_file(document_manifest_path),
        },
        "record_schema": base_manifest["record_schema"],
        "revision_protocol": generation_manifest["revision_protocol"],
        "generation_manifest": {
            "path": GENERATION_MANIFEST_PATH.as_posix(),
            "sha256": sha256_file(generation_manifest_path),
        },
        "agent_review": {
            "path": AGENT_REVIEW_PATH.as_posix(),
            "sha256": sha256_file(review_path),
            "status": review["status"],
        },
        "freeze_guard": (
            "Do not freeze until all 17 primary-law revisions receive explicit "
            "approve/revise/reject decisions and every requested revision is rerun."
        ),
        "configuration_count": 60,
        "revised_candidate_count": 17,
        "carried_forward_count": 43,
        "answerable_count": base_manifest["answerable_count"],
        "unanswerable_count": base_manifest["unanswerable_count"],
        "configurations": full_set_declarations,
    }
    candidate_manifest_path = repo_root / CANDIDATE_MANIFEST_PATH
    write_states.append(
        _write_candidate(
            candidate_manifest_path,
            candidate_manifest,
            overwrite=overwrite,
        )
    )

    return {
        "candidate_count": len(generated),
        "legal_value_addition_count": generation_manifest[
            "legal_value_addition_count"
        ],
        "legal_evidence_count": generation_manifest["legal_evidence_count"],
        "attention_required_count": review["attention_required_count"],
        "generation_manifest_path": GENERATION_MANIFEST_PATH.as_posix(),
        "review_path": AGENT_REVIEW_MARKDOWN_PATH.as_posix(),
        "candidate_manifest_path": CANDIDATE_MANIFEST_PATH.as_posix(),
        "write_states": write_states,
    }
