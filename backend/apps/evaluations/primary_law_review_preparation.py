"""Audit primary-law Documents D and prepare review-only evaluation artifacts.

The frozen formal test set must not change merely because new legal sources
exist. This module therefore performs three reproducible, non-freezing steps:

1. validate every primary-law Document D candidate and selected legal fact;
2. build an additive ``v1.2`` Document manifest candidate;
3. resolve an explicit Configuration impact plan to exact Document block IDs.

Human approval, Configuration rewriting, QRA regeneration, and freezing remain
separate later steps.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from apps.evaluations.rageval_artifacts import (
    ArtifactValidationError,
    sha256_file,
    validate_document_record,
    write_json_artifact,
)


SELECTION_PATH = Path(
    "docs/evaluation/formal_dataset/source_selection/"
    "primary_law_document_selection_v0.1.json"
)
IMPACT_PLAN_PATH = Path(
    "docs/evaluation/formal_dataset/source_selection/"
    "primary_law_configuration_impact_plan_v0.1.json"
)
BASE_DOCUMENT_MANIFEST_PATH = Path(
    "docs/evaluation/formal_dataset/documents/"
    "formal_test_document_manifest_v1.1.json"
)
CONFIGURATION_MANIFEST_PATH = Path(
    "docs/evaluation/formal_dataset/configurations/formal_test/"
    "formal_test_configuration_manifest_v1.2.json"
)
CANDIDATE_MANIFEST_PATH = Path(
    "docs/evaluation/formal_dataset/documents/"
    "formal_test_document_manifest_v1.2_candidate.json"
)
AGENT_REVIEW_JSON_PATH = Path(
    "docs/evaluation/formal_dataset/documents/primary_law_candidates/"
    "primary_law_agent_review_v0.1.json"
)
AGENT_REVIEW_MARKDOWN_PATH = Path(
    "docs/evaluation/formal_dataset/documents/primary_law_candidates/"
    "primary_law_agent_review_v0.1_zh.md"
)
IMPACT_MATRIX_JSON_PATH = Path(
    "docs/evaluation/formal_dataset/configurations/formal_test/"
    "primary_law_configuration_impact_matrix_v0.1.json"
)
IMPACT_MATRIX_MARKDOWN_PATH = Path(
    "docs/evaluation/formal_dataset/configurations/formal_test/"
    "primary_law_configuration_impact_matrix_v0.1_zh.md"
)


# These checks are deliberately narrow. They verify that legally decisive text
# survived extraction; they do not replace legal interpretation or human review.
KEY_FACT_CHECKS: dict[str, tuple[tuple[str, str, str], ...]] = {
    "D-LFS-013": (
        ("§ 14", "Unternehmer ist eine natürliche oder juristische Person", "entrepreneur definition"),
        ("§ 705", "durch den Abschluss des Gesellschaftsvertrags errichtet", "GbR formation"),
        ("§ 721", "als Gesamtschuldner persönlich", "GbR personal liability"),
    ),
    "D-LFS-014": (
        ("§ 1", "Kaufmann im Sinne dieses Gesetzbuchs", "merchant definition"),
        ("§ 105", "offene Handelsgesellschaft", "OHG definition"),
        ("§ 126", "als Gesamtschuldner persönlich", "OHG personal liability"),
        ("§ 161", "Kommanditgesellschaft", "KG definition"),
        ("§ 241a", "800 000 Euro", "bookkeeping revenue threshold"),
        ("§ 241a", "80 000 Euro", "bookkeeping profit threshold"),
    ),
    "D-LFS-015": (
        ("§ 5", "fünfundzwanzigtausend Euro", "GmbH minimum capital"),
        ("§ 5a", "UG (haftungsbeschränkt)", "UG designation"),
        ("§ 11", "haften die Handelnden persönlich", "pre-registration liability"),
        ("§ 13", "nur das Gesellschaftsvermögen", "GmbH limited liability"),
    ),
    "D-LFS-016": (
        ("§ 1", "Angehörige Freier Berufe", "PartG eligibility"),
        ("§ 8", "haften den Gläubigern", "PartG liability"),
    ),
    "D-LFS-017": (
        ("§ 7", "fünfzigtausend Euro", "AG minimum capital"),
        ("§ 76", "unter eigener Verantwortung", "management by board"),
    ),
    "D-LFS-018": (
        ("§ 4", "mindestens drei", "cooperative member minimum"),
        ("§ 17", "Juristische Person", "cooperative legal personality"),
    ),
    "D-LFS-019": (
        ("§ 14", "muss dies der zuständigen Behörde gleichzeitig anzeigen", "trade notification"),
        ("§ 35", "Gewerbeuntersagung", "prohibition for unreliability"),
    ),
    "D-LFS-020": (
        ("§ 1", "in der Handwerksrolle eingetragenen", "craft-register requirement"),
        ("Anlage A", "Friseure", "hairdresser classification"),
        ("Anlage B", "Kosmetiker", "cosmetician classification"),
        ("Anlage B", "Bodenleger", "floor-layer classification"),
    ),
    "D-LFS-021": (
        ("§ 138", "innerhalb eines Monats", "tax notification deadline"),
        ("§ 141", "800 000 Euro", "tax bookkeeping revenue threshold"),
        ("§ 141", "80 000 Euro", "tax bookkeeping profit threshold"),
    ),
    "D-LFS-022": (
        ("§ 4", "Überschuss der Betriebseinnahmen", "EÜR profit calculation"),
        ("§ 18", "Einkünfte aus freiberuflicher Tätigkeit", "professional income"),
        ("§ 20", "Gewinnanteile (Dividenden)", "distribution income"),
        ("§ 32d", "beträgt 25 Prozent", "capital-income tax rate"),
        ("§ 34a", "Nachversteuerungsbetrag beträgt 25 Prozent", "retained-profit subsequent tax"),
    ),
    "D-LFS-023": (
        ("§ 1", "Kapitalgesellschaften", "corporation-tax subject"),
        ("§ 23", "bis 2027 15 Prozent", "current corporation-tax rate"),
    ),
    "D-LFS-024": (
        ("§ 2", "Gewerbesteuer unterliegt", "trade-tax subject"),
        ("§ 11", "24 500 Euro", "trade-tax allowance"),
    ),
    "D-LFS-025": (
        ("§ 12", "19 Prozent", "standard VAT rate"),
        ("§ 12", "sieben Prozent", "reduced VAT rate"),
        ("§ 19", "25 000 Euro", "prior-year small-business threshold"),
        ("§ 19", "100 000 Euro", "current-year small-business threshold"),
    ),
    "D-FP-013": (
        ("§ 93", "mindestens 150 Tage", "remaining-benefit eligibility"),
        ("§ 93", "fachkundigen Stelle", "viability statement"),
        ("§ 94", "Dauer von sechs Monaten", "first funding phase"),
        ("§ 94", "weitere neun Monate", "second funding phase"),
        ("§ 94", "monatlich 300 Euro", "social-security supplement"),
    ),
}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(f"Cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ArtifactValidationError(f"Expected JSON object: {path}")
    return payload


def _resolve(repo_root: Path, path: Path | str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else repo_root / candidate


def _normalise_for_check(text: str) -> str:
    return " ".join(text.replace("\u00a0", " ").split()).casefold()


def _document_blocks_by_section(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(block["law_section"]): block for block in document["blocks"]}


def _audit_document(
    *,
    repo_root: Path,
    declaration: dict[str, Any],
    document: dict[str, Any],
) -> dict[str, Any]:
    document_id = str(document["document_id"])
    source_path = repo_root / "knowledge_base" / str(declaration["source_path"])
    validate_document_record(document=document, source_path=source_path)

    failures: list[str] = []
    advisories: list[str] = []
    blocks = document["blocks"]
    by_section = _document_blocks_by_section(document)
    abbreviation = str(document["law"]["abbreviation"])

    if document.get("status") != "review_candidate":
        failures.append("Document status is not review_candidate.")
    if document["source"].get("source_type") != "primary_law":
        failures.append("Source type is not primary_law.")
    if not str(document["source"].get("url") or "").startswith(
        "https://www.gesetze-im-internet.de/"
    ):
        failures.append("Source URL is not the declared official Gesetze-im-Internet URL.")
    if document["source"].get("sha256") != declaration.get("source_sha256"):
        failures.append("Document source hash differs from the selection declaration.")

    expected_units = [f"§ {item}" for item in document["extraction"]["selected_sections"]]
    expected_units.extend(
        f"Anlage {item}" for item in document["extraction"]["selected_appendices"]
    )
    actual_units = [str(block["law_section"]) for block in blocks]
    if actual_units != expected_units:
        failures.append("Block order or legal-unit coverage differs from selected units.")

    noise_markers = (
        "nichtamtliches inhaltsverzeichnis",
        "ein service des bundesministeriums",
        "fundstelle:",
        "www.gesetze-im-internet.de/",
        "\ufffd",
    )
    for block in blocks:
        block_id = str(block["block_id"])
        text = str(block["text"])
        folded = text.casefold()
        expected_prefix = f"{abbreviation} {block['law_section']}".casefold()
        if not folded.startswith(expected_prefix):
            failures.append(f"{block_id} does not start with its law and section label.")
        if not isinstance(block.get("page_start"), int) or block["page_start"] <= 0:
            failures.append(f"{block_id} has an invalid page_start.")
        if not isinstance(block.get("page_end"), int) or block["page_end"] < block["page_start"]:
            failures.append(f"{block_id} has an invalid page_end.")
        found_noise = [marker for marker in noise_markers if marker in folded]
        if found_noise:
            failures.append(f"{block_id} contains extraction noise: {found_noise}.")

    key_check_results: list[dict[str, Any]] = []
    for law_section, needle, description in KEY_FACT_CHECKS[document_id]:
        block = by_section.get(law_section)
        passed = bool(
            block
            and _normalise_for_check(needle)
            in _normalise_for_check(str(block["text"]))
        )
        key_check_results.append(
            {
                "law_section": law_section,
                "block_id": block.get("block_id") if block else None,
                "description": description,
                "needle": needle,
                "passed": passed,
            }
        )
        if not passed:
            failures.append(f"Key-fact check failed: {law_section} / {description}.")

    if not str(document["law"].get("effective_from") or "").strip():
        advisories.append(
            "effective_from is intentionally unresolved; retain retrieved_at and "
            "current_at_retrieval wording until temporal legal review."
        )
    official_note_count = sum(
        "amtlicher hinweis" in str(block["text"]).casefold() for block in blocks
    )
    if official_note_count:
        advisories.append(
            f"{official_note_count} official implementation note(s) remain inside a "
            "complete statutory unit; human reviewer should decide whether to retain them."
        )

    return {
        "document_id": document_id,
        "law_abbreviation": abbreviation,
        "path": str(declaration["output_path"]),
        "artifact_sha256": sha256_file(repo_root / str(declaration["output_path"])),
        "content_sha256": document["content_sha256"],
        "block_count": document["block_count"],
        "structural_check_count": 8 + len(blocks) * 3,
        "key_fact_check_count": len(key_check_results),
        "key_fact_checks": key_check_results,
        "failures": failures,
        "advisories": advisories,
        "agent_result": "pass_human_review_pending" if not failures else "fail",
    }


def _build_candidate_manifest(
    *,
    repo_root: Path,
    selection: dict[str, Any],
    base_manifest: dict[str, Any],
    candidates: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    documents = [dict(item) for item in base_manifest["documents"]]
    for declaration in selection["documents"]:
        document = candidates[str(declaration["document_id"])]
        path = repo_root / str(declaration["output_path"])
        documents.append(
            {
                "document_id": document["document_id"],
                "domain": document["domain"],
                "resolution": "added_review_candidate",
                "path": path.relative_to(repo_root).as_posix(),
                "file_sha256": sha256_file(path),
                "freeze_record_path": None,
                "freeze_record_sha256": None,
                "content_sha256": document["content_sha256"],
                "source_path": document["source"]["path"],
                "source_sha256": document["source"]["sha256"],
                "block_count": document["block_count"],
                "review_status": "agent_checks_passed_human_review_pending",
            }
        )

    domain_counts = Counter(str(item["domain"]) for item in documents)
    return {
        "manifest_id": "formal_test_document_manifest_v1.2_candidate",
        "version": "1.2-candidate",
        "status": "review_candidate",
        "created_at": "2026-08-03",
        "supersedes_if_frozen": base_manifest["manifest_id"],
        "predecessor_manifest": {
            "path": BASE_DOCUMENT_MANIFEST_PATH.as_posix(),
            "sha256": sha256_file(repo_root / BASE_DOCUMENT_MANIFEST_PATH),
            "status": base_manifest["status"],
        },
        "primary_law_source_selection": {
            "path": SELECTION_PATH.as_posix(),
            "sha256": sha256_file(repo_root / SELECTION_PATH),
        },
        "configuration_impact_plan": {
            "path": IMPACT_PLAN_PATH.as_posix(),
            "sha256": sha256_file(repo_root / IMPACT_PLAN_PATH),
        },
        "freeze_guard": (
            "Do not freeze until every primary-law Document D has a recorded human "
            "approve/revise/reject decision and approved revisions are regenerated."
        ),
        "document_count": len(documents),
        "carried_forward_frozen_count": len(base_manifest["documents"]),
        "added_review_candidate_count": len(selection["documents"]),
        "domain_document_counts": dict(sorted(domain_counts.items())),
        "sampling_balance_note": (
            "The added records are authority supplements, not an equal-per-domain "
            "question sample. Configuration block scopes control their formal use."
        ),
        "documents": documents,
    }


def _build_impact_matrix(
    *,
    repo_root: Path,
    plan: dict[str, Any],
    selection: dict[str, Any],
    configuration_manifest: dict[str, Any],
    candidates: dict[str, dict[str, Any]],
    candidate_manifest_path: Path,
) -> dict[str, Any]:
    declarations = {
        str(item["document_id"]): item for item in selection["documents"]
    }
    configurations = {
        str(item["configuration_id"]): item
        for item in configuration_manifest["configurations"]
    }
    impacts = {str(item["configuration_id"]): item for item in plan["impacts"]}
    if len(impacts) != len(plan["impacts"]):
        raise ArtifactValidationError("Impact plan contains duplicate configuration IDs")
    unknown_configurations = sorted(set(impacts) - set(configurations))
    if unknown_configurations:
        raise ArtifactValidationError(
            f"Impact plan references unknown configurations: {unknown_configurations}"
        )

    matrix: list[dict[str, Any]] = []
    for configuration_id, config_declaration in configurations.items():
        config_path = repo_root / str(config_declaration["path"])
        planned = impacts.get(configuration_id)
        base_entry: dict[str, Any] = {
            "configuration_id": configuration_id,
            "domain": config_declaration["domain"],
            "current_version": config_declaration["version"],
            "current_path": config_declaration["path"],
            "current_file_sha256": sha256_file(config_path),
            "answerability": config_declaration["answerability"],
            "schema_revision_required": False,
        }
        if planned is None:
            if config_declaration["answerability"] == "unanswerable":
                reason = (
                    "Primary law does not provide the missing case-specific facts; "
                    "unanswerable status remains unchanged."
                )
            elif config_declaration["domain"] == "market_validation":
                reason = "The selected primary-law corpus does not target market validation."
            else:
                reason = "No direct primary-law augmentation is required for this question."
            matrix.append(
                {
                    **base_entry,
                    "impact_type": "no_primary_law_change",
                    "recommended_action": "carry_forward_after_manifest_review",
                    "rationale": reason,
                    "configuration_revision_required": False,
                    "qra_regeneration_required": False,
                    "primary_law_sources": [],
                }
            )
            continue

        resolved_sources: list[dict[str, Any]] = []
        for evidence in planned["evidence"]:
            document_id = str(evidence["document_id"])
            if document_id not in candidates or document_id not in declarations:
                raise ArtifactValidationError(
                    f"{configuration_id} references unknown candidate {document_id}"
                )
            declaration = declarations[document_id]
            if configuration_id not in declaration["configuration_impact_candidates"]:
                raise ArtifactValidationError(
                    f"{configuration_id} is absent from {document_id}'s impact declaration"
                )
            document = candidates[document_id]
            if (
                document["domain"] != config_declaration["domain"]
                and not evidence.get("cross_domain")
            ):
                raise ArtifactValidationError(
                    f"Cross-domain evidence must be explicit: {configuration_id} / {document_id}"
                )

            by_section = _document_blocks_by_section(document)
            labels = [f"§ {item}" for item in evidence.get("law_sections", [])]
            labels.extend(f"Anlage {item}" for item in evidence.get("appendices", []))
            unknown_labels = [label for label in labels if label not in by_section]
            if unknown_labels:
                raise ArtifactValidationError(
                    f"{configuration_id} requests unknown units from {document_id}: "
                    f"{unknown_labels}"
                )
            resolved_sources.append(
                {
                    "document_id": document_id,
                    "document_path": declaration["output_path"],
                    "document_content_sha256": document["content_sha256"],
                    "law_abbreviation": document["law"]["abbreviation"],
                    "law_sections": evidence.get("law_sections", []),
                    "appendices": evidence.get("appendices", []),
                    "allowed_block_ids": [by_section[label]["block_id"] for label in labels],
                    "cross_domain": bool(evidence.get("cross_domain", False)),
                }
            )

        matrix.append(
            {
                **base_entry,
                "impact_type": planned["impact_type"],
                "recommended_action": planned["recommended_action"],
                "rationale": planned["rationale"],
                "configuration_revision_required": True,
                "qra_regeneration_required": True,
                "answerability_effect": "unchanged",
                "primary_law_sources": resolved_sources,
            }
        )

    counts = Counter(str(item["impact_type"]) for item in matrix)
    return {
        "matrix_id": "primary_law_configuration_impact_matrix_v0.1",
        "version": "0.1",
        "status": "review_candidate",
        "created_at": "2026-08-03",
        "impact_plan": {
            "path": IMPACT_PLAN_PATH.as_posix(),
            "sha256": sha256_file(repo_root / IMPACT_PLAN_PATH),
        },
        "base_configuration_manifest": {
            "path": CONFIGURATION_MANIFEST_PATH.as_posix(),
            "sha256": sha256_file(repo_root / CONFIGURATION_MANIFEST_PATH),
            "status": configuration_manifest["status"],
        },
        "candidate_document_manifest": {
            "path": candidate_manifest_path.relative_to(repo_root).as_posix(),
            "sha256": sha256_file(candidate_manifest_path),
        },
        "configuration_count": len(matrix),
        "configuration_revision_count": sum(
            bool(item["configuration_revision_required"]) for item in matrix
        ),
        "schema_revision_count": 0,
        "qra_regeneration_count": sum(
            bool(item["qra_regeneration_required"]) for item in matrix
        ),
        "impact_type_counts": dict(sorted(counts.items())),
        "matrix": matrix,
    }


def _write_agent_review_markdown(path: Path, review: dict[str, Any]) -> None:
    lines = [
        "# Primary-law Document D 自动审核报告 v0.1",
        "",
        "## 结论",
        "",
        f"- 候选文档：{review['candidate_count']} 份",
        f"- 法条及附件 blocks：{review['block_count']} 个",
        f"- 关键法律事实检查：{review['key_fact_check_count']} 项",
        f"- 阻断问题：{review['blocking_failure_count']} 项",
        "- 当前状态：`agent_checks_passed_human_review_pending`",
        "- 本报告不是人工批准，也不会冻结任何 Document、Configuration 或 QRA。",
        "",
        "## 单文档结果",
        "",
        "| Document D | 法律 | Blocks | 关键事实检查 | Agent 结果 | 人工状态 |",
        "|---|---|---:|---:|---|---|",
    ]
    for item in review["documents"]:
        lines.append(
            f"| `{item['document_id']}` | {item['law_abbreviation']} | "
            f"{item['block_count']} | {item['key_fact_check_count']} | "
            f"`{item['agent_result']}` | `pending` |"
        )
    lines.extend(
        [
            "",
            "## 审核发现",
            "",
            "1. UStG `§ 12` 已补入，用于支持 19%/7% 增值税率比较。",
            "2. EStG `§§ 20, 32d, 34a` 已补入，用于支持分红及 25% 后续征税。",
            "3. 所有来源哈希、block 顺序、页码、法条标签和关键事实检查均通过。",
            "4. 法律候选使用 `current_at_retrieval`，但未自行填写 `effective_from`；涉及税率时必须保留时间范围。",
            "5. BGB `§ 14` 内保留一条官方实施说明。它属于完整法条单元，不是目录噪声，但冻结前仍应人工决定是否保留。",
            "6. 新增法律文档是 authority supplement，不代表各领域抽样数量必须相等；正式使用范围由 Configuration 的 exact block scope 控制。",
            "",
            "## 人工审核入口",
            "",
            "请继续在 `primary_law_candidate_review_v0.1_zh.md` 的最后两列填写",
            "`approve`、`revise` 或 `reject` 及备注。只有全部得到明确结论后，才进入",
            "Configuration 改写和 QRA `v0.4` 候选生成。",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_impact_markdown(path: Path, impact_matrix: dict[str, Any]) -> None:
    impacted = [
        item
        for item in impact_matrix["matrix"]
        if item["configuration_revision_required"]
    ]
    lines = [
        "# Primary-law Configuration 影响矩阵 v0.1",
        "",
        "## 摘要",
        "",
        f"- 当前冻结 Configuration：{impact_matrix['configuration_count']} 条",
        f"- 建议增加 primary-law evidence：{len(impacted)} 条",
        f"- 不需要法律来源变更：{impact_matrix['configuration_count'] - len(impacted)} 条",
        "- Schema S 修改：0 条；现有字段足以表达这些法律事实。",
        f"- 后续需要重新生成 QRA 候选：{impact_matrix['qra_regeneration_count']} 条",
        "- 当前状态：`review_candidate`，尚未修改冻结 Configuration。",
        "",
        "## 受影响的 Configuration",
        "",
        "| Configuration | 影响类型 | 精确法律范围 | 建议动作 |",
        "|---|---|---|---|",
    ]
    for item in impacted:
        sources = []
        for source in item["primary_law_sources"]:
            units = [f"§ {unit}" for unit in source["law_sections"]]
            units.extend(f"Anlage {unit}" for unit in source["appendices"])
            cross = "（跨领域）" if source["cross_domain"] else ""
            sources.append(
                f"{source['law_abbreviation']} {'/'.join(units)}{cross}"
            )
        lines.append(
            f"| `{item['configuration_id']}` | `{item['impact_type']}` | "
            f"{'<br>'.join(sources)} | `{item['recommended_action']}` |"
        )
    lines.extend(
        [
            "",
            "## 解释",
            "",
            "- `required_authority_augmentation`：现有答案可保留，但法律事实应增加 primary-law blocks 和 provenance。",
            "- `supplementary_authority_cross_check`：原问题主要考察已选二级材料，法律文本用于核验而不是替换题意。",
            "- `augment_sources_and_revalidate_temporal_wording`：税率具有时间敏感性，改写时必须说明适用年份。",
            "- 两条 legal-form `Unanswerable` 问题仍然不可回答，因为缺少的是用户个案数据，不是法律条文。",
            "",
            "## 下一步门槛",
            "",
            "1. 人工审核 14 份 Document D。",
            "2. 根据审核备注修订并重新生成候选。",
            "3. 明确确认后冻结 Document manifest `v1.2`。",
            "4. 只改写本矩阵标记的 Configuration，并重新生成对应 QRA `v0.4`。",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def prepare_primary_law_review_artifacts(
    *,
    repo_root: Path,
    candidate_manifest_path: Path = CANDIDATE_MANIFEST_PATH,
    agent_review_json_path: Path = AGENT_REVIEW_JSON_PATH,
    agent_review_markdown_path: Path = AGENT_REVIEW_MARKDOWN_PATH,
    impact_matrix_json_path: Path = IMPACT_MATRIX_JSON_PATH,
    impact_matrix_markdown_path: Path = IMPACT_MATRIX_MARKDOWN_PATH,
) -> dict[str, Any]:
    """Generate non-freezing audit, manifest-candidate, and impact artifacts."""

    repo_root = repo_root.resolve()
    selection = _read_json(repo_root / SELECTION_PATH)
    plan = _read_json(repo_root / IMPACT_PLAN_PATH)
    base_manifest = _read_json(repo_root / BASE_DOCUMENT_MANIFEST_PATH)
    configuration_manifest = _read_json(repo_root / CONFIGURATION_MANIFEST_PATH)

    if base_manifest.get("status") != "frozen":
        raise ArtifactValidationError("Base Document manifest must be frozen")
    if configuration_manifest.get("status") != "frozen":
        raise ArtifactValidationError("Base Configuration manifest must be frozen")
    if len(selection.get("documents", [])) != len(KEY_FACT_CHECKS):
        raise ArtifactValidationError("Every selected primary-law document needs key checks")

    candidates: dict[str, dict[str, Any]] = {}
    document_reviews: list[dict[str, Any]] = []
    for declaration in selection["documents"]:
        document_path = repo_root / str(declaration["output_path"])
        document = _read_json(document_path)
        document_id = str(document["document_id"])
        if document_id in candidates:
            raise ArtifactValidationError(f"Duplicate candidate Document ID: {document_id}")
        candidates[document_id] = document
        document_reviews.append(
            _audit_document(
                repo_root=repo_root,
                declaration=declaration,
                document=document,
            )
        )

    failures = [
        failure
        for item in document_reviews
        for failure in item["failures"]
    ]
    if failures:
        raise ArtifactValidationError(
            "Primary-law audit failed: " + "; ".join(failures[:10])
        )

    candidate_manifest_path = _resolve(repo_root, candidate_manifest_path)
    candidate_manifest = _build_candidate_manifest(
        repo_root=repo_root,
        selection=selection,
        base_manifest=base_manifest,
        candidates=candidates,
    )
    write_json_artifact(candidate_manifest_path, candidate_manifest)

    agent_review = {
        "review_id": "primary_law_agent_review_v0.1",
        "version": "0.1",
        "status": "agent_checks_passed_human_review_pending",
        "created_at": "2026-08-03",
        "primary_law_selection": {
            "path": SELECTION_PATH.as_posix(),
            "sha256": sha256_file(repo_root / SELECTION_PATH),
        },
        "candidate_document_manifest": {
            "path": candidate_manifest_path.relative_to(repo_root).as_posix(),
            "sha256": sha256_file(candidate_manifest_path),
        },
        "candidate_count": len(document_reviews),
        "block_count": sum(item["block_count"] for item in document_reviews),
        "key_fact_check_count": sum(
            item["key_fact_check_count"] for item in document_reviews
        ),
        "blocking_failure_count": 0,
        "human_review_state": "pending",
        "documents": document_reviews,
    }
    agent_review_json_path = _resolve(repo_root, agent_review_json_path)
    write_json_artifact(agent_review_json_path, agent_review)
    agent_review_markdown_path = _resolve(repo_root, agent_review_markdown_path)
    _write_agent_review_markdown(agent_review_markdown_path, agent_review)

    impact_matrix = _build_impact_matrix(
        repo_root=repo_root,
        plan=plan,
        selection=selection,
        configuration_manifest=configuration_manifest,
        candidates=candidates,
        candidate_manifest_path=candidate_manifest_path,
    )
    impact_matrix_json_path = _resolve(repo_root, impact_matrix_json_path)
    write_json_artifact(impact_matrix_json_path, impact_matrix)
    impact_matrix_markdown_path = _resolve(repo_root, impact_matrix_markdown_path)
    _write_impact_markdown(impact_matrix_markdown_path, impact_matrix)

    return {
        "candidate_count": agent_review["candidate_count"],
        "block_count": agent_review["block_count"],
        "key_fact_check_count": agent_review["key_fact_check_count"],
        "configuration_count": impact_matrix["configuration_count"],
        "configuration_revision_count": impact_matrix[
            "configuration_revision_count"
        ],
        "candidate_manifest_path": candidate_manifest_path,
        "agent_review_json_path": agent_review_json_path,
        "agent_review_markdown_path": agent_review_markdown_path,
        "impact_matrix_json_path": impact_matrix_json_path,
        "impact_matrix_markdown_path": impact_matrix_markdown_path,
    }
