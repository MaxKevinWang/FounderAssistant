"""Prepare QRA semantic curation v0.4 for primary-law Configuration v1.3."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

from apps.evaluations.rageval_artifacts import (
    ArtifactValidationError,
    sha256_file,
    write_json_artifact,
)


CONFIGURATION_MANIFEST_PATH = Path(
    "docs/evaluation/formal_dataset/configurations/formal_test/"
    "formal_test_configuration_manifest_v1.3.json"
)
PREDECESSOR_CURATION_PATH = Path(
    "docs/evaluation/formal_dataset/qra/qra_semantic_curation_v0.3.json"
)
REVISION_SPEC_PATH = Path(
    "docs/evaluation/formal_dataset/qra/primary_law_qra_revision_spec_v0.4.json"
)
REVISION_PROTOCOL_PATH = Path(
    "docs/evaluation/formal_dataset/prompts/"
    "primary_law_qra_revision_protocol_v0.4.md"
)
OUTPUT_CURATION_PATH = Path(
    "docs/evaluation/formal_dataset/qra/qra_semantic_curation_v0.4.json"
)
REVISION_MANIFEST_PATH = Path(
    "docs/evaluation/formal_dataset/qra/"
    "primary_law_qra_revision_manifest_v0.4.json"
)
REVIEW_PATH = Path(
    "docs/evaluation/formal_dataset/qra/primary_law_qra_revision_review_v0.4_zh.md"
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


def _canonical_sha256(value: Any) -> str:
    """Hash one JSON value independently of whitespace or key order."""

    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_candidate(
    path: Path,
    payload: dict[str, Any],
    *,
    overwrite: bool,
) -> str:
    """Write a candidate or require explicit replacement of changed content."""

    if path.exists():
        if _read_json_object(path) == payload:
            return "unchanged"
        if not overwrite:
            raise ArtifactValidationError(
                f"QRA revision artifact differs; enable overwrite: {path}"
            )
        path.unlink()
        state = "replaced"
    else:
        state = "created"
    write_json_artifact(path, payload)
    return state


def _validate_override(
    *,
    configuration: dict[str, Any],
    entry: dict[str, Any],
) -> None:
    """Validate a semantic recipe against one frozen Configuration."""

    configuration_id = str(configuration["configuration_id"])
    if entry.get("question_type") != configuration["target_question_type"]:
        raise ArtifactValidationError(
            f"Question type mismatch for {configuration_id}"
        )
    if not str(entry.get("question") or "").strip():
        raise ArtifactValidationError(f"Missing revised question for {configuration_id}")

    provenance_ids = {
        str(item["provenance_id"]) for item in configuration["provenance"]
    }
    allowed_pairs = {
        (str(source["document_id"]), str(block_id))
        for source in configuration["source_documents"]
        for block_id in source["allowed_block_ids"]
    }
    references = list(entry.get("references") or [])
    reference_keys = [str(item.get("reference_key") or "") for item in references]
    if not references or len(set(reference_keys)) != len(reference_keys):
        raise ArtifactValidationError(
            f"Missing or duplicate references for {configuration_id}"
        )
    for reference in references:
        provenance_id = str(reference.get("provenance_id") or "")
        quote_scope = str(reference.get("quote_scope") or "exact")
        if quote_scope not in {"exact", "smallest_complete_paragraph"}:
            raise ArtifactValidationError(
                f"Unknown quote scope {quote_scope!r} for {configuration_id}"
            )
        if provenance_id and provenance_id not in provenance_ids:
            raise ArtifactValidationError(
                f"Unknown provenance {provenance_id!r} for {configuration_id}"
            )
        if provenance_id:
            continue
        pair = (
            str(reference.get("document_id") or ""),
            str(reference.get("block_id") or ""),
        )
        selector_count = sum(
            bool(str(reference.get(field) or "").strip())
            for field in ("exact_quote", "contains")
        )
        if pair not in allowed_pairs or selector_count != 1:
            raise ArtifactValidationError(
                f"Invalid direct evidence selector for {configuration_id}: "
                f"{reference.get('reference_key')!r}"
            )

    answer_units = list(entry.get("answer_units") or [])
    if not 3 <= len(answer_units) <= 5:
        raise ArtifactValidationError(
            f"Expected 3-5 answer units for {configuration_id}"
        )
    used_keys: set[str] = set()
    for unit in answer_units:
        text = str(unit.get("text") or "").strip()
        keys = set(unit.get("reference_keys") or [])
        if not text or not keys or not keys <= set(reference_keys):
            raise ArtifactValidationError(
                f"Invalid answer unit references for {configuration_id}"
            )
        used_keys.update(str(key) for key in keys)
    if used_keys != set(reference_keys):
        raise ArtifactValidationError(
            f"Unused references in revised QRA for {configuration_id}"
        )


def _render_review(manifest: dict[str, Any]) -> str:
    """Render a Chinese review index for the 17 rewritten QRA recipes."""

    lines = [
        "# Primary-law QRA v0.4 修订审核",
        "",
        f"- 日期：`{manifest['created_at']}`",
        f"- 改写 QRA：`{manifest['revised_qra_count']}`",
        f"- 完整候选集：`{manifest['expected_full_qra_count']}`",
        "- 状态：待人工审核，未冻结",
        "",
        "## 为什么仍生成 60 份文件",
        "",
        "只有 17 份 QRA 的语义配方发生变化；其余 43 份沿用 v0.3 配方。",
        "但一次正式 QRA generation manifest 必须让全部记录绑定同一份冻结的",
        "Configuration manifest，因此输出会整体版本化为 v0.4。",
        "",
        "## 受影响条目",
        "",
        "| Configuration | 版本 | 类型 | 引用数 | 答案单元 | 审核 |",
        "|---|---:|---|---:|---:|---|",
    ]
    for item in manifest["revisions"]:
        lines.append(
            f"| `{item['configuration_id']}` | "
            f"`{item['configuration_version']}` | {item['question_type']} | "
            f"{item['reference_count']} | {item['answer_unit_count']} | pending |"
        )
    lines.extend(
        [
            "",
            "审核时请重点确认问题是否自然、标准答案是否只覆盖配置证据、数值比较",
            "是否计算正确，以及 keypoint 是否没有重新引入已删除的旧法律陈述。",
            "",
        ]
    )
    return "\n".join(lines)


def prepare_primary_law_qra_revisions(
    *,
    repo_root: Path,
    created_at: str = "2026-08-03",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create curation v0.4 by replacing the 17 dependency-changed recipes."""

    repo_root = repo_root.resolve()
    configuration_manifest_path = repo_root / CONFIGURATION_MANIFEST_PATH
    predecessor_curation_path = repo_root / PREDECESSOR_CURATION_PATH
    revision_spec_path = repo_root / REVISION_SPEC_PATH
    revision_protocol_path = repo_root / REVISION_PROTOCOL_PATH
    configuration_manifest = _read_json_object(configuration_manifest_path)
    predecessor_curation = _read_json_object(predecessor_curation_path)
    revision_spec = _read_json_object(revision_spec_path)

    if (
        configuration_manifest.get("status") != "frozen"
        or configuration_manifest.get("version") != "1.3"
        or configuration_manifest.get("revised_configuration_count") != 17
    ):
        raise ArtifactValidationError("Expected frozen Configuration manifest v1.3")
    if (
        predecessor_curation.get("curation_id")
        != "formal_qra_semantic_curation_v0.3"
        or predecessor_curation.get("version") != "0.3"
    ):
        raise ArtifactValidationError("Expected predecessor QRA curation v0.3")
    if revision_spec.get("version") != "0.4":
        raise ArtifactValidationError("Expected QRA revision specification v0.4")

    revised_declarations = {
        str(item["configuration_id"]): item
        for item in configuration_manifest["configurations"]
        if item["resolution"] == "primary_law_revision"
    }
    overrides = dict(revision_spec.get("overrides") or {})
    if set(overrides) != set(revised_declarations) or len(overrides) != 17:
        raise ArtifactValidationError(
            "QRA v0.4 overrides must cover exactly the 17 revised Configurations"
        )

    configurations: dict[str, dict[str, Any]] = {}
    revisions: list[dict[str, Any]] = []
    for configuration_id, declaration in revised_declarations.items():
        configuration_path = repo_root / str(declaration["path"])
        if sha256_file(configuration_path) != declaration["file_sha256"]:
            raise ArtifactValidationError(
                f"Frozen Configuration changed: {configuration_path}"
            )
        configuration = _read_json_object(configuration_path)
        configurations[configuration_id] = configuration
        entry = overrides[configuration_id]
        _validate_override(configuration=configuration, entry=entry)
        revisions.append(
            {
                "configuration_id": configuration_id,
                "configuration_version": configuration["version"],
                "configuration_path": declaration["path"],
                "configuration_sha256": declaration["file_sha256"],
                "question_type": entry["question_type"],
                "predecessor_recipe_sha256": _canonical_sha256(
                    predecessor_curation["entries"][configuration_id]
                ),
                "revised_recipe_sha256": _canonical_sha256(entry),
                "reference_count": len(entry["references"]),
                "answer_unit_count": len(entry["answer_units"]),
                "human_review_state": "pending",
            }
        )

    curation = deepcopy(predecessor_curation)
    curation.update(
        {
            "curation_id": "formal_qra_semantic_curation_v0.4",
            "version": "0.4",
            "status": "development_candidate",
            "supersedes": "formal_qra_semantic_curation_v0.3",
            "supersedes_generation_manifest": (
                "formal_qra_generation_manifest_v0.3"
            ),
            "scope": (
                "All v0.3 semantic recipes plus 17 primary-law dependency "
                "revisions bound to frozen Configuration manifest v1.3"
            ),
            "methodology_note": (
                "Version 0.4 preserves unaffected v0.3 recipes and rewrites all "
                "17 recipes whose Configurations changed after primary-law "
                "curation. Removed secondary-source claims are not reused; "
                "current legal rules prefer frozen primary-law provenance."
            ),
            "configuration_manifest": {
                "path": CONFIGURATION_MANIFEST_PATH.as_posix(),
                "sha256": sha256_file(configuration_manifest_path),
            },
            "revision_specification": {
                "path": REVISION_SPEC_PATH.as_posix(),
                "sha256": sha256_file(revision_spec_path),
            },
            "revision_protocol": {
                "path": REVISION_PROTOCOL_PATH.as_posix(),
                "sha256": sha256_file(revision_protocol_path),
            },
            "human_review_state": "pending",
        }
    )
    for configuration_id, entry in overrides.items():
        curation["entries"][configuration_id] = deepcopy(entry)

    write_states: list[str] = []
    output_curation_path = repo_root / OUTPUT_CURATION_PATH
    write_states.append(
        _write_candidate(output_curation_path, curation, overwrite=overwrite)
    )
    revision_manifest = {
        "manifest_id": "primary_law_qra_revision_manifest_v0.4",
        "version": "0.4",
        "status": "review_candidate",
        "created_at": created_at,
        "configuration_manifest": {
            "path": CONFIGURATION_MANIFEST_PATH.as_posix(),
            "sha256": sha256_file(configuration_manifest_path),
        },
        "predecessor_curation": {
            "path": PREDECESSOR_CURATION_PATH.as_posix(),
            "sha256": sha256_file(predecessor_curation_path),
        },
        "revision_specification": {
            "path": REVISION_SPEC_PATH.as_posix(),
            "sha256": sha256_file(revision_spec_path),
        },
        "output_curation": {
            "path": OUTPUT_CURATION_PATH.as_posix(),
            "sha256": sha256_file(output_curation_path),
        },
        "revised_qra_count": 17,
        "carried_recipe_count": len(curation["entries"]) - 17,
        "expected_full_qra_count": 60,
        "human_review_claimed": False,
        "freeze_status": "not_frozen",
        "revisions": revisions,
    }
    revision_manifest_path = repo_root / REVISION_MANIFEST_PATH
    write_states.append(
        _write_candidate(
            revision_manifest_path,
            revision_manifest,
            overwrite=overwrite,
        )
    )

    review_path = repo_root / REVIEW_PATH
    review_text = _render_review(revision_manifest)
    if review_path.exists():
        existing = review_path.read_text(encoding="utf-8")
        if existing != review_text and not overwrite:
            raise ArtifactValidationError(
                f"QRA review sheet differs; enable overwrite: {review_path}"
            )
    review_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.write_text(review_text, encoding="utf-8")

    return {
        "revised_qra_count": 17,
        "curated_entry_count": len(curation["entries"]),
        "curation_path": OUTPUT_CURATION_PATH.as_posix(),
        "revision_manifest_path": REVISION_MANIFEST_PATH.as_posix(),
        "review_path": REVIEW_PATH.as_posix(),
        "created_count": write_states.count("created"),
        "replaced_count": write_states.count("replaced"),
        "unchanged_count": write_states.count("unchanged"),
    }
