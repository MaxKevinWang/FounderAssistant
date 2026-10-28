"""Generate review-candidate formal RAGEval Document D artifacts.

The formal source-selection manifest is the single control surface for this
step. It decides whether a selected source reuses an already frozen Document D
or produces a new candidate. The generator performs extraction only; it never
freezes candidates and never creates Configurations, questions, or answers.
"""

from __future__ import annotations

import json
from pathlib import Path

from apps.evaluations.corpus_paths import resolve_corpus_path
from typing import Any

from apps.evaluations.rageval_artifacts import (
    ArtifactValidationError,
    build_html_document_record,
    build_pdf_document_record,
    sha256_file,
    validate_document_record,
    write_json_artifact,
)


DEFAULT_FIGURE_DESCRIPTION_PATH = (
    "backend/rag_index/figure_descriptions/figure_descriptions.json"
)


def load_reviewed_figure_descriptions(
    figure_description_path: Path,
) -> dict[str, dict[int, list[dict[str, Any]]]]:
    """Load final reviewed PDF descriptions by data-relative source and page."""

    if not figure_description_path.exists():
        return {}

    try:
        payload = json.loads(figure_description_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(
            f"Cannot read figure descriptions: {figure_description_path}: {exc}"
        ) from exc
    if not isinstance(payload, list):
        raise ArtifactValidationError(
            f"Figure descriptions must be a JSON list: {figure_description_path}"
        )

    grouped: dict[str, dict[int, list[dict[str, Any]]]] = {}
    for figure in payload:
        if not isinstance(figure, dict):
            continue
        source_path = str(figure.get("source_path") or "").replace("\\", "/").strip()
        if not source_path.lower().endswith(".pdf"):
            continue
        try:
            page_number = int(figure.get("page_number") or 0)
        except (TypeError, ValueError):
            continue
        if page_number <= 0:
            continue
        grouped.setdefault(source_path, {}).setdefault(page_number, []).append(figure)
    return grouped


def _data_relative_path(source_record_path: str) -> str:
    """Convert a repository-relative ``knowledge_base/...`` path to a data-relative key."""

    normalized = source_record_path.replace("\\", "/").lstrip("./")
    for prefix in ("knowledge_base/", "data/"):
        if normalized.startswith(prefix):
            return normalized[len(prefix):]
    return normalized


def _read_json_object(path: Path) -> dict[str, Any]:
    """Read one UTF-8 JSON object with an artifact-oriented error."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(f"Cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ArtifactValidationError(f"Expected a JSON object: {path}")
    return payload


def _validate_source_identity(source_path: Path, selected: dict[str, Any]) -> None:
    """Ensure generation uses the exact raw source version selected for testing."""

    if not source_path.exists():
        raise ArtifactValidationError(f"Selected source is missing: {source_path}")
    expected_hash = str(selected.get("raw_sha256") or "")
    actual_hash = sha256_file(source_path)
    if not expected_hash or actual_hash != expected_hash:
        raise ArtifactValidationError(
            f"Raw source SHA-256 mismatch for {selected.get('seed_document_id')}: "
            f"expected {expected_hash!r}, got {actual_hash!r}"
        )


def _validate_reused_document(
    *,
    repo_root: Path,
    selected: dict[str, Any],
) -> dict[str, Any]:
    """Validate one explicitly reused frozen D against the selected raw source."""

    document_path_value = str(selected.get("existing_document_path") or "")
    if not document_path_value:
        raise ArtifactValidationError(
            f"{selected.get('document_id')} has no existing_document_path"
        )
    document_path = repo_root / document_path_value
    document = _read_json_object(document_path)
    source_path = resolve_corpus_path(repo_root, str(selected["path"]))

    _validate_source_identity(source_path, selected)
    validate_document_record(document=document, source_path=source_path)
    if document.get("document_id") != selected.get("document_id"):
        raise ArtifactValidationError(
            f"Reused document ID mismatch in {document_path}: "
            f"{document.get('document_id')} != {selected.get('document_id')}"
        )
    if document.get("status") != "frozen":
        raise ArtifactValidationError(
            f"Reused document must already be frozen: {document_path}"
        )
    expected_artifact_hash = str(selected.get("existing_document_sha256") or "")
    if expected_artifact_hash and sha256_file(document_path) != expected_artifact_hash:
        raise ArtifactValidationError(
            f"Reused document artifact hash changed: {document_path}"
        )
    return {
        "document_id": document["document_id"],
        "action": "reuse_existing",
        "path": document_path_value,
        "artifact_sha256": sha256_file(document_path),
        "content_sha256": document["content_sha256"],
        "source_sha256": document["source"]["sha256"],
        "status": document["status"],
        "version": document["version"],
    }


def _build_candidate(
    *,
    repo_root: Path,
    selection: dict[str, Any],
    selected: dict[str, Any],
    figures_by_source: dict[str, dict[int, list[dict[str, Any]]]],
    figure_description_path: Path,
    figure_description_record_path: str,
) -> tuple[dict[str, Any], Path]:
    """Build and validate one new HTML or PDF candidate."""

    document_id = str(selected["document_id"])
    source_record_path = str(selected["path"]).replace("\\", "/")
    source_path = resolve_corpus_path(repo_root, source_record_path)
    _validate_source_identity(source_path, selected)

    common = {
        "source_path": source_path,
        "source_record_path": source_record_path,
        "document_id": document_id,
        "domain": str(selected["domain"]),
        "title": str(selected["title"]),
        "language": str(selection["language"]),
        "jurisdiction": str(selection["jurisdiction"]),
        "seed_document_id": str(selected["seed_document_id"]),
        "created_at": str(selection["created_at"]),
        "source_url": selected.get("source_url"),
        "accessed_at": selected.get("accessed_at"),
        "publication_date": selected.get("publication_date"),
        "version": "0.1",
        "status": "review_candidate",
    }
    source_format = str(selected["format"]).lower()
    if source_format == "html":
        html_rules = selected.get("html_extraction") or {}
        document = build_html_document_record(
            **common,
            stop_heading_prefixes=tuple(
                str(value) for value in html_rules.get("stop_heading_prefixes", [])
            ),
            stop_block_prefixes=tuple(
                str(value) for value in html_rules.get("stop_block_prefixes", [])
            ),
            exclude_block_prefixes=tuple(
                str(value) for value in html_rules.get("exclude_block_prefixes", [])
            ),
        )
    elif source_format == "pdf":
        source_key = _data_relative_path(source_record_path)
        reviewed_figures = figures_by_source.get(source_key, {})
        document = build_pdf_document_record(
            **common,
            reviewed_figures_by_page=reviewed_figures,
            figure_description_path=(
                figure_description_path if reviewed_figures else None
            ),
            figure_description_record_path=(
                figure_description_record_path if reviewed_figures else None
            ),
        )
    else:
        raise ArtifactValidationError(
            f"Unsupported formal Document D source format: {source_format}"
        )

    document["source"].update(
        {
            "organization": selected.get("source_organization"),
            "source_group": selected.get("source_group"),
            "source_type": selected.get("source_type"),
        }
    )
    document["selection"] = {
        "selection_id": selection["selection_id"],
        "schema_role": selected.get("schema_role"),
        "planned_primary_qra_budget": selected.get("planned_primary_qra_budget"),
        "candidate_question_types": selected.get("candidate_question_types", []),
        "development_leakage_guard": selected.get("development_leakage_guard"),
        "temporal_validation_required_before_qra_freeze": selected.get(
            "temporal_validation_required_before_qra_freeze",
            False,
        ),
    }
    validate_document_record(document=document, source_path=source_path)

    output_path_value = str(selected.get("document_output_path") or "")
    if not output_path_value:
        raise ArtifactValidationError(
            f"{document_id} has no document_output_path in source selection"
        )
    return document, repo_root / output_path_value


def generate_formal_document_candidates(
    *,
    repo_root: Path,
    selection_path: Path,
    overwrite: bool = False,
    manifest_output_path: Path | None = None,
    generation_id: str | None = None,
) -> dict[str, Any]:
    """Generate every new D selected by the formal source manifest.

    Existing output files are accepted without ``overwrite`` only when their
    full JSON content exactly matches a deterministic rebuild.
    """

    repo_root = repo_root.resolve()
    if not selection_path.is_absolute():
        selection_path = repo_root / selection_path
    selection = _read_json_object(selection_path)

    figure_description_record_path = str(
        selection.get("figure_description_path") or DEFAULT_FIGURE_DESCRIPTION_PATH
    )
    figure_description_path = repo_root / figure_description_record_path
    figures_by_source = load_reviewed_figure_descriptions(figure_description_path)

    generated: list[dict[str, Any]] = []
    reused: list[dict[str, Any]] = []
    for domain_name, domain_payload in selection["domains"].items():
        for selected in domain_payload["selected_documents"]:
            selected = dict(selected)
            selected["domain"] = domain_name
            action = selected.get("document_action", "generate_candidate")
            if action == "reuse_existing":
                reused.append(
                    _validate_reused_document(repo_root=repo_root, selected=selected)
                )
                continue
            if action != "generate_candidate":
                raise ArtifactValidationError(
                    f"Unknown document_action {action!r} for {selected.get('document_id')}"
            )

            document, output_path = _build_candidate(
                repo_root=repo_root,
                selection=selection,
                selected=selected,
                figures_by_source=figures_by_source,
                figure_description_path=figure_description_path,
                figure_description_record_path=figure_description_record_path,
            )
            existed_before = output_path.exists()
            write_state = "created"
            if existed_before:
                existing = _read_json_object(output_path)
                if existing == document:
                    write_state = "unchanged"
                elif not overwrite:
                    raise ArtifactValidationError(
                        f"Candidate differs from existing output; rerun with --overwrite: "
                        f"{output_path}"
                    )
            if write_state == "created" or overwrite:
                write_json_artifact(output_path, document)
                if overwrite and existed_before:
                    write_state = "overwritten"

            generated.append(
                {
                    "document_id": document["document_id"],
                    "action": "generate_candidate",
                    "write_state": write_state,
                    "path": output_path.relative_to(repo_root).as_posix(),
                    "artifact_sha256": sha256_file(output_path),
                    "content_sha256": document["content_sha256"],
                    "source_sha256": document["source"]["sha256"],
                    "format": selected["format"],
                    "block_count": document["block_count"],
                    "page_text_block_count": sum(
                        block.get("block_type") == "page_text"
                        for block in document["blocks"]
                    ),
                    "figure_description_block_count": sum(
                        block.get("block_type") == "figure_description"
                        for block in document["blocks"]
                    ),
                    "reviewed_figure_description_count": document["extraction"].get(
                        "reviewed_figure_description_count",
                        0,
                    ),
                    "status": document["status"],
                    "version": document["version"],
                }
            )

    report = {
        "generation_id": generation_id or str(
            selection.get("generation_id") or "formal_document_candidates_v0.1"
        ),
        "created_at": selection["created_at"],
        "selection_id": selection["selection_id"],
        "selection_path": selection_path.relative_to(repo_root).as_posix(),
        "selection_sha256": sha256_file(selection_path),
        "figure_description_path": figure_description_record_path,
        "figure_description_sha256": (
            sha256_file(figure_description_path)
            if figure_description_path.exists()
            else None
        ),
        "selected_document_count": len(generated) + len(reused),
        "new_candidate_count": len(generated),
        "reused_frozen_count": len(reused),
        "new_candidates": generated,
        "reused_documents": reused,
    }
    include_manifest_path = bool(
        manifest_output_path is not None or selection.get("generation_manifest_path")
    )
    if manifest_output_path is None:
        manifest_output_path = Path(
            str(
                selection.get("generation_manifest_path")
                or (
                    "docs/evaluation/formal_dataset/documents/formal_candidates/"
                    "generation_manifest_v0.1.json"
                )
            )
        )
    if not manifest_output_path.is_absolute():
        manifest_output_path = repo_root / manifest_output_path
    if include_manifest_path:
        report["generation_manifest_path"] = manifest_output_path.relative_to(
            repo_root
        ).as_posix()
    write_json_artifact(manifest_output_path, report)
    return report
