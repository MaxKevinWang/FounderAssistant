"""Generate review-candidate RAGEval Documents D from selected primary laws.

Production RAG chunks and evaluation Documents D serve different purposes. The
RAG index may split a long section for retrieval, while a Document D must retain
a stable, human-reviewable legal unit for exact references. This generator
therefore reuses the legal PDF parser but writes one block per selected statutory
section or appendix.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from apps.advisors.legal_corpus import (
    LegalSourceProfile,
    extract_legal_pdf,
    load_legal_scope,
)
from apps.evaluations.rageval_artifacts import (
    ArtifactValidationError,
    sha256_file,
    validate_document_record,
    write_json_artifact,
)


DEFAULT_SELECTION_PATH = Path(
    "docs/evaluation/formal_dataset/source_selection/"
    "primary_law_document_selection_v0.1.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "docs/evaluation/formal_dataset/documents/primary_law_candidates"
)


def _read_json_object(path: Path) -> dict[str, Any]:
    """Read a UTF-8 JSON object with an artifact-oriented error."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(f"Cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ArtifactValidationError(f"Expected a JSON object: {path}")
    return payload


def _canonical_sha256(value: Any) -> str:
    """Hash JSON data using the same canonical form as other evaluation D files."""

    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _law_section_label(unit_type: str, unit_id: str) -> str:
    return f"§ {unit_id}" if unit_type == "section" else f"Anlage {unit_id}"


def _build_primary_law_document(
    *,
    repo_root: Path,
    selection: dict[str, Any],
    declaration: dict[str, Any],
    profile: LegalSourceProfile,
    scope_manifest_path: Path,
    scope_sha256: str,
) -> dict[str, Any]:
    """Build one candidate Document D without freezing it."""

    document_id = str(declaration["document_id"])
    source_relative = Path("knowledge_base") / Path(profile.source_path)
    source_path = repo_root / source_relative
    if not source_path.exists():
        raise ArtifactValidationError(f"Primary-law source is missing: {source_path}")

    expected_source_hash = str(declaration.get("source_sha256") or "")
    actual_source_hash = sha256_file(source_path)
    if not expected_source_hash or actual_source_hash != expected_source_hash:
        raise ArtifactValidationError(
            f"Source SHA-256 mismatch for {document_id}: expected "
            f"{expected_source_hash!r}, got {actual_source_hash!r}"
        )

    extraction_parameters = selection["extraction_parameters"]
    result = extract_legal_pdf(
        source_path,
        profile,
        chunk_size=int(extraction_parameters["chunk_size"]),
        chunk_overlap=int(extraction_parameters["chunk_overlap"]),
    )
    if result.audit["missing_selected_sections"] or result.audit["missing_selected_appendices"]:
        raise ArtifactValidationError(
            f"{document_id} is missing selected legal units: "
            f"sections={result.audit['missing_selected_sections']}, "
            f"appendices={result.audit['missing_selected_appendices']}"
        )

    blocks: list[dict[str, Any]] = []
    for unit in result.units:
        law_section = _law_section_label(unit.unit_type, unit.unit_id)
        section_title = f"{law_section} {unit.title}".strip()
        blocks.append(
            {
                "block_id": f"{document_id}-B{len(blocks) + 1:03d}",
                "block_type": (
                    "statute_section" if unit.unit_type == "section" else "statute_appendix"
                ),
                "heading_path": f"{profile.law_abbreviation} > {section_title}",
                "page_start": unit.page_start,
                "page_end": unit.page_end,
                "evidence_type": "primary_law",
                "law_section": law_section,
                "section_title": unit.title,
                "text": unit.text,
            }
        )
    if not blocks:
        raise ArtifactValidationError(f"No selected legal blocks extracted for {document_id}")

    source_record_path = source_relative.as_posix()
    document = {
        "document_id": document_id,
        "version": "0.1",
        "status": "review_candidate",
        "domain": profile.domain,
        "title": f"{profile.law_name} ({profile.law_abbreviation})",
        "language": "de",
        "jurisdiction": "Germany",
        "created_at": str(selection["created_at"]),
        "source": {
            "seed_document_id": str(declaration["seed_document_id"]),
            "path": source_record_path,
            "url": profile.canonical_url,
            "accessed_at": profile.retrieved_at,
            "publication_date": None,
            "sha256": actual_source_hash,
            "organization": profile.authority,
            "source_group": "gesetze_im_internet",
            "source_type": "primary_law",
        },
        "law": {
            "name": profile.law_name,
            "abbreviation": profile.law_abbreviation,
            "legal_area": profile.legal_area,
            "authority": profile.authority,
            "canonical_url": profile.canonical_url,
            "retrieved_at": profile.retrieved_at,
            "temporal_status": profile.temporal_status,
            "effective_from": profile.effective_from,
            "effective_to": profile.effective_to,
            "supersedes": profile.supersedes,
        },
        "extraction": {
            "method": "apps.advisors.legal_corpus.extract_legal_pdf",
            "scope_manifest_path": scope_manifest_path.relative_to(repo_root).as_posix(),
            "scope_manifest_sha256": scope_sha256,
            "front_matter_and_toc_removed": True,
            "repealed_sections_removed": result.audit["skipped_repealed_sections"],
            "duplicate_toc_occurrences_resolved": result.audit[
                "duplicate_toc_or_reference_occurrences_resolved"
            ],
            "text_block_granularity": "one block per selected statutory section or appendix",
            "selected_sections": result.audit["selected_sections"],
            "selected_appendices": result.audit["selected_appendices"],
            "selection_reason": profile.selection_reason,
            "block_id_pattern": f"{document_id}-B###",
        },
        "selection": {
            "selection_id": selection["selection_id"],
            "review_role": "primary_authority_candidate",
            "intended_use": declaration.get("intended_use", []),
            "configuration_impact_candidates": declaration.get(
                "configuration_impact_candidates", []
            ),
        },
        "block_count": len(blocks),
        "content_sha256": _canonical_sha256(blocks),
        "blocks": blocks,
    }
    validate_document_record(document=document, source_path=source_path)
    return document


def _validate_selection(
    *,
    selection: dict[str, Any],
    profiles: dict[str, LegalSourceProfile],
) -> None:
    """Reject duplicate IDs, missing profiles, and domain/abbreviation drift."""

    declarations = selection.get("documents")
    if not isinstance(declarations, list) or not declarations:
        raise ArtifactValidationError("Primary-law selection must contain documents")

    seen_ids: set[str] = set()
    seen_sources: set[str] = set()
    seen_outputs: set[str] = set()
    for declaration in declarations:
        document_id = str(declaration.get("document_id") or "")
        source_path = str(declaration.get("source_path") or "")
        output_path = str(declaration.get("output_path") or "")
        if not document_id or document_id in seen_ids:
            raise ArtifactValidationError(f"Duplicate or empty Document D ID: {document_id!r}")
        if not source_path or source_path in seen_sources:
            raise ArtifactValidationError(f"Duplicate or empty law source: {source_path!r}")
        if not output_path or output_path in seen_outputs:
            raise ArtifactValidationError(f"Duplicate or empty output path: {output_path!r}")
        seen_ids.add(document_id)
        seen_sources.add(source_path)
        seen_outputs.add(output_path)

        profile = profiles.get(source_path)
        if profile is None:
            raise ArtifactValidationError(f"Law source is not in the legal scope: {source_path}")
        if declaration.get("domain") != profile.domain:
            raise ArtifactValidationError(f"Domain mismatch for {document_id}")
        if declaration.get("law_abbreviation") != profile.law_abbreviation:
            raise ArtifactValidationError(f"Law abbreviation mismatch for {document_id}")
        expected_prefix = "D-LFS-" if profile.domain == "legal_form_selection" else "D-FP-"
        if not document_id.startswith(expected_prefix):
            raise ArtifactValidationError(
                f"{document_id} does not match domain prefix {expected_prefix}"
            )


def generate_primary_law_document_candidates(
    *,
    repo_root: Path,
    selection_path: Path = DEFAULT_SELECTION_PATH,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Generate every selected primary-law Document D review candidate."""

    repo_root = repo_root.resolve()
    if not selection_path.is_absolute():
        selection_path = repo_root / selection_path
    selection = _read_json_object(selection_path)

    scope_declaration = selection.get("legal_scope_manifest") or {}
    scope_manifest_path = repo_root / str(scope_declaration.get("path") or "")
    scope_sha256, profiles = load_legal_scope(scope_manifest_path)
    if scope_sha256 != scope_declaration.get("sha256"):
        raise ArtifactValidationError(
            "Legal scope manifest hash differs from the primary-law selection"
        )
    _validate_selection(selection=selection, profiles=profiles)

    allowed_output_dir = (repo_root / DEFAULT_OUTPUT_DIR).resolve()
    generated: list[dict[str, Any]] = []
    for declaration in selection["documents"]:
        profile = profiles[str(declaration["source_path"])]
        document = _build_primary_law_document(
            repo_root=repo_root,
            selection=selection,
            declaration=declaration,
            profile=profile,
            scope_manifest_path=scope_manifest_path,
            scope_sha256=scope_sha256,
        )
        output_path = (repo_root / str(declaration["output_path"])).resolve()
        if not output_path.is_relative_to(allowed_output_dir):
            raise ArtifactValidationError(
                f"Primary-law candidate output must stay under {DEFAULT_OUTPUT_DIR}: "
                f"{output_path}"
            )

        write_state = "created"
        if output_path.exists():
            existing = _read_json_object(output_path)
            if existing == document:
                write_state = "unchanged"
            elif not overwrite:
                raise ArtifactValidationError(
                    f"Candidate differs from existing output; rerun with --overwrite: "
                    f"{output_path}"
                )
            else:
                write_state = "overwritten"
        if write_state in {"created", "overwritten"}:
            write_json_artifact(output_path, document)

        generated.append(
            {
                "document_id": document["document_id"],
                "domain": document["domain"],
                "law_abbreviation": document["law"]["abbreviation"],
                "write_state": write_state,
                "path": output_path.relative_to(repo_root).as_posix(),
                "artifact_sha256": sha256_file(output_path),
                "content_sha256": document["content_sha256"],
                "source_sha256": document["source"]["sha256"],
                "block_count": document["block_count"],
                "selected_sections": document["extraction"]["selected_sections"],
                "selected_appendices": document["extraction"]["selected_appendices"],
                "status": document["status"],
                "version": document["version"],
            }
        )

    report = {
        "generation_id": "primary_law_document_candidates_v0.1",
        "created_at": selection["created_at"],
        "selection_id": selection["selection_id"],
        "selection_path": selection_path.relative_to(repo_root).as_posix(),
        "selection_sha256": sha256_file(selection_path),
        "legal_scope_manifest": {
            "path": scope_manifest_path.relative_to(repo_root).as_posix(),
            "sha256": scope_sha256,
        },
        "candidate_count": len(generated),
        "legal_form_selection_count": sum(
            item["domain"] == "legal_form_selection" for item in generated
        ),
        "funding_preparation_count": sum(
            item["domain"] == "funding_preparation" for item in generated
        ),
        "block_count": sum(item["block_count"] for item in generated),
        "candidates": generated,
    }
    manifest_path = allowed_output_dir / "generation_manifest_v0.1.json"
    write_json_artifact(manifest_path, report)
    return report
