"""Audit optimization Document D candidates and build leakage exclusions."""

from __future__ import annotations

from collections import Counter
import hashlib
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


DEFAULT_SELECTION_PATH = Path(
    "docs/evaluation/optimization_dataset/source_selection/"
    "optimization_source_selection_v0.1.json"
)
DEFAULT_SCOPE_REVIEW_PATH = Path(
    "docs/evaluation/optimization_dataset/documents/"
    "market_case_scope_review_v0.1.json"
)
DEFAULT_OUTPUT_PATH = Path(
    "docs/evaluation/optimization_dataset/documents/"
    "document_candidate_audit_v0.1.json"
)
FORMAL_QRA_MANIFEST_PATH = Path(
    "docs/evaluation/formal_dataset/qra/formal_test/"
    "formal_test_qra_manifest_v1.0.json"
)


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(f"Cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ArtifactValidationError(f"Expected a JSON object: {path}")
    return payload


def _repository_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _normalise_text(value: str) -> str:
    return " ".join(value.split())


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_or_verify(path: Path, payload: dict[str, Any], *, overwrite: bool) -> str:
    if path.exists():
        if _read_json_object(path) == payload:
            return "unchanged"
        if not overwrite:
            raise ArtifactValidationError(
                f"Document audit differs from existing output; rerun with --overwrite: {path}"
            )
        write_json_artifact(path, payload)
        return "overwritten"
    write_json_artifact(path, payload)
    return "created"


def _formal_references(repo_root: Path) -> dict[str, list[dict[str, str]]]:
    manifest = _read_json_object(repo_root / FORMAL_QRA_MANIFEST_PATH)
    references: dict[str, list[dict[str, str]]] = {}
    for declaration in manifest.get("qras", []):
        qra = _read_json_object(repo_root / str(declaration["path"]))
        for reference in qra.get("references", []):
            source_path = str(reference.get("raw_source_path") or "").replace(
                "\\", "/"
            )
            quote = _normalise_text(str(reference.get("quote") or ""))
            if not source_path or not quote:
                continue
            references.setdefault(source_path, []).append(
                {
                    "qra_id": str(qra["id"]),
                    "reference_id": str(reference.get("reference_id") or ""),
                    "prior_document_id": str(reference.get("document_id") or ""),
                    "prior_block_id": str(reference.get("block_id") or ""),
                    "quote_sha256": _text_sha256(quote),
                    "quote": quote,
                }
            )
    return references


def _leakage_exclusions(
    *,
    selected_documents: list[dict[str, Any]],
    documents: dict[str, dict[str, Any]],
    formal_references: dict[str, list[dict[str, str]]],
) -> list[dict[str, Any]]:
    exclusions: list[dict[str, Any]] = []
    for selected in selected_documents:
        if not selected.get("prior_formal_source_overlap"):
            continue
        document_id = str(selected["document_id"])
        document = documents[document_id]
        prior = formal_references.get(str(selected["path"]), [])
        matched_references: list[dict[str, Any]] = []
        unmatched_references: list[dict[str, str]] = []
        excluded_block_ids: set[str] = set()
        for reference in prior:
            matching_blocks = [
                str(block["block_id"])
                for block in document["blocks"]
                if reference["quote"] in _normalise_text(str(block["text"]))
            ]
            public_reference = {
                key: value for key, value in reference.items() if key != "quote"
            }
            if matching_blocks:
                excluded_block_ids.update(matching_blocks)
                matched_references.append(
                    {**public_reference, "new_block_ids": matching_blocks}
                )
            else:
                unmatched_references.append(public_reference)
        exclusions.append(
            {
                "document_id": document_id,
                "source_path": selected["path"],
                "prior_reference_count": len(prior),
                "matched_reference_count": len(matched_references),
                "unmatched_reference_count": len(unmatched_references),
                "excluded_block_ids": sorted(excluded_block_ids),
                "matched_references": matched_references,
                "unmatched_references": unmatched_references,
            }
        )
    return exclusions


def _domain_summary(
    selected_documents: list[dict[str, Any]],
    documents: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    domains = sorted({str(selected["domain"]) for selected in selected_documents})
    for domain in domains:
        selected = [row for row in selected_documents if row["domain"] == domain]
        domain_documents = [documents[str(row["document_id"])] for row in selected]
        rows.append(
            {
                "domain": domain,
                "document_count": len(domain_documents),
                "block_count": sum(document["block_count"] for document in domain_documents),
                "word_count": sum(
                    len(str(block["text"]).split())
                    for document in domain_documents
                    for block in document["blocks"]
                ),
                "character_count": sum(
                    len(str(block["text"]))
                    for document in domain_documents
                    for block in document["blocks"]
                ),
                "new_source_count": sum(
                    not row["prior_formal_source_overlap"] for row in selected
                ),
                "prior_formal_source_overlap_count": sum(
                    bool(row["prior_formal_source_overlap"]) for row in selected
                ),
                "planned_qra_count": sum(
                    int(row["planned_primary_qra_budget"]) for row in selected
                ),
            }
        )
    return rows


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Optimization Document D candidate audit v0.1",
        "",
        f"Status: `{report['status']}`",
        "",
        "> Automated and agent scope review only; independent human review is not claimed.",
        "",
        "## Summary",
        "",
        "| Domain | Documents | Blocks | Words | New sources | Prior-source overlap | Planned QRAs |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["summary"]["by_domain"]:
        lines.append(
            f"| {row['domain']} | {row['document_count']} | {row['block_count']} | "
            f"{row['word_count']} | {row['new_source_count']} | "
            f"{row['prior_formal_source_overlap_count']} | {row['planned_qra_count']} |"
        )
    lines.extend(["", "## Automated gates", ""])
    for name, passed in report["automated_gates"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} `{name}`")
    lines.extend(
        [
            "",
            "## Leakage control",
            "",
            f"- Overlapping source documents: {len(report['prior_formal_evidence_exclusions'])}",
            "- Any new block containing an exact quote used by the exposed formal QRA set is excluded from later Configuration scopes.",
            "- Quote text is represented by SHA-256 in the registry; the prior QRA artifacts remain the canonical source.",
            "",
            "## Review boundary",
            "",
            "Document candidates are not frozen. Configuration or QRA generation must wait for project-owner review/authorization.",
        ]
    )
    return "\n".join(lines) + "\n"


def audit_optimization_document_candidates(
    *,
    repo_root: Path,
    selection_path: Path = DEFAULT_SELECTION_PATH,
    scope_review_path: Path = DEFAULT_SCOPE_REVIEW_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Validate candidates, case scopes, and prior-formal evidence exclusions."""

    repo_root = repo_root.resolve()
    selection_path = _repository_path(repo_root, selection_path)
    scope_review_path = _repository_path(repo_root, scope_review_path)
    output_path = _repository_path(repo_root, output_path)
    selection = _read_json_object(selection_path)
    scope_review = _read_json_object(scope_review_path)
    generation_manifest_path = repo_root / str(selection["generation_manifest_path"])
    generation_manifest = _read_json_object(generation_manifest_path)

    selected_documents: list[dict[str, Any]] = []
    documents: dict[str, dict[str, Any]] = {}
    text_hashes: list[str] = []
    formats: Counter[str] = Counter()
    for domain, domain_payload in selection["domains"].items():
        for selected_value in domain_payload["selected_documents"]:
            selected = {**dict(selected_value), "domain": domain}
            selected_documents.append(selected)
            path = repo_root / str(selected["document_output_path"])
            document = _read_json_object(path)
            source_path = resolve_corpus_path(repo_root, str(selected["path"]))
            validate_document_record(document=document, source_path=source_path)
            if document["document_id"] != selected["document_id"]:
                raise ArtifactValidationError(f"Document identity mismatch: {path}")
            if document["domain"] != domain:
                raise ArtifactValidationError(f"Document domain mismatch: {path}")
            if document["status"] != "review_candidate" or document["version"] != "0.1":
                raise ArtifactValidationError(f"Document is not a v0.1 review candidate: {path}")
            documents[str(document["document_id"])] = document
            formats[str(selected["format"])] += 1
            text_hashes.append(
                _text_sha256(
                    "\n".join(_normalise_text(str(block["text"])) for block in document["blocks"])
                )
            )

    candidate_manifest = {
        str(item["document_id"]): item
        for item in generation_manifest.get("new_candidates", [])
    }
    manifest_matches = len(candidate_manifest) == len(documents) and all(
        document_id in candidate_manifest
        and sha256_file(
            repo_root
            / next(
                str(selected["document_output_path"])
                for selected in selected_documents
                if selected["document_id"] == document_id
            )
        )
        == candidate_manifest[document_id]["artifact_sha256"]
        for document_id in documents
    )

    scoped_ids: set[str] = set()
    scope_valid = True
    for scope in scope_review.get("documents", []):
        document_id = str(scope.get("document_id") or "")
        scoped_ids.add(document_id)
        document = documents.get(document_id)
        if document is None:
            scope_valid = False
            continue
        existing_ids = {str(block["block_id"]) for block in document["blocks"]}
        allowed_ids = list(scope.get("allowed_block_ids", []))
        if not allowed_ids or not set(allowed_ids).issubset(existing_ids):
            scope_valid = False
    expected_scoped_ids = {
        str(selected["document_id"])
        for selected in selected_documents
        if selected.get("case_scope")
    }
    scope_valid = scope_valid and scoped_ids == expected_scoped_ids

    references = _formal_references(repo_root)
    exclusions = _leakage_exclusions(
        selected_documents=selected_documents,
        documents=documents,
        formal_references=references,
    )
    prior_reference_count = sum(
        int(exclusion["prior_reference_count"]) for exclusion in exclusions
    )
    matched_reference_count = sum(
        int(exclusion["matched_reference_count"]) for exclusion in exclusions
    )
    unmatched_reference_count = sum(
        int(exclusion["unmatched_reference_count"]) for exclusion in exclusions
    )
    excluded_block_count = sum(
        len(exclusion["excluded_block_ids"]) for exclusion in exclusions
    )
    expected_overlap_count = sum(
        bool(selected["prior_formal_source_overlap"])
        for selected in selected_documents
    )
    by_domain = _domain_summary(selected_documents, documents)
    automated_gates = {
        "document_count_24": len(documents) == 24,
        "eight_documents_per_domain": all(
            row["document_count"] == 8 for row in by_domain
        ),
        "twenty_planned_qras_per_domain": all(
            row["planned_qra_count"] == 20 for row in by_domain
        ),
        "all_document_records_valid": len(documents) == len(selected_documents),
        "source_paths_unique": len(
            {str(selected["path"]) for selected in selected_documents}
        )
        == len(selected_documents),
        "document_texts_unique": len(set(text_hashes)) == len(text_hashes),
        "generation_manifest_matches": manifest_matches,
        "generation_replay_unchanged": all(
            item.get("write_state") == "unchanged"
            for item in candidate_manifest.values()
        ),
        "market_case_scopes_valid": scope_valid,
        "overlapping_sources_have_exclusion_registry": (
            len(exclusions) == expected_overlap_count
        ),
        "all_prior_formal_references_mapped": (
            matched_reference_count == prior_reference_count
            and unmatched_reference_count == 0
        ),
    }
    report = {
        "audit_id": "optimization_document_candidate_audit_v0.1",
        "version": "0.1",
        "status": (
            "agent_audit_passed_human_review_pending"
            if all(automated_gates.values())
            else "automated_audit_failed"
        ),
        "audited_at": selection["created_at"],
        "selection": {
            "path": selection_path.relative_to(repo_root).as_posix(),
            "sha256": sha256_file(selection_path),
        },
        "generation_manifest": {
            "path": generation_manifest_path.relative_to(repo_root).as_posix(),
            "sha256": sha256_file(generation_manifest_path),
        },
        "market_case_scope_review": {
            "path": scope_review_path.relative_to(repo_root).as_posix(),
            "sha256": sha256_file(scope_review_path),
            "human_review_claimed": bool(scope_review.get("human_review_claimed")),
        },
        "prior_formal_qra_manifest": {
            "path": FORMAL_QRA_MANIFEST_PATH.as_posix(),
            "sha256": sha256_file(repo_root / FORMAL_QRA_MANIFEST_PATH),
        },
        "summary": {
            "document_count": len(documents),
            "block_count": sum(document["block_count"] for document in documents.values()),
            "format_counts": dict(sorted(formats.items())),
            "new_source_count": sum(
                not selected["prior_formal_source_overlap"]
                for selected in selected_documents
            ),
            "prior_formal_source_overlap_count": expected_overlap_count,
            "prior_formal_reference_count": prior_reference_count,
            "matched_prior_formal_reference_count": matched_reference_count,
            "excluded_block_count": excluded_block_count,
            "by_domain": by_domain,
        },
        "automated_gates": automated_gates,
        "all_automated_gates_passed": all(automated_gates.values()),
        "human_review_complete": False,
        "prior_formal_evidence_exclusions": exclusions,
    }
    write_state = _write_or_verify(output_path, report, overwrite=overwrite)
    markdown_path = output_path.with_suffix(".md")
    markdown = _markdown(report)
    if markdown_path.exists():
        existing_markdown = markdown_path.read_text(encoding="utf-8")
        if existing_markdown != markdown and not overwrite:
            raise ArtifactValidationError(
                f"Document audit Markdown differs; rerun with --overwrite: {markdown_path}"
            )
    if not markdown_path.exists() or overwrite:
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(markdown, encoding="utf-8")
    return {
        "report": report,
        "output_path": output_path.relative_to(repo_root).as_posix(),
        "markdown_path": markdown_path.relative_to(repo_root).as_posix(),
        "write_state": write_state,
    }
