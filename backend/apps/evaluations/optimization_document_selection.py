"""Prepare an auditable source-selection manifest for optimization QRAs."""

from __future__ import annotations

import json
import re
from pathlib import Path

from apps.evaluations.corpus_paths import resolve_corpus_path
from typing import Any

from apps.evaluations.rageval_artifacts import (
    ArtifactValidationError,
    sha256_file,
    write_json_artifact,
)


DEFAULT_PLAN_PATH = Path(
    "docs/evaluation/optimization_dataset/source_selection/"
    "optimization_source_plan_v0.1.json"
)
SEED_MANIFEST_TEMPLATE = (
    "docs/evaluation/schemas/{domain}/seed_document_manifest_{domain}_v1.0.json"
)
FORMAL_SELECTION_PATH = Path(
    "docs/evaluation/formal_dataset/source_selection/"
    "formal_test_source_selection_v1.0.json"
)
FORMAL_QRA_MANIFEST_PATH = Path(
    "docs/evaluation/formal_dataset/qra/formal_test/"
    "formal_test_qra_manifest_v1.0.json"
)
RAG_METADATA_PATH = Path("backend/rag_index/metadata.json")
_CAPTURE_TIMESTAMP_RE = re.compile(r"__(\d{8})T(\d{6})Z\.html$")


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


def _captured_at(source_path: str) -> str | None:
    match = _CAPTURE_TIMESTAMP_RE.search(source_path.replace("\\", "/"))
    if not match:
        return None
    date, time = match.groups()
    return (
        f"{date[0:4]}-{date[4:6]}-{date[6:8]}T"
        f"{time[0:2]}:{time[2:4]}:{time[4:6]}Z"
    )


def _seed_documents(repo_root: Path, domains: list[str]) -> dict[tuple[str, str], dict[str, Any]]:
    documents: dict[tuple[str, str], dict[str, Any]] = {}
    for domain in domains:
        path = repo_root / SEED_MANIFEST_TEMPLATE.format(domain=domain)
        manifest = _read_json_object(path)
        for document in manifest.get("documents", []):
            key = (domain, str(document.get("document_id") or ""))
            if not key[1] or key in documents:
                raise ArtifactValidationError(f"Duplicate or empty seed document ID: {key}")
            documents[key] = dict(document)
    return documents


def _rag_documents(repo_root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    metadata_path = repo_root / RAG_METADATA_PATH
    metadata = _read_json_object(metadata_path)
    documents: dict[str, dict[str, Any]] = {}
    for document in metadata.get("documents", []):
        source_path = str(document.get("source_path") or "").replace("\\", "/")
        repository_path = f"data/{source_path.lstrip('/')}"
        if repository_path in documents:
            raise ArtifactValidationError(
                f"Duplicate source path in RAG metadata: {repository_path}"
            )
        documents[repository_path] = dict(document)
    return documents, metadata


def _write_or_verify(path: Path, payload: dict[str, Any], *, overwrite: bool) -> str:
    if path.exists():
        existing = _read_json_object(path)
        if existing == payload:
            return "unchanged"
        if not overwrite:
            raise ArtifactValidationError(
                f"Prepared selection differs from existing output; rerun with --overwrite: {path}"
            )
        write_json_artifact(path, payload)
        return "overwritten"
    write_json_artifact(path, payload)
    return "created"


def prepare_optimization_document_selection(
    *,
    repo_root: Path,
    plan_path: Path = DEFAULT_PLAN_PATH,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Resolve a reviewed source plan into a hash-pinned selection manifest."""

    repo_root = repo_root.resolve()
    plan_path = _repository_path(repo_root, plan_path)
    plan = _read_json_object(plan_path)
    domains = list(plan.get("domains", {}))
    if set(domains) != {
        "legal_form_selection",
        "funding_preparation",
        "market_validation",
    }:
        raise ArtifactValidationError("Optimization source plan must define all three domains.")

    seeds = _seed_documents(repo_root, domains)
    rag_documents, rag_metadata = _rag_documents(repo_root)
    formal_selection_path = repo_root / FORMAL_SELECTION_PATH
    formal_selection = _read_json_object(formal_selection_path)
    prior_formal_paths = {
        str(document["path"]).replace("\\", "/")
        for domain in formal_selection["domains"].values()
        for document in domain["selected_documents"]
    }

    prepared_domains: dict[str, Any] = {}
    all_document_ids: set[str] = set()
    all_source_paths: set[str] = set()
    new_source_count = 0
    reused_source_count = 0
    for domain, domain_plan in plan["domains"].items():
        selected_documents: list[dict[str, Any]] = []
        planned_budget = 0
        for specification in domain_plan.get("selected_documents", []):
            seed_id = str(specification.get("seed_document_id") or "")
            seed = dict(seeds.get((domain, seed_id), {}))
            source = {**seed, **dict(specification)}
            document_id = str(source.get("document_id") or "")
            source_path_value = str(source.get("path") or "").replace("\\", "/")
            if not document_id or document_id in all_document_ids:
                raise ArtifactValidationError(
                    f"Duplicate or empty optimization document ID: {document_id!r}"
                )
            if not source_path_value or source_path_value in all_source_paths:
                raise ArtifactValidationError(
                    f"Duplicate or empty optimization source path: {source_path_value!r}"
                )
            all_document_ids.add(document_id)
            all_source_paths.add(source_path_value)

            source_path = resolve_corpus_path(repo_root, source_path_value)
            if not source_path.exists():
                raise ArtifactValidationError(f"Optimization source is missing: {source_path}")
            actual_sha256 = sha256_file(source_path)
            seed_sha256 = str(seed.get("sha256") or "")
            if seed_sha256 and actual_sha256 != seed_sha256:
                raise ArtifactValidationError(
                    f"Seed source SHA-256 changed for {seed_id}: "
                    f"{seed_sha256} != {actual_sha256}"
                )

            rag_document = rag_documents.get(source_path_value)
            if rag_document is None:
                raise ArtifactValidationError(
                    f"Optimization source is absent from the current RAG index: {source_path_value}"
                )
            overlaps_formal = source_path_value in prior_formal_paths
            new_source_count += int(not overlaps_formal)
            reused_source_count += int(overlaps_formal)
            planned_primary_qra_budget = int(
                source.get("planned_primary_qra_budget") or 0
            )
            if planned_primary_qra_budget < 1:
                raise ArtifactValidationError(
                    f"{document_id} must have a positive planned QRA budget."
                )
            planned_budget += planned_primary_qra_budget

            output_name = str(source.get("output_name") or "").strip()
            if not output_name:
                raise ArtifactValidationError(f"{document_id} has no output_name")
            selected = {
                "document_id": document_id,
                "seed_document_id": seed_id,
                "title": str(source.get("title") or rag_document.get("source_name") or ""),
                "path": source_path_value,
                "format": str(source.get("format") or source_path.suffix.lstrip(".")),
                "source_organization": source.get("source_organization"),
                "source_group": source.get("source_group"),
                "source_type": source.get("source_type"),
                "publication_date": source.get("publication_date"),
                "source_url": source.get("source_url") or rag_document.get("url"),
                "accessed_at": source.get("accessed_at") or _captured_at(source_path_value),
                "raw_bytes": source_path.stat().st_size,
                "raw_sha256": actual_sha256,
                "rag_index_char_count": int(rag_document.get("char_count") or 0),
                "rag_index_chunk_count": int(rag_document.get("chunk_count") or 0),
                "schema_role": source.get("schema_role"),
                "planned_primary_qra_budget": planned_primary_qra_budget,
                "candidate_question_types": source.get("candidate_question_types", []),
                "selection_reason": source.get("selection_reason"),
                "case_scope": source.get("case_scope"),
                "prior_formal_source_overlap": overlaps_formal,
                "development_leakage_guard": (
                    "Exclude every exact evidence span used by the exposed formal QRA set."
                    if overlaps_formal
                    else "No raw-source overlap with the exposed formal source selection."
                ),
                "temporal_validation_required_before_qra_freeze": bool(
                    source.get("temporal_validation_required_before_qra_freeze", True)
                ),
                "document_action": "generate_candidate",
                "document_output_path": (
                    "docs/evaluation/optimization_dataset/documents/candidates/"
                    f"{output_name}"
                ),
            }
            if source.get("html_extraction"):
                selected["html_extraction"] = source["html_extraction"]
            selected_documents.append(selected)

        target = int(domain_plan.get("qra_target") or 0)
        if len(selected_documents) != int(domain_plan.get("document_target") or 0):
            raise ArtifactValidationError(
                f"{domain} selected-document count does not match its target."
            )
        if planned_budget != target:
            raise ArtifactValidationError(
                f"{domain} planned QRA budget {planned_budget} does not match target {target}."
            )
        prepared_domains[domain] = {
            key: value
            for key, value in domain_plan.items()
            if key != "selected_documents"
        }
        prepared_domains[domain]["selected_document_count"] = len(selected_documents)
        prepared_domains[domain]["selected_documents"] = selected_documents

    selection = {
        "selection_id": plan["selection_id"],
        "version": plan["version"],
        "status": "review_candidate",
        "intended_split": "optimization",
        "created_at": plan["created_at"],
        "language": plan["language"],
        "jurisdiction": plan["jurisdiction"],
        "method_basis": "RAGEval-style Document D -> Configuration C -> QRA construction",
        "generation_id": plan["generation_id"],
        "generation_manifest_path": plan["generation_manifest_path"],
        "protocol": plan["protocol"],
        "selection_policy": plan["selection_policy"],
        "source_snapshot": {
            "rag_metadata_path": RAG_METADATA_PATH.as_posix(),
            "rag_metadata_sha256": sha256_file(repo_root / RAG_METADATA_PATH),
            "rag_metadata_generated_at": rag_metadata.get("generated_at"),
            "rag_document_count": rag_metadata.get("document_count"),
            "rag_chunk_count": rag_metadata.get("chunk_count"),
            "embedding_model": rag_metadata.get("embedding_model"),
        },
        "dependencies": {
            "plan_path": plan_path.relative_to(repo_root).as_posix(),
            "plan_sha256": sha256_file(plan_path),
            "prior_formal_selection_path": FORMAL_SELECTION_PATH.as_posix(),
            "prior_formal_selection_sha256": sha256_file(formal_selection_path),
            "prior_formal_qra_manifest_path": FORMAL_QRA_MANIFEST_PATH.as_posix(),
            "prior_formal_qra_manifest_sha256": sha256_file(
                repo_root / FORMAL_QRA_MANIFEST_PATH
            ),
        },
        "document_count": len(all_document_ids),
        "new_source_count": new_source_count,
        "prior_formal_source_overlap_count": reused_source_count,
        "domains": prepared_domains,
    }
    output_path = _repository_path(repo_root, str(plan["selection_output_path"]))
    write_state = _write_or_verify(output_path, selection, overwrite=overwrite)
    return {
        "selection": selection,
        "selection_path": output_path.relative_to(repo_root).as_posix(),
        "selection_sha256": sha256_file(output_path),
        "write_state": write_state,
    }
