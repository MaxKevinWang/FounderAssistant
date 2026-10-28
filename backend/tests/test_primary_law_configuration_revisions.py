"""Regression checks for primary-law Configuration review candidates."""

import json
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

from apps.evaluations.formal_configuration_generation import DOMAIN_SCHEMA_PATHS
from apps.evaluations.rageval_artifacts import (
    sha256_file,
    validate_configuration_record,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
FORMAL_ROOT = REPO_ROOT / "docs/evaluation/formal_dataset"
BASE_MANIFEST_PATH = (
    FORMAL_ROOT
    / "configurations/formal_test/formal_test_configuration_manifest_v1.2.json"
)
DOCUMENT_MANIFEST_PATH = (
    FORMAL_ROOT / "documents/formal_test_document_manifest_v1.2.json"
)
IMPACT_MATRIX_PATH = (
    FORMAL_ROOT
    / "configurations/formal_test/primary_law_configuration_impact_matrix_v0.1.json"
)
GENERATION_MANIFEST_PATH = (
    FORMAL_ROOT
    / "configurations/primary_law_candidates/generation_manifest_v0.1.json"
)
REVIEW_PATH = (
    FORMAL_ROOT
    / "configurations/primary_law_candidates/"
    "primary_law_configuration_agent_review_v0.1.json"
)
CANDIDATE_MANIFEST_PATH = (
    FORMAL_ROOT
    / "configurations/formal_test/formal_test_configuration_manifest_v1.3_candidate.json"
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _documents() -> dict[str, dict]:
    manifest = _read_json(DOCUMENT_MANIFEST_PATH)
    return {
        item["document_id"]: _read_json(REPO_ROOT / item["path"])
        for item in manifest["documents"]
    }


def test_generation_manifest_covers_exactly_the_17_impacts() -> None:
    impact = _read_json(IMPACT_MATRIX_PATH)
    generation = _read_json(GENERATION_MANIFEST_PATH)
    expected_ids = {
        item["configuration_id"]
        for item in impact["matrix"]
        if item["configuration_revision_required"]
    }

    assert generation["status"] == "review_candidate"
    assert generation["candidate_count"] == 17
    assert {item["configuration_id"] for item in generation["candidates"]} == expected_ids
    assert generation["legal_value_addition_count"] > 17
    assert generation["legal_evidence_count"] > generation["legal_value_addition_count"]


def test_every_candidate_is_pending_and_mechanically_valid() -> None:
    generation = _read_json(GENERATION_MANIFEST_PATH)
    documents = _documents()
    record_schema_path = (
        FORMAL_ROOT / "configurations/configuration_record_schema_v1.1.json"
    )
    record_schema = _read_json(record_schema_path)
    domain_schemas = {
        domain: _read_json(REPO_ROOT / path)
        for domain, path in DOMAIN_SCHEMA_PATHS.items()
    }

    for declaration in generation["candidates"]:
        path = REPO_ROOT / declaration["path"]
        assert sha256_file(path) == declaration["file_sha256"]
        candidate = _read_json(path)
        assert candidate["status"] == "development_draft"
        assert candidate["generation_record"]["human_review_status"] == "pending"
        assert candidate["source_scope"] == "multiple_documents"
        assert candidate["supersedes"].endswith(
            f"v{declaration['predecessor_version']}"
        )
        assert not any(
            "primary_law_candidates" in source["document_id"]
            for source in candidate["source_documents"]
        )
        errors = list(
            Draft202012Validator(
                record_schema,
                format_checker=FormatChecker(),
            ).iter_errors(candidate)
        )
        assert not errors
        validate_configuration_record(
            configuration=candidate,
            schema=domain_schemas[candidate["domain"]],
            documents=documents,
        )
        law_ids = set(declaration["primary_law_document_ids"])
        assert law_ids <= {
            source["document_id"] for source in candidate["source_documents"]
        }
        assert law_ids <= {
            provenance["document_id"] for provenance in candidate["provenance"]
        }


def test_full_set_manifest_is_non_frozen_and_additive() -> None:
    base = _read_json(BASE_MANIFEST_PATH)
    candidate = _read_json(CANDIDATE_MANIFEST_PATH)
    revised = [
        item
        for item in candidate["configurations"]
        if item["resolution"] == "primary_law_review_candidate"
    ]
    carried = [
        item
        for item in candidate["configurations"]
        if item["resolution"] == "carried_forward_if_frozen"
    ]

    assert base["status"] == "frozen"
    assert candidate["status"] == "review_candidate"
    assert candidate["configuration_count"] == 60
    assert len(revised) == candidate["revised_candidate_count"] == 17
    assert len(carried) == candidate["carried_forward_count"] == 43
    assert candidate["document_manifest"]["sha256"] == sha256_file(
        DOCUMENT_MANIFEST_PATH
    )
    assert all(item["freeze_record_path"] is None for item in revised)


def test_review_exposes_conflicts_without_claiming_semantic_approval() -> None:
    review = _read_json(REVIEW_PATH)
    by_id = {
        item["configuration_id"]: item for item in review["configurations"]
    }

    assert review["status"] == "mechanically_valid_semantic_review_pending"
    assert review["blocking_failure_count"] == 0
    assert review["semantic_review_claimed"] is False
    assert review["human_review_state"] == "pending"
    assert review["attention_required_count"] >= 8
    assert any("GbR" in note for note in by_id["C-FT-LFS-F-001"]["attention_notes"])
    assert any("Jobcenter" in note for note in by_id["C-FT-FP-F-001"]["attention_notes"])
    assert any("future rate" in note for note in by_id["C-FT-LFS-NC-002"]["attention_notes"])
