"""Regression checks for semantically curated primary-law Configurations."""

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
CURATION_ROOT = (
    FORMAL_ROOT
    / "configurations/primary_law_candidates/reviewed_v0.2"
)
SPEC_PATH = (
    FORMAL_ROOT
    / "configurations/primary_law_candidates/"
    "primary_law_configuration_semantic_curation_spec_v0.2.json"
)
GENERATION_MANIFEST_PATH = CURATION_ROOT / "generation_manifest_v0.2.json"
REVIEW_PATH = CURATION_ROOT / "agent_semantic_review_v0.2.json"
FULL_MANIFEST_PATH = (
    FORMAL_ROOT
    / "configurations/formal_test/"
    "formal_test_configuration_manifest_v1.3_semantic_candidate.json"
)
DOCUMENT_MANIFEST_PATH = (
    FORMAL_ROOT / "documents/formal_test_document_manifest_v1.2.json"
)
RECORD_SCHEMA_PATH = (
    FORMAL_ROOT / "configurations/configuration_record_schema_v1.1.json"
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _documents() -> dict[str, dict]:
    manifest = _read_json(DOCUMENT_MANIFEST_PATH)
    return {
        declaration["document_id"]: _read_json(
            REPO_ROOT / declaration["path"]
        )
        for declaration in manifest["documents"]
    }


def test_curation_spec_decides_all_candidates_and_removes_34_values() -> None:
    specification = _read_json(SPEC_PATH)
    generation = _read_json(GENERATION_MANIFEST_PATH)

    assert specification["human_line_by_line_review_claimed"] is False
    assert len(specification["configurations"]) == 17
    assert sum(
        len(item["remove_values"])
        for item in specification["configurations"]
    ) == 34
    assert generation["status"] == "agent_semantic_review_passed"
    assert generation["candidate_count"] == 17
    assert generation["removed_value_count"] == 34
    assert generation["constraint_update_count"] == 7
    assert generation["human_line_by_line_review_claimed"] is False


def test_curated_candidates_are_valid_pending_and_source_minimal() -> None:
    generation = _read_json(GENERATION_MANIFEST_PATH)
    record_schema = _read_json(RECORD_SCHEMA_PATH)
    documents = _documents()
    domain_schemas = {
        domain: _read_json(REPO_ROOT / path)
        for domain, path in DOMAIN_SCHEMA_PATHS.items()
    }

    for declaration in generation["candidates"]:
        candidate_path = REPO_ROOT / declaration["path"]
        assert sha256_file(candidate_path) == declaration["file_sha256"]
        candidate = _read_json(candidate_path)
        assert candidate["status"] == "development_draft"
        assert candidate["generation_record"]["human_review_status"] == "pending"
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

        allowed_pairs = {
            (source["document_id"], block_id)
            for source in candidate["source_documents"]
            for block_id in source["allowed_block_ids"]
        }
        used_pairs = {
            (item["document_id"], item["block_id"])
            for item in candidate["provenance"]
        }
        assert allowed_pairs == used_pairs


def test_known_conflicting_or_low_quality_values_are_absent() -> None:
    generation = _read_json(GENERATION_MANIFEST_PATH)
    combined = "\n".join(
        (REPO_ROOT / declaration["path"]).read_text(encoding="utf-8")
        for declaration in generation["candidates"]
    )

    removed_phrases = (
        "GbR nein",
        "Das Jobcenter entscheidet",
        "Stellungnahme der fachkundigen Stelle zur Tragfähigkeit der "
        "Existenzgründung nach § 93 Abs.",
        "ElsterFormular",
        "15,83 Prozent",
        "KG faktisch eine haftungsbeschränkte Gesellschaft",
        "3 EStG geregelt",
    )
    for phrase in removed_phrases:
        assert phrase not in combined

    assert "Gesellschaftsregister anmelden" in combined
    assert "mindestens 150 Tage Anspruch auf Arbeitslosengeld" in combined
    assert "Veranlagungszeiträume bis 2027" in combined


def test_agent_review_is_transparent_and_full_manifest_is_additive() -> None:
    review = _read_json(REVIEW_PATH)
    manifest = _read_json(FULL_MANIFEST_PATH)
    revised = [
        item
        for item in manifest["configurations"]
        if item["resolution"] == "primary_law_semantic_candidate"
    ]
    carried = [
        item
        for item in manifest["configurations"]
        if item["resolution"] == "carried_forward"
    ]

    assert review["status"] == "agent_semantic_review_passed"
    assert review["unresolved_blocker_count"] == 0
    assert review["agent_semantic_review_claimed"] is True
    assert review["human_line_by_line_review_claimed"] is False
    assert review["human_review_state"] == "pending"
    assert manifest["status"] == "review_candidate"
    assert manifest["configuration_count"] == 60
    assert len(revised) == manifest["revised_candidate_count"] == 17
    assert len(carried) == manifest["carried_forward_count"] == 43
    assert all(item["freeze_record_path"] is None for item in revised)
