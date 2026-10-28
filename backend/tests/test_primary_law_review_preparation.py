"""Regression checks for non-freezing primary-law review artifacts."""

import json
from pathlib import Path

from apps.evaluations.rageval_artifacts import sha256_file


REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_DOCUMENT_MANIFEST_PATH = (
    REPO_ROOT
    / "docs/evaluation/formal_dataset/documents/"
    "formal_test_document_manifest_v1.1.json"
)
CANDIDATE_DOCUMENT_MANIFEST_PATH = (
    REPO_ROOT
    / "docs/evaluation/formal_dataset/documents/"
    "formal_test_document_manifest_v1.2_candidate.json"
)
AGENT_REVIEW_PATH = (
    REPO_ROOT
    / "docs/evaluation/formal_dataset/documents/primary_law_candidates/"
    "primary_law_agent_review_v0.1.json"
)
IMPACT_MATRIX_PATH = (
    REPO_ROOT
    / "docs/evaluation/formal_dataset/configurations/formal_test/"
    "primary_law_configuration_impact_matrix_v0.1.json"
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_primary_law_agent_review_passes_without_freezing() -> None:
    review = _read_json(AGENT_REVIEW_PATH)

    assert review["status"] == "agent_checks_passed_human_review_pending"
    assert review["human_review_state"] == "pending"
    assert review["candidate_count"] == 14
    assert review["block_count"] == 257
    assert review["key_fact_check_count"] == 46
    assert review["blocking_failure_count"] == 0
    assert all(item["agent_result"] == "pass_human_review_pending" for item in review["documents"])
    assert all(not item["failures"] for item in review["documents"])
    assert all(
        check["passed"]
        for item in review["documents"]
        for check in item["key_fact_checks"]
    )


def test_document_manifest_v1_2_is_additive_and_remains_a_candidate() -> None:
    base = _read_json(BASE_DOCUMENT_MANIFEST_PATH)
    candidate = _read_json(CANDIDATE_DOCUMENT_MANIFEST_PATH)

    assert base["status"] == "frozen"
    assert candidate["status"] == "review_candidate"
    assert candidate["document_count"] == 38
    assert candidate["carried_forward_frozen_count"] == 24
    assert candidate["added_review_candidate_count"] == 14
    assert candidate["domain_document_counts"] == {
        "funding_preparation": 9,
        "legal_form_selection": 21,
        "market_validation": 8,
    }

    base_ids = [item["document_id"] for item in base["documents"]]
    candidate_ids = [item["document_id"] for item in candidate["documents"]]
    assert candidate_ids[: len(base_ids)] == base_ids
    assert len(candidate_ids) == len(set(candidate_ids))

    for declaration in candidate["documents"]:
        path = REPO_ROOT / declaration["path"]
        assert sha256_file(path) == declaration["file_sha256"]
        if declaration["resolution"] == "added_review_candidate":
            assert declaration["freeze_record_path"] is None
            assert declaration["review_status"] == (
                "agent_checks_passed_human_review_pending"
            )


def test_configuration_impact_matrix_resolves_exact_candidate_blocks() -> None:
    matrix = _read_json(IMPACT_MATRIX_PATH)

    assert matrix["status"] == "review_candidate"
    assert matrix["configuration_count"] == 60
    assert matrix["configuration_revision_count"] == 17
    assert matrix["qra_regeneration_count"] == 17
    assert matrix["schema_revision_count"] == 0
    assert matrix["impact_type_counts"] == {
        "no_primary_law_change": 43,
        "required_authority_augmentation": 14,
        "supplementary_authority_cross_check": 3,
    }

    candidate_cache: dict[str, dict] = {}
    for item in matrix["matrix"]:
        assert sha256_file(REPO_ROOT / item["current_path"]) == item["current_file_sha256"]
        for source in item["primary_law_sources"]:
            document_id = source["document_id"]
            document = candidate_cache.setdefault(
                document_id,
                _read_json(REPO_ROOT / source["document_path"]),
            )
            known_block_ids = {block["block_id"] for block in document["blocks"]}
            assert set(source["allowed_block_ids"]) <= known_block_ids
            assert source["document_content_sha256"] == document["content_sha256"]


def test_tax_rate_and_distribution_gaps_are_present_in_review_scope() -> None:
    candidate_dir = (
        REPO_ROOT
        / "docs/evaluation/formal_dataset/documents/primary_law_candidates"
    )
    estg = _read_json(candidate_dir / "D-LFS-022_estg_v0.1.json")
    ustg = _read_json(candidate_dir / "D-LFS-025_ustg_v0.1.json")
    estg_sections = {block["law_section"] for block in estg["blocks"]}
    ustg_sections = {block["law_section"] for block in ustg["blocks"]}

    assert {"§ 20", "§ 32d", "§ 34a"} <= estg_sections
    assert "§ 12" in ustg_sections

    matrix = _read_json(IMPACT_MATRIX_PATH)
    by_configuration = {
        item["configuration_id"]: item for item in matrix["matrix"]
    }
    nc_sources = {
        source["law_abbreviation"]: source
        for source in by_configuration["C-FT-LFS-NC-002"][
            "primary_law_sources"
        ]
    }
    assert nc_sources["UStG"]["law_sections"] == ["12"]
    assert nc_sources["KStG"]["law_sections"] == ["23"]
    assert nc_sources["EStG"]["law_sections"] == ["34a"]
