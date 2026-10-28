"""Regression checks for frozen primary-law Configuration manifest v1.3."""

import json
from pathlib import Path

from apps.evaluations.rageval_artifacts import sha256_file


REPO_ROOT = Path(__file__).resolve().parents[2]
FORMAL_ROOT = REPO_ROOT / "docs/evaluation/formal_dataset"
MANIFEST_PATH = (
    FORMAL_ROOT
    / "configurations/formal_test/formal_test_configuration_manifest_v1.3.json"
)
FREEZE_RECORD_PATH = MANIFEST_PATH.with_name(
    "formal_test_configuration_manifest_v1.3_freeze_record.json"
)
PREDECESSOR_PATH = (
    FORMAL_ROOT
    / "configurations/formal_test/formal_test_configuration_manifest_v1.2.json"
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_manifest_v1_3_is_frozen_balanced_and_hash_bound() -> None:
    manifest = _read_json(MANIFEST_PATH)
    freeze_record = _read_json(FREEZE_RECORD_PATH)

    assert manifest["status"] == "frozen"
    assert manifest["version"] == "1.3"
    assert manifest["configuration_count"] == 60
    assert manifest["revised_configuration_count"] == 17
    assert manifest["carried_forward_count"] == 43
    assert manifest["answerable_count"] == 54
    assert manifest["unanswerable_count"] == 6
    assert freeze_record["file_sha256"] == sha256_file(MANIFEST_PATH)
    assert freeze_record["human_line_by_line_review_claimed"] is False
    assert freeze_record["legal_expert_review_claimed"] is False


def test_revisions_are_frozen_without_false_human_approval() -> None:
    manifest = _read_json(MANIFEST_PATH)
    revised = [
        item
        for item in manifest["configurations"]
        if item["resolution"] == "primary_law_revision"
    ]

    assert len(revised) == 17
    for declaration in revised:
        configuration_path = REPO_ROOT / declaration["path"]
        freeze_record_path = REPO_ROOT / declaration["freeze_record_path"]
        assert sha256_file(configuration_path) == declaration["file_sha256"]
        assert (
            sha256_file(freeze_record_path)
            == declaration["freeze_record_sha256"]
        )
        configuration = _read_json(configuration_path)
        freeze_record = _read_json(freeze_record_path)
        assert configuration["status"] == "frozen"
        assert configuration["generation_record"]["human_review_status"] == "pending"
        assert (
            freeze_record["freeze_authorization"][
                "human_line_by_line_review_claimed"
            ]
            is False
        )
        assert (
            freeze_record["freeze_authorization"]["legal_expert_review_claimed"]
            is False
        )
        assert freeze_record["agent_semantic_review"]["unresolved_blocker_count"] == 0


def test_43_unaffected_configurations_are_byte_identical_carry_forwards() -> None:
    predecessor = _read_json(PREDECESSOR_PATH)
    manifest = _read_json(MANIFEST_PATH)
    predecessor_by_id = {
        item["configuration_id"]: item for item in predecessor["configurations"]
    }
    carried = [
        item
        for item in manifest["configurations"]
        if item["resolution"] == "carried_forward"
    ]

    assert len(carried) == 43
    for declaration in carried:
        old = predecessor_by_id[declaration["configuration_id"]]
        assert declaration["path"] == old["path"]
        assert declaration["file_sha256"] == old["file_sha256"]
        assert declaration["freeze_record_path"] == old["freeze_record_path"]
        assert (
            declaration["freeze_record_sha256"]
            == old["freeze_record_sha256"]
        )


def test_authorization_binds_the_frozen_semantic_review() -> None:
    manifest = _read_json(MANIFEST_PATH)
    authorization_path = REPO_ROOT / manifest["freeze_authorization"]["path"]
    review_path = REPO_ROOT / manifest["agent_semantic_review"]["path"]
    authorization = _read_json(authorization_path)

    assert sha256_file(authorization_path) == manifest["freeze_authorization"]["sha256"]
    assert sha256_file(review_path) == manifest["agent_semantic_review"]["sha256"]
    assert authorization["source_instruction"] == "继续"
    assert authorization["human_line_by_line_review_claimed"] is False
    assert authorization["legal_expert_review_claimed"] is False
    assert (
        authorization["agent_semantic_review"]["sha256"]
        == sha256_file(review_path)
    )
