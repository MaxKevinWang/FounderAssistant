"""Integrity checks for the additive primary-law Document D freeze."""

import json
from pathlib import Path

from apps.evaluations.corpus_paths import resolve_corpus_path

from apps.evaluations.rageval_artifacts import sha256_file, validate_document_record


REPO_ROOT = Path(__file__).resolve().parents[2]
DOCUMENT_ROOT = REPO_ROOT / "docs/evaluation/formal_dataset/documents"
CANDIDATE_MANIFEST_PATH = DOCUMENT_ROOT / "formal_test_document_manifest_v1.2_candidate.json"
FROZEN_MANIFEST_PATH = DOCUMENT_ROOT / "formal_test_document_manifest_v1.2.json"
AUTHORIZATION_PATH = (
    DOCUMENT_ROOT
    / "primary_law_candidates/primary_law_freeze_authorization_v1.0.json"
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_primary_law_authorization_is_explicit_and_transparent() -> None:
    authorization = _read_json(AUTHORIZATION_PATH)

    assert authorization["status"] == "authorized_to_freeze_after_agent_review"
    assert authorization["project_owner_instruction"] == "继续"
    assert authorization["human_line_by_line_review_claimed"] is False
    assert authorization["agent_review"]["blocking_failure_count"] == 0
    assert authorization["agent_review"]["key_fact_check_count"] == 46
    assert len(authorization["authorized_document_ids"]) == 14


def test_document_manifest_v1_2_adds_only_frozen_primary_law_documents() -> None:
    candidate = _read_json(CANDIDATE_MANIFEST_PATH)
    frozen = _read_json(FROZEN_MANIFEST_PATH)
    predecessor_path = REPO_ROOT / frozen["predecessor_manifest"]["path"]
    predecessor = _read_json(predecessor_path)

    assert frozen["status"] == "frozen"
    assert frozen["document_count"] == 38
    assert frozen["carried_forward_frozen_count"] == 24
    assert frozen["added_primary_law_count"] == 14
    assert frozen["domain_document_counts"] == {
        "funding_preparation": 9,
        "legal_form_selection": 21,
        "market_validation": 8,
    }
    assert frozen["documents"][:24] == predecessor["documents"]
    assert frozen["authorization"]["human_line_by_line_review_claimed"] is False
    assert frozen["candidate_manifest"]["sha256"] == sha256_file(
        CANDIDATE_MANIFEST_PATH
    )
    assert candidate["status"] == "review_candidate"


def test_frozen_primary_law_documents_preserve_candidate_content() -> None:
    candidate = _read_json(CANDIDATE_MANIFEST_PATH)
    frozen = _read_json(FROZEN_MANIFEST_PATH)
    candidate_by_id = {
        item["document_id"]: item
        for item in candidate["documents"]
        if item["resolution"] == "added_review_candidate"
    }
    added = frozen["documents"][24:]

    assert len(added) == 14
    for declaration in added:
        assert declaration["resolution"] == "added_frozen_primary_law"
        document_id = declaration["document_id"]
        candidate_path = REPO_ROOT / candidate_by_id[document_id]["path"]
        frozen_path = REPO_ROOT / declaration["path"]
        freeze_record_path = REPO_ROOT / declaration["freeze_record_path"]
        candidate_document = _read_json(candidate_path)
        frozen_document = _read_json(frozen_path)
        freeze_record = _read_json(freeze_record_path)

        assert frozen_document["version"] == "1.0"
        assert frozen_document["status"] == "frozen"
        assert frozen_document["blocks"] == candidate_document["blocks"]
        assert frozen_document["content_sha256"] == candidate_document["content_sha256"]
        assert declaration["file_sha256"] == sha256_file(frozen_path)
        assert declaration["freeze_record_sha256"] == sha256_file(freeze_record_path)
        assert freeze_record["authorization"]["human_line_by_line_review_claimed"] is False
        source_path = resolve_corpus_path(REPO_ROOT, frozen_document["source"]["path"])
        validate_document_record(document=frozen_document, source_path=source_path)


def test_document_manifest_v1_2_freeze_record_binds_all_inputs() -> None:
    manifest = _read_json(FROZEN_MANIFEST_PATH)
    record_path = DOCUMENT_ROOT / "formal_test_document_manifest_v1.2_freeze_record.json"
    record = _read_json(record_path)

    assert record["status"] == "frozen"
    assert record["file_sha256"] == sha256_file(FROZEN_MANIFEST_PATH)
    assert record["candidate_manifest_sha256"] == sha256_file(CANDIDATE_MANIFEST_PATH)
    assert record["authorization_sha256"] == sha256_file(AUTHORIZATION_PATH)
    assert record["human_line_by_line_review_claimed"] is False
    assert record["added_document_ids"] == [
        item["document_id"] for item in manifest["documents"][24:]
    ]
