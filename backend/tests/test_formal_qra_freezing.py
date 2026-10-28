"""Integrity checks for the immutable formal-test QRA v1.0 set."""

from collections import Counter
import json
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

from apps.evaluations.rageval_artifacts import (
    sha256_file,
    validate_qra_record,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
FORMAL_ROOT = REPO_ROOT / "docs/evaluation/formal_dataset"
MANIFEST_PATH = FORMAL_ROOT / "qra/formal_test/formal_test_qra_manifest_v1.0.json"
MANIFEST_RECORD_PATH = (
    FORMAL_ROOT
    / "qra/formal_test/formal_test_qra_manifest_v1.0_freeze_record.json"
)

EXPECTED_TYPE_COUNTS = {
    "Factual": 5,
    "Summarization": 3,
    "Multi-hop Reasoning": 3,
    "Information Integration": 3,
    "Numerical Comparison": 2,
    "Temporal Sequence": 2,
    "Unanswerable": 2,
}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_frozen_manifest_has_a_complete_transparent_review_chain() -> None:
    manifest = _read_json(MANIFEST_PATH)
    freeze_record = _read_json(MANIFEST_RECORD_PATH)

    assert manifest["manifest_id"] == "formal_test_qra_manifest_v1.0"
    assert manifest["version"] == "1.0"
    assert manifest["status"] == "frozen"
    assert manifest["split"] == "test"
    assert manifest["review_status"] == "single_checked"
    assert manifest["double_checked"] is False
    assert manifest["qra_count"] == 60
    assert manifest["primary_law_revised_count"] == 17
    assert manifest["semantic_carry_forward_count"] == 43
    assert manifest["answerable_count"] == 54
    assert manifest["unanswerable_count"] == 6
    assert manifest["review_limitations"] == {
        "documented_human_reviewers": 1,
        "independent_second_human_review_claimed": False,
        "post_correction_human_line_by_line_review_claimed": False,
        "legal_expert_review_claimed": False,
    }
    assert freeze_record["file_sha256"] == sha256_file(MANIFEST_PATH)
    assert freeze_record["independent_second_human_review_claimed"] is False


def test_all_frozen_qras_are_hash_bound_valid_and_balanced() -> None:
    manifest = _read_json(MANIFEST_PATH)
    record_schema = _read_json(REPO_ROOT / manifest["record_schema"]["path"])
    configuration_manifest = _read_json(
        REPO_ROOT / manifest["configuration_manifest"]["path"]
    )
    document_manifest = _read_json(REPO_ROOT / manifest["document_manifest"]["path"])
    configurations = {
        declaration["configuration_id"]: _read_json(
            REPO_ROOT / declaration["path"]
        )
        for declaration in configuration_manifest["configurations"]
    }
    documents = {
        declaration["document_id"]: _read_json(REPO_ROOT / declaration["path"])
        for declaration in document_manifest["documents"]
    }

    records = []
    for declaration in manifest["qras"]:
        qra_path = REPO_ROOT / declaration["path"]
        freeze_record_path = REPO_ROOT / declaration["freeze_record_path"]
        qra = _read_json(qra_path)
        freeze_record = _read_json(freeze_record_path)
        assert sha256_file(qra_path) == declaration["file_sha256"]
        assert sha256_file(freeze_record_path) == declaration[
            "freeze_record_sha256"
        ]
        assert freeze_record["artifacts"]["frozen_qra"]["sha256"] == sha256_file(
            qra_path
        )
        assert qra["version"] == "1.0"
        assert qra["status"] == "frozen"
        assert qra["split"] == "test"
        assert qra["review_status"] == "single_checked"
        assert not list(
            Draft202012Validator(
                record_schema,
                format_checker=FormatChecker(),
            ).iter_errors(qra)
        )
        validate_qra_record(
            qra=qra,
            configuration=configurations[qra["configuration_id"]],
            documents=documents,
        )
        records.append(qra)

    assert len(records) == 60
    assert len({record["id"] for record in records}) == 60
    assert Counter(record["answerable"] for record in records) == {
        True: 54,
        False: 6,
    }
    assert Counter(item["resolution"] for item in manifest["qras"]) == {
        "primary_law_revised": 17,
        "semantic_carry_forward": 43,
    }
    for domain in ("legal_form_selection", "funding_preparation", "market_validation"):
        domain_records = [record for record in records if record["domain"] == domain]
        assert len(domain_records) == 20
        assert Counter(
            record["question_type"] for record in domain_records
        ) == EXPECTED_TYPE_COUNTS
