"""Integrity checks for the frozen 60-item formal Configuration C set."""

from collections import Counter
from copy import deepcopy
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
FROZEN_ROOT = FORMAL_ROOT / "configurations/formal_test"
MANIFEST_PATH = FROZEN_ROOT / "formal_test_configuration_manifest_v1.0.json"
MANIFEST_FREEZE_PATH = (
    FROZEN_ROOT / "formal_test_configuration_manifest_v1.0_freeze_record.json"
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


def _load_documents(manifest: dict) -> dict[str, dict]:
    document_manifest = _read_json(
        REPO_ROOT / manifest["document_manifest"]["path"]
    )
    return {
        declaration["document_id"]: _read_json(REPO_ROOT / declaration["path"])
        for declaration in document_manifest["documents"]
    }


def test_formal_configurations_are_immutable_valid_and_balanced() -> None:
    manifest = _read_json(MANIFEST_PATH)
    manifest_freeze = _read_json(MANIFEST_FREEZE_PATH)
    record_schema = _read_json(REPO_ROOT / manifest["record_schema"]["path"])
    domain_schemas = {
        domain: _read_json(REPO_ROOT / path)
        for domain, path in DOMAIN_SCHEMA_PATHS.items()
    }
    documents = _load_documents(manifest)

    assert manifest["status"] == "frozen"
    assert manifest["version"] == "1.0"
    assert manifest["configuration_count"] == 60
    assert manifest["answerable_count"] == 54
    assert manifest["unanswerable_count"] == 6
    assert manifest_freeze["file_sha256"] == sha256_file(MANIFEST_PATH)

    configurations = []
    expected_files = set()
    expected_freeze_files = set()
    for declaration in manifest["configurations"]:
        configuration_path = REPO_ROOT / declaration["path"]
        freeze_path = REPO_ROOT / declaration["freeze_record_path"]
        expected_files.add(configuration_path)
        expected_freeze_files.add(freeze_path)

        configuration = _read_json(configuration_path)
        freeze_record = _read_json(freeze_path)
        schema_errors = list(
            Draft202012Validator(
                record_schema,
                format_checker=FormatChecker(),
            ).iter_errors(configuration)
        )

        assert not schema_errors
        assert sha256_file(configuration_path) == declaration["file_sha256"]
        assert sha256_file(freeze_path) == declaration["freeze_record_sha256"]
        assert configuration["configuration_id"] == declaration["configuration_id"]
        assert configuration["version"] == "1.0"
        assert configuration["status"] == "frozen"
        assert configuration["generation_record"]["human_review_status"] == "approved"
        assert freeze_record["status"] == "frozen"
        assert freeze_record["file_sha256"] == declaration["file_sha256"]
        assert freeze_record["approval"]["basis"] == (
            "explicit_project_owner_freeze_instruction"
        )
        validate_configuration_record(
            configuration=configuration,
            schema=domain_schemas[configuration["domain"]],
            documents=documents,
        )
        configurations.append(configuration)

    actual_files = {
        path
        for path in FROZEN_ROOT.glob("*/*.json")
        if path.name.endswith("_v1.0.json")
    }
    actual_freeze_files = set(
        FROZEN_ROOT.glob("*/*_v1.0_freeze_record.json")
    )
    assert actual_files == expected_files
    assert actual_freeze_files == expected_freeze_files

    assert Counter(item["answerability"] for item in configurations) == {
        "answerable": 54,
        "unanswerable": 6,
    }
    for domain in DOMAIN_SCHEMA_PATHS:
        domain_items = [item for item in configurations if item["domain"] == domain]
        assert len(domain_items) == 20
        assert Counter(
            item["target_question_type"] for item in domain_items
        ) == EXPECTED_TYPE_COUNTS


def test_frozen_configuration_semantics_equal_the_approved_candidate() -> None:
    manifest = _read_json(MANIFEST_PATH)

    for declaration in manifest["configurations"]:
        frozen = _read_json(REPO_ROOT / declaration["path"])
        freeze_record = _read_json(REPO_ROOT / declaration["freeze_record_path"])
        candidate = _read_json(REPO_ROOT / freeze_record["candidate_file"])

        expected = deepcopy(candidate)
        expected["version"] = "1.0"
        expected["status"] = "frozen"
        expected["supersedes"] = f"{candidate['configuration_id']} v0.1"
        expected["generation_record"]["human_review_status"] = "approved"
        assert frozen == expected
