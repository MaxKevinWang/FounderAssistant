"""Integrity checks for the frozen optimization Configuration C set."""

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

from apps.evaluations.rageval_artifacts import sha256_file


REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_ROOT = (
    REPO_ROOT / "docs/evaluation/optimization_dataset/configurations/frozen"
)
MANIFEST_PATH = FROZEN_ROOT / "optimization_configuration_manifest_v1.0.json"
AUTHORIZATION_PATH = (
    REPO_ROOT
    / "docs/evaluation/optimization_dataset/configurations/"
    "optimization_configuration_freeze_authorization_v1.0.json"
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


def test_optimization_configuration_manifest_is_frozen_and_balanced() -> None:
    manifest = _read_json(MANIFEST_PATH)
    authorization = manifest["freeze_authorization"]

    assert manifest["manifest_id"] == "optimization_configuration_manifest_v1.0"
    assert manifest["status"] == "frozen"
    assert manifest["configuration_count"] == 60
    assert manifest["answerable_count"] == 54
    assert manifest["unanswerable_count"] == 6
    assert authorization["sha256"] == sha256_file(AUTHORIZATION_PATH)

    configurations = []
    for declaration in manifest["configurations"]:
        path = REPO_ROOT / declaration["path"]
        freeze_path = REPO_ROOT / declaration["freeze_record_path"]
        configuration = _read_json(path)
        freeze_record = _read_json(freeze_path)

        assert configuration["configuration_id"].startswith("C-OPT-")
        assert configuration["version"] == "1.0"
        assert configuration["status"] == "frozen"
        assert configuration["generation_record"]["human_review_status"] == "approved"
        assert sha256_file(path) == declaration["file_sha256"]
        assert sha256_file(freeze_path) == declaration["freeze_record_sha256"]
        assert freeze_record["file_sha256"] == declaration["file_sha256"]
        assert freeze_record["freeze_authorization"] == authorization
        configurations.append(configuration)

    assert Counter(item["answerability"] for item in configurations) == {
        "answerable": 54,
        "unanswerable": 6,
    }
    for domain in {item["domain"] for item in configurations}:
        domain_items = [item for item in configurations if item["domain"] == domain]
        assert len(domain_items) == 20
        assert Counter(
            item["target_question_type"] for item in domain_items
        ) == EXPECTED_TYPE_COUNTS


def test_frozen_semantics_equal_authorized_candidates() -> None:
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
