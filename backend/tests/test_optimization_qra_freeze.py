"""Integrity checks for the frozen optimization QRA release."""

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

from apps.evaluations.formal_development_benchmark import (
    load_frozen_development_qras,
)
from apps.evaluations.rageval_artifacts import sha256_file


REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_ROOT = REPO_ROOT / "docs/evaluation/optimization_dataset/qra/frozen"
MANIFEST_PATH = FROZEN_ROOT / "optimization_qra_manifest_v1.0.json"
MANIFEST_FREEZE_PATH = (
    FROZEN_ROOT / "optimization_qra_manifest_v1.0_freeze_record.json"
)
AUTHORIZATION_PATH = (
    REPO_ROOT
    / "docs/evaluation/optimization_dataset/qra/"
    "optimization_qra_freeze_authorization_v1.0.json"
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_optimization_qra_release_is_frozen_balanced_and_loadable() -> None:
    manifest = _read_json(MANIFEST_PATH)
    manifest_freeze = _read_json(MANIFEST_FREEZE_PATH)

    assert manifest["manifest_id"] == "optimization_qra_manifest_v1.0"
    assert manifest["status"] == "frozen"
    assert manifest["review_status"] == "single_checked"
    assert manifest["double_checked"] is False
    assert manifest["qra_count"] == 60
    assert manifest["answerable_count"] == 54
    assert manifest["unanswerable_count"] == 6
    assert manifest["freeze_authorization"]["sha256"] == sha256_file(
        AUTHORIZATION_PATH
    )
    assert manifest_freeze["file_sha256"] == sha256_file(MANIFEST_PATH)

    records = load_frozen_development_qras(FROZEN_ROOT, root=REPO_ROOT)
    assert len(records) == 60
    assert len({record["id"] for record in records}) == 60
    assert Counter(record["answerable"] for record in records) == {
        True: 54,
        False: 6,
    }
    assert {
        record["_release_manifest"]["manifest_id"] for record in records
    } == {"optimization_qra_manifest_v1.0"}


def test_frozen_qra_semantics_equal_the_authorized_candidates() -> None:
    manifest = _read_json(MANIFEST_PATH)
    expected_paths = set()

    for declaration in manifest["qras"]:
        frozen_path = REPO_ROOT / declaration["path"]
        freeze_path = REPO_ROOT / declaration["freeze_record_path"]
        frozen = _read_json(frozen_path)
        freeze_record = _read_json(freeze_path)
        candidate = _read_json(REPO_ROOT / freeze_record["artifacts"]["candidate_qra"]["path"])

        expected = deepcopy(candidate)
        expected["version"] = "1.0"
        expected["status"] = "frozen"
        expected["review_status"] = "single_checked"
        expected["generation_record"]["reproducibility_note"] = (
            "Immutable optimization-test QRA v1.0 frozen under explicit "
            "project-owner authorization. Independent second review and expert "
            "fact certification are not claimed."
        )
        expected["generation_record"]["mechanical_validation"] = (
            "passed: JSON Schema, frozen Configuration and Document identity, exact "
            "Reference location, answer alignment, Keypoint coverage, set balance, "
            "candidate uniqueness, and project-owner freeze authorization"
        )

        assert frozen == expected
        assert sha256_file(frozen_path) == declaration["file_sha256"]
        assert sha256_file(freeze_path) == declaration["freeze_record_sha256"]
        assert freeze_record["freeze_authorization"] == manifest["freeze_authorization"]
        expected_paths.add(frozen_path)

    actual_paths = {
        path
        for path in FROZEN_ROOT.glob("*/*.json")
        if path.name.endswith("_v1.0.json")
    }
    assert actual_paths == expected_paths
