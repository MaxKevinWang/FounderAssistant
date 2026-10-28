"""Repository-level integrity checks for formal Document D candidates."""

import json
from pathlib import Path

from apps.evaluations.corpus_paths import resolve_corpus_path

from apps.evaluations.rageval_artifacts import (
    sha256_file,
    validate_document_record,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SELECTION_PATH = (
    REPO_ROOT
    / "docs/evaluation/formal_dataset/source_selection/"
    "formal_test_source_selection_v0.1.json"
)
CANDIDATE_DIR = (
    REPO_ROOT / "docs/evaluation/formal_dataset/documents/formal_candidates"
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_formal_document_selection_and_artifacts_are_consistent() -> None:
    selection = _read_json(SELECTION_PATH)
    records = [
        record
        for domain in selection["domains"].values()
        for record in domain["selected_documents"]
    ]

    assert len(records) == 24
    assert len({record["document_id"] for record in records}) == 24
    assert sum(
        record["document_action"] == "generate_candidate" for record in records
    ) == 21
    assert sum(record["document_action"] == "reuse_existing" for record in records) == 3

    candidate_paths: set[Path] = set()
    for record in records:
        source_path = resolve_corpus_path(REPO_ROOT, record["path"])
        assert sha256_file(source_path) == record["raw_sha256"]

        if record["document_action"] == "generate_candidate":
            artifact_path = REPO_ROOT / record["document_output_path"]
            candidate_paths.add(artifact_path)
            expected_status = "review_candidate"
            expected_version = "0.1"
        else:
            artifact_path = REPO_ROOT / record["existing_document_path"]
            expected_status = "frozen"
            expected_version = "1.0"
            assert sha256_file(artifact_path) == record["existing_document_sha256"]

        document = _read_json(artifact_path)
        validate_document_record(document=document, source_path=source_path)
        assert document["document_id"] == record["document_id"]
        assert document["status"] == expected_status
        assert document["version"] == expected_version

    actual_candidate_paths = set(CANDIDATE_DIR.glob("D-*_v0.1.json"))
    assert actual_candidate_paths == candidate_paths


def test_formal_generation_manifest_covers_all_selected_documents() -> None:
    manifest = _read_json(CANDIDATE_DIR / "generation_manifest_v0.1.json")

    assert manifest["selected_document_count"] == 24
    assert manifest["new_candidate_count"] == 21
    assert manifest["reused_frozen_count"] == 3
    assert len(manifest["new_candidates"]) == 21
    assert len(manifest["reused_documents"]) == 3
    assert sum(item["block_count"] for item in manifest["new_candidates"]) == 972
    assert sum(
        item["reviewed_figure_description_count"]
        for item in manifest["new_candidates"]
    ) == 24
