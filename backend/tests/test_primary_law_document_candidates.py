"""Integrity checks for primary-law evaluation Document D candidates."""

import json
from pathlib import Path

from apps.evaluations.rageval_artifacts import (
    sha256_file,
    validate_document_record,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SELECTION_PATH = (
    REPO_ROOT
    / "docs/evaluation/formal_dataset/source_selection/"
    "primary_law_document_selection_v0.1.json"
)
CANDIDATE_DIR = (
    REPO_ROOT
    / "docs/evaluation/formal_dataset/documents/primary_law_candidates"
)
CONFIGURATION_MANIFEST_PATH = (
    REPO_ROOT
    / "docs/evaluation/formal_dataset/configurations/formal_test/"
    "formal_test_configuration_manifest_v1.2.json"
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_primary_law_document_selection_and_candidates_are_consistent() -> None:
    selection = _read_json(SELECTION_PATH)
    declarations = selection["documents"]
    configuration_manifest = _read_json(CONFIGURATION_MANIFEST_PATH)
    known_configuration_ids = {
        item["configuration_id"] for item in configuration_manifest["configurations"]
    }

    assert len(declarations) == 14
    assert len({item["document_id"] for item in declarations}) == 14
    assert len({item["source_path"] for item in declarations}) == 14
    assert sum(item["domain"] == "legal_form_selection" for item in declarations) == 13
    assert sum(item["domain"] == "funding_preparation" for item in declarations) == 1

    expected_paths: set[Path] = set()
    for declaration in declarations:
        source_path = REPO_ROOT / "knowledge_base" / declaration["source_path"]
        candidate_path = REPO_ROOT / declaration["output_path"]
        expected_paths.add(candidate_path)

        assert sha256_file(source_path) == declaration["source_sha256"]
        assert set(declaration["configuration_impact_candidates"]) <= known_configuration_ids
        document = _read_json(candidate_path)
        validate_document_record(document=document, source_path=source_path)
        assert document["document_id"] == declaration["document_id"]
        assert document["version"] == "0.1"
        assert document["status"] == "review_candidate"
        assert document["source"]["source_type"] == "primary_law"
        assert document["law"]["abbreviation"] == declaration["law_abbreviation"]
        assert document["extraction"]["front_matter_and_toc_removed"] is True
        assert document["blocks"]

        law_sections = [block["law_section"] for block in document["blocks"]]
        assert len(law_sections) == len(set(law_sections))
        assert all(
            block["block_type"] in {"statute_section", "statute_appendix"}
            for block in document["blocks"]
        )
        assert all(
            "nichtamtliches inhaltsverzeichnis" not in block["text"].casefold()
            for block in document["blocks"]
        )

    actual_paths = {
        path
        for path in CANDIDATE_DIR.glob("D-*_v0.1.json")
        if path.name != "generation_manifest_v0.1.json"
    }
    assert actual_paths == expected_paths


def test_primary_law_generation_manifest_covers_all_candidates() -> None:
    manifest = _read_json(CANDIDATE_DIR / "generation_manifest_v0.1.json")

    assert manifest["candidate_count"] == 14
    assert manifest["legal_form_selection_count"] == 13
    assert manifest["funding_preparation_count"] == 1
    assert manifest["block_count"] == 257
    assert len(manifest["candidates"]) == 14
    assert all(item["status"] == "review_candidate" for item in manifest["candidates"])
