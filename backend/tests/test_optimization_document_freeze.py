"""Repository-level integrity checks for frozen optimization Documents."""

from collections import Counter
import json
from pathlib import Path

from apps.evaluations.corpus_paths import resolve_corpus_path

from apps.evaluations.rageval_artifacts import (
    sha256_file,
    validate_document_record,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
OPTIMIZATION_ROOT = REPO_ROOT / "docs/evaluation/optimization_dataset"
DOCUMENT_MANIFEST_PATH = (
    OPTIMIZATION_ROOT / "documents/optimization_document_manifest_v1.0.json"
)
SOURCE_SELECTION_PATH = (
    OPTIMIZATION_ROOT
    / "source_selection/optimization_source_selection_v1.0.json"
)
SOURCE_SELECTION_FREEZE_PATH = (
    OPTIMIZATION_ROOT
    / "source_selection/optimization_source_selection_v1.0_freeze_record.json"
)
AUTHORIZATION_PATH = (
    OPTIMIZATION_ROOT
    / "documents/optimization_document_freeze_authorization_v1.0.json"
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_optimization_document_freeze_is_complete_and_hash_pinned() -> None:
    manifest = _read_json(DOCUMENT_MANIFEST_PATH)
    authorization = _read_json(AUTHORIZATION_PATH)

    assert manifest["manifest_id"] == "optimization_document_manifest_v1.0"
    assert manifest["status"] == "frozen"
    assert manifest["version"] == "1.0"
    assert manifest["document_count"] == 24
    assert manifest["newly_frozen_count"] == 24
    assert manifest["reused_frozen_count"] == 0
    assert manifest["freeze_authorization"] == {
        "path": AUTHORIZATION_PATH.relative_to(REPO_ROOT).as_posix(),
        "sha256": sha256_file(AUTHORIZATION_PATH),
    }
    assert authorization["review_disclosure"] == {
        "agent_automated_audit_passed": True,
        "independent_human_review_claimed": False,
        "human_line_by_line_review_claimed": False,
    }

    declarations = manifest["documents"]
    assert len({item["document_id"] for item in declarations}) == 24
    assert Counter(item["domain"] for item in declarations) == {
        "legal_form_selection": 8,
        "funding_preparation": 8,
        "market_validation": 8,
    }

    for declaration in declarations:
        document_path = REPO_ROOT / declaration["path"]
        freeze_record_path = REPO_ROOT / declaration["freeze_record_path"]
        source_path = resolve_corpus_path(REPO_ROOT, declaration["source_path"])
        document = _read_json(document_path)
        freeze_record = _read_json(freeze_record_path)

        assert declaration["resolution"] == "newly_frozen"
        assert sha256_file(document_path) == declaration["file_sha256"]
        assert sha256_file(freeze_record_path) == declaration["freeze_record_sha256"]
        assert sha256_file(source_path) == declaration["source_sha256"]
        assert document["status"] == "frozen"
        assert document["version"] == "1.0"
        assert document["content_sha256"] == declaration["content_sha256"]
        assert freeze_record["freeze_authorization"] == manifest[
            "freeze_authorization"
        ]
        validate_document_record(document=document, source_path=source_path)


def test_optimization_source_selection_resolves_the_frozen_manifest() -> None:
    selection = _read_json(SOURCE_SELECTION_PATH)
    freeze_record = _read_json(SOURCE_SELECTION_FREEZE_PATH)

    assert selection["selection_id"] == "optimization_source_selection_v1.0"
    assert selection["status"] == "frozen"
    assert selection["version"] == "1.0"
    assert selection["intended_split"] == "optimization"
    assert selection["document_manifest"]["sha256"] == sha256_file(
        DOCUMENT_MANIFEST_PATH
    )
    assert freeze_record["file_sha256"] == sha256_file(SOURCE_SELECTION_PATH)
    assert freeze_record["document_manifest_sha256"] == sha256_file(
        DOCUMENT_MANIFEST_PATH
    )

    selected = [
        item
        for domain in selection["domains"].values()
        for item in domain["selected_documents"]
    ]
    assert len(selected) == 24
    assert all(item["resolution"] == "newly_frozen" for item in selected)
    assert all(item["resolved_document_path"] for item in selected)
