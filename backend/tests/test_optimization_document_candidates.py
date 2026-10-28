"""Repository-level integrity checks for optimization Document D candidates."""

import json
from pathlib import Path

from apps.evaluations.corpus_paths import resolve_corpus_path

from apps.evaluations.rageval_artifacts import (
    sha256_file,
    validate_document_record,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = REPO_ROOT / "docs/evaluation/optimization_dataset"
SELECTION_PATH = (
    DATASET_ROOT
    / "source_selection/optimization_source_selection_v0.1.json"
)
CANDIDATE_DIR = DATASET_ROOT / "documents/candidates"
AUDIT_PATH = DATASET_ROOT / "documents/document_candidate_audit_v0.1.json"
SCOPE_PATH = DATASET_ROOT / "documents/market_case_scope_review_v0.1.json"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_optimization_selection_and_documents_are_consistent() -> None:
    selection = _read_json(SELECTION_PATH)
    records = [
        {**record, "domain": domain}
        for domain, payload in selection["domains"].items()
        for record in payload["selected_documents"]
    ]

    assert selection["status"] == "review_candidate"
    assert selection["intended_split"] == "optimization"
    assert selection["document_count"] == 24
    assert selection["new_source_count"] == 14
    assert selection["prior_formal_source_overlap_count"] == 10
    assert len(records) == len({record["document_id"] for record in records}) == 24
    assert len({record["path"] for record in records}) == 24

    for domain, payload in selection["domains"].items():
        assert payload["selected_document_count"] == 8
        assert len(payload["selected_documents"]) == 8
        assert sum(
            record["planned_primary_qra_budget"]
            for record in payload["selected_documents"]
        ) == 20
        assert all(record["document_id"].startswith("OD-") for record in payload["selected_documents"])
        assert all(record["document_action"] == "generate_candidate" for record in payload["selected_documents"])
        assert all(domain in record["path"] for record in payload["selected_documents"])

    candidate_paths: set[Path] = set()
    for record in records:
        source_path = resolve_corpus_path(REPO_ROOT, record["path"])
        candidate_path = REPO_ROOT / record["document_output_path"]
        candidate_paths.add(candidate_path)
        assert sha256_file(source_path) == record["raw_sha256"]
        document = _read_json(candidate_path)
        validate_document_record(document=document, source_path=source_path)
        assert document["document_id"] == record["document_id"]
        assert document["domain"] == record["domain"]
        assert document["status"] == "review_candidate"
        assert document["version"] == "0.1"

    assert set(CANDIDATE_DIR.glob("OD-*_v0.1.json")) == candidate_paths


def test_optimization_generation_manifest_is_reproducible() -> None:
    manifest = _read_json(CANDIDATE_DIR / "generation_manifest_v0.1.json")

    assert manifest["generation_id"] == "optimization_document_candidates_v0.1"
    assert manifest["selected_document_count"] == 24
    assert manifest["new_candidate_count"] == 24
    assert manifest["reused_frozen_count"] == 0
    assert len(manifest["new_candidates"]) == 24
    assert sum(item["block_count"] for item in manifest["new_candidates"]) == 1572
    assert all(item["write_state"] == "unchanged" for item in manifest["new_candidates"])


def test_optimization_document_audit_and_market_scopes_pass() -> None:
    audit = _read_json(AUDIT_PATH)
    scope = _read_json(SCOPE_PATH)

    assert audit["status"] == "agent_audit_passed_human_review_pending"
    assert audit["all_automated_gates_passed"] is True
    assert audit["human_review_complete"] is False
    assert audit["summary"]["document_count"] == 24
    assert audit["summary"]["block_count"] == 1572
    assert audit["summary"]["prior_formal_reference_count"] == 76
    assert audit["summary"]["matched_prior_formal_reference_count"] == 76
    assert audit["summary"]["excluded_block_count"] == 55
    assert len(audit["prior_formal_evidence_exclusions"]) == 10
    assert all(
        exclusion["unmatched_reference_count"] == 0
        for exclusion in audit["prior_formal_evidence_exclusions"]
    )

    assert scope["status"] == "agent_review_candidate"
    assert scope["human_review_claimed"] is False
    assert {item["document_id"] for item in scope["documents"]} == {
        "OD-MV-005",
        "OD-MV-006",
        "OD-MV-007",
        "OD-MV-008",
    }
