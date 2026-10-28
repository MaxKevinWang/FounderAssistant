"""Integrity, provenance, and isolation checks for optimization QRA candidates."""

from collections import Counter
import json
from pathlib import Path
import re

from apps.evaluations.rageval_artifacts import sha256_file, validate_qra_record


REPO_ROOT = Path(__file__).resolve().parents[2]
QRA_ROOT = REPO_ROOT / "docs/evaluation/optimization_dataset/qra/candidates"
MANIFEST_PATH = QRA_ROOT / "generation_manifest_v0.1.json"
PRIOR_QRA_MANIFEST_PATH = (
    REPO_ROOT
    / "docs/evaluation/formal_dataset/qra/formal_test/"
    "formal_test_qra_manifest_v1.0.json"
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


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def _load_declared_records(manifest: dict) -> list[dict]:
    return [
        _read_json(REPO_ROOT / declaration["path"])
        for declaration in manifest["qras"]
    ]


def test_optimization_qra_set_is_complete_balanced_and_pending_review() -> None:
    manifest = _read_json(MANIFEST_PATH)
    curation_path = REPO_ROOT / manifest["semantic_curation"]["path"]
    curation = _read_json(curation_path)
    records = _load_declared_records(manifest)

    assert manifest["manifest_id"] == "optimization_qra_generation_manifest_v0.1"
    assert manifest["status"] == "review_candidate"
    assert manifest["review_status"] == "pending"
    assert manifest["qra_count"] == 60
    assert manifest["answerable_count"] == 54
    assert manifest["unanswerable_count"] == 6
    assert manifest["semantic_curation"]["entry_count"] == 60
    assert manifest["semantic_curation"]["sha256"] == sha256_file(curation_path)
    assert curation["curation_id"] == "optimization_qra_semantic_curation_v0.1"
    assert set(curation["entries"]) == {
        record["configuration_id"] for record in records
    }

    assert len({_normalize(record["question"]) for record in records}) == 60
    assert len({_normalize(record["answer"]) for record in records}) == 60
    assert Counter(record["answerable"] for record in records) == {
        True: 54,
        False: 6,
    }
    for domain in {record["domain"] for record in records}:
        domain_records = [record for record in records if record["domain"] == domain]
        assert len(domain_records) == 20
        assert Counter(
            record["question_type"] for record in domain_records
        ) == EXPECTED_TYPE_COUNTS


def test_optimization_qra_references_resolve_to_frozen_allowed_evidence() -> None:
    manifest = _read_json(MANIFEST_PATH)
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

    expected_paths = set()
    for declaration in manifest["qras"]:
        path = REPO_ROOT / declaration["path"]
        expected_paths.add(path)
        record = _read_json(path)
        configuration = configurations[record["configuration_id"]]

        assert sha256_file(path) == declaration["file_sha256"]
        assert record["id"].startswith("OPT")
        assert record["status"] == "development_candidate"
        assert record["review_status"] == "pending"
        assert record["configuration_version"] == "1.0"
        assert "/optimization_dataset/" in record["configuration_path"]
        assert 3 <= len(record["keypoints"]) <= 5
        assert all(document_id.startswith("OD-") for document_id in record["document_ids"])

        allowed_pairs = {
            (source["document_id"], block_id)
            for source in configuration["source_documents"]
            for block_id in source["allowed_block_ids"]
        }
        assert {
            (reference["document_id"], reference["block_id"])
            for reference in record["references"]
        }.issubset(allowed_pairs)
        validate_qra_record(
            qra=record,
            configuration=configuration,
            documents=documents,
        )

    actual_paths = {
        path
        for path in QRA_ROOT.glob("*/*.json")
        if path.name.endswith("_v0.1.json")
    }
    assert actual_paths == expected_paths


def test_optimization_questions_and_answers_do_not_duplicate_formal_test_items() -> None:
    optimization = _load_declared_records(_read_json(MANIFEST_PATH))
    prior = _load_declared_records(_read_json(PRIOR_QRA_MANIFEST_PATH))

    assert not (
        {_normalize(record["question"]) for record in optimization}
        & {_normalize(record["question"]) for record in prior}
    )
    assert not (
        {_normalize(record["answer"]) for record in optimization}
        & {_normalize(record["answer"]) for record in prior}
    )
