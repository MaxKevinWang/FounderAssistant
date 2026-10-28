"""Integrity and regression checks for the approved v1.1 upstream repairs."""

from collections import Counter
import json
from pathlib import Path

from apps.evaluations.corpus_paths import resolve_corpus_path

import pytest

from jsonschema import Draft202012Validator, FormatChecker

from apps.evaluations.formal_configuration_generation import DOMAIN_SCHEMA_PATHS
from apps.evaluations.formal_upstream_revisions import (
    SEMANTIC_AUDIT_PATH,
    freeze_formal_upstream_revisions,
)
from apps.evaluations.rageval_artifacts import (
    sha256_file,
    validate_configuration_record,
    validate_document_record,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
FORMAL_ROOT = REPO_ROOT / "docs/evaluation/formal_dataset"
DOCUMENT_MANIFEST_PATH = (
    FORMAL_ROOT / "documents/formal_test_document_manifest_v1.1.json"
)
DOCUMENT_MANIFEST_FREEZE_PATH = (
    FORMAL_ROOT / "documents/formal_test_document_manifest_v1.1_freeze_record.json"
)
CONFIGURATION_MANIFEST_PATH = (
    FORMAL_ROOT
    / "configurations/formal_test/formal_test_configuration_manifest_v1.1.json"
)
CONFIGURATION_MANIFEST_FREEZE_PATH = (
    FORMAL_ROOT
    / "configurations/formal_test/"
    "formal_test_configuration_manifest_v1.1_freeze_record.json"
)

REVISED_CONFIGURATION_IDS = {
    "C-FT-LFS-F-001",
    "C-FT-LFS-II-001",
    "C-FT-LFS-II-003",
    "C-FT-LFS-NC-001",
    "C-FT-LFS-S-001",
    "C-FT-LFS-U-001",
    "C-FT-LFS-U-002",
    "C-FT-MV-S-003",
}

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


def test_v1_1_document_release_corrects_only_the_audited_table() -> None:
    manifest = _read_json(DOCUMENT_MANIFEST_PATH)
    freeze_record = _read_json(DOCUMENT_MANIFEST_FREEZE_PATH)

    assert manifest["status"] == "frozen"
    assert manifest["version"] == "1.1"
    assert manifest["document_count"] == 24
    assert manifest["revised_document_count"] == 1
    assert manifest["carried_forward_count"] == 23
    assert freeze_record["file_sha256"] == sha256_file(DOCUMENT_MANIFEST_PATH)

    for declaration in manifest["documents"]:
        document_path = REPO_ROOT / declaration["path"]
        document_freeze_path = REPO_ROOT / declaration["freeze_record_path"]
        document = _read_json(document_path)

        assert sha256_file(document_path) == declaration["file_sha256"]
        assert sha256_file(document_freeze_path) == declaration[
            "freeze_record_sha256"
        ]
        assert document["content_sha256"] == declaration["content_sha256"]
        validate_document_record(
            document=document,
            source_path=resolve_corpus_path(REPO_ROOT, declaration["source_path"]),
        )

    declaration = next(
        item for item in manifest["documents"] if item["document_id"] == "D-LFS-005"
    )
    revised = _read_json(REPO_ROOT / declaration["path"])
    predecessor = _read_json(
        FORMAL_ROOT / "documents/D-LFS-005_rechtsformen_v1.0.json"
    )
    revised_block = next(
        block for block in revised["blocks"] if block["block_id"] == "D-LFS-005-B007"
    )

    assert revised["version"] == "1.1"
    assert revised["supersedes"] == "D-LFS-005 v1.0"
    assert revised["block_count"] == predecessor["block_count"]
    assert [
        block for block in revised["blocks"] if block != predecessor["blocks"][
            revised["blocks"].index(block)
        ]
    ] == [revised_block]
    assert "zehn Rechtsformen" in revised_block["text"]
    assert "GmbH & Co. KG" in revised_block["text"]
    assert "PartG, OHG, KG" in revised_block["text"]
    assert "OGH" not in revised_block["text"]
    assert "Partnerschaftsregister" in revised_block["text"]
    assert "Genossenschaftsregister" in revised_block["text"]


def test_v1_1_configuration_release_is_grounded_balanced_and_frozen() -> None:
    manifest = _read_json(CONFIGURATION_MANIFEST_PATH)
    manifest_freeze = _read_json(CONFIGURATION_MANIFEST_FREEZE_PATH)
    document_manifest = _read_json(
        REPO_ROOT / manifest["document_manifest"]["path"]
    )
    documents = {
        declaration["document_id"]: _read_json(REPO_ROOT / declaration["path"])
        for declaration in document_manifest["documents"]
    }
    record_schema = _read_json(REPO_ROOT / manifest["record_schema"]["path"])
    domain_schemas = {
        domain: _read_json(REPO_ROOT / path)
        for domain, path in DOMAIN_SCHEMA_PATHS.items()
    }

    assert manifest["status"] == "frozen"
    assert manifest["version"] == "1.1"
    assert manifest["configuration_count"] == 60
    assert manifest["revised_configuration_count"] == 8
    assert manifest["carried_forward_count"] == 52
    assert manifest_freeze["file_sha256"] == sha256_file(
        CONFIGURATION_MANIFEST_PATH
    )
    assert set(manifest_freeze["revised_configuration_ids"]) == (
        REVISED_CONFIGURATION_IDS
    )

    configurations = []
    for declaration in manifest["configurations"]:
        path = REPO_ROOT / declaration["path"]
        freeze_path = REPO_ROOT / declaration["freeze_record_path"]
        configuration = _read_json(path)

        assert sha256_file(path) == declaration["file_sha256"]
        assert sha256_file(freeze_path) == declaration["freeze_record_sha256"]
        assert not list(
            Draft202012Validator(
                record_schema,
                format_checker=FormatChecker(),
            ).iter_errors(configuration)
        )
        assert configuration["version"] == declaration["version"]
        assert configuration["status"] == "frozen"
        if configuration["configuration_id"] in REVISED_CONFIGURATION_IDS:
            assert configuration["version"] == "1.1"
            assert configuration["supersedes"].endswith(" v1.0")
        else:
            assert configuration["version"] == "1.0"
        validate_configuration_record(
            configuration=configuration,
            schema=domain_schemas[configuration["domain"]],
            documents=documents,
        )
        configurations.append(configuration)

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

    by_id = {item["configuration_id"]: item for item in configurations}
    legal_form = by_id["C-FT-LFS-F-001"]
    market_summary = by_id["C-FT-MV-S-003"]
    assert len(legal_form["values"]["rulesAndExceptions"]) == 5
    assert len(legal_form["provenance"]) == 5
    assert "zehn Rechtsformen" in legal_form["values"]["rulesAndExceptions"][0][
        "rule"
    ]
    assert market_summary["question_generation_constraints"]["focus"] == (
        "Kundennutzen, Ertragsmodell und Kostenstruktur eines Geschäftsmodells"
    )
    assert "Vier Hauptelemente" not in json.dumps(
        market_summary,
        ensure_ascii=False,
    )


@pytest.mark.skipif(
    not (REPO_ROOT / SEMANTIC_AUDIT_PATH).exists(),
    reason="Historical re-freeze requires the Chinese audit excluded from this submission.",
)
def test_v1_1_upstream_freeze_is_idempotent() -> None:
    report = freeze_formal_upstream_revisions(
        repo_root=REPO_ROOT,
        frozen_at="2026-08-02",
    )

    assert report["created_count"] == 0
    assert report["unchanged_count"] == 22
    assert set(report["revised_configuration_ids"]) == REVISED_CONFIGURATION_IDS
