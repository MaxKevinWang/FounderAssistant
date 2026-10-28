"""Repository-level integrity checks for the 60-item formal evaluation set."""

from collections import Counter
import json
from pathlib import Path

from apps.evaluations.corpus_paths import resolve_corpus_path
import re

from jsonschema import Draft202012Validator, FormatChecker

from apps.evaluations.rageval_artifacts import (
    sha256_file,
    validate_configuration_record,
    validate_document_record,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
FORMAL_ROOT = REPO_ROOT / "docs/evaluation/formal_dataset"
DOCUMENT_MANIFEST_PATH = (
    FORMAL_ROOT / "documents/formal_test_document_manifest_v1.0.json"
)
SOURCE_SELECTION_PATH = (
    FORMAL_ROOT / "source_selection/formal_test_source_selection_v1.0.json"
)
SOURCE_SELECTION_FREEZE_PATH = (
    FORMAL_ROOT
    / "source_selection/formal_test_source_selection_v1.0_freeze_record.json"
)
ALLOCATION_PATH = (
    FORMAL_ROOT
    / "configurations/formal_configuration_allocation_v0.1.json"
)
CONFIGURATION_SCHEMA_PATH = (
    FORMAL_ROOT / "configurations/configuration_record_schema_v1.1.json"
)
CONFIGURATION_CANDIDATE_ROOT = (
    FORMAL_ROOT / "configurations/formal_candidates"
)
CONFIGURATION_MANIFEST_PATH = (
    CONFIGURATION_CANDIDATE_ROOT / "generation_manifest_v0.1.json"
)

DOMAIN_SCHEMA_PATHS = {
    "legal_form_selection": (
        REPO_ROOT
        / "docs/evaluation/schemas/legal_form_selection/"
        "schema_S_legal_form_selection_v1.0.json"
    ),
    "funding_preparation": (
        REPO_ROOT
        / "docs/evaluation/schemas/funding_preparation/"
        "schema_S_funding_preparation_v1.0.json"
    ),
    "market_validation": (
        REPO_ROOT
        / "docs/evaluation/schemas/market_validation/"
        "schema_S_market_validation_v1.0.json"
    ),
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

REVIEWED_FIGURE_CONFIG_BLOCKS = {
    "C-FT-LFS-F-001": {"D-LFS-005-B007"},
    "C-FT-LFS-NC-001": {"D-LFS-005-B003"},
    "C-FT-FP-NC-001": {"D-FP-006-B011"},
    "C-FT-FP-NC-002": {"D-FP-008-B007"},
    "C-FT-MV-NC-001": {"D-MV-006-B008"},
    "C-FT-MV-NC-002": {"D-MV-007-B004"},
}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_frozen_documents() -> tuple[dict, dict[str, dict]]:
    manifest = _read_json(DOCUMENT_MANIFEST_PATH)
    documents = {
        declaration["document_id"]: _read_json(REPO_ROOT / declaration["path"])
        for declaration in manifest["documents"]
    }
    return manifest, documents


def test_formal_document_manifest_and_freeze_records_are_consistent() -> None:
    manifest, documents = _load_frozen_documents()

    assert manifest["status"] == "frozen"
    assert manifest["version"] == "1.0"
    assert manifest["document_count"] == 24
    assert manifest["newly_frozen_count"] == 21
    assert manifest["reused_frozen_count"] == 3
    assert len(documents) == 24
    assert Counter(document["domain"] for document in documents.values()) == {
        "legal_form_selection": 8,
        "funding_preparation": 8,
        "market_validation": 8,
    }

    for declaration in manifest["documents"]:
        document_path = REPO_ROOT / declaration["path"]
        freeze_record_path = REPO_ROOT / declaration["freeze_record_path"]
        source_path = resolve_corpus_path(REPO_ROOT, declaration["source_path"])
        document = documents[declaration["document_id"]]
        freeze_record = _read_json(freeze_record_path)

        assert sha256_file(document_path) == declaration["file_sha256"]
        assert sha256_file(freeze_record_path) == declaration["freeze_record_sha256"]
        assert sha256_file(source_path) == declaration["source_sha256"]
        assert document["status"] == "frozen"
        assert document["version"] == "1.0"
        assert document["content_sha256"] == declaration["content_sha256"]
        assert freeze_record["file_sha256"] == declaration["file_sha256"]
        assert freeze_record["content_sha256"] == declaration["content_sha256"]
        assert freeze_record["source_sha256"] == declaration["source_sha256"]
        validate_document_record(document=document, source_path=source_path)

    source_selection = _read_json(SOURCE_SELECTION_PATH)
    source_freeze = _read_json(SOURCE_SELECTION_FREEZE_PATH)
    assert source_selection["status"] == "frozen"
    assert source_selection["version"] == "1.0"
    assert source_freeze["file_sha256"] == sha256_file(SOURCE_SELECTION_PATH)
    assert source_freeze["document_manifest_sha256"] == sha256_file(
        DOCUMENT_MANIFEST_PATH
    )


def test_formal_configuration_allocation_has_expected_balance() -> None:
    allocation = _read_json(ALLOCATION_PATH)
    items = allocation["items"]

    assert allocation["status"] == "review_candidate"
    assert len(items) == 60
    assert len({item["configuration_id"] for item in items}) == 60
    assert Counter(item["answerability"] for item in items) == {
        "answerable": 54,
        "unanswerable": 6,
    }

    for domain in DOMAIN_SCHEMA_PATHS:
        domain_items = [item for item in items if item["domain"] == domain]
        assert len(domain_items) == 20
        assert Counter(
            item["target_question_type"] for item in domain_items
        ) == EXPECTED_TYPE_COUNTS

        primary_document_counts = Counter(
            item["primary_document_id"]
            for item in domain_items
            if item["answerability"] == "answerable"
        )
        assert len(primary_document_counts) == 8
        assert max(primary_document_counts.values()) <= 3


def test_formal_configuration_candidates_are_grounded_and_complete() -> None:
    document_manifest, documents = _load_frozen_documents()
    configuration_manifest = _read_json(CONFIGURATION_MANIFEST_PATH)
    record_schema = _read_json(CONFIGURATION_SCHEMA_PATH)
    domain_schemas = {
        domain: _read_json(path)
        for domain, path in DOMAIN_SCHEMA_PATHS.items()
    }
    declarations = configuration_manifest["configurations"]
    expected_paths = {
        REPO_ROOT / declaration["path"] for declaration in declarations
    }
    actual_paths = set(CONFIGURATION_CANDIDATE_ROOT.glob("*/*.json"))

    assert configuration_manifest["configuration_count"] == 60
    assert configuration_manifest["human_review_status"] == "pending"
    assert len(declarations) == 60
    assert actual_paths == expected_paths
    assert configuration_manifest["document_manifest"]["sha256"] == sha256_file(
        DOCUMENT_MANIFEST_PATH
    )
    assert configuration_manifest["allocation"]["sha256"] == sha256_file(
        ALLOCATION_PATH
    )

    corpus_by_domain = {
        domain: {
            declaration["document_id"]
            for declaration in document_manifest["documents"]
            if declaration["domain"] == domain
        }
        for domain in DOMAIN_SCHEMA_PATHS
    }

    for declaration in declarations:
        candidate_path = REPO_ROOT / declaration["path"]
        configuration = _read_json(candidate_path)
        schema_errors = list(
            Draft202012Validator(
                record_schema,
                format_checker=FormatChecker(),
            ).iter_errors(configuration)
        )

        assert not schema_errors
        assert sha256_file(candidate_path) == declaration["file_sha256"]
        assert configuration["status"] == "development_draft"
        assert configuration["version"] == "0.1"
        assert configuration["generation_record"]["human_review_status"] == "pending"
        assert configuration["configuration_id"] == declaration["configuration_id"]
        validate_configuration_record(
            configuration=configuration,
            schema=domain_schemas[configuration["domain"]],
            documents=documents,
        )

        declared_document_ids = {
            source["document_id"]
            for source in configuration["source_documents"]
        }
        if configuration["answerability"] == "unanswerable":
            assert declared_document_ids == corpus_by_domain[configuration["domain"]]
            assert len(declared_document_ids) == 8
        else:
            assert len(declared_document_ids) in {1, 2}

        for provenance in configuration["provenance"]:
            normalized_quote = re.sub(
                r"\s+",
                " ",
                provenance["exact_quote"],
            ).strip()
            assert not normalized_quote.startswith("GründerZeiten ")
            assert not normalized_quote.startswith("Figure title:")
            assert not re.search(
                r"\b(?:bzw|ca|u|z)\.$",
                normalized_quote,
                flags=re.IGNORECASE,
            )


def test_high_risk_question_types_use_reviewed_or_ordered_blocks() -> None:
    configurations = {
        path.stem.removesuffix("_v0.1"): _read_json(path)
        for path in CONFIGURATION_CANDIDATE_ROOT.glob("*/*.json")
    }

    for configuration_id, expected_blocks in REVIEWED_FIGURE_CONFIG_BLOCKS.items():
        configuration = configurations[configuration_id]
        assert {
            item["block_id"] for item in configuration["provenance"]
        } == expected_blocks

    design_thinking = configurations["C-FT-MV-TS-002"]
    assert [item["block_id"] for item in design_thinking["provenance"]] == [
        "D-MV-009-B017",
        "D-MV-009-B020",
        "D-MV-009-B023",
        "D-MV-009-B026",
        "D-MV-009-B028",
    ]
    assert design_thinking["question_generation_constraints"]["must_require_order"]

    grant_phases = configurations["C-FT-FP-TS-002"]
    assert [item["block_id"] for item in grant_phases["provenance"]] == [
        "D-FP-007-B005",
        "D-FP-007-B005",
    ]
    assert grant_phases["question_generation_constraints"]["must_require_order"]
