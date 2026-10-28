"""Integrity and evidence-policy checks for optimization Configurations."""

from collections import Counter
import json
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

from apps.evaluations.rageval_artifacts import (
    sha256_file,
    validate_configuration_record,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
OPTIMIZATION_ROOT = REPO_ROOT / "docs/evaluation/optimization_dataset"
ALLOCATION_PATH = (
    OPTIMIZATION_ROOT
    / "configurations/optimization_configuration_allocation_v0.1.json"
)
CONFIGURATION_ROOT = OPTIMIZATION_ROOT / "configurations/candidates"
CONFIGURATION_MANIFEST_PATH = CONFIGURATION_ROOT / "generation_manifest_v0.1.json"
DOCUMENT_MANIFEST_PATH = (
    OPTIMIZATION_ROOT / "documents/optimization_document_manifest_v1.0.json"
)
EXCLUSION_REGISTRY_PATH = (
    OPTIMIZATION_ROOT / "documents/document_candidate_audit_v0.1.json"
)
MARKET_SCOPE_PATH = (
    OPTIMIZATION_ROOT / "documents/market_case_scope_review_v0.1.json"
)
CONFIGURATION_SCHEMA_PATH = (
    REPO_ROOT
    / "docs/evaluation/formal_dataset/configurations/"
    "configuration_record_schema_v1.1.json"
)
DOMAIN_SCHEMA_PATHS = {
    domain: (
        REPO_ROOT
        / f"docs/evaluation/schemas/{domain}/schema_S_{domain}_v1.0.json"
    )
    for domain in (
        "legal_form_selection",
        "funding_preparation",
        "market_validation",
    )
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


def _load_documents() -> tuple[dict, dict[str, dict]]:
    manifest = _read_json(DOCUMENT_MANIFEST_PATH)
    documents = {
        declaration["document_id"]: _read_json(REPO_ROOT / declaration["path"])
        for declaration in manifest["documents"]
    }
    return manifest, documents


def test_optimization_allocation_is_balanced_and_distinct() -> None:
    allocation = _read_json(ALLOCATION_PATH)
    items = allocation["items"]

    assert allocation["status"] == "review_candidate"
    assert allocation["intended_split"] == "optimization"
    assert len(items) == 60
    assert len({item["configuration_id"] for item in items}) == 60
    assert all(item["configuration_id"].startswith("C-OPT-") for item in items)
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
        primary_counts = Counter(
            item["primary_document_id"]
            for item in domain_items
            if item["answerability"] == "answerable"
        )
        assert len(primary_counts) == 8
        assert max(primary_counts.values()) <= 3


def test_optimization_candidates_are_valid_and_hash_pinned() -> None:
    document_manifest, documents = _load_documents()
    allocation = _read_json(ALLOCATION_PATH)
    manifest = _read_json(CONFIGURATION_MANIFEST_PATH)
    record_schema = _read_json(CONFIGURATION_SCHEMA_PATH)
    domain_schemas = {
        domain: _read_json(path) for domain, path in DOMAIN_SCHEMA_PATHS.items()
    }

    assert manifest["manifest_id"] == (
        "optimization_configuration_generation_manifest_v0.1"
    )
    assert manifest["status"] == "review_candidate"
    assert manifest["configuration_count"] == 60
    assert manifest["human_review_status"] == "pending"
    assert manifest["allocation"]["sha256"] == sha256_file(ALLOCATION_PATH)
    assert manifest["document_manifest"]["sha256"] == sha256_file(
        DOCUMENT_MANIFEST_PATH
    )
    assert manifest["evidence_policy"][
        "excluded_prior_evidence_block_count"
    ] == 55
    assert manifest["evidence_policy"]["market_case_document_count"] == 4

    expected_paths = {
        REPO_ROOT / declaration["path"]
        for declaration in manifest["configurations"]
    }
    actual_paths = set(CONFIGURATION_ROOT.glob("*/*.json"))
    assert actual_paths == expected_paths

    corpus_by_domain = {
        domain: {
            declaration["document_id"]
            for declaration in document_manifest["documents"]
            if declaration["domain"] == domain
        }
        for domain in DOMAIN_SCHEMA_PATHS
    }
    item_by_id = {
        item["configuration_id"]: item for item in allocation["items"]
    }
    for declaration in manifest["configurations"]:
        path = REPO_ROOT / declaration["path"]
        configuration = _read_json(path)
        errors = list(
            Draft202012Validator(
                record_schema,
                format_checker=FormatChecker(),
            ).iter_errors(configuration)
        )

        assert not errors
        assert sha256_file(path) == declaration["file_sha256"]
        assert configuration["status"] == "development_draft"
        assert configuration["version"] == "0.1"
        assert configuration["generation_record"]["human_review_status"] == (
            "pending"
        )
        assert len(
            {
                source["document_id"]
                for source in configuration["source_documents"]
            }
        ) == len(configuration["source_documents"])
        validate_configuration_record(
            configuration=configuration,
            schema=domain_schemas[configuration["domain"]],
            documents=documents,
        )

        item = item_by_id[configuration["configuration_id"]]
        if configuration["answerability"] == "unanswerable":
            assert {
                source["document_id"]
                for source in configuration["source_documents"]
            } == corpus_by_domain[configuration["domain"]]
        else:
            assert configuration["source_scope"] == item["source_scope"]


def test_evidence_exclusions_and_market_case_scopes_are_hard_gates() -> None:
    manifest = _read_json(CONFIGURATION_MANIFEST_PATH)
    exclusion_registry = _read_json(EXCLUSION_REGISTRY_PATH)
    market_review = _read_json(MARKET_SCOPE_PATH)
    excluded = {
        item["document_id"]: set(item["excluded_block_ids"])
        for item in exclusion_registry["prior_formal_evidence_exclusions"]
    }
    market_allowed = {
        item["document_id"]: set(item["allowed_block_ids"])
        for item in market_review["documents"]
    }

    for declaration in manifest["configurations"]:
        configuration = _read_json(REPO_ROOT / declaration["path"])
        for source in configuration["source_documents"]:
            document_id = source["document_id"]
            allowed_blocks = set(source["allowed_block_ids"])
            assert not (allowed_blocks & excluded.get(document_id, set()))
            if document_id in market_allowed:
                assert allowed_blocks <= market_allowed[document_id]

        for provenance in configuration["provenance"]:
            document_id = provenance["document_id"]
            block_id = provenance["block_id"]
            assert block_id not in excluded.get(document_id, set())
            if document_id in market_allowed:
                assert block_id in market_allowed[document_id]


def test_high_risk_sequences_and_counts_use_the_reviewed_blocks() -> None:
    configurations = {
        path.stem.removesuffix("_v0.1"): _read_json(path)
        for path in CONFIGURATION_ROOT.glob("*/*.json")
    }
    expected_blocks = {
        "C-OPT-LFS-TS-001": {
            "OD-LFS-001-B035",
            "OD-LFS-001-B037",
            "OD-LFS-001-B041",
            "OD-LFS-001-B047",
        },
        "C-OPT-FP-NC-001": {"OD-FP-001-B013"},
        "C-OPT-FP-NC-002": {
            "OD-FP-003-B009",
            "OD-FP-003-B010",
            "OD-FP-003-B011",
        },
        "C-OPT-MV-NC-001": {"OD-MV-002-B029", "OD-MV-002-B038"},
        "C-OPT-MV-NC-002": {
            "OD-MV-002-B043",
            "OD-MV-002-B044",
            "OD-MV-002-B045",
            "OD-MV-002-B047",
            "OD-MV-002-B048",
            "OD-MV-002-B049",
        },
        "C-OPT-MV-TS-001": {
            "OD-MV-004-B018",
            "OD-MV-004-B019",
            "OD-MV-004-B020",
        },
        "C-OPT-MV-TS-002": {
            "OD-MV-008-B024",
            "OD-MV-008-B028",
            "OD-MV-008-B038",
            "OD-MV-008-B047",
        },
    }

    for configuration_id, block_ids in expected_blocks.items():
        configuration = configurations[configuration_id]
        assert {
            provenance["block_id"]
            for provenance in configuration["provenance"]
        } == block_ids

    for configuration in configurations.values():
        if configuration["target_question_type"] == "Temporal Sequence":
            assert configuration["question_generation_constraints"][
                "must_require_order"
            ]
