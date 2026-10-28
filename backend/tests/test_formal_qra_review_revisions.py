"""Regression checks for Configuration v1.2 and formal QRA candidate v0.3."""

from collections import Counter
import json
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

from apps.evaluations.formal_configuration_generation import DOMAIN_SCHEMA_PATHS
from apps.evaluations.formal_qra_review_revisions import (
    REVISED_CONFIGURATION_IDS,
    prepare_formal_qra_review_revision,
)
from apps.evaluations.rageval_artifacts import (
    sha256_file,
    validate_configuration_record,
    validate_qra_record,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
FORMAL_ROOT = REPO_ROOT / "docs/evaluation/formal_dataset"
CONFIGURATION_MANIFEST_PATH = (
    FORMAL_ROOT
    / "configurations/formal_test/formal_test_configuration_manifest_v1.2.json"
)
CONFIGURATION_FREEZE_PATH = (
    FORMAL_ROOT
    / "configurations/formal_test/"
    "formal_test_configuration_manifest_v1.2_freeze_record.json"
)
QRA_ROOT = FORMAL_ROOT / "qra/formal_candidates"
QRA_MANIFEST_PATH = QRA_ROOT / "generation_manifest_v0.3.json"
QRA_CURATION_PATH = FORMAL_ROOT / "qra/qra_semantic_curation_v0.3.json"

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


def _configuration_context() -> tuple[dict, dict, dict[str, dict], dict]:
    manifest = _read_json(CONFIGURATION_MANIFEST_PATH)
    document_manifest = _read_json(
        REPO_ROOT / manifest["document_manifest"]["path"]
    )
    documents = {
        declaration["document_id"]: _read_json(REPO_ROOT / declaration["path"])
        for declaration in document_manifest["documents"]
    }
    record_schema = _read_json(REPO_ROOT / manifest["record_schema"]["path"])
    return manifest, document_manifest, documents, record_schema


def _qras_by_id() -> dict[str, dict]:
    manifest = _read_json(QRA_MANIFEST_PATH)
    return {
        declaration["qra_id"]: _read_json(REPO_ROOT / declaration["path"])
        for declaration in manifest["qras"]
    }


def test_configuration_v1_2_is_complete_grounded_and_append_only() -> None:
    manifest, _, documents, record_schema = _configuration_context()
    freeze_record = _read_json(CONFIGURATION_FREEZE_PATH)
    domain_schemas = {
        domain: _read_json(REPO_ROOT / path)
        for domain, path in DOMAIN_SCHEMA_PATHS.items()
    }

    assert manifest["status"] == "frozen"
    assert manifest["version"] == "1.2"
    assert manifest["supersedes"] == "formal_test_configuration_manifest_v1.1"
    assert manifest["configuration_count"] == 60
    assert manifest["revised_configuration_count"] == 8
    assert manifest["carried_forward_count"] == 52
    assert freeze_record["file_sha256"] == sha256_file(
        CONFIGURATION_MANIFEST_PATH
    )
    assert set(freeze_record["revised_configuration_ids"]) == (
        REVISED_CONFIGURATION_IDS
    )

    records = []
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
        validate_configuration_record(
            configuration=configuration,
            schema=domain_schemas[configuration["domain"]],
            documents=documents,
        )
        records.append(configuration)

    assert len({item["configuration_id"] for item in records}) == 60
    assert Counter(item["answerability"] for item in records) == {
        "answerable": 54,
        "unanswerable": 6,
    }
    for domain in DOMAIN_SCHEMA_PATHS:
        domain_records = [item for item in records if item["domain"] == domain]
        assert len(domain_records) == 20
        assert Counter(item["target_question_type"] for item in domain_records) == (
            EXPECTED_TYPE_COUNTS
        )

    by_id = {item["configuration_id"]: item for item in records}
    for configuration_id in REVISED_CONFIGURATION_IDS:
        expected_version = "1.2" if configuration_id == "C-FT-LFS-U-002" else "1.1"
        assert by_id[configuration_id]["version"] == expected_version
        assert by_id[configuration_id]["supersedes"].startswith(configuration_id)


def test_configuration_revisions_remove_the_audited_targets() -> None:
    manifest = _read_json(CONFIGURATION_MANIFEST_PATH)
    configurations = {
        declaration["configuration_id"]: _read_json(
            REPO_ROOT / declaration["path"]
        )
        for declaration in manifest["configurations"]
    }

    legal_summary = configurations["C-FT-LFS-S-002"]
    assert {
        block_id
        for source in legal_summary["source_documents"]
        for block_id in source["allowed_block_ids"]
    } == {"D-LFS-009-B005", "D-LFS-009-B036", "D-LFS-009-B039"}
    assert "D-LFS-009-B043" not in json.dumps(legal_summary)

    legal_temporal = configurations["C-FT-LFS-TS-001"]
    assert [item["document_id"] for item in legal_temporal["source_documents"]] == [
        "D-LFS-008"
    ]
    assert "ELSTER" in json.dumps(legal_temporal, ensure_ascii=False)
    assert "mailen" not in json.dumps(legal_temporal, ensure_ascii=False)

    legal_unanswerable = configurations["C-FT-LFS-U-002"]
    scenario = next(
        item["value"]
        for item in legal_unanswerable["protocol_assumptions"]
        if item["json_pointer"].endswith("hypothetical_scenario")
    )
    assert "UG" in scenario and "Einzelunternehmen" in scenario
    assert "KI-Beratung" not in scenario

    funding_temporal = configurations["C-FT-FP-TS-002"]
    assert "Förderphasen" not in funding_temporal["question_generation_constraints"][
        "focus"
    ]
    assert {
        block_id
        for source in funding_temporal["source_documents"]
        for block_id in source["allowed_block_ids"]
    } == {"D-FP-007-B005", "D-FP-007-B008", "D-FP-007-B009"}

    funding_reasoning = configurations["C-FT-FP-MHR-002"]
    assert "Marktannahmen" not in funding_reasoning[
        "question_generation_constraints"
    ]["focus"]
    assert "privatem Lebensunterhalt" in funding_reasoning[
        "question_generation_constraints"
    ]["focus"]

    credit_process = configurations["C-FT-FP-MHR-003"]
    assert [
        item["block_id"] for item in credit_process["provenance"]
    ] == [
        "D-FP-011-B135",
        "D-FP-011-B137",
        "D-FP-011-B138",
        "D-FP-011-B139",
        "D-FP-011-B140",
    ]

    assert "Empathy Map" in configurations["C-FT-MV-F-004"][
        "question_generation_constraints"
    ]["focus"]
    assert "Eignung eines MVP" in configurations["C-FT-MV-F-005"][
        "question_generation_constraints"
    ]["focus"]


def test_qra_v0_3_is_complete_grounded_and_pending_review() -> None:
    qra_manifest = _read_json(QRA_MANIFEST_PATH)
    configuration_manifest, _, documents, _ = _configuration_context()
    qra_record_schema = _read_json(REPO_ROOT / qra_manifest["record_schema"]["path"])
    configurations = {
        declaration["configuration_id"]: _read_json(
            REPO_ROOT / declaration["path"]
        )
        for declaration in configuration_manifest["configurations"]
    }

    assert qra_manifest["version"] == "0.3"
    assert qra_manifest["status"] == "review_candidate"
    assert qra_manifest["review_status"] == "pending"
    assert qra_manifest["qra_count"] == 60
    assert qra_manifest["answerable_count"] == 54
    assert qra_manifest["unanswerable_count"] == 6
    assert qra_manifest["semantic_curation"]["entry_count"] == 51
    assert qra_manifest["semantic_curation"]["sha256"] == sha256_file(
        QRA_CURATION_PATH
    )

    declared_paths = {
        REPO_ROOT / declaration["path"] for declaration in qra_manifest["qras"]
    }
    assert set(QRA_ROOT.glob("*/*_v0.3.json")) == declared_paths

    records = []
    for declaration in qra_manifest["qras"]:
        path = REPO_ROOT / declaration["path"]
        qra = _read_json(path)
        configuration = configurations[qra["configuration_id"]]

        assert sha256_file(path) == declaration["file_sha256"]
        assert qra["version"] == "0.3"
        assert qra["status"] == "development_candidate"
        assert qra["split"] == "test"
        assert qra["review_status"] == "pending"
        assert not list(
            Draft202012Validator(
                qra_record_schema,
                format_checker=FormatChecker(),
            ).iter_errors(qra)
        )
        validate_qra_record(
            qra=qra,
            configuration=configuration,
            documents=documents,
        )
        records.append(qra)

    assert len({item["id"] for item in records}) == 60
    assert len({item["question"] for item in records}) == 60
    for domain in DOMAIN_SCHEMA_PATHS:
        domain_records = [item for item in records if item["domain"] == domain]
        assert len(domain_records) == 20
        assert Counter(item["question_type"] for item in domain_records) == (
            EXPECTED_TYPE_COUNTS
        )


def test_qra_v0_3_locks_final_audit_regressions() -> None:
    qras = _qras_by_id()

    assert "seit dem 1. Januar 2021" in qras["FTLFS-TS-001"]["question"]
    assert {item["document_id"] for item in qras["FTLFS-TS-001"]["references"]} == {
        "D-LFS-008"
    }
    assert "24.500" not in qras["FTLFS-F-002"]["answer"]
    assert "D-LFS-009-B043" not in {
        item["block_id"] for item in qras["FTLFS-S-002"]["references"]
    }
    assert len(qras["FTLFS-S-003"]["keypoints"]) == 3
    assert not qras["FTLFS-II-001"]["references"][0]["quote"].endswith("d.")

    assert "Business Model Canvas" not in qras["FTFP-II-002"]["answer"]
    assert "beiden Förderphasen" not in qras["FTFP-TS-002"]["question"]
    assert qras["FTFP-TS-002"]["keypoints"][-1]["reference_ids"] == [
        "R2",
        "R4",
    ]
    assert "Marktannahmen" not in qras["FTFP-MHR-002"]["question"]
    assert qras["FTFP-MHR-003"]["references"][0]["block_id"] == (
        "D-FP-011-B135"
    )
    assert "2021" in qras["FTFP-NC-002"]["question"]

    assert "Empathy Map" in qras["FTMV-F-004"]["question"]
    assert "Alter, Geschlecht" not in qras["FTMV-F-004"]["answer"]
    assert "Finanzierungsbedarf" in qras["FTMV-F-005"]["question"]
    assert "Definition und Mindestumfang" not in qras["FTMV-F-005"]["question"]
    assert "Vertriebs- und Marketingkanäle" in qras["FTMV-MHR-003"]["answer"]
    assert "2020" in qras["FTMV-NC-001"]["question"]
    assert "2020" in qras["FTMV-NC-002"]["question"]
    assert not qras["FTMV-F-001"]["references"][0]["quote"].endswith("für")
    assert "detaillierten Plan" in qras["FTMV-TS-001"]["references"][2]["quote"]

    # The three previously leaked development targets must not survive in v0.3.
    assert "KI-Beratung" not in qras["FTLFS-U-002"]["question"]
    assert "Förderphasen" not in qras["FTFP-TS-002"]["question"]
    assert "Definition und Mindestumfang eines MVP" not in qras["FTMV-F-005"][
        "question"
    ]


def test_review_revision_preparation_is_idempotent() -> None:
    report = prepare_formal_qra_review_revision(
        repo_root=REPO_ROOT,
        revised_at="2026-08-02",
    )

    assert report["configuration"]["created_count"] == 0
    assert report["configuration"]["unchanged_count"] == 18
    assert report["curation"]["write_state"] == "unchanged"
    assert report["curation"]["entry_count"] == 51
