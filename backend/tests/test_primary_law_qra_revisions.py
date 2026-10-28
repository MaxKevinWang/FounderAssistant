"""Regression checks for primary-law QRA review candidate v0.4."""

from collections import Counter
import json
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

from apps.evaluations.formal_configuration_generation import DOMAIN_SCHEMA_PATHS
from apps.evaluations.rageval_artifacts import sha256_file, validate_qra_record


REPO_ROOT = Path(__file__).resolve().parents[2]
FORMAL_ROOT = REPO_ROOT / "docs/evaluation/formal_dataset"
CONFIGURATION_MANIFEST_PATH = (
    FORMAL_ROOT
    / "configurations/formal_test/formal_test_configuration_manifest_v1.3.json"
)
REVISION_MANIFEST_PATH = (
    FORMAL_ROOT / "qra/primary_law_qra_revision_manifest_v0.4.json"
)
CURATION_PATH = FORMAL_ROOT / "qra/qra_semantic_curation_v0.4.json"
QRA_MANIFEST_PATH = (
    FORMAL_ROOT / "qra/formal_candidates/generation_manifest_v0.4.json"
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


def test_revision_recipe_set_matches_the_17_configuration_changes() -> None:
    configuration_manifest = _read_json(CONFIGURATION_MANIFEST_PATH)
    revision_manifest = _read_json(REVISION_MANIFEST_PATH)
    curation = _read_json(CURATION_PATH)
    revised_ids = {
        item["configuration_id"]
        for item in configuration_manifest["configurations"]
        if item["resolution"] == "primary_law_revision"
    }

    assert revision_manifest["status"] == "review_candidate"
    assert revision_manifest["revised_qra_count"] == 17
    assert revision_manifest["expected_full_qra_count"] == 60
    assert revision_manifest["human_review_claimed"] is False
    assert revision_manifest["freeze_status"] == "not_frozen"
    assert {
        item["configuration_id"] for item in revision_manifest["revisions"]
    } == revised_ids
    assert curation["curation_id"] == "formal_qra_semantic_curation_v0.4"
    assert curation["version"] == "0.4"
    assert curation["status"] == "development_candidate"
    assert curation["human_review_state"] == "pending"
    assert len(curation["entries"]) == 51
    assert revised_ids <= set(curation["entries"])


def test_qra_v0_4_is_complete_grounded_and_pending_review() -> None:
    qra_manifest = _read_json(QRA_MANIFEST_PATH)
    configuration_manifest = _read_json(CONFIGURATION_MANIFEST_PATH)
    document_manifest = _read_json(
        REPO_ROOT / qra_manifest["document_manifest"]["path"]
    )
    qra_schema = _read_json(REPO_ROOT / qra_manifest["record_schema"]["path"])
    configurations = {
        declaration["configuration_id"]: _read_json(
            REPO_ROOT / declaration["path"]
        )
        for declaration in configuration_manifest["configurations"]
    }
    documents = {
        declaration["document_id"]: _read_json(
            REPO_ROOT / declaration["path"]
        )
        for declaration in document_manifest["documents"]
    }

    assert qra_manifest["version"] == "0.4"
    assert qra_manifest["status"] == "review_candidate"
    assert qra_manifest["review_status"] == "pending"
    assert qra_manifest["supersedes"] == "formal_qra_generation_manifest_v0.3"
    assert qra_manifest["qra_count"] == 60
    assert qra_manifest["answerable_count"] == 54
    assert qra_manifest["unanswerable_count"] == 6
    assert qra_manifest["configuration_manifest"]["sha256"] == sha256_file(
        CONFIGURATION_MANIFEST_PATH
    )
    assert qra_manifest["semantic_curation"]["sha256"] == sha256_file(
        CURATION_PATH
    )
    assert "17 primary-law" in qra_manifest["semantic_curation"]["scope"]

    records = []
    for declaration in qra_manifest["qras"]:
        qra_path = REPO_ROOT / declaration["path"]
        qra = _read_json(qra_path)
        configuration = configurations[qra["configuration_id"]]
        assert sha256_file(qra_path) == declaration["file_sha256"]
        assert qra["version"] == "0.4"
        assert qra["status"] == "development_candidate"
        assert qra["review_status"] == "pending"
        assert qra["configuration_version"] == configuration["version"]
        assert 3 <= len(qra["keypoints"]) <= 5
        assert not list(
            Draft202012Validator(
                qra_schema,
                format_checker=FormatChecker(),
            ).iter_errors(qra)
        )
        validate_qra_record(
            qra=qra,
            configuration=configuration,
            documents=documents,
        )
        records.append(qra)

    assert len({record["id"] for record in records}) == 60
    assert len({record["question"] for record in records}) == 60
    assert Counter(record["answerable"] for record in records) == {
        True: 54,
        False: 6,
    }
    for domain in DOMAIN_SCHEMA_PATHS:
        domain_records = [record for record in records if record["domain"] == domain]
        assert len(domain_records) == 20
        assert Counter(
            record["question_type"] for record in domain_records
        ) == EXPECTED_TYPE_COUNTS


def test_revised_qras_use_primary_law_without_removed_claims() -> None:
    qra_manifest = _read_json(QRA_MANIFEST_PATH)
    revision_manifest = _read_json(REVISION_MANIFEST_PATH)
    revised_ids = {
        item["configuration_id"] for item in revision_manifest["revisions"]
    }
    revised_qras = [
        _read_json(REPO_ROOT / declaration["path"])
        for declaration in qra_manifest["qras"]
        if declaration["configuration_id"] in revised_ids
    ]
    combined = "\n".join(
        " ".join(
            [qra["question"], qra["answer"]]
            + [item["text"] for item in qra["keypoints"]]
        )
        for qra in revised_qras
    )

    assert len(revised_qras) == 17
    removed_phrases = (
        "GbR nein",
        "Das Jobcenter entscheidet",
        "15,83 Prozent",
        "KG faktisch eine haftungsbeschränkte Gesellschaft",
        "ElsterFormular",
        "wenige Formalitäten",
    )
    for phrase in removed_phrases:
        assert phrase not in combined

    assert all(
        any(
            reference["document_id"] == "D-FP-013"
            or (
                reference["document_id"].startswith("D-LFS-")
                and int(reference["document_id"].rsplit("-", 1)[1]) >= 13
            )
            for reference in qra["references"]
        )
        for qra in revised_qras
    )


def test_numerical_revision_uses_correct_differences_and_time_scope() -> None:
    qra_manifest = _read_json(QRA_MANIFEST_PATH)
    declaration = next(
        item
        for item in qra_manifest["qras"]
        if item["qra_id"] == "FTLFS-NC-002"
    )
    qra = _read_json(REPO_ROOT / declaration["path"])

    assert "bis 2027" in qra["question"]
    assert "12 Prozentpunkte" in qra["answer"]
    assert "4 Prozentpunkte" in qra["answer"]
    assert "8 Prozentpunkte" in qra["answer"]
    assert "10 Prozentpunkte" in qra["answer"]
    assert "nicht zeitlos" in qra["answer"]
