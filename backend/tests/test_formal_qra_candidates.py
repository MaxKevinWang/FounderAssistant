"""Repository-level validation for the 60 formal-test QRA review candidates."""

from collections import Counter
import json
from pathlib import Path
import re

from jsonschema import Draft202012Validator, FormatChecker

from apps.evaluations.rageval_artifacts import (
    sha256_file,
    validate_qra_record,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
FORMAL_ROOT = REPO_ROOT / "docs/evaluation/formal_dataset"
QRA_ROOT = FORMAL_ROOT / "qra/formal_candidates"
QRA_MANIFEST_PATH = QRA_ROOT / "generation_manifest_v0.2.json"
QRA_REVIEW_PATH = QRA_ROOT / "qra_candidate_review_v0.2.md"
QRA_CURATION_PATH = FORMAL_ROOT / "qra/qra_semantic_curation_v0.2.json"

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


def test_formal_qra_candidates_are_complete_grounded_and_pending_review() -> None:
    manifest = _read_json(QRA_MANIFEST_PATH)
    configuration_manifest = _read_json(
        REPO_ROOT / manifest["configuration_manifest"]["path"]
    )
    document_manifest = _read_json(
        REPO_ROOT / manifest["document_manifest"]["path"]
    )
    record_schema = _read_json(REPO_ROOT / manifest["record_schema"]["path"])
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

    assert manifest["status"] == "review_candidate"
    assert manifest["split"] == "test"
    assert manifest["review_status"] == "pending"
    assert manifest["qra_count"] == 60
    assert manifest["answerable_count"] == 54
    assert manifest["unanswerable_count"] == 6
    assert sha256_file(
        REPO_ROOT / manifest["configuration_manifest"]["path"]
    ) == manifest["configuration_manifest"]["sha256"]
    assert QRA_REVIEW_PATH.exists()

    expected_paths = {
        REPO_ROOT / declaration["path"] for declaration in manifest["qras"]
    }
    actual_paths = set(QRA_ROOT.glob("*/*_v0.2.json"))
    assert actual_paths == expected_paths

    records = []
    for declaration in manifest["qras"]:
        path = REPO_ROOT / declaration["path"]
        qra = _read_json(path)
        configuration = configurations[qra["configuration_id"]]
        schema_errors = list(
            Draft202012Validator(
                record_schema,
                format_checker=FormatChecker(),
            ).iter_errors(qra)
        )

        assert not schema_errors
        assert sha256_file(path) == declaration["file_sha256"]
        assert qra["status"] == "development_candidate"
        assert qra["version"] == "0.2"
        assert qra["split"] == "test"
        assert qra["review_status"] == "pending"
        assert qra["configuration_version"] == configuration["version"]
        assert qra["configuration_sha256"] == sha256_file(
            REPO_ROOT / qra["configuration_path"]
        )
        assert 3 <= len(qra["keypoints"]) <= 5
        assert len(qra["keypoints"]) == declaration["keypoint_count"]
        assert len(qra["references"]) == declaration["reference_count"]
        validate_qra_record(
            qra=qra,
            configuration=configuration,
            documents=documents,
        )
        records.append(qra)

    assert len({item["id"] for item in records}) == 60
    assert len({item["question"] for item in records}) == 60
    assert Counter(item["answerable"] for item in records) == {
        True: 54,
        False: 6,
    }
    for domain in {item["domain"] for item in records}:
        domain_items = [item for item in records if item["domain"] == domain]
        assert len(domain_items) == 20
        assert Counter(item["question_type"] for item in domain_items) == (
            EXPECTED_TYPE_COUNTS
        )


def test_high_risk_qra_types_have_expected_evidence_shape() -> None:
    manifest = _read_json(QRA_MANIFEST_PATH)
    records = [
        _read_json(REPO_ROOT / declaration["path"])
        for declaration in manifest["qras"]
    ]

    for qra in records:
        if qra["question_type"] == "Numerical Comparison":
            assert len(qra["keypoints"]) >= 3
            assert all(re.search(r"\d", item["text"]) for item in qra["keypoints"])
            assert all(
                re.search(
                    r"\b(?:Abstand|entspricht|gegenüber|größer|höher|lag|lagen|"
                    r"liegt|Prozentpunkte?|übertraf|Unterschied|vor)\b",
                    item["text"],
                    flags=re.IGNORECASE,
                )
                for item in qra["keypoints"]
            )
        if qra["question_type"] == "Information Integration":
            assert len({item["document_id"] for item in qra["references"]}) >= 2
        if qra["question_type"] == "Unanswerable":
            assert not qra["answerable"]
            assert len(qra["keypoints"]) == 3
            assert "keine abschließende Entscheidung" in qra["answer"]
            assert qra["keypoints"][0]["text"].startswith(
                "Mit den angegebenen Falldaten"
            )
            assert qra["keypoints"][2]["text"].startswith(
                "Die herangezogenen Quellen nennen nur allgemeine Prüfkriterien"
            )


def test_semantic_curation_is_versioned_complete_and_encoding_safe() -> None:
    """Keep reviewed repairs reproducible and reject console-encoding damage."""

    manifest = _read_json(QRA_MANIFEST_PATH)
    curation = _read_json(QRA_CURATION_PATH)
    records = [
        _read_json(REPO_ROOT / declaration["path"])
        for declaration in manifest["qras"]
    ]
    high_risk_ids = {
        qra["configuration_id"]
        for qra in records
        if qra["question_type"] in {"Numerical Comparison", "Temporal Sequence"}
    }

    assert curation["curation_id"] == "formal_qra_semantic_curation_v0.2"
    assert curation["version"] == "0.2"
    assert curation["status"] == "development_candidate"
    assert len(curation["entries"]) == 44
    assert high_risk_ids.issubset(curation["entries"])
    assert manifest["semantic_curation"]["entry_count"] == 44
    assert manifest["semantic_curation"]["sha256"] == sha256_file(
        QRA_CURATION_PATH
    )

    for qra in records:
        generated_text = " ".join(
            [qra["question"], qra["answer"]]
            + [keypoint["text"] for keypoint in qra["keypoints"]]
        )
        assert "\u00ad" not in generated_text
        assert "\ufffd" not in generated_text
        assert not re.search(
            r"[A-Za-zÄÖÜäöüß]\?[A-Za-zÄÖÜäöüß]",
            generated_text,
        )
        assert not any(item["text"].endswith(":.") for item in qra["keypoints"])

        curated_entry = curation["entries"].get(qra["configuration_id"])
        if curated_entry is not None:
            assert qra["question"] == curated_entry["question"]
            assert qra["generation_record"]["method"] == (
                "Versioned semantic curation over exact frozen evidence"
            )


def test_qra_v0_1_release_remains_hash_verifiable_history() -> None:
    """The superseded candidate release remains available for audit comparison."""

    manifest_path = QRA_ROOT / "generation_manifest_v0.1.json"
    manifest = _read_json(manifest_path)

    assert manifest["version"] == "0.1"
    assert manifest["qra_count"] == 60
    for declaration in manifest["qras"]:
        path = REPO_ROOT / declaration["path"]
        assert path.name.endswith("_v0.1.json")
        assert sha256_file(path) == declaration["file_sha256"]


def test_upstream_repaired_qras_bind_v1_1_and_sufficient_exact_evidence() -> None:
    """Lock the two review-driven repairs into the current candidate release."""

    manifest = _read_json(QRA_MANIFEST_PATH)
    declarations = {item["qra_id"]: item for item in manifest["qras"]}
    legal_form = _read_json(REPO_ROOT / declarations["FTLFS-F-001"]["path"])
    market_summary = _read_json(REPO_ROOT / declarations["FTMV-S-003"]["path"])

    assert legal_form["configuration_version"] == "1.1"
    assert "zehn Rechtsformen" in legal_form["question"]
    assert len(legal_form["references"]) == 4
    assert all(
        reference["block_id"] == "D-LFS-005-B007"
        for reference in legal_form["references"]
    )
    assert "GmbH & Co. KG" in legal_form["answer"]
    assert "Partnerschaftsregister" in legal_form["answer"]

    assert market_summary["configuration_version"] == "1.1"
    assert "Vier Hauptelemente" not in market_summary["question"]
    cost_reference = market_summary["references"][-1]["quote"]
    assert "laufenden Kosten" in cost_reference
    assert "Investitionen" in cost_reference
    assert market_summary["keypoints"][-1]["reference_ids"] == ["R4"]
