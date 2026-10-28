"""Tests for source-grounded QRA, Reference, and Keypoint validation."""

import pytest

from apps.evaluations.rageval_artifacts import (
    ArtifactValidationError,
    validate_qra_record,
)


def _minimal_qra_inputs() -> tuple[dict, dict, dict]:
    configuration = {
        "configuration_id": "C-TEST-001",
        "version": "0.1",
        "target_question_type": "Temporal Sequence",
        "source_documents": [
            {
                "document_id": "D-TEST-001",
                "content_sha256": "content-hash",
                "allowed_block_ids": ["D-TEST-001-B001"],
            }
        ],
    }
    document = {
        "document_id": "D-TEST-001",
        "content_sha256": "content-hash",
        "source": {
            "path": "data/test.html",
            "url": "https://example.test/article",
            "sha256": "source-hash",
        },
        "blocks": [
            {
                "block_id": "D-TEST-001-B001",
                "heading_path": "Method",
                "text": "First, interview potential customers.",
            }
        ],
    }
    qra = {
        "id": "TEST-TS-001",
        "configuration_id": "C-TEST-001",
        "configuration_version": "0.1",
        "question_type": "Temporal Sequence",
        "question": "What should happen first?",
        "answerable": True,
        "answer": "First, interview potential customers.",
        "document_ids": ["D-TEST-001"],
        "references": [
            {
                "reference_id": "R1",
                "document_id": "D-TEST-001",
                "document_content_sha256": "content-hash",
                "raw_source_path": "data/test.html",
                "raw_source_sha256": "source-hash",
                "source_url": "https://example.test/article",
                "heading_path": "Method",
                "block_id": "D-TEST-001-B001",
                "quote": "First, interview potential customers.",
            }
        ],
        "keypoints": [
            {
                "id": "K1",
                "text": "Interview potential customers first.",
                "required": True,
                "reference_ids": ["R1"],
            },
            {
                "id": "K2",
                "text": "The interview targets potential customers.",
                "required": True,
                "reference_ids": ["R1"],
            },
            {
                "id": "K3",
                "text": "The interview is the first action.",
                "required": True,
                "reference_ids": ["R1"],
            },
        ],
        "answer_reference_alignment": [
            {
                "claim_id": "A1",
                "answer_span": "First, interview potential customers.",
                "reference_ids": ["R1"],
            }
        ],
    }
    return qra, configuration, {"D-TEST-001": document}


def test_qra_validator_accepts_fully_grounded_record() -> None:
    qra, configuration, documents = _minimal_qra_inputs()

    validate_qra_record(
        qra=qra,
        configuration=configuration,
        documents=documents,
    )


def test_qra_validator_accepts_empty_url_for_source_without_url() -> None:
    qra, configuration, documents = _minimal_qra_inputs()
    documents["D-TEST-001"]["source"]["url"] = None
    qra["references"][0]["source_url"] = ""

    validate_qra_record(
        qra=qra,
        configuration=configuration,
        documents=documents,
    )


def test_qra_validator_rejects_quote_absent_from_block() -> None:
    qra, configuration, documents = _minimal_qra_inputs()
    qra["references"][0]["quote"] = "Invented source text"

    with pytest.raises(ArtifactValidationError, match="Reference quote not found"):
        validate_qra_record(
            qra=qra,
            configuration=configuration,
            documents=documents,
        )


def test_qra_validator_rejects_reference_outside_configuration_scope() -> None:
    qra, configuration, documents = _minimal_qra_inputs()
    documents["D-TEST-001"]["blocks"].append(
        {
            "block_id": "D-TEST-001-B002",
            "heading_path": "Unrelated",
            "text": "This block is outside the configured scope.",
        }
    )
    qra["references"][0].update(
        {
            "block_id": "D-TEST-001-B002",
            "heading_path": "Unrelated",
            "quote": "outside the configured scope",
        }
    )

    with pytest.raises(ArtifactValidationError, match="outside Configuration scope"):
        validate_qra_record(
            qra=qra,
            configuration=configuration,
            documents=documents,
        )


def test_qra_validator_rejects_keypoint_without_valid_reference() -> None:
    qra, configuration, documents = _minimal_qra_inputs()
    qra["keypoints"][0]["reference_ids"] = ["R-UNKNOWN"]

    with pytest.raises(ArtifactValidationError, match="unknown References"):
        validate_qra_record(
            qra=qra,
            configuration=configuration,
            documents=documents,
        )


def test_qra_validator_rejects_alignment_span_absent_from_answer() -> None:
    qra, configuration, documents = _minimal_qra_inputs()
    qra["answer_reference_alignment"][0]["answer_span"] = "Missing answer span"

    with pytest.raises(ArtifactValidationError, match="Answer span not found"):
        validate_qra_record(
            qra=qra,
            configuration=configuration,
            documents=documents,
        )
