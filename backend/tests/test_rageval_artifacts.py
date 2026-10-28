"""Tests for deterministic RAGEval Document D and Configuration validation."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from apps.evaluations.rageval_artifacts import (
    ArtifactValidationError,
    build_html_document_record,
    build_pdf_document_record,
    validate_configuration_record,
    validate_document_record,
)


def test_html_document_builder_stops_before_related_links(tmp_path: Path) -> None:
    source = tmp_path / "source.html"
    source.write_text(
        """
        <main>
          <h1>Article</h1>
          <h2>Method</h2>
          <p>Useful validation evidence appears in this paragraph.</p>
          <h2>Related links</h2>
          <p>This promotional content must not enter Document D.</p>
        </main>
        """,
        encoding="utf-8",
    )

    document = build_html_document_record(
        source_path=source,
        document_id="D-TEST-001",
        domain="market_validation",
        title="Article",
        language="en",
        jurisdiction="Test",
        seed_document_id="SEED-001",
        source_url="https://example.test/article",
        accessed_at="2026-01-01T00:00:00Z",
        created_at="2026-01-02",
        stop_heading_prefixes=("Related links",),
    )

    text = "\n".join(block["text"] for block in document["blocks"])
    assert "Useful validation evidence" in text
    assert "promotional content" not in text
    assert document["blocks"][0]["block_id"] == "D-TEST-001-B001"


def test_html_document_builder_stops_at_non_heading_cta(tmp_path: Path) -> None:
    source = tmp_path / "source.html"
    source.write_text(
        """
        <main>
          <h1>Article</h1>
          <p>Grounded article content.</p>
          <p>Next recommended articles:</p>
          <ul><li>Promotional link</li></ul>
        </main>
        """,
        encoding="utf-8",
    )

    document = build_html_document_record(
        source_path=source,
        document_id="D-TEST-001",
        domain="market_validation",
        title="Article",
        language="en",
        jurisdiction="Test",
        seed_document_id="SEED-001",
        source_url="https://example.test/article",
        accessed_at="2026-01-01T00:00:00Z",
        created_at="2026-01-02",
        stop_block_prefixes=("Next recommended articles:",),
    )

    text = "\n".join(block["text"] for block in document["blocks"])
    assert "Grounded article content" in text
    assert "Next recommended articles" not in text
    assert "Promotional link" not in text
    assert document["extraction"]["stop_block_prefixes"] == [
        "Next recommended articles:"
    ]


def test_html_document_builder_excludes_noise_and_keeps_later_content(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.html"
    source.write_text(
        """
        <main>
          <h1>Article</h1>
          <p>First grounded section.</p>
          <h2>Book a consultation</h2>
          <p>Monthly and free</p>
          <h2>Eligibility</h2>
          <p>Second grounded section.</p>
        </main>
        """,
        encoding="utf-8",
    )

    document = build_html_document_record(
        source_path=source,
        document_id="D-TEST-001",
        domain="funding_preparation",
        title="Article",
        language="en",
        jurisdiction="Test",
        seed_document_id="SEED-001",
        source_url="https://example.test/article",
        accessed_at="2026-01-01T00:00:00Z",
        created_at="2026-01-02",
        exclude_block_prefixes=("Book a consultation", "Monthly and free"),
    )

    text = "\n".join(block["text"] for block in document["blocks"])
    assert "First grounded section" in text
    assert "Book a consultation" not in text
    assert "Monthly and free" not in text
    assert "Eligibility" in text
    assert "Second grounded section" in text
    assert document["extraction"]["exclude_block_prefixes"] == [
        "Book a consultation",
        "Monthly and free",
    ]


def test_html_document_builder_can_emit_review_candidate(tmp_path: Path) -> None:
    source = tmp_path / "source.html"
    source.write_text("<main><h1>Article</h1><p>Grounded content.</p></main>")

    document = build_html_document_record(
        source_path=source,
        document_id="D-TEST-001",
        domain="market_validation",
        title="Article",
        language="en",
        jurisdiction="Test",
        seed_document_id="SEED-001",
        source_url="https://example.test/article",
        accessed_at="2026-01-01T00:00:00Z",
        created_at="2026-01-02",
        publication_date="2025-12",
        version="0.1",
        status="review_candidate",
    )

    validate_document_record(document=document, source_path=source)
    assert document["version"] == "0.1"
    assert document["status"] == "review_candidate"
    assert document["source"]["publication_date"] == "2025-12"


def test_pdf_document_builder_separates_page_text_and_reviewed_figure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.pdf"
    source.write_bytes(b"deterministic test source")
    extracted = SimpleNamespace(
        pages=[
            SimpleNamespace(text="Cleaned text from page one.", page_number=1),
            SimpleNamespace(text="Cleaned text from page two.", page_number=2),
        ]
    )
    monkeypatch.setattr(
        "apps.evaluations.rageval_artifacts._extract_pdf_document",
        lambda _source_path: extracted,
    )

    document = build_pdf_document_record(
        source_path=source,
        document_id="D-TEST-002",
        domain="funding_preparation",
        title="PDF guide",
        language="en",
        jurisdiction="Test",
        seed_document_id="SEED-002",
        created_at="2026-01-02",
        reviewed_figures_by_page={
            2: [
                {
                    "title": "Break-even chart",
                    "figure_type": "chart",
                    "description_markdown": "Revenue crosses total costs.",
                    "extracted_values": ["Break-even at 100 units"],
                }
            ]
        },
    )

    validate_document_record(document=document, source_path=source)
    assert [block["block_type"] for block in document["blocks"]] == [
        "page_text",
        "page_text",
        "figure_description",
    ]
    assert document["blocks"][2]["page"] == 2
    assert document["blocks"][2]["evidence_type"] == "reviewed_figure_description"
    assert "Break-even at 100 units" in document["blocks"][2]["text"]


def test_document_validator_rejects_tampered_content_hash(tmp_path: Path) -> None:
    source = tmp_path / "source.html"
    source.write_text("<main><p>Grounded content.</p></main>")
    document = build_html_document_record(
        source_path=source,
        document_id="D-TEST-003",
        domain="market_validation",
        title="Article",
        language="en",
        jurisdiction="Test",
        seed_document_id="SEED-003",
        source_url="https://example.test/article",
        accessed_at="2026-01-01T00:00:00Z",
        created_at="2026-01-02",
    )
    document["blocks"][0]["text"] = "Tampered after hashing."

    with pytest.raises(ArtifactValidationError, match="content_sha256"):
        validate_document_record(document=document, source_path=source)


def _minimal_configuration_and_documents() -> tuple[dict, dict, dict]:
    schema = {"validationProcess": []}
    document = {
        "document_id": "D-TEST-001",
        "content_sha256": "content-hash",
        "blocks": [
            {
                "block_id": "D-TEST-001-B001",
                "heading_path": "Method",
                "text": "First, interview potential customers.",
            }
        ],
    }
    configuration = {
        "configuration_id": "C-TEST-001",
        "schema_id": "S-TEST-001",
        "domain": "market_validation",
        "target_question_type": "Temporal Sequence",
        "source_documents": [
            {
                "document_id": "D-TEST-001",
                "content_sha256": "content-hash",
                "allowed_block_ids": ["D-TEST-001-B001"],
            }
        ],
        "values": {
            "validationProcess": [
                {
                    "sequence": "1",
                    "step": "Interview potential customers",
                }
            ]
        },
        "provenance": [
            {
                "provenance_id": "P001",
                "json_pointer": "/values/validationProcess/0/sequence",
                "document_id": "D-TEST-001",
                "block_id": "D-TEST-001-B001",
                "heading_path": "Method",
                "support_type": "direct",
                "derivation": "",
                "exact_quote": "First",
            },
            {
                "provenance_id": "P002",
                "json_pointer": "/values/validationProcess/0/step",
                "document_id": "D-TEST-001",
                "block_id": "D-TEST-001-B001",
                "heading_path": "Method",
                "support_type": "faithful_paraphrase",
                "derivation": "",
                "exact_quote": "interview potential customers",
            }
        ],
    }
    return configuration, schema, {"D-TEST-001": document}


def test_configuration_validator_accepts_exact_grounding() -> None:
    configuration, schema, documents = _minimal_configuration_and_documents()

    validate_configuration_record(
        configuration=configuration,
        schema=schema,
        documents=documents,
    )


def test_configuration_validator_rejects_quote_absent_from_block() -> None:
    configuration, schema, documents = _minimal_configuration_and_documents()
    configuration["provenance"][0]["exact_quote"] = "invented evidence"

    with pytest.raises(ArtifactValidationError, match="Exact quote not found"):
        validate_configuration_record(
            configuration=configuration,
            schema=schema,
            documents=documents,
        )


def test_configuration_validator_rejects_populated_value_without_provenance() -> None:
    configuration, schema, documents = _minimal_configuration_and_documents()
    configuration["values"]["validationProcess"][0]["method"] = (
        "Conduct a semi-structured interview"
    )

    with pytest.raises(
        ArtifactValidationError,
        match="Populated Configuration values have no exact provenance pointer",
    ):
        validate_configuration_record(
            configuration=configuration,
            schema=schema,
            documents=documents,
        )


def test_configuration_validator_rejects_block_outside_allowed_scope() -> None:
    configuration, schema, documents = _minimal_configuration_and_documents()
    documents["D-TEST-001"]["blocks"].append(
        {
            "block_id": "D-TEST-001-B002",
            "heading_path": "Unrelated",
            "text": "Second, use an unrelated source block.",
        }
    )
    configuration["provenance"][1].update(
        {
            "block_id": "D-TEST-001-B002",
            "heading_path": "Unrelated",
            "exact_quote": "unrelated source block",
        }
    )

    with pytest.raises(ArtifactValidationError, match="outside the allowed scope"):
        validate_configuration_record(
            configuration=configuration,
            schema=schema,
            documents=documents,
        )


def test_configuration_validator_rejects_heading_path_mismatch() -> None:
    configuration, schema, documents = _minimal_configuration_and_documents()
    configuration["provenance"][0]["heading_path"] = "Wrong heading"

    with pytest.raises(ArtifactValidationError, match="Heading path mismatch"):
        validate_configuration_record(
            configuration=configuration,
            schema=schema,
            documents=documents,
        )


def test_configuration_validator_rejects_unknown_schema_key() -> None:
    configuration, schema, documents = _minimal_configuration_and_documents()
    configuration["values"] = {"inventedStructure": []}

    with pytest.raises(ArtifactValidationError, match="absent from Schema S"):
        validate_configuration_record(
            configuration=configuration,
            schema=schema,
            documents=documents,
        )
