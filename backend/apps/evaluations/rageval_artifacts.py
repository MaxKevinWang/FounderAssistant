"""Build and validate the first source-grounded RAGEval evaluation artifacts.

This module deliberately stays small. It reuses the structure-aware HTML
parser from the RAG ingestion pipeline so that evaluation documents and corpus
documents start from the same cleaned representation. It does not generate
questions or answers.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from apps.advisors.rag_pipeline import (
    _StructuredHTMLParser,
    _extract_pdf_document,
    _format_reviewed_figures_for_page_text,
    _is_noisy_html_block,
    _read_text_file,
)


RAGEVAL_QUESTION_TYPES = {
    "Factual",
    "Summarization",
    "Multi-hop Reasoning",
    "Information Integration",
    "Numerical Comparison",
    "Temporal Sequence",
    "Unanswerable",
}

PROVENANCE_SUPPORT_TYPES = {
    "direct",
    "faithful_paraphrase",
    "derived",
}


class ArtifactValidationError(ValueError):
    """Raised when a generated evaluation artifact breaks grounding rules."""


def sha256_file(path: Path) -> str:
    """Return a lowercase SHA-256 digest for one immutable source file."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for part in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    """Hash JSON data with deterministic key ordering and separators."""

    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_html_document_record(
    *,
    source_path: Path,
    document_id: str,
    domain: str,
    title: str,
    language: str,
    jurisdiction: str,
    seed_document_id: str,
    source_url: str,
    accessed_at: str,
    created_at: str,
    version: str = "1.0",
    status: str = "frozen",
    publication_date: str | None = None,
    source_record_path: str | None = None,
    stop_heading_prefixes: tuple[str, ...] = (),
    stop_block_prefixes: tuple[str, ...] = (),
    exclude_block_prefixes: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Create a block-addressable evaluation Document ``D`` from cleaned HTML.

    Heading blocks are retained because they make source locations readable.
    Content after a configured trailing heading, such as a related-links
    section, is excluded deterministically. Block IDs are assigned only after
    filtering, so rebuilding the same source with the same rules produces the
    same ordered block sequence.
    """

    parser = _StructuredHTMLParser()
    parser.feed(_read_text_file(source_path))
    parser.close()

    stop_heading_prefixes_folded = tuple(
        prefix.casefold() for prefix in stop_heading_prefixes
    )
    stop_block_prefixes_folded = tuple(
        prefix.casefold() for prefix in stop_block_prefixes
    )
    exclude_block_prefixes_folded = tuple(
        prefix.casefold() for prefix in exclude_block_prefixes
    )
    blocks: list[dict[str, str]] = []
    for parsed_block in parser.blocks:
        if not parsed_block.text.strip() or _is_noisy_html_block(parsed_block):
            continue
        block_text = parsed_block.text.casefold()
        if parsed_block.block_type == "heading" and any(
            block_text.startswith(prefix) for prefix in stop_heading_prefixes_folded
        ):
            break
        if any(
            block_text.startswith(prefix) for prefix in stop_block_prefixes_folded
        ):
            break
        if any(
            block_text.startswith(prefix) for prefix in exclude_block_prefixes_folded
        ):
            continue

        block_id = f"{document_id}-B{len(blocks) + 1:03d}"
        blocks.append(
            {
                "block_id": block_id,
                "block_type": parsed_block.block_type,
                "heading_path": parsed_block.heading_path,
                "text": parsed_block.text,
            }
        )

    if not blocks:
        raise ArtifactValidationError(f"No evaluation blocks extracted from {source_path}")

    return {
        "document_id": document_id,
        "version": version,
        "status": status,
        "domain": domain,
        "title": title,
        "language": language,
        "jurisdiction": jurisdiction,
        "created_at": created_at,
        "source": {
            "seed_document_id": seed_document_id,
            "path": source_record_path or source_path.as_posix(),
            "url": source_url,
            "accessed_at": accessed_at,
            "publication_date": publication_date,
            "sha256": sha256_file(source_path),
        },
        "extraction": {
            "method": "apps.advisors.rag_pipeline._StructuredHTMLParser",
            "noise_filter": "apps.advisors.rag_pipeline._is_noisy_html_block",
            "stop_heading_prefixes": list(stop_heading_prefixes),
            "stop_block_prefixes": list(stop_block_prefixes),
            "exclude_block_prefixes": list(exclude_block_prefixes),
            "block_id_pattern": f"{document_id}-B###",
        },
        "block_count": len(blocks),
        "content_sha256": _canonical_sha256(blocks),
        "blocks": blocks,
    }


def build_pdf_document_record(
    *,
    source_path: Path,
    document_id: str,
    domain: str,
    title: str,
    language: str,
    jurisdiction: str,
    seed_document_id: str,
    created_at: str,
    source_url: str | None = None,
    accessed_at: str | None = None,
    publication_date: str | None = None,
    reviewed_figures_by_page: dict[int, list[dict[str, Any]]] | None = None,
    figure_description_path: Path | None = None,
    source_record_path: str | None = None,
    figure_description_record_path: str | None = None,
    version: str = "0.1",
    status: str = "review_candidate",
) -> dict[str, Any]:
    """Create a page-addressable evaluation Document ``D`` from one PDF.

    This deliberately reuses the production RAG pipeline's PDF extraction,
    hyphenation repair, normalization, and publication-boilerplate removal.
    Evaluation blocks remain page-level rather than being split at production
    embedding boundaries. Reviewed figure descriptions are emitted as separate
    blocks on their original page so a later Configuration can distinguish
    text evidence from multimodal-review evidence.
    """

    extracted = _extract_pdf_document(source_path)
    reviewed_figures_by_page = reviewed_figures_by_page or {}
    blocks: list[dict[str, Any]] = []

    for page in extracted.pages:
        if page.page_number is None:
            raise ArtifactValidationError(
                f"PDF extraction returned a page without a page number: {source_path}"
            )

        page_text = page.text.strip()
        if page_text:
            block_id = f"{document_id}-B{len(blocks) + 1:03d}"
            blocks.append(
                {
                    "block_id": block_id,
                    "block_type": "page_text",
                    "heading_path": f"Page {page.page_number}",
                    "page": page.page_number,
                    "evidence_type": "text",
                    "text": page_text,
                }
            )

        figure_text = _format_reviewed_figures_for_page_text(
            reviewed_figures_by_page.get(page.page_number, [])
        )
        if figure_text:
            block_id = f"{document_id}-B{len(blocks) + 1:03d}"
            blocks.append(
                {
                    "block_id": block_id,
                    "block_type": "figure_description",
                    "heading_path": f"Page {page.page_number} > Reviewed visual content",
                    "page": page.page_number,
                    "evidence_type": "reviewed_figure_description",
                    "text": figure_text,
                }
            )

    if not blocks:
        raise ArtifactValidationError(f"No evaluation blocks extracted from {source_path}")

    figure_count = sum(
        len(figures)
        for figures in reviewed_figures_by_page.values()
        if isinstance(figures, list)
    )
    figure_description_source: dict[str, Any] | None = None
    if figure_description_path is not None:
        figure_description_source = {
            "path": (
                figure_description_record_path
                or figure_description_path.as_posix()
            ),
            "sha256": sha256_file(figure_description_path),
        }

    return {
        "document_id": document_id,
        "version": version,
        "status": status,
        "domain": domain,
        "title": title,
        "language": language,
        "jurisdiction": jurisdiction,
        "created_at": created_at,
        "source": {
            "seed_document_id": seed_document_id,
            "path": source_record_path or source_path.as_posix(),
            "url": source_url,
            "accessed_at": accessed_at,
            "publication_date": publication_date,
            "sha256": sha256_file(source_path),
        },
        "extraction": {
            "method": "apps.advisors.rag_pipeline._extract_pdf_document",
            "hyphenation_repair": "apps.advisors.rag_pipeline._repair_pdf_hyphenation",
            "normalization": "apps.advisors.rag_pipeline._normalize_text",
            "publication_boilerplate_filter": (
                "apps.advisors.rag_pipeline._strip_pdf_publication_boilerplate"
            ),
            "page_aware": True,
            "text_block_granularity": "one block per retained PDF page",
            "reviewed_figure_description_count": figure_count,
            "figure_description_source": figure_description_source,
            "block_id_pattern": f"{document_id}-B###",
        },
        "block_count": len(blocks),
        "content_sha256": _canonical_sha256(blocks),
        "blocks": blocks,
    }


def validate_document_record(
    *,
    document: dict[str, Any],
    source_path: Path | None = None,
) -> None:
    """Validate deterministic identity and block integrity for Document ``D``."""

    required = {
        "document_id",
        "version",
        "status",
        "domain",
        "title",
        "language",
        "jurisdiction",
        "source",
        "extraction",
        "block_count",
        "content_sha256",
        "blocks",
    }
    missing = sorted(required - document.keys())
    if missing:
        raise ArtifactValidationError(f"Missing Document fields: {missing}")

    document_id = str(document["document_id"])
    blocks = document["blocks"]
    if not isinstance(blocks, list) or not blocks:
        raise ArtifactValidationError(f"{document_id} must contain at least one block")
    if document["block_count"] != len(blocks):
        raise ArtifactValidationError(
            f"{document_id} block_count does not match the blocks list"
        )

    seen_block_ids: set[str] = set()
    for index, block in enumerate(blocks, start=1):
        expected_id = f"{document_id}-B{index:03d}"
        block_id = block.get("block_id")
        if block_id != expected_id:
            raise ArtifactValidationError(
                f"{document_id} has non-sequential block ID {block_id!r}; "
                f"expected {expected_id!r}"
            )
        if block_id in seen_block_ids:
            raise ArtifactValidationError(f"Duplicate block ID: {block_id}")
        seen_block_ids.add(block_id)
        if not str(block.get("text") or "").strip():
            raise ArtifactValidationError(f"{block_id} has empty text")
        if block.get("block_type") in {"page_text", "figure_description"}:
            page = block.get("page")
            if not isinstance(page, int) or page <= 0:
                raise ArtifactValidationError(
                    f"{block_id} must contain a positive PDF page number"
                )

    expected_content_hash = _canonical_sha256(blocks)
    if document["content_sha256"] != expected_content_hash:
        raise ArtifactValidationError(
            f"{document_id} content_sha256 does not match its blocks"
        )

    if source_path is not None:
        if not source_path.exists():
            raise ArtifactValidationError(f"Document source is missing: {source_path}")
        expected_source_hash = sha256_file(source_path)
        if document["source"].get("sha256") != expected_source_hash:
            raise ArtifactValidationError(
                f"{document_id} source SHA-256 does not match {source_path}"
            )


def write_json_artifact(path: Path, payload: dict[str, Any]) -> None:
    """Write stable, human-reviewable UTF-8 JSON with a final newline."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _resolve_json_pointer(payload: Any, pointer: str) -> Any:
    """Resolve the RFC 6901 subset used by Configuration provenance records."""

    if not pointer.startswith("/"):
        raise ArtifactValidationError(f"JSON pointer must start with '/': {pointer}")

    current = payload
    for raw_part in pointer[1:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if isinstance(current, list):
            try:
                current = current[int(part)]
            except (ValueError, IndexError) as exc:
                raise ArtifactValidationError(f"Invalid list pointer: {pointer}") from exc
        elif isinstance(current, dict) and part in current:
            current = current[part]
        else:
            raise ArtifactValidationError(f"Pointer does not exist: {pointer}")
    return current


def _escape_json_pointer_part(value: str) -> str:
    """Escape one object key for use in an RFC 6901 JSON pointer."""

    return value.replace("~", "~0").replace("/", "~1")


def _iter_populated_value_pointers(value: Any, pointer: str) -> list[str]:
    """Return reviewable pointers for every populated semantic value.

    Lists of scalar values are treated as one semantic unit because one source
    sentence often supports the complete list. Lists of records are traversed
    item by item so that every populated field in every record needs evidence.
    Empty strings, ``None``, empty lists, and empty dictionaries are ignored.
    """

    if isinstance(value, dict):
        pointers: list[str] = []
        for key, child in value.items():
            child_pointer = f"{pointer}/{_escape_json_pointer_part(str(key))}"
            pointers.extend(_iter_populated_value_pointers(child, child_pointer))
        return pointers

    if isinstance(value, list):
        if not value:
            return []
        if all(not isinstance(item, (dict, list)) for item in value):
            return [pointer] if any(item not in {"", None} for item in value) else []

        pointers = []
        for index, child in enumerate(value):
            pointers.extend(_iter_populated_value_pointers(child, f"{pointer}/{index}"))
        return pointers

    return [pointer] if value not in {"", None} else []


def validate_configuration_record(
    *,
    configuration: dict[str, Any],
    schema: dict[str, Any],
    documents: dict[str, dict[str, Any]],
) -> None:
    """Validate structure, document identity, and exact-quote grounding.

    The validator intentionally does not judge whether a quote semantically
    entails a value. That remains a human or LLM review task. It guarantees the
    lower-level properties needed before such review: the schema field exists,
    the document and block exist, the JSON pointer resolves, and the cited
    quote occurs verbatim in the cited block.
    """

    required = {
        "configuration_id",
        "schema_id",
        "domain",
        "target_question_type",
        "source_documents",
        "values",
        "provenance",
    }
    missing = sorted(required - configuration.keys())
    if missing:
        raise ArtifactValidationError(f"Missing Configuration fields: {missing}")

    if configuration["target_question_type"] not in RAGEVAL_QUESTION_TYPES:
        raise ArtifactValidationError(
            f"Unsupported RAGEval question type: {configuration['target_question_type']}"
        )

    unknown_schema_keys = sorted(set(configuration["values"]) - set(schema))
    if unknown_schema_keys:
        raise ArtifactValidationError(
            f"Configuration values contain keys absent from Schema S: {unknown_schema_keys}"
        )

    source_document_ids = [
        entry["document_id"] for entry in configuration["source_documents"]
    ]
    if len(source_document_ids) != len(set(source_document_ids)):
        raise ArtifactValidationError("Configuration declares duplicate source documents")

    declared_documents = {
        entry["document_id"]: entry for entry in configuration["source_documents"]
    }
    for document_id, declaration in declared_documents.items():
        document = documents.get(document_id)
        if document is None:
            raise ArtifactValidationError(f"Unknown source document: {document_id}")
        if declaration.get("content_sha256") != document.get("content_sha256"):
            raise ArtifactValidationError(
                f"Document content hash mismatch for {document_id}"
            )
        known_block_ids = {block["block_id"] for block in document["blocks"]}
        unknown_allowed_blocks = sorted(
            set(declaration.get("allowed_block_ids", [])) - known_block_ids
        )
        if unknown_allowed_blocks:
            raise ArtifactValidationError(
                f"Unknown allowed blocks for {document_id}: {unknown_allowed_blocks}"
            )

    evidence_by_top_level_key: set[str] = set()
    evidence_pointers: set[str] = set()
    provenance_ids: set[str] = set()
    for item in configuration["provenance"]:
        provenance_id = item.get("provenance_id", "")
        if not provenance_id or provenance_id in provenance_ids:
            raise ArtifactValidationError(
                f"Missing or duplicate provenance ID: {provenance_id!r}"
            )
        provenance_ids.add(provenance_id)

        pointer = item.get("json_pointer", "")
        _resolve_json_pointer(configuration, pointer)
        if not pointer.startswith("/values/"):
            raise ArtifactValidationError(
                f"Provenance pointer must identify a value: {pointer}"
            )
        evidence_pointers.add(pointer)

        pointer_parts = pointer.strip("/").split("/")
        if len(pointer_parts) >= 2 and pointer_parts[0] == "values":
            evidence_by_top_level_key.add(pointer_parts[1])

        document_id = item.get("document_id", "")
        if document_id not in declared_documents:
            raise ArtifactValidationError(
                f"Provenance uses undeclared document: {document_id}"
            )
        document = documents[document_id]
        blocks = {block["block_id"]: block for block in document["blocks"]}
        block_id = item.get("block_id", "")
        if block_id not in blocks:
            raise ArtifactValidationError(
                f"Unknown block {block_id} in document {document_id}"
            )
        allowed_block_ids = declared_documents[document_id].get("allowed_block_ids")
        if allowed_block_ids is not None and block_id not in allowed_block_ids:
            raise ArtifactValidationError(
                f"Block {block_id} is outside the allowed scope for {document_id}"
            )
        if item.get("heading_path", "") != blocks[block_id].get("heading_path", ""):
            raise ArtifactValidationError(
                f"Heading path mismatch for {document_id}/{block_id}"
            )

        support_type = item.get("support_type", "")
        if support_type not in PROVENANCE_SUPPORT_TYPES:
            raise ArtifactValidationError(
                f"Unsupported provenance support type: {support_type!r}"
            )
        if support_type == "derived" and not item.get("derivation", "").strip():
            raise ArtifactValidationError(
                f"Derived provenance requires an explanation: {provenance_id}"
            )

        exact_quote = item.get("exact_quote", "")
        if not exact_quote or exact_quote not in blocks[block_id]["text"]:
            raise ArtifactValidationError(
                f"Exact quote not found in {document_id}/{block_id}"
            )

    ungrounded_top_level_keys = sorted(
        set(configuration["values"]) - evidence_by_top_level_key
    )
    if ungrounded_top_level_keys:
        raise ArtifactValidationError(
            "Configuration top-level value groups have no provenance: "
            f"{ungrounded_top_level_keys}"
        )

    populated_value_pointers = set(
        _iter_populated_value_pointers(configuration["values"], "/values")
    )
    ungrounded_value_pointers = sorted(
        populated_value_pointers - evidence_pointers
    )
    if ungrounded_value_pointers:
        raise ArtifactValidationError(
            "Populated Configuration values have no exact provenance pointer: "
            f"{ungrounded_value_pointers}"
        )


def validate_qra_record(
    *,
    qra: dict[str, Any],
    configuration: dict[str, Any],
    documents: dict[str, dict[str, Any]],
) -> None:
    """Validate a QRA item and its Reference/Keypoint traceability.

    This check covers deterministic relationships only. It proves that the QRA
    uses the intended Configuration and Document blocks, that every quote and
    answer span exists verbatim, and that Keypoints cite known References.
    Human review is still required to judge semantic entailment, naturalness,
    difficulty, and whether the answer is complete.
    """

    required = {
        "id",
        "configuration_id",
        "configuration_version",
        "question_type",
        "question",
        "answerable",
        "answer",
        "document_ids",
        "references",
        "keypoints",
        "answer_reference_alignment",
    }
    missing = sorted(required - qra.keys())
    if missing:
        raise ArtifactValidationError(f"Missing QRA fields: {missing}")

    if qra["configuration_id"] != configuration.get("configuration_id"):
        raise ArtifactValidationError("QRA Configuration ID mismatch")
    if qra["configuration_version"] != configuration.get("version"):
        raise ArtifactValidationError("QRA Configuration version mismatch")
    if qra["question_type"] != configuration.get("target_question_type"):
        raise ArtifactValidationError("QRA question type differs from Configuration")
    if not qra["question"].strip() or not qra["answer"].strip():
        raise ArtifactValidationError("QRA question and answer must not be empty")

    configured_documents = {
        item["document_id"]: item for item in configuration["source_documents"]
    }
    if set(qra["document_ids"]) != set(configured_documents):
        raise ArtifactValidationError(
            "QRA document IDs differ from Configuration source documents"
        )

    reference_ids: set[str] = set()
    for reference in qra["references"]:
        reference_id = reference.get("reference_id", "")
        if not reference_id or reference_id in reference_ids:
            raise ArtifactValidationError(
                f"Missing or duplicate Reference ID: {reference_id!r}"
            )
        reference_ids.add(reference_id)

        document_id = reference.get("document_id", "")
        declaration = configured_documents.get(document_id)
        document = documents.get(document_id)
        if declaration is None or document is None:
            raise ArtifactValidationError(
                f"Reference uses an undeclared Document: {document_id}"
            )
        if (
            reference.get("document_content_sha256")
            != document.get("content_sha256")
            or reference.get("document_content_sha256")
            != declaration.get("content_sha256")
        ):
            raise ArtifactValidationError(
                f"Reference Document hash mismatch: {reference_id}"
            )

        source = document.get("source", {})
        expected_source_fields = {
            "raw_source_path": source.get("path"),
            "raw_source_sha256": source.get("sha256"),
            # QRA JSON stores an absent PDF/text URL as an empty string because
            # its schema deliberately keeps Reference metadata scalar-valued.
            "source_url": source.get("url") or "",
        }
        for field, expected in expected_source_fields.items():
            if reference.get(field) != expected:
                raise ArtifactValidationError(
                    f"Reference source metadata mismatch for {reference_id}: {field}"
                )

        blocks = {block["block_id"]: block for block in document["blocks"]}
        block_id = reference.get("block_id", "")
        if block_id not in blocks:
            raise ArtifactValidationError(
                f"Unknown Reference block {document_id}/{block_id}"
            )
        allowed_blocks = declaration.get("allowed_block_ids")
        if allowed_blocks is not None and block_id not in allowed_blocks:
            raise ArtifactValidationError(
                f"Reference block {block_id} is outside Configuration scope"
            )
        block = blocks[block_id]
        if reference.get("heading_path", "") != block.get("heading_path", ""):
            raise ArtifactValidationError(
                f"Reference heading path mismatch: {reference_id}"
            )
        quote = reference.get("quote", "")
        if not quote or quote not in block["text"]:
            raise ArtifactValidationError(
                f"Reference quote not found in {document_id}/{block_id}"
            )

    if not 3 <= len(qra["keypoints"]) <= 5:
        raise ArtifactValidationError("QRA must contain between 3 and 5 Keypoints")

    keypoint_ids: set[str] = set()
    referenced_by_keypoints: set[str] = set()
    for keypoint in qra["keypoints"]:
        keypoint_id = keypoint.get("id", "")
        if not keypoint_id or keypoint_id in keypoint_ids:
            raise ArtifactValidationError(
                f"Missing or duplicate Keypoint ID: {keypoint_id!r}"
            )
        keypoint_ids.add(keypoint_id)
        keypoint_reference_ids = set(keypoint.get("reference_ids", []))
        unknown_references = sorted(keypoint_reference_ids - reference_ids)
        if not keypoint_reference_ids or unknown_references:
            raise ArtifactValidationError(
                f"Keypoint {keypoint_id} uses unknown References: "
                f"{unknown_references}"
            )
        referenced_by_keypoints.update(keypoint_reference_ids)

    claim_ids: set[str] = set()
    referenced_by_answer: set[str] = set()
    for claim in qra["answer_reference_alignment"]:
        claim_id = claim.get("claim_id", "")
        if not claim_id or claim_id in claim_ids:
            raise ArtifactValidationError(
                f"Missing or duplicate answer claim ID: {claim_id!r}"
            )
        claim_ids.add(claim_id)
        answer_span = claim.get("answer_span", "")
        if not answer_span or answer_span not in qra["answer"]:
            raise ArtifactValidationError(f"Answer span not found: {claim_id}")
        claim_reference_ids = set(claim.get("reference_ids", []))
        unknown_references = sorted(claim_reference_ids - reference_ids)
        if not claim_reference_ids or unknown_references:
            raise ArtifactValidationError(
                f"Answer claim {claim_id} uses unknown References: "
                f"{unknown_references}"
            )
        referenced_by_answer.update(claim_reference_ids)

    orphan_references = sorted(
        reference_ids - referenced_by_keypoints - referenced_by_answer
    )
    if orphan_references:
        raise ArtifactValidationError(
            f"QRA contains unused References: {orphan_references}"
        )
