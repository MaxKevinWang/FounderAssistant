"""Prepare the review-driven Configuration v1.2 and QRA curation v0.3.

The formal evaluation artifacts are append-only.  This module therefore does
not edit the frozen Configuration v1.1 release or the QRA v0.2 candidates.  It
issues narrowly scoped Configuration revisions, builds a complete mixed-version
manifest, and derives a new semantic-curation artifact for another human review.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

from apps.evaluations.corpus_paths import resolve_corpus_path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

from apps.evaluations.formal_configuration_generation import DOMAIN_SCHEMA_PATHS
from apps.evaluations.rageval_artifacts import (
    ArtifactValidationError,
    sha256_file,
    validate_configuration_record,
    validate_document_record,
    write_json_artifact,
)


FORMAL_ROOT = "docs/evaluation/formal_dataset"
BASE_CONFIGURATION_MANIFEST_PATH = (
    f"{FORMAL_ROOT}/configurations/formal_test/"
    "formal_test_configuration_manifest_v1.1.json"
)
REVISED_CONFIGURATION_MANIFEST_PATH = (
    f"{FORMAL_ROOT}/configurations/formal_test/"
    "formal_test_configuration_manifest_v1.2.json"
)
BASE_CURATION_PATH = f"{FORMAL_ROOT}/qra/qra_semantic_curation_v0.2.json"
REVISED_CURATION_PATH = f"{FORMAL_ROOT}/qra/qra_semantic_curation_v0.3.json"
REVISION_PROTOCOL_PATH = (
    f"{FORMAL_ROOT}/prompts/qra_review_revision_protocol_v1.0.md"
)

REVISED_CONFIGURATION_IDS = {
    "C-FT-LFS-S-002",
    "C-FT-LFS-TS-001",
    "C-FT-LFS-U-002",
    "C-FT-FP-MHR-002",
    "C-FT-FP-MHR-003",
    "C-FT-FP-TS-002",
    "C-FT-MV-F-004",
    "C-FT-MV-F-005",
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


def _read_json_object(path: Path) -> dict[str, Any]:
    """Read one UTF-8 JSON object with an artifact-oriented error."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(f"Cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ArtifactValidationError(f"Expected a JSON object: {path}")
    return payload


def _write_or_verify(path: Path, payload: dict[str, Any]) -> str:
    """Write a new immutable artifact, or verify an identical prior run."""

    if path.exists():
        if _read_json_object(path) != payload:
            raise ArtifactValidationError(
                f"Versioned artifact already exists with different content: {path}"
            )
        return "unchanged"
    write_json_artifact(path, payload)
    return "created"


def _compact(value: str) -> str:
    """Collapse source line wrapping while preserving the source wording."""

    return " ".join(value.split())


def _next_minor(version: str) -> str:
    """Increment the minor component of a two-part artifact version."""

    major, minor = version.split(".", maxsplit=1)
    return f"{major}.{int(minor) + 1}"


def _block(
    documents: dict[str, dict[str, Any]],
    document_id: str,
    block_id: str,
) -> dict[str, Any]:
    """Return one stable Document D block."""

    try:
        return next(
            item
            for item in documents[document_id]["blocks"]
            if item["block_id"] == block_id
        )
    except (KeyError, StopIteration) as exc:
        raise ArtifactValidationError(
            f"Unknown formal evidence block: {document_id}/{block_id}"
        ) from exc


def _quote(
    documents: dict[str, dict[str, Any]],
    document_id: str,
    block_id: str,
    *,
    start: str | None = None,
    end: str | None = None,
) -> str:
    """Extract an exact, reviewable span using stable boundary text."""

    text = _block(documents, document_id, block_id)["text"]
    start_index = 0
    if start is not None:
        try:
            start_index = text.index(start)
        except ValueError as exc:
            raise ArtifactValidationError(
                f"Quote start not found in {document_id}/{block_id}: {start!r}"
            ) from exc
    end_index = len(text)
    if end is not None:
        try:
            end_index = text.index(end, start_index) + len(end)
        except ValueError as exc:
            raise ArtifactValidationError(
                f"Quote end not found in {document_id}/{block_id}: {end!r}"
            ) from exc
    quote = text[start_index:end_index]
    if not quote.strip():
        raise ArtifactValidationError(f"Empty quote in {document_id}/{block_id}")
    return quote


def _source(
    documents: dict[str, dict[str, Any]],
    document_id: str,
    block_ids: list[str],
) -> dict[str, Any]:
    """Build a Configuration source declaration from the frozen Document set."""

    for block_id in block_ids:
        _block(documents, document_id, block_id)
    return {
        "document_id": document_id,
        "content_sha256": documents[document_id]["content_sha256"],
        "allowed_block_ids": block_ids,
    }


def _provenance(
    *,
    provenance_id: str,
    json_pointer: str,
    documents: dict[str, dict[str, Any]],
    document_id: str,
    block_id: str,
    exact_quote: str,
) -> dict[str, Any]:
    """Build one direct evidence mapping for a populated Configuration value."""

    block = _block(documents, document_id, block_id)
    if exact_quote not in block["text"]:
        raise ArtifactValidationError(
            f"Non-exact revision quote: {document_id}/{block_id}"
        )
    return {
        "provenance_id": provenance_id,
        "json_pointer": json_pointer,
        "document_id": document_id,
        "block_id": block_id,
        "heading_path": str(block.get("heading_path") or ""),
        "support_type": "direct",
        "derivation": "",
        "exact_quote": exact_quote,
    }


def _allowed_information(sources: list[dict[str, Any]]) -> str:
    """Describe the exact revised source scope without copying full block text."""

    scopes = []
    for source in sources:
        scopes.append(
            f"{source['document_id']} ({', '.join(source['allowed_block_ids'])})"
        )
    return "Nur die konfigurierten Informationen aus " + "; ".join(scopes) + "."


def _replace_evidence(
    configuration: dict[str, Any],
    *,
    documents: dict[str, dict[str, Any]],
    sources: list[tuple[str, list[str]]],
    values_key: str,
    value_field: str,
    evidence: list[tuple[str, str, str]],
) -> None:
    """Replace source scope, values, and exact provenance as one atomic change."""

    source_records = [
        _source(documents, document_id, block_ids)
        for document_id, block_ids in sources
    ]
    configuration["source_scope"] = (
        "single_document" if len(source_records) == 1 else "multiple_documents"
    )
    configuration["source_documents"] = source_records
    configuration["question_generation_constraints"]["allowed_information"] = (
        _allowed_information(source_records)
    )

    value_records = []
    provenance_records = []
    for index, (document_id, block_id, exact_quote) in enumerate(evidence):
        value_records.append({value_field: _compact(exact_quote)})
        provenance_records.append(
            _provenance(
                provenance_id=f"P{index + 1:03d}",
                json_pointer=f"/values/{values_key}/{index}/{value_field}",
                documents=documents,
                document_id=document_id,
                block_id=block_id,
                exact_quote=exact_quote,
            )
        )
    configuration["values"] = {values_key: value_records}
    configuration["provenance"] = provenance_records


def _revise_lfs_summary(
    configuration: dict[str, Any],
    documents: dict[str, dict[str, Any]],
) -> None:
    """Replace the unsafe blanket liability claim with form-specific evidence."""

    configuration["question_generation_constraints"].update(
        {
            "focus": (
                "Beteiligtenstruktur und unterschiedliche Haftungsmodelle bei "
                "Personengesellschaften"
            ),
            "required_reasoning_operation": (
                "Die Beteiligtenstruktur und die dokumentierten Haftungsunterschiede "
                "zwischen KG und PartG knapp zusammenfassen."
            ),
            "must_require_order": False,
        }
    )
    _replace_evidence(
        configuration,
        documents=documents,
        sources=[
            (
                "D-LFS-009",
                ["D-LFS-009-B005", "D-LFS-009-B036", "D-LFS-009-B039"],
            )
        ],
        values_key="rulesAndExceptions",
        value_field="rule",
        evidence=[
            (
                "D-LFS-009",
                "D-LFS-009-B005",
                _quote(documents, "D-LFS-009", "D-LFS-009-B005"),
            ),
            (
                "D-LFS-009",
                "D-LFS-009-B036",
                _quote(
                    documents,
                    "D-LFS-009",
                    "D-LFS-009-B036",
                    start="Bei der KG haftet",
                    end="bestimmten Einlage.",
                ),
            ),
            (
                "D-LFS-009",
                "D-LFS-009-B039",
                _quote(
                    documents,
                    "D-LFS-009",
                    "D-LFS-009-B039",
                    start="Die Partnerschaftsgesellschaft hat",
                    end="vollumfänglich haften.",
                ),
            ),
        ],
    )


def _revise_lfs_temporal(
    configuration: dict[str, Any],
    documents: dict[str, dict[str, Any]],
) -> None:
    """Replace the superseded paper/mail tax-registration path with ELSTER."""

    configuration["question_generation_constraints"].update(
        {
            "focus": "Elektronische steuerliche Erfassung einer Gründung über ELSTER",
            "required_reasoning_operation": (
                "Frist, elektronische Übermittlung und anschließende Berechnung der "
                "ersten Steuerforderungen in Abhängigkeitsreihenfolge wiedergeben."
            ),
            "must_require_order": True,
        }
    )
    _replace_evidence(
        configuration,
        documents=documents,
        sources=[("D-LFS-008", ["D-LFS-008-B001"])],
        values_key="formationProcess",
        value_field="step",
        evidence=[
            (
                "D-LFS-008",
                "D-LFS-008-B001",
                _quote(
                    documents,
                    "D-LFS-008",
                    "D-LFS-008-B001",
                    start="Gründerinnen und Gründer sind seit dem 1. Januar 2021",
                    end="beim Finanzamt anzumelden.",
                ),
            ),
            (
                "D-LFS-008",
                "D-LFS-008-B001",
                _quote(
                    documents,
                    "D-LFS-008",
                    "D-LFS-008-B001",
                    start="Für diesen Zweck müssen sie den",
                    end="www.elster.de",
                ),
            ),
            (
                "D-LFS-008",
                "D-LFS-008-B001",
                _quote(
                    documents,
                    "D-LFS-008",
                    "D-LFS-008-B001",
                    start="Aus den Angaben zur Rechtsform",
                    end="erste Steuerforderungen.",
                ),
            ),
        ],
    )


def _revise_lfs_unanswerable(
    configuration: dict[str, Any],
    documents: dict[str, dict[str, Any]],
) -> None:
    """Use a distinct, intentionally underspecified legal-form tax comparison."""

    assumptions = {
        item["json_pointer"]: item for item in configuration["protocol_assumptions"]
    }
    assumptions["/question_generation_constraints/hypothetical_scenario"] = {
        "json_pointer": "/question_generation_constraints/hypothetical_scenario",
        "value": (
            "Eine Gründerin möchte wissen, ob ihr bei identischem Umsatz mit einer "
            "UG oder einem Einzelunternehmen exakt mehr Netto verbleibt. Sie nennt "
            "weder den erwarteten steuerpflichtigen Gewinn noch Standort, "
            "Ausschüttungs- oder Vergütungsplanung und persönliche Steuerdaten"
        ),
        "basis": (
            "Author-defined underspecified scenario for corpus-wide "
            "unanswerability review."
        ),
    }
    assumptions["/question_generation_constraints/missing_facts"] = {
        "json_pointer": "/question_generation_constraints/missing_facts",
        "value": (
            "erwarteter steuerpflichtiger Gewinn, kommunaler Hebesatz, geplante "
            "Geschäftsführungsvergütung, Ausschüttung oder Gewinnthesaurierung sowie "
            "persönlicher Einkommensteuersatz und weitere Einkünfte"
        ),
        "basis": (
            "These facts are intentionally omitted; general tax descriptions cannot "
            "determine an exact case-specific net result."
        ),
    }
    configuration["protocol_assumptions"] = list(assumptions.values())
    configuration["question_generation_constraints"].update(
        {
            "focus": (
                "Exakter Nettovergleich zwischen UG und Einzelunternehmen im "
                "konkreten Gründungsfall"
            ),
            "required_reasoning_operation": (
                "Eine exakte Einzelfallprognose verweigern, die fehlenden "
                "Berechnungsparameter benennen und nur allgemeine Steuerregeln nennen."
            ),
            "forbidden_information": (
                "Keine definitive Rechtsformempfehlung, keine erfundenen Steuerdaten "
                "und keine Ergänzung der absichtlich fehlenden Falldaten."
            ),
            "must_require_order": False,
        }
    )

    # Preserve all eight declared Documents for the corpus-wide answerability test,
    # but ground the populated values only in the relevant general tax blocks.
    configuration["question_generation_constraints"]["allowed_information"] = (
        "Nur allgemeine Regeln aus dem eingefrorenen legal_form_selection-Korpus; "
        "insbesondere die belegten Steuerunterschiede aus D-LFS-008."
    )
    evidence = [
        (
            "D-LFS-008",
            "D-LFS-008-B002",
            _quote(
                documents,
                "D-LFS-008",
                "D-LFS-008-B002",
                start="Bei Einzelunternehmen oder Gesellschaftern",
                end="im Unternehmen verbleiben oder nicht.",
            ),
        ),
        (
            "D-LFS-008",
            "D-LFS-008-B003",
            _quote(
                documents,
                "D-LFS-008",
                "D-LFS-008-B003",
                start="Die Körperschaftsteuer fällt",
                end="ob ausgeschüttet wird oder nicht.",
            ),
        ),
        (
            "D-LFS-008",
            "D-LFS-008-B003",
            _quote(
                documents,
                "D-LFS-008",
                "D-LFS-008-B003",
                start="Das Ergebnis der endgültigen",
                end="unterschiedliche Konditionen.",
            ),
        ),
        (
            "D-LFS-008",
            "D-LFS-008-B003",
            _quote(
                documents,
                "D-LFS-008",
                "D-LFS-008-B003",
                start="Die Gewerbesteuer wird bei Personenunternehmen",
                end="von der Gewerbesteuer entlastet.",
            ),
        ),
    ]
    configuration["values"] = {
        "rulesAndExceptions": [
            {"rule": _compact(exact_quote)} for _, _, exact_quote in evidence
        ]
    }
    configuration["provenance"] = [
        _provenance(
            provenance_id=f"P{index + 1:03d}",
            json_pointer=f"/values/rulesAndExceptions/{index}/rule",
            documents=documents,
            document_id=document_id,
            block_id=block_id,
            exact_quote=exact_quote,
        )
        for index, (document_id, block_id, exact_quote) in enumerate(evidence)
    ]


def _revise_fp_temporal(
    configuration: dict[str, Any],
    documents: dict[str, dict[str, Any]],
) -> None:
    """Replace the leaked phase question with application preparation."""

    configuration["question_generation_constraints"].update(
        {
            "focus": (
                "Vorbereitung, fachkundige Stellungnahme und fristgerechter Antrag "
                "auf Gründungszuschuss"
            ),
            "required_reasoning_operation": (
                "Die dokumentierten Voraussetzungen als zeitliche und sachliche "
                "Abhängigkeit vom Unterlagenpaket bis zum Antrag rekonstruieren."
            ),
            "must_require_order": True,
        }
    )
    _replace_evidence(
        configuration,
        documents=documents,
        sources=[
            (
                "D-FP-007",
                ["D-FP-007-B005", "D-FP-007-B008", "D-FP-007-B009"],
            )
        ],
        values_key="applicationProcess",
        value_field="step",
        evidence=[
            (
                "D-FP-007",
                "D-FP-007-B008",
                _quote(
                    documents,
                    "D-FP-007",
                    "D-FP-007-B008",
                    start="Gründungszuschuss beantragen Agentur für Arbeit",
                    end="(vor Aufnahme der Selbständigkeit)",
                ),
            ),
            (
                "D-FP-007",
                "D-FP-007-B005",
                _quote(
                    documents,
                    "D-FP-007",
                    "D-FP-007-B005",
                    start="Antrag: Wer einen Antrag",
                    end="Tätigkeit beim Finanzamt",
                ),
            ),
            (
                "D-FP-007",
                "D-FP-007-B009",
                _quote(
                    documents,
                    "D-FP-007",
                    "D-FP-007-B009",
                    start="Arbeitslose, die mithilfe",
                    end="Tragfähigkeit des Vorhabens.",
                ),
            ),
            (
                "D-FP-007",
                "D-FP-007-B009",
                _quote(
                    documents,
                    "D-FP-007",
                    "D-FP-007-B009",
                    start="Diese Stellungnahme wurde",
                    end="Angaben abgegeben.",
                ),
            ),
        ],
    )


def _revise_fp_profitability_focus(configuration: dict[str, Any]) -> None:
    """Align the reasoning target with the already frozen financial evidence."""

    configuration["question_generation_constraints"]["focus"] = (
        "Zusammenhang von Umsatz, Kostenstruktur, privatem Lebensunterhalt, "
        "Kostenrisiko und Rentabilität"
    )


def _revise_fp_credit_process(
    configuration: dict[str, Any],
    documents: dict[str, dict[str, Any]],
) -> None:
    """Replace disconnected credit facts with one complete decision chain."""

    configuration["question_generation_constraints"].update(
        {
            "focus": (
                "Kreditprozess vom Finanzierungsbedarf bis zur Auswahl eines Angebots"
            ),
            "required_reasoning_operation": (
                "Die fünf dokumentierten Prozessschritte zu einer kausalen "
                "Entscheidungskette verbinden."
            ),
            "must_require_order": False,
        }
    )
    block_ids = [
        "D-FP-011-B135",
        "D-FP-011-B137",
        "D-FP-011-B138",
        "D-FP-011-B139",
        "D-FP-011-B140",
    ]
    _replace_evidence(
        configuration,
        documents=documents,
        sources=[("D-FP-011", block_ids)],
        values_key="applicationProcess",
        value_field="step",
        evidence=[
            ("D-FP-011", block_id, _quote(documents, "D-FP-011", block_id))
            for block_id in block_ids
        ],
    )


def _revise_mv_empathy_map(
    configuration: dict[str, Any],
    documents: dict[str, dict[str, Any]],
) -> None:
    """Replace a development-set duplicate with an Empathy Map target."""

    configuration["question_generation_constraints"].update(
        {
            "focus": "Zweck, Durchführung und Nutzung einer Empathy Map",
            "required_reasoning_operation": (
                "Die konfigurierten Einzelfakten zur Empathy Map präzise wiedergeben."
            ),
            "must_require_order": False,
        }
    )
    _replace_evidence(
        configuration,
        documents=documents,
        sources=[
            (
                "D-MV-001",
                [
                    "D-MV-001-B057",
                    "D-MV-001-B058",
                    "D-MV-001-B060",
                    "D-MV-001-B068",
                ],
            )
        ],
        values_key="customerAndMarketResearch",
        value_field="finding",
        evidence=[
            (
                "D-MV-001",
                block_id,
                _quote(documents, "D-MV-001", block_id),
            )
            for block_id in [
                "D-MV-001-B057",
                "D-MV-001-B058",
                "D-MV-001-B060",
                "D-MV-001-B068",
            ]
        ],
    )


def _revise_mv_mvp_suitability(
    configuration: dict[str, Any],
    documents: dict[str, dict[str, Any]],
) -> None:
    """Replace the duplicated MVP definition with suitability and financing."""

    configuration["question_generation_constraints"].update(
        {
            "focus": "Eignung eines MVP und Darstellung eines finanzierten Markttests",
            "required_reasoning_operation": (
                "Die dokumentierten Eignungsbedingungen und die Anforderung bei "
                "Finanzierungsbedarf präzise wiedergeben."
            ),
            "must_require_order": False,
        }
    )
    _replace_evidence(
        configuration,
        documents=documents,
        sources=[("D-MV-003", ["D-MV-003-B008", "D-MV-003-B025"])],
        values_key="businessModelHypotheses",
        value_field="assumption",
        evidence=[
            (
                "D-MV-003",
                "D-MV-003-B008",
                _quote(
                    documents,
                    "D-MV-003",
                    "D-MV-003-B008",
                    start="Nicht jede Gründung",
                    end="braucht ein MVP.",
                ),
            ),
            (
                "D-MV-003",
                "D-MV-003-B008",
                _quote(
                    documents,
                    "D-MV-003",
                    "D-MV-003-B008",
                    start="Aber vor allem technikaffine",
                    end="nicht verzichten.",
                ),
            ),
            (
                "D-MV-003",
                "D-MV-003-B008",
                _quote(
                    documents,
                    "D-MV-003",
                    "D-MV-003-B008",
                    start="Man kann sagen:",
                    end="Methode angeraten.",
                ),
            ),
            (
                "D-MV-003",
                "D-MV-003-B025",
                _quote(
                    documents,
                    "D-MV-003",
                    "D-MV-003-B025",
                    start="Falls das bei dir",
                    end="wie dieses aussehen kann.",
                ),
            ),
        ],
    )


def _revise_configuration(
    *,
    base: dict[str, Any],
    documents: dict[str, dict[str, Any]],
    revised_at: str,
    prompt_hash: str,
) -> dict[str, Any]:
    """Create one approved, immutable Configuration correction."""

    revised = deepcopy(base)
    revised_version = _next_minor(base["version"])
    revised["version"] = revised_version
    revised["status"] = "frozen"
    revised["supersedes"] = f"{base['configuration_id']} v{base['version']}"

    revision_functions = {
        "C-FT-LFS-S-002": _revise_lfs_summary,
        "C-FT-LFS-TS-001": _revise_lfs_temporal,
        "C-FT-LFS-U-002": _revise_lfs_unanswerable,
        "C-FT-FP-TS-002": _revise_fp_temporal,
        "C-FT-FP-MHR-003": _revise_fp_credit_process,
        "C-FT-MV-F-004": _revise_mv_empathy_map,
        "C-FT-MV-F-005": _revise_mv_mvp_suitability,
    }
    configuration_id = revised["configuration_id"]
    if configuration_id == "C-FT-FP-MHR-002":
        _revise_fp_profitability_focus(revised)
    else:
        revision_functions[configuration_id](revised, documents)

    revised["generation_record"] = {
        "method": "Deterministic review-driven Configuration correction",
        "generated_at": revised_at,
        "prompt_path": REVISION_PROTOCOL_PATH,
        "prompt_sha256": prompt_hash,
        "mechanical_validation": (
            "passed: predecessor hash, JSON Schema v1.1, domain Schema keys, "
            "frozen Document hashes, allowed blocks, exact quotes, provenance "
            "pointers, and unchanged formal-set allocation"
        ),
        "human_review_status": "approved",
    }
    return revised


def _configuration_freeze_record(
    *,
    repo_root: Path,
    revised_path: Path,
    revised: dict[str, Any],
    predecessor_path: Path,
    document_declarations: dict[str, dict[str, Any]],
    record_schema_path: Path,
    revised_at: str,
) -> dict[str, Any]:
    """Create the immutable dependency record for one revised Configuration."""

    source_documents = []
    for source in revised["source_documents"]:
        declaration = document_declarations[source["document_id"]]
        source_documents.append(
            {
                "document_id": source["document_id"],
                "path": declaration["path"],
                "file_sha256": declaration["file_sha256"],
                "content_sha256": declaration["content_sha256"],
                "freeze_record_path": declaration["freeze_record_path"],
                "freeze_record_sha256": declaration["freeze_record_sha256"],
            }
        )
    return {
        "artifact_type": "RAGEval Configuration C",
        "configuration_id": revised["configuration_id"],
        "version": revised["version"],
        "status": "frozen",
        "frozen_at": revised_at,
        "file": revised_path.relative_to(repo_root).as_posix(),
        "file_sha256": sha256_file(revised_path),
        "supersedes": revised["supersedes"],
        "predecessor_file": predecessor_path.relative_to(repo_root).as_posix(),
        "predecessor_file_sha256": sha256_file(predecessor_path),
        "record_schema": {
            "path": record_schema_path.relative_to(repo_root).as_posix(),
            "sha256": sha256_file(record_schema_path),
        },
        "domain_schema": {
            "path": DOMAIN_SCHEMA_PATHS[revised["domain"]],
            "sha256": sha256_file(repo_root / DOMAIN_SCHEMA_PATHS[revised["domain"]]),
        },
        "source_documents": source_documents,
        "revision_protocol": {
            "path": REVISION_PROTOCOL_PATH,
            "sha256": sha256_file(repo_root / REVISION_PROTOCOL_PATH),
        },
        "approval": {
            "basis": "explicit_project_owner_correction_instruction",
            "approved_at": revised_at,
            "scope": (
                "repair the final semantic-audit findings while preserving the "
                "formal allocation and frozen Document D release"
            ),
        },
        "change_policy": (
            "Do not edit this frozen Configuration or freeze record in place; "
            "create a new version for every later change."
        ),
    }


def _schema_errors(
    payload: dict[str, Any],
    record_schema: dict[str, Any],
) -> list[str]:
    """Return compact JSON Schema validation errors."""

    errors = sorted(
        Draft202012Validator(
            record_schema,
            format_checker=FormatChecker(),
        ).iter_errors(payload),
        key=lambda error: list(error.absolute_path),
    )
    return [
        f"{'/'.join(str(part) for part in error.absolute_path)}: {error.message}"
        for error in errors
    ]


def _assert_balance(configurations: list[dict[str, Any]]) -> None:
    """Ensure corrections preserve the frozen 20-by-3 allocation."""

    if len(configurations) != 60:
        raise ArtifactValidationError("Formal Configuration set must contain 60 items")
    if len({item["configuration_id"] for item in configurations}) != 60:
        raise ArtifactValidationError("Formal Configuration IDs are not unique")
    if Counter(item["answerability"] for item in configurations) != {
        "answerable": 54,
        "unanswerable": 6,
    }:
        raise ArtifactValidationError("Formal answerability allocation changed")
    for domain in DOMAIN_SCHEMA_PATHS:
        domain_items = [item for item in configurations if item["domain"] == domain]
        if len(domain_items) != 20 or Counter(
            item["target_question_type"] for item in domain_items
        ) != EXPECTED_TYPE_COUNTS:
            raise ArtifactValidationError(
                f"Formal question-type allocation changed for {domain}"
            )


def _load_documents(
    *,
    repo_root: Path,
    manifest: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Load and verify the frozen Document release bound by Configurations."""

    documents: dict[str, dict[str, Any]] = {}
    declarations: dict[str, dict[str, Any]] = {}
    for declaration in manifest["documents"]:
        path = repo_root / declaration["path"]
        if sha256_file(path) != declaration["file_sha256"]:
            raise ArtifactValidationError(f"Frozen Document hash mismatch: {path}")
        document = _read_json_object(path)
        validate_document_record(
            document=document,
            source_path=resolve_corpus_path(repo_root, document["source"]["path"]),
        )
        documents[document["document_id"]] = document
        declarations[document["document_id"]] = declaration
    return documents, declarations


def freeze_configuration_review_revisions(
    *,
    repo_root: Path,
    revised_at: str,
) -> dict[str, Any]:
    """Issue the mixed-version frozen Configuration manifest v1.2."""

    repo_root = repo_root.resolve()
    prompt_path = repo_root / REVISION_PROTOCOL_PATH
    if not prompt_path.exists():
        raise ArtifactValidationError(f"Revision protocol is missing: {prompt_path}")
    prompt_hash = sha256_file(prompt_path)

    base_manifest_path = repo_root / BASE_CONFIGURATION_MANIFEST_PATH
    base_manifest = _read_json_object(base_manifest_path)
    if (
        base_manifest.get("status") != "frozen"
        or base_manifest.get("version") != "1.1"
        or base_manifest.get("configuration_count") != 60
    ):
        raise ArtifactValidationError(
            f"Unexpected predecessor Configuration manifest: {base_manifest_path}"
        )

    document_manifest_path = repo_root / base_manifest["document_manifest"]["path"]
    if sha256_file(document_manifest_path) != base_manifest["document_manifest"][
        "sha256"
    ]:
        raise ArtifactValidationError("Frozen Document manifest hash mismatch")
    document_manifest = _read_json_object(document_manifest_path)
    documents, document_declarations = _load_documents(
        repo_root=repo_root,
        manifest=document_manifest,
    )

    record_schema_path = repo_root / base_manifest["record_schema"]["path"]
    if sha256_file(record_schema_path) != base_manifest["record_schema"]["sha256"]:
        raise ArtifactValidationError("Configuration record schema hash mismatch")
    record_schema = _read_json_object(record_schema_path)
    domain_schemas = {
        domain: _read_json_object(repo_root / path)
        for domain, path in DOMAIN_SCHEMA_PATHS.items()
    }

    write_states: list[str] = []
    declarations = []
    configurations = []
    revised_ids = []
    for declaration in base_manifest["configurations"]:
        predecessor_path = repo_root / declaration["path"]
        if sha256_file(predecessor_path) != declaration["file_sha256"]:
            raise ArtifactValidationError(
                f"Frozen Configuration hash mismatch: {predecessor_path}"
            )
        base = _read_json_object(predecessor_path)
        configuration_id = base["configuration_id"]
        if configuration_id not in REVISED_CONFIGURATION_IDS:
            carried = deepcopy(declaration)
            carried["resolution"] = "carried_forward"
            declarations.append(carried)
            configurations.append(base)
            continue

        revised = _revise_configuration(
            base=base,
            documents=documents,
            revised_at=revised_at,
            prompt_hash=prompt_hash,
        )
        errors = _schema_errors(revised, record_schema)
        if errors:
            raise ArtifactValidationError(
                f"Invalid revised Configuration {configuration_id}: "
                f"{'; '.join(errors[:5])}"
            )
        validate_configuration_record(
            configuration=revised,
            schema=domain_schemas[revised["domain"]],
            documents=documents,
        )

        revised_path = predecessor_path.with_name(
            f"{configuration_id}_v{revised['version']}.json"
        )
        write_states.append(_write_or_verify(revised_path, revised))
        freeze_path = revised_path.with_name(
            f"{revised_path.stem}_freeze_record.json"
        )
        freeze_record = _configuration_freeze_record(
            repo_root=repo_root,
            revised_path=revised_path,
            revised=revised,
            predecessor_path=predecessor_path,
            document_declarations=document_declarations,
            record_schema_path=record_schema_path,
            revised_at=revised_at,
        )
        write_states.append(_write_or_verify(freeze_path, freeze_record))
        declarations.append(
            {
                "configuration_id": configuration_id,
                "version": revised["version"],
                "resolution": "revised_frozen",
                "domain": revised["domain"],
                "target_question_type": revised["target_question_type"],
                "answerability": revised["answerability"],
                "path": revised_path.relative_to(repo_root).as_posix(),
                "file_sha256": sha256_file(revised_path),
                "freeze_record_path": freeze_path.relative_to(repo_root).as_posix(),
                "freeze_record_sha256": sha256_file(freeze_path),
                "source_document_ids": [
                    source["document_id"] for source in revised["source_documents"]
                ],
                "provenance_count": len(revised["provenance"]),
            }
        )
        configurations.append(revised)
        revised_ids.append(configuration_id)

    if set(revised_ids) != REVISED_CONFIGURATION_IDS:
        raise ArtifactValidationError(
            f"Unexpected Configuration revision set: {sorted(revised_ids)}"
        )
    _assert_balance(configurations)

    revised_manifest = {
        "manifest_id": "formal_test_configuration_manifest_v1.2",
        "version": "1.2",
        "status": "frozen",
        "frozen_at": revised_at,
        "supersedes": "formal_test_configuration_manifest_v1.1",
        "predecessor_manifest": {
            "path": BASE_CONFIGURATION_MANIFEST_PATH,
            "sha256": sha256_file(base_manifest_path),
        },
        "candidate_manifest": base_manifest["candidate_manifest"],
        "document_manifest": base_manifest["document_manifest"],
        "record_schema": base_manifest["record_schema"],
        "revision_protocol": {
            "path": REVISION_PROTOCOL_PATH,
            "sha256": prompt_hash,
        },
        "configuration_count": 60,
        "revised_configuration_count": len(revised_ids),
        "carried_forward_count": 60 - len(revised_ids),
        "answerable_count": 54,
        "unanswerable_count": 6,
        "configurations": declarations,
    }
    manifest_path = repo_root / REVISED_CONFIGURATION_MANIFEST_PATH
    write_states.append(_write_or_verify(manifest_path, revised_manifest))
    manifest_freeze_path = manifest_path.with_name(
        "formal_test_configuration_manifest_v1.2_freeze_record.json"
    )
    manifest_freeze = {
        "artifact_type": "Formal test Configuration C manifest",
        "manifest_id": revised_manifest["manifest_id"],
        "version": "1.2",
        "status": "frozen",
        "frozen_at": revised_at,
        "file": manifest_path.relative_to(repo_root).as_posix(),
        "file_sha256": sha256_file(manifest_path),
        "predecessor_manifest": BASE_CONFIGURATION_MANIFEST_PATH,
        "predecessor_manifest_sha256": sha256_file(base_manifest_path),
        "configuration_count": 60,
        "revised_configuration_ids": sorted(revised_ids),
        "approval_basis": "explicit_project_owner_correction_instruction",
    }
    write_states.append(_write_or_verify(manifest_freeze_path, manifest_freeze))
    return {
        "manifest_path": manifest_path.relative_to(repo_root).as_posix(),
        "revised_configuration_ids": sorted(revised_ids),
        "created_count": write_states.count("created"),
        "unchanged_count": write_states.count("unchanged"),
    }


def _reference(
    reference_key: str,
    *,
    document_id: str | None = None,
    block_id: str | None = None,
    exact_quote: str | None = None,
    provenance_id: str | None = None,
) -> dict[str, Any]:
    """Build one semantic-curation reference selector."""

    reference: dict[str, Any] = {"reference_key": reference_key}
    if provenance_id is not None:
        reference["provenance_id"] = provenance_id
    else:
        reference.update(
            {
                "document_id": document_id,
                "block_id": block_id,
                "exact_quote": exact_quote,
            }
        )
    return reference


def _unit(text: str, *reference_keys: str) -> dict[str, Any]:
    """Build one answer/keypoint unit for semantic curation."""

    return {"text": text, "reference_keys": list(reference_keys)}


def _entry(
    question_type: str,
    question: str,
    references: list[dict[str, Any]],
    answer_units: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build one complete QRA semantic recipe."""

    return {
        "question_type": question_type,
        "question": question,
        "references": references,
        "answer_units": answer_units,
    }


def _curation_replacements(
    documents: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Return all review-driven QRA replacements for v0.3."""

    return {
        "C-FT-LFS-F-002": _entry(
            "Factual",
            (
                "Welche grundlegenden Unterschiede nennt das Material zwischen "
                "Gewerbetreibenden und Angehörigen freier Berufe?"
            ),
            [
                _reference(
                    "E1",
                    document_id="D-LFS-006",
                    block_id="D-LFS-006-B001",
                    exact_quote=_quote(
                        documents,
                        "D-LFS-006",
                        "D-LFS-006-B001",
                        start="Wenn von Selbständigen",
                        end="freien Berufe \ngemeint.",
                    ),
                ),
                _reference(
                    "E2",
                    document_id="D-LFS-006",
                    block_id="D-LFS-006-B001",
                    exact_quote=_quote(
                        documents,
                        "D-LFS-006",
                        "D-LFS-006-B001",
                        start="Je nachdem jedoch",
                        end="zur Altersvorsorge.",
                    ),
                ),
                _reference(
                    "E3",
                    document_id="D-LFS-006",
                    block_id="D-LFS-006-B004",
                    exact_quote=_quote(
                        documents,
                        "D-LFS-006",
                        "D-LFS-006-B004",
                        start="Zu den Katalogberufen gehören",
                    ),
                ),
            ],
            [
                _unit(
                    "Selbstständige werden grundsätzlich entweder den "
                    "Gewerbetreibenden oder den Angehörigen freier Berufe zugeordnet.",
                    "E1",
                ),
                _unit(
                    "Die Einordnung beeinflusst Gründungsformalitäten, verfügbare "
                    "Rechtsformen sowie besondere Bedingungen für Berufszugang, "
                    "Berufsausübung und Altersvorsorge.",
                    "E2",
                ),
                _unit(
                    "Als Katalogberufe nennt das Material unter anderem Heilberufe, "
                    "rechts-, steuer- und wirtschaftsberatende Berufe, technische "
                    "Berufe sowie informationsvermittelnde und sprachliche Berufe.",
                    "E3",
                ),
            ],
        ),
        "C-FT-LFS-F-003": _entry(
            "Factual",
            (
                "Welchen Unternehmen oder Personen ordnet das Material Umsatz-, "
                "Einkommen-, Körperschaft- und Gewerbesteuer zu?"
            ),
            [
                _reference(
                    "E1",
                    document_id="D-LFS-008",
                    block_id="D-LFS-008-B002",
                    exact_quote=_quote(
                        documents,
                        "D-LFS-008",
                        "D-LFS-008-B002",
                        start="Jedes \nUnternehmen (außer Kleinunternehmen)",
                        end="in Rechnung zu stellen.",
                    ),
                ),
                _reference(
                    "E2",
                    document_id="D-LFS-008",
                    block_id="D-LFS-008-B002",
                    exact_quote=_quote(
                        documents,
                        "D-LFS-008",
                        "D-LFS-008-B002",
                        start="Einkommensteuer\nEinkommensteuer muss",
                        end="im Unternehmen verbleiben oder nicht.",
                    ),
                ),
                _reference(
                    "E3",
                    document_id="D-LFS-008",
                    block_id="D-LFS-008-B003",
                    exact_quote=_quote(
                        documents,
                        "D-LFS-008",
                        "D-LFS-008-B003",
                        start="Körperschaftsteuer\nDie Körperschaftsteuer fällt",
                        end="ob ausgeschüttet wird oder nicht.",
                    ),
                ),
                _reference(
                    "E4",
                    document_id="D-LFS-008",
                    block_id="D-LFS-008-B003",
                    exact_quote=_quote(
                        documents,
                        "D-LFS-008",
                        "D-LFS-008-B003",
                        start="Gewerbesteuer\nDie Gewerbesteuer betrifft",
                        end="Ausgenommen sind freie Berufe \nund Landwirtschaft.",
                    ),
                ),
            ],
            [
                _unit(
                    "Grundsätzlich stellt jedes Unternehmen außer Kleinunternehmen "
                    "seiner Kundschaft Umsatzsteuer in Rechnung.",
                    "E1",
                ),
                _unit(
                    "Einkommensteuer zahlen natürliche Personen; bei Einzelunternehmen "
                    "und Gesellschaftern von Personengesellschaften werden Gewinne "
                    "nach dem persönlichen Steuersatz besteuert.",
                    "E2",
                ),
                _unit(
                    "Körperschaftsteuer fällt für Kapitalgesellschaften wie GmbH, UG "
                    "und AG sowie für Genossenschaften an.",
                    "E3",
                ),
                _unit(
                    "Gewerbesteuer betrifft laut Material Industrie, Handel, Handwerk "
                    "und Dienstleistungen; freie Berufe und Landwirtschaft sind "
                    "ausgenommen.",
                    "E4",
                ),
            ],
        ),
        "C-FT-LFS-F-005": _entry(
            "Factual",
            (
                "Welche Gruppen nennt das Material als typische Nutzer einer "
                "Einnahmenüberschussrechnung, und welche ausgeschlossenen oder "
                "grenzüberschreitenden Fälle beschreibt es?"
            ),
            [
                _reference(
                    "E1",
                    document_id="D-LFS-011",
                    block_id="D-LFS-011-B106",
                    exact_quote=_quote(
                        documents,
                        "D-LFS-011",
                        "D-LFS-011-B106",
                        end="Einnahmenüberschussrechnung (EÜR).",
                    ),
                ),
                _reference("E2", provenance_id="P002"),
                _reference(
                    "E3",
                    document_id="D-LFS-011",
                    block_id="D-LFS-011-B008",
                    exact_quote=_quote(
                        documents,
                        "D-LFS-011",
                        "D-LFS-011-B008",
                        start="Bei den Rechtsformen",
                        end="doppelte Buchführungspflicht.",
                    ),
                ),
                _reference(
                    "E4",
                    document_id="D-LFS-011",
                    block_id="D-LFS-011-B008",
                    exact_quote=_quote(
                        documents,
                        "D-LFS-011",
                        "D-LFS-011-B008",
                        end="doppelte Buchführung nicht herum.",
                    ),
                ),
            ],
            [
                _unit(
                    "Freiberuflich Tätige und kleinere Gewerbe werden als typische "
                    "Nutzer der Einnahmenüberschussrechnung genannt.",
                    "E1",
                ),
                _unit(
                    "Für Gewerbetreibende nennt das Material weniger als 800.000 Euro "
                    "Umsatz und weniger als 80.000 Euro Gewinn pro Jahr als gemeinsam "
                    "einzuhaltende Werte.",
                    "E2",
                ),
                _unit(
                    "Als Überschreitungsbeispiel führt das Material weniger als "
                    "800.000 Euro Umsatz, aber mehr als 80.000 Euro Gewinn an; dann "
                    "besteht bereits doppelte Buchführungspflicht.",
                    "E3",
                ),
                _unit(
                    "Kapitalgesellschaften wie GmbH, UG und AG dürfen laut Material "
                    "keine Einnahmenüberschussrechnung erstellen.",
                    "E4",
                ),
            ],
        ),
        "C-FT-LFS-II-001": _entry(
            "Information Integration",
            (
                "Welches gemeinsame Bild ergeben die beiden Dokumente zu Kapital "
                "und Auswahlkriterien bei Personengesellschaften?"
            ),
            [
                _reference(
                    "E1",
                    document_id="D-LFS-009",
                    block_id="D-LFS-009-B006",
                    exact_quote=_quote(
                        documents,
                        "D-LFS-009",
                        "D-LFS-009-B006",
                        end="Doch du bestimmst die Summe.",
                    ),
                ),
                _reference("E2", provenance_id="P002"),
                _reference("E3", provenance_id="P004"),
            ],
            [
                _unit(
                    "Für Personengesellschaften ist keine Mindestkapitaleinlage "
                    "vorgeschrieben; die Beteiligten bestimmen selbst, ob und wie "
                    "viel Kapital sie einbringen.",
                    "E1",
                ),
                _unit(
                    "Das entspricht dem Grundprinzip, dass bei einer "
                    "Personengesellschaft die beteiligten Personen und nicht das "
                    "Kapital im Mittelpunkt stehen.",
                    "E2",
                ),
                _unit(
                    "Für die konkrete Rechtsformwahl betrachtet das Vergleichsmaterial "
                    "neben dem Mindestkapital auch Haftungsbeschränkung, Formalitäten "
                    "und Registereintrag.",
                    "E3",
                ),
            ],
        ),
        "C-FT-LFS-S-002": _entry(
            "Summarization",
            (
                "Wie unterscheiden sich die dokumentierten Beteiligten- und "
                "Haftungsmodelle bei Personengesellschaften?"
            ),
            [
                _reference("E1", provenance_id="P001"),
                _reference("E2", provenance_id="P002"),
                _reference("E3", provenance_id="P003"),
            ],
            [
                _unit(
                    "Eine Personengesellschaft verbindet mindestens zwei Personen mit "
                    "gemeinsamem unternehmerischem Ziel; beteiligt sein können "
                    "natürliche oder juristische Personen, etwa auch eine GmbH.",
                    "E1",
                ),
                _unit(
                    "Bei der KG haftet der Komplementär persönlich und unbeschränkt, "
                    "während Kommanditisten nur mit ihrer im Vertrag bestimmten "
                    "Einlage haften.",
                    "E2",
                ),
                _unit(
                    "Bei der PartG ist eine Haftungsbeschränkung für berufliche Fehler "
                    "möglich; für andere Schulden wie Bankkredite bleibt die Haftung "
                    "vollumfänglich.",
                    "E3",
                ),
            ],
        ),
        "C-FT-LFS-S-003": _entry(
            "Summarization",
            "Welche Grundprinzipien der Körperschaftsteuer nennt das Material?",
            [
                _reference("E1", provenance_id="P001"),
                _reference("E2", provenance_id="P002"),
                _reference("E3", provenance_id="P003"),
            ],
            [
                _unit(
                    "Der Körperschaftsteuersatz beträgt laut Material 15 Prozent; "
                    "mit Solidaritätszuschlag wird ein effektiver Satz von 15,83 "
                    "Prozent genannt.",
                    "E1",
                ),
                _unit(
                    "Die Körperschaftsteuerpflicht kann bereits in der Gründungsphase "
                    "und damit vor der Eintragung ins Handelsregister einsetzen.",
                    "E2",
                ),
                _unit(
                    "Bestimmte Personenhandelsgesellschaften können auf Antrag wie "
                    "eine Kapitalgesellschaft besteuert werden.",
                    "E3",
                ),
            ],
        ),
        "C-FT-LFS-TS-001": _entry(
            "Temporal Sequence",
            (
                "Welche Reihenfolge beschreibt das Material seit dem 1. Januar 2021 "
                "für die steuerliche Erfassung einer Gründung?"
            ),
            [
                _reference("E1", provenance_id="P001"),
                _reference("E2", provenance_id="P002"),
                _reference("E3", provenance_id="P003"),
            ],
            [
                _unit(
                    "Gründende müssen sich seit dem 1. Januar 2021 innerhalb eines "
                    "Monats beim Finanzamt anmelden.",
                    "E1",
                ),
                _unit(
                    "Dafür übermitteln sie den Fragebogen zur steuerlichen Erfassung "
                    "elektronisch über das ELSTER-Portal.",
                    "E2",
                ),
                _unit(
                    "Aus den Angaben zu Rechtsform sowie erwarteten Umsätzen und "
                    "Gewinnen berechnet das Finanzamt anschließend erste "
                    "Steuerforderungen.",
                    "E3",
                ),
            ],
        ),
        "C-FT-LFS-U-002": _entry(
            "Unanswerable",
            (
                "Eine Gründerin erwartet bei einer UG und einem Einzelunternehmen "
                "denselben Umsatz, nennt aber weder steuerpflichtigen Gewinn, Standort, "
                "Vergütungs- und Ausschüttungsplanung noch persönliche Steuerdaten. "
                "Lässt sich bestimmen, bei welcher Rechtsform ihr exakt mehr Netto "
                "verbleibt? Welche Angaben fehlen?"
            ),
            [
                _reference("E1", provenance_id="P001"),
                _reference("E2", provenance_id="P002"),
                _reference("E3", provenance_id="P003"),
                _reference("E4", provenance_id="P004"),
            ],
            [
                _unit(
                    "Mit den angegebenen Falldaten ist keine abschließende Entscheidung "
                    "über das exakte Nettoergebnis möglich.",
                    "E1",
                    "E2",
                    "E3",
                    "E4",
                ),
                _unit(
                    "Für eine belastbare Berechnung fehlen insbesondere erwarteter "
                    "steuerpflichtiger Gewinn, kommunaler Hebesatz, Vergütung, "
                    "Ausschüttung oder Thesaurierung sowie persönliche Steuerdaten.",
                    "E1",
                    "E2",
                    "E3",
                    "E4",
                ),
                _unit(
                    "Die Quellen beschreiben nur allgemeine Unterschiede bei "
                    "Einkommen-, Körperschaft- und Gewerbesteuer; diese ersetzen die "
                    "fehlenden fallbezogenen Rechengrößen nicht.",
                    "E1",
                    "E2",
                    "E3",
                    "E4",
                ),
            ],
        ),
        "C-FT-FP-II-002": _entry(
            "Information Integration",
            "Welche Funktion hat die Executive Summary innerhalb des Businessplans?",
            [
                _reference("E1", provenance_id="P003"),
                _reference("E2", provenance_id="P002"),
                _reference("E3", provenance_id="P001"),
            ],
            [
                _unit(
                    "Der Businessplan ist der ausführliche Fahrplan, der die wichtigen "
                    "Überlegungen zur Umsetzung der Geschäftsidee enthält.",
                    "E1",
                ),
                _unit(
                    "Die Executive Summary verdichtet die wichtigsten Punkte dieses "
                    "Plans kurz und verständlich.",
                    "E2",
                ),
                _unit(
                    "Als Einstieg soll sie bereits in den ersten Absätzen Interesse "
                    "wecken und ankündigen, was der Detailteil ausführt.",
                    "E3",
                ),
            ],
        ),
        "C-FT-FP-MHR-001": _entry(
            "Multi-hop Reasoning",
            (
                "Wie wirken Rückzahlungserwartung, Sicherheiten, Bürgschaften, Rating "
                "und Haftungsfreistellung bei einer Kreditentscheidung zusammen?"
            ),
            [
                _reference(
                    "E1",
                    document_id="D-FP-006",
                    block_id="D-FP-006-B001",
                    exact_quote=_quote(
                        documents,
                        "D-FP-006",
                        "D-FP-006-B001",
                        start="Banken und Sparkassen entscheiden",
                        end="die Kreditsicherheiten.",
                    ),
                ),
                _reference(
                    "E2",
                    document_id="D-FP-006",
                    block_id="D-FP-006-B001",
                    exact_quote=_quote(
                        documents,
                        "D-FP-006",
                        "D-FP-006-B001",
                        start="Bürgschaften\nEine besondere Form",
                        end="nicht zurückzahlen können.",
                    ),
                ),
                _reference(
                    "E3",
                    document_id="D-FP-006",
                    block_id="D-FP-006-B001",
                    exact_quote=_quote(
                        documents,
                        "D-FP-006",
                        "D-FP-006-B001",
                        start="Rating\nOb Bank oder Sparkasse",
                        end="alle Kreditnehmenden durchführen.",
                    ),
                ),
                _reference(
                    "E4",
                    document_id="D-FP-006",
                    block_id="D-FP-006-B004",
                    exact_quote=_quote(
                        documents,
                        "D-FP-006",
                        "D-FP-006-B004",
                        start="Allerdings kann dieser Kredit",
                        end="(= Haftungsfreistellung).",
                    ),
                ),
            ],
            [
                _unit(
                    "Eine Bank verleiht Geld nur bei erwarteter Rückzahlung; "
                    "Kreditsicherheiten können verwertet werden, falls die "
                    "Kreditnehmenden nicht zahlen.",
                    "E1",
                ),
                _unit(
                    "Bei einer Bürgschaft garantieren Dritte die Rückzahlung und "
                    "müssen bei Ausfall die Schuld begleichen.",
                    "E2",
                ),
                _unit(
                    "Sicherheiten und Bürgschaften reichen für die Entscheidung nicht "
                    "allein aus; zusätzlich fließt das Rating ein.",
                    "E3",
                ),
                _unit(
                    "Beim genannten KfW-Kredit kann eine Haftungsfreistellung das "
                    "Risiko der Hausbank senken, indem die KfW 80 Prozent ihrer "
                    "Haftung übernimmt.",
                    "E4",
                ),
            ],
        ),
        "C-FT-FP-MHR-002": _entry(
            "Multi-hop Reasoning",
            (
                "Wie führen Umsatz, Kostenstruktur und privater Lebensunterhalt "
                "gemeinsam zur Beurteilung von Gewinn, Risiko und Rentabilität?"
            ),
            [
                _reference("E1", provenance_id="P001"),
                _reference("E2", provenance_id="P002"),
                _reference("E3", provenance_id="P003"),
                _reference("E4", provenance_id="P004"),
            ],
            [
                _unit(
                    "Der Gewinn ergibt sich, indem die Kosten vom Umsatz abgezogen "
                    "werden.",
                    "E1",
                ),
                _unit(
                    "Je stärker Kosten mit dem Umsatz steigen und fallen, desto "
                    "geringer beschreibt das Material das Kostenrisiko.",
                    "E2",
                ),
                _unit(
                    "Auch der private Lebensunterhalt muss langfristig aus den "
                    "Umsatzerlösen gedeckt werden.",
                    "E3",
                ),
                _unit(
                    "Bleibt nach Abzug aller Kosten etwas vom Umsatz übrig, gilt das "
                    "Unternehmen in der vereinfachten Darstellung als rentabel.",
                    "E4",
                ),
            ],
        ),
        "C-FT-FP-MHR-003": _entry(
            "Multi-hop Reasoning",
            (
                "Wie führt der dokumentierte Kreditprozess vom Finanzierungsbedarf "
                "bis zur Auswahl eines Angebots?"
            ),
            [
                _reference(f"E{index}", provenance_id=f"P{index:03d}")
                for index in range(1, 6)
            ],
            [
                _unit(
                    "Zuerst werden Höhe und Zeitraum des Finanzierungsbedarfs mithilfe "
                    "des Finanz- oder Investitionsplans bestimmt.",
                    "E1",
                ),
                _unit(
                    "Nach Auswahl einer passenden Finanzierungsquelle folgt die "
                    "Anfrage mit persönlichen Angaben, Vorhabendetails und verlangten "
                    "Unterlagen wie dem Businessplan.",
                    "E2",
                ),
                _unit(
                    "Danach werden Rückfragen beantwortet, Beratungsgespräche geführt "
                    "und zusätzliche Unterlagen nachgereicht.",
                    "E3",
                ),
                _unit(
                    "Liegen mehrere Kreditzusagen vor, werden die Angebote miteinander "
                    "verglichen.",
                    "E4",
                ),
                _unit(
                    "Die abschließende Entscheidung berücksichtigt Kreditkonditionen "
                    "und mögliche persönliche Präferenzen wie die Beziehung zur "
                    "Hausbank.",
                    "E5",
                ),
            ],
        ),
        "C-FT-FP-S-003": _entry(
            "Summarization",
            "Welche Inhalte und Zielgruppenanforderungen nennt das Material für eine Executive Summary?",
            [
                _reference("E1", provenance_id="P001"),
                _reference("E2", provenance_id="P002"),
                _reference("E3", provenance_id="P003"),
                _reference("E4", provenance_id="P004"),
            ],
            [
                _unit(
                    "Investoren achten besonders auf benötigtes Investment, "
                    "wirtschaftliche Machbarkeit und Renditeaussichten.",
                    "E1",
                ),
                _unit(
                    "Arbeitsagentur und fachkundige Stellen prüfen Tragfähigkeit, "
                    "Ernsthaftigkeit sowie Markt- und Zielgruppenkenntnis; die Idee und "
                    "Zahlen sollen für fachfremde Prüfer verständlich sein.",
                    "E2",
                ),
                _unit(
                    "Im Finanzierungsteil sollen Kapitalbedarf sowie erwartete Umsatz- "
                    "und Gewinnentwicklung der ersten drei Jahre sichtbar werden.",
                    "E3",
                ),
                _unit(
                    "Als fünf Inhaltsgruppen nennt das Material Geschäftsidee, "
                    "Finanzierung, Gründer- oder Teamprofil, Markt und Wettbewerb sowie "
                    "Vertrieb und Marketing.",
                    "E4",
                ),
            ],
        ),
        "C-FT-FP-TS-002": _entry(
            "Temporal Sequence",
            (
                "Welche Vorbereitungsschritte und Abhängigkeiten nennt das Material "
                "für einen Antrag auf Gründungszuschuss?"
            ),
            [
                _reference("E1", provenance_id="P001"),
                _reference("E2", provenance_id="P002"),
                _reference("E3", provenance_id="P003"),
                _reference("E4", provenance_id="P004"),
            ],
            [
                _unit(
                    "Der Antrag ist bei der Agentur für Arbeit vor Aufnahme der "
                    "Selbstständigkeit zu stellen.",
                    "E1",
                ),
                _unit(
                    "Dafür wird ein Unterlagenpaket vorbereitet, unter anderem mit "
                    "Kurzbeschreibung, Lebenslauf, Kapitalbedarfs- und "
                    "Finanzierungsplan, Umsatz- und Rentabilitätsvorschau sowie "
                    "fachkundiger Stellungnahme.",
                    "E2",
                ),
                _unit(
                    "Die für den Antrag erforderliche fachkundige Stelle beurteilt die "
                    "Tragfähigkeit des Vorhabens.",
                    "E3",
                ),
                _unit(
                    "Ihre Stellungnahme entsteht nach Prüfung der vorgelegten "
                    "Unterlagen und der Angaben der antragstellenden Person und wird "
                    "Teil des Antrags.",
                    "E2",
                    "E4",
                ),
            ],
        ),
        "C-FT-MV-F-001": _entry(
            "Factual",
            (
                "Welche konkreten Prüfungen empfiehlt das Material, bevor eine "
                "Geschäftsidee umgesetzt wird?"
            ),
            [
                _reference(
                    "E1",
                    document_id="D-MV-005",
                    block_id="D-MV-005-B002",
                    exact_quote=_quote(
                        documents,
                        "D-MV-005",
                        "D-MV-005-B002",
                        start="Finden Sie heraus",
                        end="Idee finden werden.",
                    ),
                ),
                _reference(
                    "E2",
                    document_id="D-MV-005",
                    block_id="D-MV-005-B002",
                    exact_quote=_quote(
                        documents,
                        "D-MV-005",
                        "D-MV-005-B002",
                        start="Wer sind Ihre Kunden?",
                        end="von Ihnen erwarten.",
                    ),
                ),
                _reference(
                    "E3",
                    document_id="D-MV-005",
                    block_id="D-MV-005-B002",
                    exact_quote=_quote(
                        documents,
                        "D-MV-005",
                        "D-MV-005-B002",
                        start="Recherchieren Sie",
                        end="bereits Ähnliches an?",
                    ),
                ),
                _reference(
                    "E4",
                    document_id="D-MV-005",
                    block_id="D-MV-005-B002",
                    exact_quote=_quote(
                        documents,
                        "D-MV-005",
                        "D-MV-005-B002",
                        start="Vor allem:",
                        end="Ihrer Wettbewerber?",
                    ),
                ),
            ],
            [
                _unit(
                    "Zuerst ist zu prüfen, ob sich genügend Kundinnen und Kunden für "
                    "die Geschäftsidee finden lassen.",
                    "E1",
                ),
                _unit(
                    "Danach werden die Erwartungen der zukünftigen Kundschaft "
                    "ermittelt.",
                    "E2",
                ),
                _unit(
                    "Außerdem sollen vorhandene Wettbewerber und ähnliche Angebote "
                    "recherchiert werden.",
                    "E3",
                ),
                _unit(
                    "Schließlich wird geklärt, wodurch sich das eigene Angebot von den "
                    "Wettbewerbsangeboten unterscheidet.",
                    "E4",
                ),
            ],
        ),
        "C-FT-MV-F-004": _entry(
            "Factual",
            "Welche Angaben nennt das Material zu Zweck, Durchführung und Nutzen einer Empathy Map?",
            [
                _reference(f"E{index}", provenance_id=f"P{index:03d}")
                for index in range(1, 5)
            ],
            [
                _unit(
                    "Ziel ist ein tiefes Verständnis der Lebenswirklichkeit, Ziele und "
                    "Werte potenzieller Kundinnen und Kunden.",
                    "E1",
                ),
                _unit(
                    "Die Empathy Map ermittelt systematisch Gefühle und Gedanken einer "
                    "Persona und konzentriert sich weniger auf statistische Daten.",
                    "E2",
                ),
                _unit(
                    "Das Team versetzt sich in die Persona, notiert, was sie sieht, "
                    "hört, fühlt, denkt und tut, kann Befragungsergebnisse einbeziehen "
                    "und legt vorher die betrachtete Situation fest.",
                    "E3",
                ),
                _unit(
                    "Die Perspektive hilft, das Angebot am Bedarf auszurichten und "
                    "überzeugende Argumente für die Zielgruppe zu entwickeln.",
                    "E4",
                ),
            ],
        ),
        "C-FT-MV-F-005": _entry(
            "Factual",
            (
                "Wann ist ein MVP laut Material besonders sinnvoll, und was sollte "
                "bei einem dafür nötigen Finanzierungsbedarf im Businessplan stehen?"
            ),
            [
                _reference(f"E{index}", provenance_id=f"P{index:03d}")
                for index in range(1, 5)
            ],
            [
                _unit("Nicht jede Gründung benötigt ein MVP.", "E1"),
                _unit(
                    "Besonders genannt werden technikaffine Startups mit teuren "
                    "Entwicklungs- und Produktionszyklen sowie innovative "
                    "Geschäftsmodelle in schnelllebigen Branchen.",
                    "E2",
                ),
                _unit(
                    "Je innovativer die Geschäftsidee und je dynamischer die Branche, "
                    "desto eher wird die Methode empfohlen.",
                    "E3",
                ),
                _unit(
                    "Wenn bereits die Minimalversion Finanzierung benötigt, sollte der "
                    "Businessplan darlegen, wann und wie ein solider Markttest mittels "
                    "MVP durchgeführt wird.",
                    "E4",
                ),
            ],
        ),
        "C-FT-MV-MHR-003": _entry(
            "Multi-hop Reasoning",
            (
                "Wie führen Methoden der Zielgruppenanalyse von belastbarem "
                "Nachfragewissen zu Marketing- und Vertriebsentscheidungen?"
            ),
            [
                _reference("E1", provenance_id="P002"),
                _reference("E2", provenance_id="P001"),
                _reference("E3", provenance_id="P003"),
                _reference("E4", provenance_id="P004"),
            ],
            [
                _unit(
                    "Befragungen, Beobachtungen und technische Auswertungen des "
                    "Online-Kundenverhaltens liefern Erkenntnisse über die Zielgruppe.",
                    "E1",
                ),
                _unit(
                    "Diese Ergebnisse stützen im Businessplan die Annahme, dass das "
                    "Angebot auf entsprechende Nachfrage trifft.",
                    "E2",
                ),
                _unit(
                    "Auf dieser Basis lässt sich beurteilen, welche Vertriebs- und "
                    "Marketingkanäle am besten geeignet sind.",
                    "E3",
                ),
                _unit(
                    "Ohne Zielgruppenanalyse besteht dagegen das Risiko, dass "
                    "Vertriebsbemühungen ins Leere laufen.",
                    "E4",
                ),
            ],
        ),
        "C-FT-MV-S-001": _entry(
            "Summarization",
            (
                "Welche Vorbereitungsschritte führen von der Prüfung der "
                "Geschäftsidee bis zur Finanzierungsplanung?"
            ),
            [
                _reference(
                    "E1",
                    document_id="D-MV-005",
                    block_id="D-MV-005-B002",
                    exact_quote=_quote(
                        documents,
                        "D-MV-005",
                        "D-MV-005-B002",
                        start="Finden Sie heraus",
                        end="Was unterscheidet Ihr Angebot von dem Ihrer Wettbewerber?",
                    ),
                ),
                _reference(
                    "E2",
                    document_id="D-MV-005",
                    block_id="D-MV-005-B002",
                    exact_quote=_quote(
                        documents,
                        "D-MV-005",
                        "D-MV-005-B002",
                        start="In einer ersten Beratung",
                        end="Kenntnisse \naus?",
                    ),
                ),
                _reference(
                    "E3",
                    document_id="D-MV-005",
                    block_id="D-MV-005-B004",
                    exact_quote=_quote(
                        documents,
                        "D-MV-005",
                        "D-MV-005-B004",
                        start="Machen Sie aus Ihrer Idee",
                        end="einen detaillierten Plan.",
                    ),
                ),
                _reference(
                    "E4",
                    document_id="D-MV-005",
                    block_id="D-MV-005-B004",
                    exact_quote=_quote(
                        documents,
                        "D-MV-005",
                        "D-MV-005-B004",
                        start="Startkapital beschaffen",
                        end="zur Verfügung haben.",
                    ),
                ),
            ],
            [
                _unit(
                    "Die Geschäftsidee wird anhand von Nachfrage, "
                    "Kundenerwartungen, Wettbewerb und eigener Abgrenzung geprüft.",
                    "E1",
                ),
                _unit(
                    "Eine frühe Beratung prüft zusätzlich Erfolgsaussichten, "
                    "Markteinschätzung, finanzielle Überlegungen, persönliches Risiko "
                    "und Qualifikation.",
                    "E2",
                ),
                _unit(
                    "Anschließend wird die Geschäftsidee in einen detaillierten Plan "
                    "überführt.",
                    "E3",
                ),
                _unit(
                    "Darauf aufbauend wird festgestellt, welches eigene Startkapital "
                    "zur Verfügung steht.",
                    "E4",
                ),
            ],
        ),
        "C-FT-MV-TS-001": _entry(
            "Temporal Sequence",
            (
                "Welche Abfolge beschreibt das Material von der ersten "
                "Nachfrageprüfung bis zum Businessplan für Geldgeber?"
            ),
            [
                _reference("E1", provenance_id="P001"),
                _reference("E2", provenance_id="P002"),
                _reference(
                    "E3",
                    document_id="D-MV-005",
                    block_id="D-MV-005-B004",
                    exact_quote=_quote(
                        documents,
                        "D-MV-005",
                        "D-MV-005-B004",
                        start="Machen Sie aus Ihrer Idee",
                        end="einen detaillierten Plan.",
                    ),
                ),
                _reference("E4", provenance_id="P004"),
            ],
            [
                _unit(
                    "Zuerst wird geprüft, ob sich genügend Kundinnen und Kunden für "
                    "die Geschäftsidee finden lassen.",
                    "E1",
                ),
                _unit(
                    "Danach wird ermittelt, was diese zukünftigen Kundinnen und Kunden "
                    "konkret erwarten.",
                    "E2",
                ),
                _unit(
                    "Anschließend wird die Geschäftsidee in einen detaillierten Plan "
                    "überführt.",
                    "E3",
                ),
                _unit(
                    "Der daraus entstehende Businessplan wird schließlich den "
                    "Geldgebern vorgelegt.",
                    "E4",
                ),
            ],
        ),
    }


def write_qra_curation_revision(
    *,
    repo_root: Path,
) -> dict[str, Any]:
    """Merge reviewed semantic corrections into a complete v0.3 curation file."""

    repo_root = repo_root.resolve()
    configuration_manifest = _read_json_object(
        repo_root / REVISED_CONFIGURATION_MANIFEST_PATH
    )
    document_manifest_path = repo_root / configuration_manifest["document_manifest"][
        "path"
    ]
    document_manifest = _read_json_object(document_manifest_path)
    documents, _ = _load_documents(
        repo_root=repo_root,
        manifest=document_manifest,
    )

    base_path = repo_root / BASE_CURATION_PATH
    base = _read_json_object(base_path)
    revised = deepcopy(base)
    revised.update(
        {
            "curation_id": "formal_qra_semantic_curation_v0.3",
            "version": "0.3",
            "status": "development_candidate",
            "supersedes": "formal_qra_semantic_curation_v0.2",
            "scope": (
                "All high-risk question types plus final-audit repairs for the "
                "formal QRA review candidates"
            ),
            "methodology_note": (
                "Version 0.3 preserves all unaffected v0.2 recipes, replaces leaked "
                "or weak targets, expands truncated exact citations, frames dated "
                "figures by document year, and binds the frozen Configuration v1.2 "
                "manifest. It remains a pending human-review artifact and is not a "
                "QRA freeze."
            ),
        }
    )
    revised["entries"].update(_curation_replacements(documents))

    # These numerical questions remain unchanged except for explicit source-time
    # framing, preventing historical figures from being read as current values.
    revised["entries"]["C-FT-FP-NC-002"]["question"] = (
        "Wie unterscheiden sich die im Deutschen Startup Monitor 2021 "
        "dokumentierten Finanzierungsquellen nach ihrer prozentualen Nutzung?"
    )
    revised["entries"]["C-FT-MV-NC-001"]["question"] = (
        "Wie unterscheiden sich in der Abbildung für 2020 die Teilnahme an "
        "sozialen Netzwerken und die Informationssuche insgesamt sowie nach "
        "Altersgruppen?"
    )
    revised["entries"]["C-FT-MV-NC-002"]["question"] = (
        "Wie unterscheiden sich in der Abbildung für 2020 Durchschnitt und "
        "technikaffine Gruppe bei den vier Cross-Channel-Services?"
    )

    output_path = repo_root / REVISED_CURATION_PATH
    state = _write_or_verify(output_path, revised)
    return {
        "curation_path": output_path.relative_to(repo_root).as_posix(),
        "entry_count": len(revised["entries"]),
        "write_state": state,
    }


def prepare_formal_qra_review_revision(
    *,
    repo_root: Path,
    revised_at: str,
) -> dict[str, Any]:
    """Prepare all upstream artifacts required to generate QRA v0.3."""

    configuration_report = freeze_configuration_review_revisions(
        repo_root=repo_root,
        revised_at=revised_at,
    )
    curation_report = write_qra_curation_revision(repo_root=repo_root)
    return {
        "configuration": configuration_report,
        "curation": curation_report,
    }
