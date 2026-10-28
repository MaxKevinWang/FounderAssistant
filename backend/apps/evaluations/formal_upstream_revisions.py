"""Issue immutable v1.1 repairs for audited formal-evaluation dependencies.

The v1.0 Document and Configuration files are historical records and are never
edited here. This module creates a corrected Document D, propagates its content
hash to every dependent Configuration C, repairs one additional Configuration
focus mismatch, and freezes complete replacement manifests.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
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


DOCUMENT_ROOT = "docs/evaluation/formal_dataset/documents"
BASE_DOCUMENT_MANIFEST_PATH = (
    f"{DOCUMENT_ROOT}/formal_test_document_manifest_v1.0.json"
)
REVISED_DOCUMENT_MANIFEST_PATH = (
    f"{DOCUMENT_ROOT}/formal_test_document_manifest_v1.1.json"
)
BASE_CONFIGURATION_ROOT = (
    "docs/evaluation/formal_dataset/configurations/formal_test"
)
BASE_CONFIGURATION_MANIFEST_PATH = (
    f"{BASE_CONFIGURATION_ROOT}/formal_test_configuration_manifest_v1.0.json"
)
REVISED_CONFIGURATION_MANIFEST_PATH = (
    f"{BASE_CONFIGURATION_ROOT}/formal_test_configuration_manifest_v1.1.json"
)
REVISION_PROMPT_PATH = (
    "docs/evaluation/formal_dataset/prompts/upstream_revision_v1.1.md"
)
SEMANTIC_AUDIT_PATH = "docs/evaluation/formal_qra_semantic_audit_v0.1_zh.md"

REVISED_DOCUMENT_ID = "D-LFS-005"
SEMANTICALLY_REVISED_CONFIGURATION_IDS = {
    "C-FT-LFS-F-001",
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

CORRECTED_TABLE_TEXT = """[Reviewed visual content from this PDF page]

[Reviewed figure description inserted from multimodal PDF review]
Figure title: Vergleich der Anforderungen an die Rechtsformen
Figure type: table

Description:
Die Tabelle vergleicht zehn Rechtsformen in dieser Spaltenreihenfolge: Einzelunternehmen, GbR, PartG, OHG, KG, GmbH & Co. KG, GmbH, UG (haftungsbeschränkt), AG und eG. Die Bewertung erfolgt anhand von Kriterien wie Mindestkapital nötig, Haftungsbeschränkung, Wenige Formalitäten und Eintrag ins Handelsregister.

Reviewed table values:
- Mindestkapital nötig: Ja bei GmbH & Co. KG, GmbH, UG (haftungsbeschränkt) und AG; nein bei Einzelunternehmen, GbR, PartG, OHG, KG und eG.
- Haftungsbeschränkung: Ja bei GmbH & Co. KG, GmbH, UG (haftungsbeschränkt), AG und eG; möglich bei PartG; teilweise bei KG; nein bei Einzelunternehmen, GbR und OHG.
- Wenige Formalitäten: Ja bei Einzelunternehmen, GbR und PartG sowie bei GmbH und UG (haftungsbeschränkt) jeweils mit Musterprotokoll; nein bei OHG, KG, GmbH & Co. KG, AG und eG.
- Registereintrag: Einzelunternehmen ja*, GbR nein, PartG ins Partnerschaftsregister, OHG ja, KG ja, GmbH & Co. KG ja, GmbH ja, UG (haftungsbeschränkt) ja, AG ja und eG ins Genossenschaftsregister. *Für Freiberufler oder Kleingewerbetreibende gilt beim Einzelunternehmen: nein."""


def _read_json_object(path: Path) -> dict[str, Any]:
    """Read one UTF-8 JSON object with an artifact-oriented error."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(f"Cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ArtifactValidationError(f"Expected a JSON object: {path}")
    return payload


def _canonical_sha256(value: Any) -> str:
    """Match the canonical JSON hashing used for Document block lists."""

    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_or_verify(path: Path, payload: dict[str, Any]) -> str:
    """Write a new immutable revision, or verify an identical prior run."""

    if path.exists():
        if _read_json_object(path) != payload:
            raise ArtifactValidationError(
                f"Frozen revision already exists with different content: {path}"
            )
        return "unchanged"
    write_json_artifact(path, payload)
    return "created"


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


def _assert_frozen_manifest(
    manifest: dict[str, Any],
    *,
    version: str,
    count_key: str,
    expected_count: int,
    path: Path,
) -> None:
    """Reject an unexpected predecessor lifecycle before deriving a revision."""

    if (
        manifest.get("status") != "frozen"
        or manifest.get("version") != version
        or manifest.get(count_key) != expected_count
    ):
        raise ArtifactValidationError(f"Unexpected frozen manifest: {path}")


def _load_documents(
    *,
    repo_root: Path,
    manifest: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Load and fully verify every Document declared by one frozen manifest."""

    documents: dict[str, dict[str, Any]] = {}
    declarations: dict[str, dict[str, Any]] = {}
    for declaration in manifest["documents"]:
        document_path = repo_root / declaration["path"]
        if sha256_file(document_path) != declaration["file_sha256"]:
            raise ArtifactValidationError(
                f"Frozen Document hash mismatch: {document_path}"
            )
        freeze_path = repo_root / declaration["freeze_record_path"]
        if sha256_file(freeze_path) != declaration["freeze_record_sha256"]:
            raise ArtifactValidationError(
                f"Frozen Document freeze-record hash mismatch: {freeze_path}"
            )
        document = _read_json_object(document_path)
        validate_document_record(
            document=document,
            source_path=resolve_corpus_path(repo_root, document["source"]["path"]),
        )
        if document["content_sha256"] != declaration["content_sha256"]:
            raise ArtifactValidationError(
                f"Document manifest content hash mismatch: {document_path}"
            )
        documents[document["document_id"]] = document
        declarations[document["document_id"]] = declaration
    if len(documents) != 24:
        raise ArtifactValidationError("Formal Document set must contain 24 unique IDs")
    return documents, declarations


def _revise_legal_form_document(
    *,
    base_document: dict[str, Any],
    frozen_at: str,
) -> dict[str, Any]:
    """Correct only the audited page-five table description."""

    revised = deepcopy(base_document)
    revised["version"] = "1.1"
    revised["status"] = "frozen"
    revised["supersedes"] = f"{REVISED_DOCUMENT_ID} v1.0"
    revised["frozen_at"] = frozen_at

    changed_blocks = 0
    for block in revised["blocks"]:
        if block["block_id"] == "D-LFS-005-B007":
            block["text"] = CORRECTED_TABLE_TEXT
            changed_blocks += 1
    if changed_blocks != 1:
        raise ArtifactValidationError(
            "Expected exactly one D-LFS-005-B007 block to revise"
        )
    revised["content_sha256"] = _canonical_sha256(revised["blocks"])
    return revised


def _document_freeze_record(
    *,
    repo_root: Path,
    revised_path: Path,
    revised: dict[str, Any],
    predecessor_path: Path,
    frozen_at: str,
) -> dict[str, Any]:
    """Build an auditable freeze record for the corrected Document."""

    return {
        "artifact_type": "RAGEval Document D",
        "document_id": revised["document_id"],
        "version": "1.1",
        "status": "frozen",
        "frozen_at": frozen_at,
        "file": revised_path.relative_to(repo_root).as_posix(),
        "file_sha256": sha256_file(revised_path),
        "content_sha256": revised["content_sha256"],
        "supersedes": f"{REVISED_DOCUMENT_ID} v1.0",
        "predecessor_file": predecessor_path.relative_to(repo_root).as_posix(),
        "predecessor_file_sha256": sha256_file(predecessor_path),
        "source_file": revised["source"]["path"],
        "source_sha256": revised["source"]["sha256"],
        "block_count": revised["block_count"],
        "changed_block_ids": ["D-LFS-005-B007"],
        "revision_protocol": {
            "path": REVISION_PROMPT_PATH,
            "sha256": sha256_file(repo_root / REVISION_PROMPT_PATH),
        },
        "semantic_audit": {
            "path": SEMANTIC_AUDIT_PATH,
            "finding_sections": ["6.1", "6.2"],
            "note": (
                "The audit is a maintained explanatory document, not an immutable "
                "dependency of this freeze record."
            ),
        },
        "approval": {
            "basis": "explicit_project_owner_freeze_instruction",
            "approved_at": frozen_at,
            "scope": (
                "direct visual correction of the page-five legal-form table, "
                "including column order and all cell values"
            ),
        },
        "change_policy": (
            "Do not edit this frozen Document or freeze record in place; issue "
            "a new version for every later correction."
        ),
    }


def _revise_lfs_factual_configuration(
    configuration: dict[str, Any],
) -> None:
    """Replace ambiguous row arrays with legal-form-aligned table facts."""

    configuration["question_generation_constraints"]["focus"] = (
        "Vergleich von zehn Rechtsformen nach Kapital, Haftung, Formalitäten "
        "und Registereintrag"
    )
    configuration["values"] = {
        "rulesAndExceptions": [
            {
                "rule": (
                    "Die Tabelle vergleicht zehn Rechtsformen in dieser "
                    "Spaltenreihenfolge: Einzelunternehmen, GbR, PartG, OHG, KG, "
                    "GmbH & Co. KG, GmbH, UG (haftungsbeschränkt), AG und eG."
                )
            },
            {
                "rule": (
                    "Mindestkapital nötig: Ja bei GmbH & Co. KG, GmbH, UG "
                    "(haftungsbeschränkt) und AG; nein bei Einzelunternehmen, GbR, "
                    "PartG, OHG, KG und eG."
                )
            },
            {
                "rule": (
                    "Haftungsbeschränkung: Ja bei GmbH & Co. KG, GmbH, UG "
                    "(haftungsbeschränkt), AG und eG; möglich bei PartG; teilweise "
                    "bei KG; nein bei Einzelunternehmen, GbR und OHG."
                )
            },
            {
                "rule": (
                    "Wenige Formalitäten: Ja bei Einzelunternehmen, GbR und PartG "
                    "sowie bei GmbH und UG (haftungsbeschränkt) jeweils mit "
                    "Musterprotokoll; nein bei OHG, KG, GmbH & Co. KG, AG und eG."
                )
            },
            {
                "rule": (
                    "Registereintrag: Einzelunternehmen ja*, GbR nein, PartG ins "
                    "Partnerschaftsregister, OHG ja, KG ja, GmbH & Co. KG ja, GmbH "
                    "ja, UG (haftungsbeschränkt) ja, AG ja und eG ins "
                    "Genossenschaftsregister. *Für Freiberufler oder "
                    "Kleingewerbetreibende gilt beim Einzelunternehmen: nein."
                )
            },
        ]
    }
    configuration["provenance"] = [
        {
            "provenance_id": f"P{index:03d}",
            "json_pointer": f"/values/rulesAndExceptions/{index - 1}/rule",
            "document_id": REVISED_DOCUMENT_ID,
            "block_id": "D-LFS-005-B007",
            "heading_path": "Page 5 > Reviewed visual content",
            "support_type": "direct",
            "derivation": "",
            "exact_quote": item["rule"],
        }
        for index, item in enumerate(
            configuration["values"]["rulesAndExceptions"],
            start=1,
        )
    ]


def _revise_market_summary_configuration(
    configuration: dict[str, Any],
) -> None:
    """Align the summary focus and values with the three evidenced aspects."""

    configuration["question_generation_constraints"]["focus"] = (
        "Kundennutzen, Ertragsmodell und Kostenstruktur eines Geschäftsmodells"
    )
    assumptions = configuration["values"]["businessModelHypotheses"]
    assumptions[0]["assumption"] = (
        "Kunden kaufen allerdings am Ende den Nutzen deines Produktes."
    )
    provenance = configuration["provenance"][0]
    provenance["exact_quote"] = assumptions[0]["assumption"]


def _revise_configuration(
    *,
    base: dict[str, Any],
    revised_document_hash: str,
    prompt_hash: str,
    frozen_at: str,
) -> dict[str, Any]:
    """Create one approved v1.1 Configuration revision."""

    revised = deepcopy(base)
    revised["version"] = "1.1"
    revised["status"] = "frozen"
    revised["supersedes"] = f"{base['configuration_id']} v1.0"
    for source in revised["source_documents"]:
        if source["document_id"] == REVISED_DOCUMENT_ID:
            source["content_sha256"] = revised_document_hash

    if revised["configuration_id"] == "C-FT-LFS-F-001":
        _revise_lfs_factual_configuration(revised)
    if revised["configuration_id"] == "C-FT-MV-S-003":
        _revise_market_summary_configuration(revised)

    revised["generation_record"] = {
        "method": (
            "Deterministic upstream correction and transitive dependency-hash "
            "propagation"
        ),
        "generated_at": frozen_at,
        "prompt_path": REVISION_PROMPT_PATH,
        "prompt_sha256": prompt_hash,
        "mechanical_validation": (
            "passed: predecessor hashes, JSON Schema v1.1, domain Schema keys, "
            "frozen Document hashes, allowed-block scope, exact quotes, provenance "
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
    frozen_at: str,
) -> dict[str, Any]:
    """Build one v1.1 Configuration freeze record."""

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
        "version": "1.1",
        "status": "frozen",
        "frozen_at": frozen_at,
        "file": revised_path.relative_to(repo_root).as_posix(),
        "file_sha256": sha256_file(revised_path),
        "supersedes": f"{revised['configuration_id']} v1.0",
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
            "path": REVISION_PROMPT_PATH,
            "sha256": sha256_file(repo_root / REVISION_PROMPT_PATH),
        },
        "approval": {
            "basis": "explicit_project_owner_freeze_instruction",
            "approved_at": frozen_at,
            "scope": (
                "upstream semantic correction and/or transitive frozen-Document "
                "content-hash update"
            ),
        },
        "change_policy": (
            "Do not edit this frozen Configuration or freeze record in place; "
            "create and review a new version for every later change."
        ),
    }


def _assert_expected_balance(configurations: list[dict[str, Any]]) -> None:
    """Ensure revisions did not alter the approved formal-test allocation."""

    if len(configurations) != 60:
        raise ArtifactValidationError("Formal Configuration set must contain 60 items")
    if len({item["configuration_id"] for item in configurations}) != 60:
        raise ArtifactValidationError("Formal Configuration IDs are not unique")
    if Counter(item["answerability"] for item in configurations) != {
        "answerable": 54,
        "unanswerable": 6,
    }:
        raise ArtifactValidationError("Formal answerability allocation has changed")
    for domain in DOMAIN_SCHEMA_PATHS:
        domain_items = [item for item in configurations if item["domain"] == domain]
        if len(domain_items) != 20 or Counter(
            item["target_question_type"] for item in domain_items
        ) != EXPECTED_TYPE_COUNTS:
            raise ArtifactValidationError(
                f"Formal question-type allocation changed for {domain}"
            )


def freeze_formal_upstream_revisions(
    *,
    repo_root: Path,
    frozen_at: str,
) -> dict[str, Any]:
    """Create and freeze the complete v1.1 dependency-consistent release."""

    repo_root = repo_root.resolve()
    prompt_path = repo_root / REVISION_PROMPT_PATH
    audit_path = repo_root / SEMANTIC_AUDIT_PATH
    if not prompt_path.exists() or not audit_path.exists():
        raise ArtifactValidationError("Revision protocol or semantic audit is missing")
    prompt_hash = sha256_file(prompt_path)
    write_states: list[str] = []

    base_document_manifest_path = repo_root / BASE_DOCUMENT_MANIFEST_PATH
    base_document_manifest = _read_json_object(base_document_manifest_path)
    _assert_frozen_manifest(
        base_document_manifest,
        version="1.0",
        count_key="document_count",
        expected_count=24,
        path=base_document_manifest_path,
    )
    base_documents, base_document_declarations = _load_documents(
        repo_root=repo_root,
        manifest=base_document_manifest,
    )

    base_document_path = repo_root / base_document_declarations[REVISED_DOCUMENT_ID][
        "path"
    ]
    revised_document = _revise_legal_form_document(
        base_document=base_documents[REVISED_DOCUMENT_ID],
        frozen_at=frozen_at,
    )
    revised_document_path = (
        repo_root / DOCUMENT_ROOT / "D-LFS-005_rechtsformen_v1.1.json"
    )
    validate_document_record(
        document=revised_document,
        source_path=resolve_corpus_path(repo_root, revised_document["source"]["path"]),
    )
    write_states.append(_write_or_verify(revised_document_path, revised_document))
    revised_document_freeze_path = revised_document_path.with_name(
        f"{revised_document_path.stem}_freeze_record.json"
    )
    document_freeze_record = _document_freeze_record(
        repo_root=repo_root,
        revised_path=revised_document_path,
        revised=revised_document,
        predecessor_path=base_document_path,
        frozen_at=frozen_at,
    )
    write_states.append(
        _write_or_verify(revised_document_freeze_path, document_freeze_record)
    )

    revised_document_declarations = []
    for declaration in base_document_manifest["documents"]:
        carried = deepcopy(declaration)
        if declaration["document_id"] == REVISED_DOCUMENT_ID:
            carried.update(
                {
                    "resolution": "revised_frozen",
                    "path": revised_document_path.relative_to(repo_root).as_posix(),
                    "file_sha256": sha256_file(revised_document_path),
                    "freeze_record_path": revised_document_freeze_path.relative_to(
                        repo_root
                    ).as_posix(),
                    "freeze_record_sha256": sha256_file(
                        revised_document_freeze_path
                    ),
                    "content_sha256": revised_document["content_sha256"],
                }
            )
        else:
            carried["resolution"] = "carried_forward"
        revised_document_declarations.append(carried)

    revised_document_manifest = {
        "manifest_id": "formal_test_document_manifest_v1.1",
        "version": "1.1",
        "status": "frozen",
        "frozen_at": frozen_at,
        "supersedes": "formal_test_document_manifest_v1.0",
        "predecessor_manifest": {
            "path": BASE_DOCUMENT_MANIFEST_PATH,
            "sha256": sha256_file(base_document_manifest_path),
        },
        "source_selection_candidate": base_document_manifest[
            "source_selection_candidate"
        ],
        "revision_protocol": {
            "path": REVISION_PROMPT_PATH,
            "sha256": prompt_hash,
        },
        "document_count": 24,
        "revised_document_count": 1,
        "carried_forward_count": 23,
        "documents": revised_document_declarations,
    }
    revised_document_manifest_path = repo_root / REVISED_DOCUMENT_MANIFEST_PATH
    write_states.append(
        _write_or_verify(revised_document_manifest_path, revised_document_manifest)
    )
    revised_document_manifest_freeze_path = revised_document_manifest_path.with_name(
        "formal_test_document_manifest_v1.1_freeze_record.json"
    )
    document_manifest_freeze = {
        "artifact_type": "Formal test Document D manifest",
        "manifest_id": revised_document_manifest["manifest_id"],
        "version": "1.1",
        "status": "frozen",
        "frozen_at": frozen_at,
        "file": revised_document_manifest_path.relative_to(repo_root).as_posix(),
        "file_sha256": sha256_file(revised_document_manifest_path),
        "predecessor_manifest": BASE_DOCUMENT_MANIFEST_PATH,
        "predecessor_manifest_sha256": sha256_file(base_document_manifest_path),
        "document_count": 24,
        "revised_document_ids": [REVISED_DOCUMENT_ID],
        "approval_basis": "explicit_project_owner_freeze_instruction",
    }
    write_states.append(
        _write_or_verify(
            revised_document_manifest_freeze_path,
            document_manifest_freeze,
        )
    )

    revised_documents, revised_document_declarations_by_id = _load_documents(
        repo_root=repo_root,
        manifest=revised_document_manifest,
    )

    base_configuration_manifest_path = repo_root / BASE_CONFIGURATION_MANIFEST_PATH
    base_configuration_manifest = _read_json_object(
        base_configuration_manifest_path
    )
    _assert_frozen_manifest(
        base_configuration_manifest,
        version="1.0",
        count_key="configuration_count",
        expected_count=60,
        path=base_configuration_manifest_path,
    )
    if (
        sha256_file(base_document_manifest_path)
        != base_configuration_manifest["document_manifest"]["sha256"]
    ):
        raise ArtifactValidationError(
            "Base Configuration manifest does not bind the expected Document manifest"
        )

    record_schema_path = (
        repo_root / base_configuration_manifest["record_schema"]["path"]
    )
    if (
        sha256_file(record_schema_path)
        != base_configuration_manifest["record_schema"]["sha256"]
    ):
        raise ArtifactValidationError("Configuration record schema hash mismatch")
    record_schema = _read_json_object(record_schema_path)
    domain_schemas = {
        domain: _read_json_object(repo_root / path)
        for domain, path in DOMAIN_SCHEMA_PATHS.items()
    }

    revised_configuration_declarations = []
    resolved_configurations: list[dict[str, Any]] = []
    revised_configuration_ids: list[str] = []
    for declaration in base_configuration_manifest["configurations"]:
        base_path = repo_root / declaration["path"]
        if sha256_file(base_path) != declaration["file_sha256"]:
            raise ArtifactValidationError(
                f"Frozen Configuration hash mismatch: {base_path}"
            )
        base = _read_json_object(base_path)
        depends_on_revised_document = any(
            source["document_id"] == REVISED_DOCUMENT_ID
            for source in base["source_documents"]
        )
        needs_revision = (
            depends_on_revised_document
            or base["configuration_id"]
            in SEMANTICALLY_REVISED_CONFIGURATION_IDS
        )
        if not needs_revision:
            carried = deepcopy(declaration)
            carried["version"] = "1.0"
            carried["resolution"] = "carried_forward"
            revised_configuration_declarations.append(carried)
            resolved_configurations.append(base)
            continue

        revised = _revise_configuration(
            base=base,
            revised_document_hash=revised_document["content_sha256"],
            prompt_hash=prompt_hash,
            frozen_at=frozen_at,
        )
        errors = _schema_errors(revised, record_schema)
        if errors:
            raise ArtifactValidationError(
                f"Invalid revised Configuration {revised['configuration_id']}: "
                f"{'; '.join(errors[:5])}"
            )
        validate_configuration_record(
            configuration=revised,
            schema=domain_schemas[revised["domain"]],
            documents=revised_documents,
        )
        revised_path = base_path.with_name(
            f"{revised['configuration_id']}_v1.1.json"
        )
        write_states.append(_write_or_verify(revised_path, revised))
        revised_freeze_path = revised_path.with_name(
            f"{revised_path.stem}_freeze_record.json"
        )
        freeze_record = _configuration_freeze_record(
            repo_root=repo_root,
            revised_path=revised_path,
            revised=revised,
            predecessor_path=base_path,
            document_declarations=revised_document_declarations_by_id,
            record_schema_path=record_schema_path,
            frozen_at=frozen_at,
        )
        write_states.append(_write_or_verify(revised_freeze_path, freeze_record))
        revised_declaration = {
            "configuration_id": revised["configuration_id"],
            "version": "1.1",
            "resolution": "revised_frozen",
            "domain": revised["domain"],
            "target_question_type": revised["target_question_type"],
            "answerability": revised["answerability"],
            "path": revised_path.relative_to(repo_root).as_posix(),
            "file_sha256": sha256_file(revised_path),
            "freeze_record_path": revised_freeze_path.relative_to(
                repo_root
            ).as_posix(),
            "freeze_record_sha256": sha256_file(revised_freeze_path),
            "source_document_ids": [
                source["document_id"] for source in revised["source_documents"]
            ],
            "provenance_count": len(revised["provenance"]),
        }
        revised_configuration_declarations.append(revised_declaration)
        resolved_configurations.append(revised)
        revised_configuration_ids.append(revised["configuration_id"])

    if set(revised_configuration_ids) != {
        "C-FT-LFS-F-001",
        "C-FT-LFS-II-001",
        "C-FT-LFS-II-003",
        "C-FT-LFS-NC-001",
        "C-FT-LFS-S-001",
        "C-FT-LFS-U-001",
        "C-FT-LFS-U-002",
        "C-FT-MV-S-003",
    }:
        raise ArtifactValidationError(
            "Unexpected transitive Configuration revision set: "
            f"{sorted(revised_configuration_ids)}"
        )
    _assert_expected_balance(resolved_configurations)

    revised_configuration_manifest = {
        "manifest_id": "formal_test_configuration_manifest_v1.1",
        "version": "1.1",
        "status": "frozen",
        "frozen_at": frozen_at,
        "supersedes": "formal_test_configuration_manifest_v1.0",
        "predecessor_manifest": {
            "path": BASE_CONFIGURATION_MANIFEST_PATH,
            "sha256": sha256_file(base_configuration_manifest_path),
        },
        "candidate_manifest": base_configuration_manifest["candidate_manifest"],
        "document_manifest": {
            "path": REVISED_DOCUMENT_MANIFEST_PATH,
            "sha256": sha256_file(revised_document_manifest_path),
        },
        "record_schema": base_configuration_manifest["record_schema"],
        "revision_protocol": {
            "path": REVISION_PROMPT_PATH,
            "sha256": prompt_hash,
        },
        "configuration_count": 60,
        "revised_configuration_count": len(revised_configuration_ids),
        "carried_forward_count": 60 - len(revised_configuration_ids),
        "answerable_count": 54,
        "unanswerable_count": 6,
        "configurations": revised_configuration_declarations,
    }
    revised_configuration_manifest_path = (
        repo_root / REVISED_CONFIGURATION_MANIFEST_PATH
    )
    write_states.append(
        _write_or_verify(
            revised_configuration_manifest_path,
            revised_configuration_manifest,
        )
    )
    revised_configuration_manifest_freeze_path = (
        revised_configuration_manifest_path.with_name(
            "formal_test_configuration_manifest_v1.1_freeze_record.json"
        )
    )
    configuration_manifest_freeze = {
        "artifact_type": "Formal test Configuration C manifest",
        "manifest_id": revised_configuration_manifest["manifest_id"],
        "version": "1.1",
        "status": "frozen",
        "frozen_at": frozen_at,
        "file": revised_configuration_manifest_path.relative_to(
            repo_root
        ).as_posix(),
        "file_sha256": sha256_file(revised_configuration_manifest_path),
        "predecessor_manifest": BASE_CONFIGURATION_MANIFEST_PATH,
        "predecessor_manifest_sha256": sha256_file(
            base_configuration_manifest_path
        ),
        "configuration_count": 60,
        "revised_configuration_ids": sorted(revised_configuration_ids),
        "approval_basis": "explicit_project_owner_freeze_instruction",
    }
    write_states.append(
        _write_or_verify(
            revised_configuration_manifest_freeze_path,
            configuration_manifest_freeze,
        )
    )

    return {
        "document_manifest_path": revised_document_manifest_path.relative_to(
            repo_root
        ).as_posix(),
        "configuration_manifest_path": (
            revised_configuration_manifest_path.relative_to(repo_root).as_posix()
        ),
        "revised_document_ids": [REVISED_DOCUMENT_ID],
        "revised_configuration_ids": sorted(revised_configuration_ids),
        "created_count": write_states.count("created"),
        "unchanged_count": write_states.count("unchanged"),
    }
