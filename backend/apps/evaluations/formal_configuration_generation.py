"""Generate source-grounded formal Configuration C review candidates."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from apps.evaluations.corpus_paths import resolve_corpus_path
import re
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

from apps.evaluations.rageval_artifacts import (
    ArtifactValidationError,
    sha256_file,
    validate_configuration_record,
    validate_document_record,
    write_json_artifact,
)


CONFIGURATION_SCHEMA_PATH = (
    "docs/evaluation/formal_dataset/configurations/"
    "configuration_record_schema_v1.1.json"
)
CONFIGURATION_PROMPT_PATH = (
    "docs/evaluation/formal_dataset/prompts/"
    "configuration_generation_prompt_v1.1.md"
)
FORMAL_CONFIGURATION_OUTPUT_DIR = (
    "docs/evaluation/formal_dataset/configurations/formal_candidates"
)

DOMAIN_SCHEMA_PATHS = {
    "legal_form_selection": (
        "docs/evaluation/schemas/legal_form_selection/"
        "schema_S_legal_form_selection_v1.0.json"
    ),
    "funding_preparation": (
        "docs/evaluation/schemas/funding_preparation/"
        "schema_S_funding_preparation_v1.0.json"
    ),
    "market_validation": (
        "docs/evaluation/schemas/market_validation/"
        "schema_S_market_validation_v1.0.json"
    ),
}

VALUE_SLOTS = {
    "legal_rule": ("rulesAndExceptions", "rule"),
    "legal_decision": ("decisionCriteria", "comparisonResult"),
    "legal_process": ("formationProcess", "step"),
    "legal_obligation": ("postFormationObligations", "obligation"),
    "funding_rule": ("rulesAndExceptions", "rule"),
    "funding_assessment": ("financingAssessment", "assessmentCriterion"),
    "funding_process": ("applicationProcess", "step"),
    "funding_comparison": ("comparisonAndSelection", "comparisonResult"),
    "market_finding": ("customerAndMarketResearch", "finding"),
    "market_hypothesis": ("businessModelHypotheses", "assumption"),
    "market_process": ("validationProcess", "step"),
    "market_trend": ("marketSizeAndTrendEvidence", "growthOrTrend"),
    "market_decision": ("evidenceBasedDecisions", "decision"),
}

REASONING_OPERATIONS = {
    "Factual": "Die konfigurierten Einzelfakten präzise aus den Quellen wiedergeben.",
    "Summarization": (
        "Die konfigurierten Kernaussagen knapp zusammenfassen, ohne neue Aspekte "
        "hinzuzufügen."
    ),
    "Multi-hop Reasoning": (
        "Die konfigurierten Aussagen zu einer nachvollziehbaren Begründungs- oder "
        "Folgenkette verbinden."
    ),
    "Information Integration": (
        "Die konfigurierten Aussagen aus allen deklarierten Dokumenten "
        "zusammenführen und ihre Beziehung erklären."
    ),
    "Numerical Comparison": (
        "Die dokumentierten Zahlen oder Größen vergleichen und nur daraus "
        "ableitbare Unterschiede erläutern."
    ),
    "Temporal Sequence": (
        "Die dokumentierten Schritte in Quellenreihenfolge rekonstruieren und "
        "ihren Zusammenhang erklären."
    ),
    "Unanswerable": (
        "Eine definitive Antwort verweigern, die fehlenden Fallinformationen "
        "benennen und nur die dokumentierten allgemeinen Regeln erläutern."
    ),
}

_EXCLUDED_EVIDENCE_PHRASES = (
    "reviewed visual content from this pdf page",
    "reviewed figure description inserted from multimodal pdf review",
    "logge dich",
    "kostenlosen account",
    "das könnte dich auch interessieren",
    "was dich interessieren könnte",
    "weitere videoratgeber",
    "melde dein unternehmen online an",
    "weitere informationen:",
    "lies mehr dazu",
    "das könnte dich auch interessieren",
)

_EXCLUDED_EVIDENCE_PREFIXES = (
    "gründerzeiten ",
    "figure title:",
    "weitere informationen",
    "impressum",
    "bildnachweis",
    "bestellmöglichkeit",
)

_INCOMPLETE_ENDINGS = re.compile(
    r"(?:\b(?:bzw|ca|d|dr|etc|ggf|nr|u|usw|vgl|z)\.|\bz\.\s*b\.)$",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class EvidenceExcerpt:
    """One exact source span selected for a Configuration value."""

    document_id: str
    block_id: str
    heading_path: str
    quote: str
    normalized_value: str
    block_index: int
    offset: int
    score: int


def _read_json_object(path: Path) -> dict[str, Any]:
    """Read one UTF-8 JSON object."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(f"Cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ArtifactValidationError(f"Expected a JSON object: {path}")
    return value


def _canonical_sha256(value: Any) -> str:
    """Hash JSON-compatible data deterministically."""

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalize_inline(value: str) -> str:
    """Collapse source whitespace while retaining the original quote separately."""

    return re.sub(r"\s+", " ", value).strip()


def _sentence_spans(text: str) -> list[tuple[int, str]]:
    """Split prose at sentence boundaries without breaking common abbreviations."""

    spans: list[tuple[int, str]] = []
    start = 0
    for boundary in re.finditer(r"[.!?](?=\s|$)", text):
        candidate = text[start : boundary.end()]
        normalized = _normalize_inline(candidate)
        if _INCOMPLETE_ENDINGS.search(normalized):
            continue

        leading_whitespace = len(candidate) - len(candidate.lstrip())
        quote = candidate.strip()
        if quote:
            spans.append((start + leading_whitespace, quote))
        start = boundary.end()

    trailing = text[start:]
    leading_whitespace = len(trailing) - len(trailing.lstrip())
    quote = trailing.strip()
    if quote:
        spans.append((start + leading_whitespace, quote))
    return spans


def _candidate_spans(text: str) -> list[tuple[int, str]]:
    """Return exact sentence, line, and bullet spans suitable for evidence."""

    spans = _sentence_spans(text)

    # Plain PDF lines are often OCR continuations rather than semantic units.
    # Only retain line-level candidates when the source marks them as bullets.
    for match in re.finditer(r"(?m)^[ \t]*[-•▪]\s+.{20,}$", text):
        spans.append((match.start(), match.group(0).strip()))

    for match in re.finditer(r"(?:^|\s)[•▪]\s*([^•▪\n]{20,})", text):
        spans.append((match.start(1), match.group(1).strip()))

    unique: dict[str, tuple[int, str]] = {}
    for offset, quote in spans:
        normalized = _normalize_inline(quote)
        if len(normalized) < 12 or len(normalized) > 900:
            continue
        lowered = normalized.casefold()
        if any(phrase in lowered for phrase in _EXCLUDED_EVIDENCE_PHRASES):
            continue
        if lowered.startswith(_EXCLUDED_EVIDENCE_PREFIXES):
            continue
        if re.match(r"^\d+\s+gründerzeiten\b", lowered):
            continue
        if re.match(r"^[A-Za-zÄÖÜäöüß]\.\s", normalized):
            continue
        if normalized[0].islower():
            continue
        if _INCOMPLETE_ENDINGS.search(normalized):
            continue
        if not re.search(r"[A-Za-zÄÖÜäöüß]", normalized):
            continue
        unique.setdefault(normalized, (offset, quote))
    return list(unique.values())


def _score_excerpt(
    *,
    normalized: str,
    keywords: list[str],
    preferred: bool,
    question_type: str,
) -> int:
    """Score an exact span against the human-authored allocation terms."""

    lowered = normalized.casefold()
    hits = sum(keyword.casefold() in lowered for keyword in keywords)
    score = hits * 6
    if preferred:
        score += 8
    if question_type == "Numerical Comparison" and re.search(
        r"\d[\d.,]*\s*(?:%|prozent|euro|eur|monate?|jahre?)?",
        lowered,
    ):
        score += 5
    if question_type == "Temporal Sequence" and re.search(
        r"\b(?:zuerst|danach|anschließend|schritt|phase|innerhalb|vor|nach)\b",
        lowered,
    ):
        score += 4
    if 55 <= len(normalized) <= 420:
        score += 2
    return score


def _document_excerpt_candidates(
    *,
    document: dict[str, Any],
    evidence_plan: dict[str, Any],
    question_type: str,
    permitted_block_ids: set[str],
) -> list[EvidenceExcerpt]:
    """Rank exact excerpts from one frozen Document D."""

    keywords = [str(value) for value in evidence_plan["keywords"]]
    preferred_blocks = set(evidence_plan.get("preferred_block_ids", []))
    preferred_only = bool(evidence_plan.get("preferred_only", False))
    candidates: list[EvidenceExcerpt] = []
    for block_index, block in enumerate(document["blocks"]):
        if block.get("block_type") == "heading":
            continue
        if block["block_id"] not in permitted_block_ids:
            continue
        if preferred_only and block["block_id"] not in preferred_blocks:
            continue
        text = str(block["text"])
        for offset, quote in _candidate_spans(text):
            normalized = _normalize_inline(quote)
            lowered = normalized.casefold()
            if not any(keyword.casefold() in lowered for keyword in keywords):
                continue
            score = _score_excerpt(
                normalized=normalized,
                keywords=keywords,
                preferred=block["block_id"] in preferred_blocks,
                question_type=question_type,
            )
            if score <= 0:
                continue
            candidates.append(
                EvidenceExcerpt(
                    document_id=document["document_id"],
                    block_id=block["block_id"],
                    heading_path=str(block.get("heading_path") or ""),
                    quote=quote,
                    normalized_value=normalized,
                    block_index=block_index,
                    offset=offset,
                    score=score,
                )
            )
    return sorted(
        candidates,
        key=lambda item: (-item.score, item.block_index, item.offset),
    )


def _overlaps_selected(
    candidate: EvidenceExcerpt,
    selected: list[EvidenceExcerpt],
) -> bool:
    """Avoid nested copies of the same source sentence in one Configuration."""

    value = candidate.normalized_value.casefold()
    for existing in selected:
        other = existing.normalized_value.casefold()
        if (
            candidate.document_id == existing.document_id
            and candidate.block_id == existing.block_id
            and (value in other or other in value)
        ):
            return True
    return False


def _select_evidence(
    *,
    item: dict[str, Any],
    documents: dict[str, dict[str, Any]],
    used_values: dict[str, set[str]],
    permitted_blocks: dict[str, set[str]],
) -> list[EvidenceExcerpt]:
    """Select the planned number of high-scoring, minimally repeated excerpts."""

    selected: list[EvidenceExcerpt] = []
    for source_plan in item["evidence_plan"]:
        document_id = source_plan["document_id"]
        document = documents[document_id]
        disallowed_preferred = set(source_plan.get("preferred_block_ids", [])) - (
            permitted_blocks[document_id]
        )
        if disallowed_preferred:
            raise ArtifactValidationError(
                f"{item['configuration_id']} prefers prohibited blocks from "
                f"{document_id}: {sorted(disallowed_preferred)}"
            )
        candidates = _document_excerpt_candidates(
            document=document,
            evidence_plan=source_plan,
            question_type=item["target_question_type"],
            permitted_block_ids=permitted_blocks[document_id],
        )
        required = int(source_plan["excerpt_count"])
        chosen: list[EvidenceExcerpt] = []
        prior_values = used_values.setdefault(document_id, set())
        ranked_candidates = sorted(
            candidates,
            key=lambda candidate: (
                -(candidate.score - (3 if candidate.normalized_value in prior_values else 0)),
                candidate.block_index,
                candidate.offset,
            ),
        )
        for candidate in ranked_candidates:
            if len(chosen) >= required:
                break
            if _overlaps_selected(candidate, selected + chosen):
                continue
            chosen.append(candidate)

        if len(chosen) != required:
            raise ArtifactValidationError(
                f"{item['configuration_id']} could select only {len(chosen)}/"
                f"{required} excerpts from {document_id}; "
                f"preferred={source_plan.get('preferred_block_ids', [])}, "
                f"keywords={source_plan.get('keywords', [])}"
            )
        selected.extend(chosen)
        used_values.setdefault(document_id, set()).update(
            excerpt.normalized_value for excerpt in chosen
        )

    return sorted(
        selected,
        key=lambda excerpt: (
            next(
                index
                for index, plan in enumerate(item["evidence_plan"])
                if plan["document_id"] == excerpt.document_id
            ),
            excerpt.block_index,
            excerpt.offset,
        ),
    )


def _build_values_and_provenance(
    *,
    item: dict[str, Any],
    excerpts: list[EvidenceExcerpt],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Map exact evidence into a valid partial domain-Schema instantiation."""

    try:
        top_level_key, field = VALUE_SLOTS[item["value_slot"]]
    except KeyError as exc:
        raise ArtifactValidationError(
            f"Unknown value_slot for {item['configuration_id']}: "
            f"{item['value_slot']}"
        ) from exc

    values = {
        top_level_key: [
            {field: excerpt.normalized_value}
            for excerpt in excerpts
        ]
    }
    provenance = []
    for index, excerpt in enumerate(excerpts):
        provenance.append(
            {
                "provenance_id": f"P{index + 1:03d}",
                "json_pointer": f"/values/{top_level_key}/{index}/{field}",
                "document_id": excerpt.document_id,
                "block_id": excerpt.block_id,
                "heading_path": excerpt.heading_path,
                "support_type": "direct",
                "derivation": "",
                "exact_quote": excerpt.quote,
            }
        )
    return values, provenance


def _allowed_document_declarations(
    *,
    item: dict[str, Any],
    excerpts: list[EvidenceExcerpt],
    documents: dict[str, dict[str, Any]],
    permitted_blocks: dict[str, set[str]],
) -> list[dict[str, Any]]:
    """Declare exact block scope, or the complete domain corpus for U items."""

    if item["answerability"] == "unanswerable":
        declarations = []
        for document_id in item["corpus_document_ids"]:
            document = documents[document_id]
            declarations.append(
                {
                    "document_id": document_id,
                    "content_sha256": document["content_sha256"],
                    "allowed_block_ids": [
                        block["block_id"]
                        for block in document["blocks"]
                        if block["block_id"] in permitted_blocks[document_id]
                    ],
                }
            )
        return declarations

    declarations = []
    document_ids = list(
        dict.fromkeys(plan["document_id"] for plan in item["evidence_plan"])
    )
    for document_id in document_ids:
        document = documents[document_id]
        block_order = {
            block["block_id"]: index
            for index, block in enumerate(document["blocks"])
        }
        allowed = sorted(
            {
                excerpt.block_id
                for excerpt in excerpts
                if excerpt.document_id == document_id
            },
            key=block_order.__getitem__,
        )
        declarations.append(
            {
                "document_id": document_id,
                "content_sha256": document["content_sha256"],
                "allowed_block_ids": allowed,
            }
        )
    return declarations


def _constraints(
    item: dict[str, Any],
    source_documents: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build conservative German question-generation constraints."""

    document_scope = ", ".join(
        f"{document['document_id']} ({', '.join(document['allowed_block_ids'])})"
        for document in source_documents
    )
    if item["answerability"] == "unanswerable":
        forbidden = (
            "Keine definitive Einzelfallentscheidung, kein erfundener Betrag, "
            "Zeitpunkt, Marktwert oder Behördenbescheid und keine Ergänzung der "
            "absichtlich fehlenden Falldaten."
        )
    else:
        forbidden = (
            "Keine externen Fakten, keine nicht konfigurierten Rechts- oder "
            "Programmdetails, keine erfundenen Zahlen und keine Informationen "
            "außerhalb der erlaubten Blocks."
        )
    return {
        "focus": item["topic"],
        "required_reasoning_operation": REASONING_OPERATIONS[
            item["target_question_type"]
        ],
        "allowed_information": f"Nur die konfigurierten Informationen aus {document_scope}.",
        "forbidden_information": forbidden,
        "must_require_order": item["target_question_type"] == "Temporal Sequence",
    }


def _build_configuration(
    *,
    item: dict[str, Any],
    excerpts: list[EvidenceExcerpt],
    documents: dict[str, dict[str, Any]],
    permitted_blocks: dict[str, set[str]],
    record_schema_hash: str,
    domain_schema_hash: str,
    prompt_hash: str,
    prompt_path: str,
    allocation_path: str,
    allocation_hash: str,
    generated_at: str,
    generation_method: str,
) -> dict[str, Any]:
    """Build one mechanically valid v0.1 formal Configuration candidate."""

    values, provenance = _build_values_and_provenance(
        item=item,
        excerpts=excerpts,
    )
    source_documents = _allowed_document_declarations(
        item=item,
        excerpts=excerpts,
        documents=documents,
        permitted_blocks=permitted_blocks,
    )
    protocol_assumptions = [
        {
            "json_pointer": "/jurisdiction",
            "value": "Germany",
            "basis": (
                "The frozen thesis protocol defines Germany as the default "
                "jurisdiction; source evidence is not used for this metadata value."
            ),
        }
    ]
    if item["answerability"] == "unanswerable":
        protocol_assumptions.extend(
            [
                {
                    "json_pointer": (
                        "/question_generation_constraints/hypothetical_scenario"
                    ),
                    "value": item["hypothetical_scenario"],
                    "basis": (
                        "Author-defined underspecified scenario for corpus-wide "
                        "unanswerability review."
                    ),
                },
                {
                    "json_pointer": (
                        "/question_generation_constraints/missing_facts"
                    ),
                    "value": item["missing_facts"],
                    "basis": (
                        "These facts are intentionally omitted; a definitive answer "
                        "must not be inferred from general corpus guidance."
                    ),
                },
            ]
        )

    return {
        "configuration_id": item["configuration_id"],
        "version": "0.1",
        "status": "development_draft",
        "record_schema_id": "configuration_record_schema_v1.1",
        "record_schema_sha256": record_schema_hash,
        "schema_id": f"schema_S_{item['domain']}_v1.0",
        "schema_sha256": domain_schema_hash,
        "domain": item["domain"],
        "language": "de",
        "jurisdiction": "Germany",
        "target_question_type": item["target_question_type"],
        "answerability": item["answerability"],
        "source_scope": item["source_scope"],
        "source_documents": source_documents,
        "protocol_assumptions": protocol_assumptions,
        "question_generation_constraints": _constraints(item, source_documents),
        "values": values,
        "provenance": provenance,
        "generation_record": {
            "method": generation_method,
            "generated_at": generated_at,
            "prompt_path": prompt_path,
            "prompt_sha256": prompt_hash,
            "mechanical_validation": (
                "passed: allocation quotas, JSON Schema v1.1, domain Schema keys, "
                "frozen Document hashes, allowed-block scope, heading paths, exact "
                f"quotes and provenance pointers; allocation={allocation_path} "
                f"sha256={allocation_hash}"
            ),
            "human_review_status": "pending",
        },
    }


def _hash_pinned_dependency(
    *,
    repo_root: Path,
    dependencies: dict[str, Any],
    path_key: str,
    sha256_key: str,
) -> tuple[Path, dict[str, Any]] | None:
    """Load an optional dependency and reject stale path/hash declarations."""

    path_value = dependencies.get(path_key)
    if not path_value:
        return None
    path = repo_root / str(path_value)
    expected_hash = str(dependencies.get(sha256_key) or "")
    if not expected_hash or sha256_file(path) != expected_hash:
        raise ArtifactValidationError(
            f"Hash mismatch for Configuration dependency: {path_value}"
        )
    return path, _read_json_object(path)


def _permitted_blocks(
    *,
    repo_root: Path,
    allocation: dict[str, Any],
    documents: dict[str, dict[str, Any]],
) -> tuple[dict[str, set[str]], dict[str, Any]]:
    """Resolve hard evidence exclusions before excerpt selection begins."""

    permitted = {
        document_id: {block["block_id"] for block in document["blocks"]}
        for document_id, document in documents.items()
    }
    dependencies = allocation["dependencies"]
    policy_audit: dict[str, Any] = {
        "excluded_prior_evidence_block_count": 0,
        "market_case_document_count": 0,
    }

    exclusion_dependency = _hash_pinned_dependency(
        repo_root=repo_root,
        dependencies=dependencies,
        path_key="evidence_exclusion_registry_path",
        sha256_key="evidence_exclusion_registry_sha256",
    )
    if exclusion_dependency is not None:
        exclusion_path, exclusion_registry = exclusion_dependency
        excluded_count = 0
        for entry in exclusion_registry.get("prior_formal_evidence_exclusions", []):
            document_id = str(entry["document_id"])
            if document_id not in permitted:
                raise ArtifactValidationError(
                    f"Evidence exclusion references unknown Document: {document_id}"
                )
            excluded = set(entry.get("excluded_block_ids", []))
            unknown = excluded - permitted[document_id]
            if unknown:
                raise ArtifactValidationError(
                    f"Evidence exclusion has unknown blocks for {document_id}: "
                    f"{sorted(unknown)}"
                )
            permitted[document_id] -= excluded
            excluded_count += len(excluded)
        policy_audit["evidence_exclusion_registry"] = {
            "path": exclusion_path.relative_to(repo_root).as_posix(),
            "sha256": sha256_file(exclusion_path),
        }
        policy_audit["excluded_prior_evidence_block_count"] = excluded_count

    market_dependency = _hash_pinned_dependency(
        repo_root=repo_root,
        dependencies=dependencies,
        path_key="market_case_scope_review_path",
        sha256_key="market_case_scope_review_sha256",
    )
    if market_dependency is not None:
        market_path, market_review = market_dependency
        case_count = 0
        for entry in market_review.get("documents", []):
            document_id = str(entry["document_id"])
            if document_id not in permitted:
                raise ArtifactValidationError(
                    f"Market case review references unknown Document: {document_id}"
                )
            allowed = set(entry.get("allowed_block_ids", []))
            unknown = allowed - permitted[document_id]
            if unknown:
                raise ArtifactValidationError(
                    f"Market case review has prohibited or unknown blocks for "
                    f"{document_id}: {sorted(unknown)}"
                )
            permitted[document_id] &= allowed
            case_count += 1
        policy_audit["market_case_scope_review"] = {
            "path": market_path.relative_to(repo_root).as_posix(),
            "sha256": sha256_file(market_path),
        }
        policy_audit["market_case_document_count"] = case_count

    empty_documents = sorted(
        document_id for document_id, block_ids in permitted.items() if not block_ids
    )
    if empty_documents:
        raise ArtifactValidationError(
            f"Evidence policy leaves Documents without blocks: {empty_documents}"
        )
    return permitted, policy_audit


def generate_formal_configurations(
    *,
    repo_root: Path,
    allocation_path: Path,
    replace_existing: bool = False,
) -> dict[str, Any]:
    """Generate and validate all 60 formal Configuration C candidates."""

    repo_root = repo_root.resolve()
    if not allocation_path.is_absolute():
        allocation_path = repo_root / allocation_path
    allocation = _read_json_object(allocation_path)
    if allocation.get("status") != "review_candidate":
        raise ArtifactValidationError(
            f"Expected review-candidate allocation: {allocation_path}"
        )

    manifest_path = repo_root / allocation["dependencies"]["document_manifest_path"]
    if sha256_file(manifest_path) != allocation["dependencies"][
        "document_manifest_sha256"
    ]:
        raise ArtifactValidationError("Frozen Document manifest hash mismatch")
    document_manifest = _read_json_object(manifest_path)

    documents: dict[str, dict[str, Any]] = {}
    document_paths: dict[str, Path] = {}
    for declaration in document_manifest["documents"]:
        path = repo_root / declaration["path"]
        document = _read_json_object(path)
        validate_document_record(
            document=document,
            source_path=resolve_corpus_path(repo_root, document["source"]["path"]),
        )
        if sha256_file(path) != declaration["file_sha256"]:
            raise ArtifactValidationError(f"Frozen Document file hash mismatch: {path}")
        documents[document["document_id"]] = document
        document_paths[document["document_id"]] = path

    permitted_blocks, evidence_policy = _permitted_blocks(
        repo_root=repo_root,
        allocation=allocation,
        documents=documents,
    )

    record_schema_path = repo_root / CONFIGURATION_SCHEMA_PATH
    record_schema = _read_json_object(record_schema_path)
    record_schema_hash = sha256_file(record_schema_path)
    prompt_path_value = str(
        allocation["dependencies"].get(
            "configuration_prompt_path", CONFIGURATION_PROMPT_PATH
        )
    )
    prompt_path = repo_root / prompt_path_value
    prompt_hash = sha256_file(prompt_path)
    allocation_hash = sha256_file(allocation_path)

    domain_schemas = {
        domain: _read_json_object(repo_root / path)
        for domain, path in DOMAIN_SCHEMA_PATHS.items()
    }
    domain_schema_hashes = {
        domain: sha256_file(repo_root / path)
        for domain, path in DOMAIN_SCHEMA_PATHS.items()
    }

    output_root = repo_root / str(
        allocation.get("output_directory", FORMAL_CONFIGURATION_OUTPUT_DIR)
    )
    generated: list[dict[str, Any]] = []
    used_values: dict[str, set[str]] = {}
    review_rows: list[str] = []
    unanswerable_audits: list[dict[str, Any]] = []

    for item in allocation["items"]:
        excerpts = _select_evidence(
            item=item,
            documents=documents,
            used_values=used_values,
            permitted_blocks=permitted_blocks,
        )
        configuration = _build_configuration(
            item=item,
            excerpts=excerpts,
            documents=documents,
            permitted_blocks=permitted_blocks,
            record_schema_hash=record_schema_hash,
            domain_schema_hash=domain_schema_hashes[item["domain"]],
            prompt_hash=prompt_hash,
            prompt_path=prompt_path_value,
            allocation_path=allocation_path.relative_to(repo_root).as_posix(),
            allocation_hash=allocation_hash,
            generated_at=allocation["created_at"],
            generation_method=str(
                allocation.get(
                    "generation_method",
                    "Codex-authored formal allocation with deterministic "
                    "exact-source evidence extraction",
                )
            ),
        )
        errors = sorted(
            Draft202012Validator(
                record_schema,
                format_checker=FormatChecker(),
            ).iter_errors(configuration),
            key=lambda error: list(error.absolute_path),
        )
        if errors:
            detail = "; ".join(
                f"{'/'.join(str(part) for part in error.absolute_path)}: "
                f"{error.message}"
                for error in errors[:5]
            )
            raise ArtifactValidationError(
                f"JSON Schema validation failed for "
                f"{configuration['configuration_id']}: {detail}"
            )
        validate_configuration_record(
            configuration=configuration,
            schema=domain_schemas[item["domain"]],
            documents=documents,
        )

        domain_dir = output_root / item["domain"]
        output_path = domain_dir / (
            f"{configuration['configuration_id']}_v0.1.json"
        )
        if output_path.exists():
            existing = _read_json_object(output_path)
            if existing != configuration and not replace_existing:
                raise ArtifactValidationError(
                    "Configuration candidate differs from existing file; "
                    f"review the change or rerun with replacement enabled: {output_path}"
                )
            if existing == configuration:
                write_state = "unchanged"
            else:
                write_json_artifact(output_path, configuration)
                write_state = "replaced"
        else:
            write_json_artifact(output_path, configuration)
            write_state = "created"

        generated.append(
            {
                "configuration_id": configuration["configuration_id"],
                "domain": configuration["domain"],
                "target_question_type": configuration["target_question_type"],
                "answerability": configuration["answerability"],
                "primary_document_id": item["primary_document_id"],
                "path": output_path.relative_to(repo_root).as_posix(),
                "file_sha256": sha256_file(output_path),
                "configuration_sha256": _canonical_sha256(configuration),
                "source_document_ids": [
                    source["document_id"]
                    for source in configuration["source_documents"]
                ],
                "allowed_block_count": sum(
                    len(source["allowed_block_ids"])
                    for source in configuration["source_documents"]
                ),
                "provenance_count": len(configuration["provenance"]),
                "write_state": write_state,
            }
        )

        review_rows.append(
            f"### {configuration['configuration_id']} - "
            f"{configuration['target_question_type']}\n\n"
            f"- Domain: `{configuration['domain']}`\n"
            f"- Answerability: `{configuration['answerability']}`\n"
            f"- Topic: {item['topic']}\n"
            f"- Sources: "
            f"{', '.join(source['document_id'] for source in configuration['source_documents'])}\n"
            f"- Candidate: `{output_path.relative_to(repo_root).as_posix()}`\n\n"
            + "\n".join(
                f"{index + 1}. `{excerpt.document_id}/{excerpt.block_id}`: "
                f"{excerpt.normalized_value}"
                for index, excerpt in enumerate(excerpts)
            )
            + "\n"
        )

        if item["answerability"] == "unanswerable":
            unanswerable_audits.append(
                {
                    "configuration_id": configuration["configuration_id"],
                    "domain": item["domain"],
                    "scenario": item["hypothetical_scenario"],
                    "missing_facts": item["missing_facts"],
                    "frozen_corpus_document_ids": item["corpus_document_ids"],
                    "frozen_corpus_document_count": len(
                        item["corpus_document_ids"]
                    ),
                    "mechanical_scope_check": "passed",
                    "semantic_unanswerability_review": "pending_human_review",
                    "general_evidence": [
                        {
                            "document_id": excerpt.document_id,
                            "block_id": excerpt.block_id,
                            "exact_quote": excerpt.quote,
                        }
                        for excerpt in excerpts
                    ],
                }
            )

    manifest = {
        "manifest_id": allocation.get(
            "generation_manifest_id",
            "formal_configuration_generation_manifest_v0.1",
        ),
        "version": "0.1",
        "status": "review_candidate",
        "generated_at": allocation["created_at"],
        "allocation": {
            "path": allocation_path.relative_to(repo_root).as_posix(),
            "sha256": allocation_hash,
        },
        "document_manifest": {
            "path": manifest_path.relative_to(repo_root).as_posix(),
            "sha256": sha256_file(manifest_path),
        },
        "record_schema": {
            "path": CONFIGURATION_SCHEMA_PATH,
            "sha256": record_schema_hash,
        },
        "configuration_count": len(generated),
        "human_review_status": "pending",
        "configurations": generated,
    }
    if len(evidence_policy) > 2:
        manifest["evidence_policy"] = evidence_policy
    manifest_path_out = output_root / "generation_manifest_v0.1.json"
    write_json_artifact(manifest_path_out, manifest)

    audit_payload = {
        "audit_id": allocation.get(
            "unanswerable_audit_id",
            "formal_unanswerable_configuration_scope_v0.1",
        ),
        "version": "0.1",
        "status": "review_candidate",
        "generated_at": allocation["created_at"],
        "scope": (
            "Mechanical verification that every formal Unanswerable "
            "Configuration declares all eight frozen domain Documents."
        ),
        "configuration_count": len(unanswerable_audits),
        "items": unanswerable_audits,
        "limitation": (
            "This mechanical scope audit does not replace independent semantic "
            "review of corpus-wide unanswerability before QRA freeze."
        ),
    }
    audit_path = output_root / "unanswerable_scope_audit_v0.1.json"
    write_json_artifact(audit_path, audit_payload)

    review_path = output_root / "configuration_evidence_review_v0.1.md"
    review_path.write_text(
        f"# {allocation.get('review_title', 'Formal Configuration evidence review v0.1')}\n\n"
        "All 60 files are mechanically validated `development_draft` candidates. "
        "Review the topic, semantic fit of each extracted statement, temporal "
        "validity, and whether the configured facts can support a distinct "
        "Question/Reference Answer with 3-5 keypoints.\n\n"
        + "\n\n".join(review_rows).rstrip()
        + "\n",
        encoding="utf-8",
    )

    return {
        "configuration_count": len(generated),
        "manifest_path": manifest_path_out.relative_to(repo_root).as_posix(),
        "review_path": review_path.relative_to(repo_root).as_posix(),
        "unanswerable_audit_path": audit_path.relative_to(repo_root).as_posix(),
        "created_count": sum(item["write_state"] == "created" for item in generated),
        "replaced_count": sum(
            item["write_state"] == "replaced" for item in generated
        ),
        "unchanged_count": sum(
            item["write_state"] == "unchanged" for item in generated
        ),
    }
