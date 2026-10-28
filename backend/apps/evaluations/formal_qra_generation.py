"""Generate mechanically grounded formal QRA review candidates from frozen C files."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path

from apps.evaluations.corpus_paths import resolve_corpus_path
import re
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

from apps.evaluations.formal_configuration_generation import DOMAIN_SCHEMA_PATHS
from apps.evaluations.rageval_artifacts import (
    ArtifactValidationError,
    sha256_file,
    validate_configuration_record,
    validate_document_record,
    validate_qra_record,
    write_json_artifact,
)


QRA_RECORD_SCHEMA_PATH = (
    "docs/evaluation/formal_dataset/qra/qra_record_schema_v0.1.json"
)
QRA_PROMPT_PATH = (
    "docs/evaluation/formal_dataset/prompts/qra_generation_prompt_v1.0.md"
)
FORMAL_QRA_OUTPUT_DIR = "docs/evaluation/formal_dataset/qra/formal_candidates"
QRA_SEMANTIC_CURATION_PATH = (
    "docs/evaluation/formal_dataset/qra/qra_semantic_curation_v0.1.json"
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

QUESTION_TEMPLATES = {
    "Factual": "Welche konkreten Angaben nennt das Material zum Thema „{focus}“?",
    "Summarization": (
        "Wie lassen sich die zentralen Aussagen zum Thema „{focus}“ zusammenfassen?"
    ),
    "Multi-hop Reasoning": (
        "Wie hängen die beschriebenen Aspekte zum Thema „{focus}“ zusammen?"
    ),
    "Information Integration": (
        "Wie lassen sich die Angaben aus den herangezogenen Materialien zum "
        "Thema „{focus}“ zu einer gemeinsamen Antwort verbinden?"
    ),
    "Numerical Comparison": (
        "Welche dokumentierten Zahlen sind für das Thema „{focus}“ relevant, und wie "
        "unterscheiden sie sich?"
    ),
    "Temporal Sequence": (
        "Welche Reihenfolge wird zum Thema „{focus}“ beschrieben, und welche Schritte "
        "sind dabei zu beachten?"
    ),
}

_NOISE_PREFIXES = (
    "[reviewed visual content",
    "[reviewed figure description",
    "figure title:",
    "figure type:",
    "extracted values:",
    "main text:",
    "quelle:",
    "gründerzeiten ",
    "page ",
    "weiterführende informationen",
    "leitfaden:",
    "professionelle businessplan-gliederung:",
)
_NOISE_PHRASES = (
    "kostenlosen account",
    "das könnte dich auch interessieren",
    "bildnachweis",
    "bestellmöglichkeit",
    "publikationsversand",
    "deutscher startup monitor",
)
_INCOMPLETE_ENDINGS = re.compile(
    r"(?:\b(?:abs|bzw|ca|d|dr|etc|für|nach|nr|s|u|usw|vgl|von|z)\."
    r"|\bje\s+nach\.?|\bz\.\s*b\.)$",
    flags=re.IGNORECASE,
)
_FOCUS_STOPWORDS = {
    "aber",
    "auch",
    "dabei",
    "durch",
    "eine",
    "einem",
    "einer",
    "eines",
    "für",
    "gegen",
    "ihre",
    "nach",
    "oder",
    "sowie",
    "über",
    "verschiedener",
    "zwischen",
}


@dataclass(frozen=True)
class EvidenceAtom:
    """One exact source span and its concise judge-facing formulation."""

    document_id: str
    block_id: str
    heading_path: str
    page: int | None
    exact_quote: str
    keypoint_text: str
    block_order: int
    segment_order: int
    provenance_id: str | None
    provenance_order: int | None
    score: int


@dataclass(frozen=True)
class CuratedQRA:
    """A versioned semantic override resolved back to exact allowed evidence."""

    question: str
    atoms: tuple[EvidenceAtom, ...]
    units: tuple[tuple[str, tuple[str, ...]], ...]


def _read_json_object(path: Path) -> dict[str, Any]:
    """Read one UTF-8 JSON object."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(f"Cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ArtifactValidationError(f"Expected a JSON object: {path}")
    return payload


def _compact(value: str) -> str:
    """Collapse source whitespace for questions, answers, and review text."""

    return re.sub(r"\s+", " ", value).strip()


def _focus_terms(focus: str) -> set[str]:
    """Return stable content words used only to rank allowed evidence."""

    return {
        token.casefold()
        for token in re.findall(r"[A-Za-zÄÖÜäöüß]{4,}", focus)
        if token.casefold() not in _FOCUS_STOPWORDS
    }


def _clean_keypoint(exact_quote: str) -> str:
    """Turn an exact quote into a concise, source-faithful answer unit."""

    value = _compact(exact_quote)
    # PDF extraction often leaves a discretionary soft hyphen followed by a
    # line-break space inside one word (for example ``steuerli­ chen``).
    value = re.sub(r"(?<=\w)\u00ad\s*(?=\w)", "", value)
    value = re.sub(r"^(?:[-•▪○]|\d+[.)])\s*", "", value)
    value = re.sub(r"^Extracted values:\s*", "", value, flags=re.IGNORECASE)
    # Some source blocks use a short editorial question as an inline heading and
    # put the supported statement immediately after it.
    value = re.sub(r"^[^?]{1,140}\?\s+(?=[A-ZÄÖÜ])", "", value)
    value = re.sub(r"\s+([,.;:!?])", r"\1", value)
    value = re.sub(r":\.$", ".", value)
    if value and value[-1] not in ".!?":
        value += "."
    return value


def _is_usable_atom(exact_quote: str, keypoint_text: str) -> bool:
    """Reject extraction markers, fragments, and publication boilerplate."""

    normalized = _compact(exact_quote)
    lowered = normalized.casefold()
    if len(keypoint_text) < 16 or len(keypoint_text) > 700:
        return False
    if lowered.startswith(_NOISE_PREFIXES):
        return False
    if any(phrase in lowered for phrase in _NOISE_PHRASES):
        return False
    if "www." in lowered or "http://" in lowered or "https://" in lowered:
        return False
    if "gründerzeiten" in lowered and len(normalized) < 180:
        return False
    if "würden den kostenpreis" in lowered and len(normalized) < 100:
        return False
    if re.match(r"^(?:de\s+|\d+\s+(?:estg|ustg|gewstg)\b)", lowered):
        return False
    if _INCOMPLETE_ENDINGS.search(normalized):
        return False
    if normalized.endswith((",", ";")):
        return False
    if normalized.endswith("?") and normalized.count("?") == 1:
        return False
    has_finite_verb = bool(
        re.search(
            r"\b(?:darf|dürfen|enthält|erfolgt|gibt|gehört|gehören|hängt|"
            r"ist|kann|können|liegt|muss|müssen|sind|soll|sollen|steht|"
            r"werden|wird|zählt|zahlen|zeigt|zeigen)\b",
            lowered,
        )
    )
    structured_value = bool(
        (
            re.search(r"\d", normalized)
            and (":" in normalized or "%" in normalized or "prozent" in lowered)
        )
        or (
            ":" in normalized
            and len(re.findall(r"[A-Za-zÄÖÜäöüß]+", normalized)) >= 4
        )
    )
    ordered_label = bool(
        re.match(
            r"^(?:schritt\s+\d+|phase\s+\d+|empathize|define|ideate|"
            r"prototype|test)\b",
            lowered,
        )
    )
    complete_sentence = len(normalized) >= 45 and normalized.endswith((".", "!"))
    if (
        not has_finite_verb
        and not structured_value
        and not ordered_label
        and not complete_sentence
    ):
        return False
    if not re.search(r"[A-Za-zÄÖÜäöüß]", normalized):
        return False
    if re.fullmatch(r"(?:https?://|www\.)\S+", normalized):
        return False
    return True


def _exact_segments(text: str) -> list[tuple[int, str]]:
    """Find exact sentence and list-item spans without normalizing source quotes."""

    candidates: list[tuple[int, str]] = [(0, text.strip())]

    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped:
            start = offset + line.index(stripped)
            if (
                re.match(r"^(?:[-•▪○]|\d+[.)])\s+", stripped)
                or re.match(r"^[^:]{2,80}:\s*\S+", stripped)
                or re.search(r"\d", stripped)
            ):
                candidates.append((start, stripped))
        offset += len(line)

    for match in re.finditer(
        r"(?:^|(?<=[.!?]))\s*([^.!?\n][^.!?]*[.!?])(?=\s|$)",
        text,
        flags=re.MULTILINE,
    ):
        exact = match.group(1).strip()
        if exact:
            candidates.append((match.start(1), exact))

    for match in re.finditer(r"[•▪○]\s*([^•▪○\n]{12,})", text):
        exact = match.group(1).strip()
        if exact:
            candidates.append((match.start(1), exact))

    unique: dict[str, tuple[int, str]] = {}
    for segment_order, exact in candidates:
        key = _compact(exact).casefold()
        if key:
            unique.setdefault(key, (segment_order, exact))
    return list(unique.values())


def _smallest_complete_paragraph(text: str, exact_quote: str) -> str:
    """Expand a short locator quote to its smallest complete source paragraph.

    Frozen Configurations sometimes retain a deliberately small provenance
    fragment, such as a legal threshold or the first words of a statutory
    sentence.  That is sufficient to locate the source block, but it can be too
    short to stand alone as QRA evidence.  Curated recipes may opt into this
    expansion without changing the allowed Document or block scope.
    """

    paragraphs = [
        match.group(0).strip()
        for match in re.finditer(
            r"(?:^|(?<=\n))[^\n].*?(?=\n\s*\n|\Z)",
            text,
            flags=re.DOTALL,
        )
        if exact_quote in match.group(0)
    ]
    if not paragraphs:
        raise ArtifactValidationError(
            "Could not expand provenance quote to a complete paragraph"
        )
    return min(paragraphs, key=len)


def _atom_score(
    *,
    keypoint_text: str,
    focus_terms: set[str],
    question_type: str,
    provenance_id: str | None,
) -> int:
    """Rank only already-allowed evidence; scoring never expands source scope."""

    lowered = keypoint_text.casefold()
    score = 8 if provenance_id else 0
    score += 8 * sum(term in lowered for term in focus_terms)
    if 35 <= len(keypoint_text) <= 360:
        score += 5
    if len(keypoint_text) > 260:
        score -= 15
    if len(re.findall(r"[•▪○]", keypoint_text)) > 1:
        score -= 20
    if re.search(
        r"\b(?:darf|dürfen|enthält|erfolgt|gibt|gehört|gehören|hängt|"
        r"ist|kann|können|liegt|muss|müssen|sind|soll|sollen|steht|"
        r"werden|wird|zählt|zahlen|zeigt|zeigen)\b",
        lowered,
    ):
        score += 15
    if question_type == "Numerical Comparison" and re.search(r"\d", keypoint_text):
        score += 35
    if question_type == "Temporal Sequence" and re.search(
        r"\b(?:schritt|phase|zuerst|danach|anschließend|vor|nach|zuletzt)\b",
        lowered,
    ):
        score += 20
    if question_type == "Multi-hop Reasoning" and re.search(
        r"\b(?:abhäng|dadurch|daher|deshalb|führt|folg|je\s+höher|"
        r"je\s+stärker|wenn|weil|zusammenhang)\b",
        lowered,
    ):
        score += 20
    if question_type == "Unanswerable" and re.search(
        r"\b(?:abhäng|entscheid|kondition|prüf|variier|voraussetz)",
        lowered,
    ):
        score += 25
    if re.match(r"^(?:[-•▪○]|\d+[.)])\s+", keypoint_text):
        score += 5
    return score


def _atoms_from_span(
    *,
    document_id: str,
    block: dict[str, Any],
    block_order: int,
    span: str,
    provenance_id: str | None,
    provenance_order: int | None,
    focus_terms: set[str],
    question_type: str,
) -> list[EvidenceAtom]:
    """Split one exact allowed span into independently judgeable atoms."""

    atoms = []
    for segment_order, exact_quote in _exact_segments(span):
        keypoint_text = _clean_keypoint(exact_quote)
        if not _is_usable_atom(exact_quote, keypoint_text):
            continue
        score = _atom_score(
            keypoint_text=keypoint_text,
            focus_terms=focus_terms,
            question_type=question_type,
            provenance_id=provenance_id,
        )
        if _compact(exact_quote).casefold() == _compact(span).casefold():
            # A Configuration provenance span is already an authored evidence
            # choice. Prefer keeping it intact over an incidental neighbouring
            # sentence found while splitting the same block.
            score += 30
        atoms.append(
            EvidenceAtom(
                document_id=document_id,
                block_id=block["block_id"],
                heading_path=str(block.get("heading_path") or ""),
                page=block.get("page"),
                exact_quote=exact_quote,
                keypoint_text=keypoint_text,
                block_order=block_order,
                segment_order=segment_order,
                provenance_id=provenance_id,
                provenance_order=provenance_order,
                score=score,
            )
        )
    return atoms


def _candidate_atoms(
    *,
    configuration: dict[str, Any],
    documents: dict[str, dict[str, Any]],
) -> list[EvidenceAtom]:
    """Collect atoms from provenance first, then from the same allowed blocks."""

    question_type = configuration["target_question_type"]
    focus_text = configuration["question_generation_constraints"]["focus"]
    if configuration["target_question_type"] == "Unanswerable":
        focus_text += " " + " ".join(
            str(item["value"])
            for item in configuration["protocol_assumptions"]
            if item["json_pointer"]
            == "/question_generation_constraints/missing_facts"
        )
    focus_terms = _focus_terms(focus_text)
    blocks_by_document: dict[str, dict[str, tuple[int, dict[str, Any]]]] = {}
    for document_id, document in documents.items():
        blocks_by_document[document_id] = {
            block["block_id"]: (index, block)
            for index, block in enumerate(document["blocks"])
        }

    candidates: list[EvidenceAtom] = []
    for provenance_order, provenance in enumerate(configuration["provenance"]):
        document_id = provenance["document_id"]
        block_order, block = blocks_by_document[document_id][provenance["block_id"]]
        candidates.extend(
            _atoms_from_span(
                document_id=document_id,
                block=block,
                block_order=block_order,
                span=provenance["exact_quote"],
                provenance_id=provenance["provenance_id"],
                provenance_order=provenance_order,
                focus_terms=focus_terms,
                question_type=question_type,
            )
        )

    # Full allowed blocks are necessary for structured figures where one selected
    # provenance line is only one cell or chart value from the reviewed block.
    for source in configuration["source_documents"]:
        document_id = source["document_id"]
        for block_id in source["allowed_block_ids"]:
            block_order, block = blocks_by_document[document_id][block_id]
            candidates.extend(
                _atoms_from_span(
                    document_id=document_id,
                    block=block,
                    block_order=block_order,
                    span=block["text"],
                    provenance_id=None,
                    provenance_order=None,
                    focus_terms=focus_terms,
                    question_type=question_type,
                )
            )

    unique: dict[tuple[str, str, str], EvidenceAtom] = {}
    for atom in candidates:
        key = (atom.document_id, atom.block_id, _compact(atom.exact_quote).casefold())
        existing = unique.get(key)
        if existing is None or atom.score > existing.score:
            unique[key] = atom
    return list(unique.values())


def _overlaps_selected(atom: EvidenceAtom, selected: list[EvidenceAtom]) -> bool:
    """Avoid selecting a whole paragraph and one of its nested list items."""

    value = _compact(atom.exact_quote).casefold()
    for existing in selected:
        if atom.document_id != existing.document_id or atom.block_id != existing.block_id:
            continue
        other = _compact(existing.exact_quote).casefold()
        if value in other or other in value:
            return True
    return False


def _select_atoms(
    *,
    configuration: dict[str, Any],
    documents: dict[str, dict[str, Any]],
) -> list[EvidenceAtom]:
    """Select three to five evidence atoms while preserving multi-source coverage."""

    candidates = _candidate_atoms(
        configuration=configuration,
        documents=documents,
    )
    question_type = configuration["target_question_type"]
    target_count = {
        "Factual": 4,
        "Summarization": 4,
        "Multi-hop Reasoning": 4,
        "Information Integration": 4,
        "Numerical Comparison": 4,
        "Temporal Sequence": 5,
        "Unanswerable": 3,
    }[question_type]

    ranked = sorted(
        candidates,
        key=lambda atom: (
            -atom.score,
            atom.document_id,
            atom.block_order,
            atom.segment_order,
            len(atom.keypoint_text),
        ),
    )
    if question_type == "Numerical Comparison":
        numeric = [atom for atom in ranked if re.search(r"\d", atom.keypoint_text)]
        non_numeric = [atom for atom in ranked if not re.search(r"\d", atom.keypoint_text)]
        ranked = numeric + non_numeric

    selected: list[EvidenceAtom] = []
    if question_type not in {"Numerical Comparison", "Temporal Sequence"}:
        # Configuration C already records the intended evidence choices. Cover
        # every usable provenance item before falling back to the wider allowed
        # blocks; otherwise a high-scoring but irrelevant neighbouring sentence
        # can displace the authored evidence.
        provenance_orders = sorted(
            {
                atom.provenance_order
                for atom in ranked
                if atom.provenance_order is not None
            }
        )
        for provenance_order in provenance_orders:
            provenance_candidates = [
                atom
                for atom in ranked
                if atom.provenance_order == provenance_order
                and not _overlaps_selected(atom, selected)
            ]
            if provenance_candidates:
                selected.append(provenance_candidates[0])

    if question_type == "Temporal Sequence":
        allowed_blocks = [
            (source["document_id"], block_id)
            for source in configuration["source_documents"]
            for block_id in source["allowed_block_ids"]
        ]
        if len(allowed_blocks) >= 3:
            for document_id, block_id in allowed_blocks:
                block_candidates = [
                    atom
                    for atom in ranked
                    if atom.document_id == document_id and atom.block_id == block_id
                ]
                if block_candidates:
                    selected.append(block_candidates[0])
        else:
            provenance_orders = sorted(
                {
                    atom.provenance_order
                    for atom in ranked
                    if atom.provenance_order is not None
                }
            )
            for provenance_order in provenance_orders:
                provenance_candidates = [
                    atom
                    for atom in ranked
                    if atom.provenance_order == provenance_order
                ]
                if provenance_candidates:
                    selected.append(provenance_candidates[0])

    if question_type == "Information Integration":
        for source in configuration["source_documents"]:
            document_id = source["document_id"]
            for atom in ranked:
                if atom.document_id == document_id and not _overlaps_selected(
                    atom, selected
                ):
                    selected.append(atom)
                    break

    for atom in ranked:
        if len(selected) >= target_count:
            break
        if atom in selected or _overlaps_selected(atom, selected):
            continue
        selected.append(atom)

    if len(selected) < 3:
        raise ArtifactValidationError(
            f"{configuration['configuration_id']} has only {len(selected)} usable "
            "atomic evidence spans; revise the frozen Configuration before QRA freeze"
        )

    if question_type == "Temporal Sequence":
        document_order = {
            source["document_id"]: index
            for index, source in enumerate(configuration["source_documents"])
        }
        selected.sort(
            key=lambda atom: (
                atom.provenance_order
                if atom.provenance_order is not None
                else 10_000 + document_order[atom.document_id] * 1_000
                + atom.block_order,
                atom.segment_order,
            )
        )
    elif question_type not in {"Numerical Comparison"}:
        selected.sort(
            key=lambda atom: (
                atom.provenance_order
                if atom.provenance_order is not None
                else 10_000 + atom.block_order,
                atom.segment_order,
            )
        )
    return selected[:5]


def _qra_id(configuration_id: str) -> str:
    """Map a namespaced Configuration ID to a schema-compatible QRA ID."""

    match = re.fullmatch(
        r"C-([A-Z]+)-([A-Z]+)-([A-Z]+)-(\d{3})",
        configuration_id,
    )
    if match is None:
        raise ArtifactValidationError(
            f"Unexpected Configuration ID: {configuration_id}"
        )
    return (
        f"{match.group(1)}{match.group(2)}-"
        f"{match.group(3)}-{match.group(4)}"
    )


def _resolve_curated_qra(
    *,
    configuration: dict[str, Any],
    documents: dict[str, dict[str, Any]],
    entry: dict[str, Any],
) -> CuratedQRA:
    """Resolve one reviewed semantic recipe to exact Configuration evidence."""

    if entry.get("question_type") != configuration["target_question_type"]:
        raise ArtifactValidationError(
            f"Semantic curation type mismatch for {configuration['configuration_id']}"
        )
    question = str(entry.get("question") or "").strip()
    if not question:
        raise ArtifactValidationError(
            f"Semantic curation has no question: {configuration['configuration_id']}"
        )

    allowed_pairs = {
        (source["document_id"], block_id)
        for source in configuration["source_documents"]
        for block_id in source["allowed_block_ids"]
    }
    blocks: dict[tuple[str, str], tuple[int, dict[str, Any]]] = {}
    for document_id, block_id in allowed_pairs:
        block_order, block = next(
            (index, value)
            for index, value in enumerate(documents[document_id]["blocks"])
            if value["block_id"] == block_id
        )
        blocks[(document_id, block_id)] = (block_order, block)

    provenance_by_id = {
        value["provenance_id"]: (index, value)
        for index, value in enumerate(configuration["provenance"])
    }
    atoms: list[EvidenceAtom] = []
    reference_ids_by_key: dict[str, str] = {}
    for index, reference in enumerate(entry.get("references", []), start=1):
        reference_key = str(reference.get("reference_key") or "").strip()
        if not reference_key or reference_key in reference_ids_by_key:
            raise ArtifactValidationError(
                f"Invalid semantic curation reference key for "
                f"{configuration['configuration_id']}: {reference_key!r}"
            )

        provenance_id = reference.get("provenance_id")
        provenance_order: int | None = None
        if provenance_id:
            if provenance_id not in provenance_by_id:
                raise ArtifactValidationError(
                    f"Unknown provenance {provenance_id} in semantic curation for "
                    f"{configuration['configuration_id']}"
                )
            provenance_order, provenance = provenance_by_id[provenance_id]
            document_id = provenance["document_id"]
            block_id = provenance["block_id"]
            exact_quote = provenance["exact_quote"]
        else:
            document_id = str(reference.get("document_id") or "")
            block_id = str(reference.get("block_id") or "")
            exact_quote = str(reference.get("exact_quote") or "")

        pair = (document_id, block_id)
        if pair not in allowed_pairs:
            raise ArtifactValidationError(
                f"Semantic curation exceeds the frozen source scope for "
                f"{configuration['configuration_id']}: {document_id}/{block_id}"
            )
        block_order, block = blocks[pair]
        quote_scope = str(reference.get("quote_scope") or "exact")
        if quote_scope not in {"exact", "smallest_complete_paragraph"}:
            raise ArtifactValidationError(
                f"Unknown quote scope for {configuration['configuration_id']}: "
                f"{quote_scope}"
            )
        if not exact_quote and reference.get("contains"):
            needle = str(reference["contains"])
            matching_segments = [
                exact
                for _, exact in _exact_segments(block["text"])
                if needle in exact
            ]
            if not matching_segments:
                raise ArtifactValidationError(
                    f"Semantic curation selector did not match for "
                    f"{configuration['configuration_id']}: {reference_key}"
                )
            # The whole block is also an extraction candidate. The shortest
            # matching exact span gives the narrowest auditable citation.
            exact_quote = min(matching_segments, key=len)
        if quote_scope == "smallest_complete_paragraph":
            exact_quote = _smallest_complete_paragraph(block["text"], exact_quote)
        if not exact_quote or exact_quote not in block["text"]:
            raise ArtifactValidationError(
                f"Semantic curation quote is not exact for "
                f"{configuration['configuration_id']}: {reference_key}"
            )

        atoms.append(
            EvidenceAtom(
                document_id=document_id,
                block_id=block_id,
                heading_path=str(block.get("heading_path") or ""),
                page=block.get("page"),
                exact_quote=exact_quote,
                keypoint_text=_clean_keypoint(exact_quote),
                block_order=block_order,
                segment_order=0,
                provenance_id=str(provenance_id) if provenance_id else None,
                provenance_order=provenance_order,
                score=1_000,
            )
        )
        reference_ids_by_key[reference_key] = f"R{index}"

    if not atoms:
        raise ArtifactValidationError(
            f"Semantic curation has no references: {configuration['configuration_id']}"
        )

    units: list[tuple[str, tuple[str, ...]]] = []
    used_reference_keys: set[str] = set()
    for unit in entry.get("answer_units", []):
        text = str(unit.get("text") or "").strip()
        reference_keys = list(unit.get("reference_keys") or [])
        if not text or not reference_keys:
            raise ArtifactValidationError(
                f"Invalid semantic answer unit for {configuration['configuration_id']}"
            )
        unknown = set(reference_keys) - set(reference_ids_by_key)
        if unknown:
            raise ArtifactValidationError(
                f"Unknown semantic reference keys for "
                f"{configuration['configuration_id']}: {sorted(unknown)}"
            )
        used_reference_keys.update(reference_keys)
        units.append(
            (
                text,
                tuple(reference_ids_by_key[key] for key in reference_keys),
            )
        )

    if not 3 <= len(units) <= 5:
        raise ArtifactValidationError(
            f"Semantic curation must define 3-5 answer units for "
            f"{configuration['configuration_id']}"
        )
    if used_reference_keys != set(reference_ids_by_key):
        raise ArtifactValidationError(
            f"Semantic curation contains unused references for "
            f"{configuration['configuration_id']}"
        )
    return CuratedQRA(
        question=question,
        atoms=tuple(atoms),
        units=tuple(units),
    )


def _question(configuration: dict[str, Any]) -> str:
    """Create a natural question without exposing evaluation terminology."""

    focus = configuration["question_generation_constraints"]["focus"].strip()
    question_type = configuration["target_question_type"]
    if question_type != "Unanswerable":
        return QUESTION_TEMPLATES[question_type].format(focus=focus)

    assumptions = {
        item["json_pointer"]: item["value"]
        for item in configuration["protocol_assumptions"]
    }
    scenario = assumptions.get(
        "/question_generation_constraints/hypothetical_scenario",
        "Für den beschriebenen Gründungsfall fehlen entscheidende Angaben.",
    )
    return (
        f"{str(scenario).rstrip('.')} – lässt sich hierzu eine abschließende "
        f"Entscheidung zum Thema „{focus}“ treffen? Welche Angaben fehlen für eine "
        "belastbare Entscheidung?"
    )


def _source_type(source_path: str) -> str:
    """Map the frozen raw-source suffix to the QRA schema vocabulary."""

    suffix = Path(source_path).suffix.casefold()
    if suffix == ".pdf":
        return "pdf"
    if suffix in {".html", ".htm"}:
        return "html"
    return "text"


def _evidence_type(block: dict[str, Any]) -> str:
    """Map Document D block metadata to the QRA evidence vocabulary."""

    value = f"{block.get('block_type', '')} {block.get('evidence_type', '')}".casefold()
    if "figure" in value or "visual" in value:
        return "figure_description"
    if "table" in value:
        return "table"
    return "text"


def _references(
    *,
    atoms: list[EvidenceAtom],
    documents: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Resolve selected atoms to complete immutable QRA Reference records."""

    references = []
    for index, atom in enumerate(atoms, start=1):
        document = documents[atom.document_id]
        block = next(
            value for value in document["blocks"] if value["block_id"] == atom.block_id
        )
        source = document["source"]
        references.append(
            {
                "reference_id": f"R{index}",
                "document_id": atom.document_id,
                "document_content_sha256": document["content_sha256"],
                "raw_source_path": source["path"],
                "raw_source_sha256": source["sha256"],
                "source_type": _source_type(source["path"]),
                "source_url": str(source.get("url") or ""),
                "page": block.get("page"),
                "heading_path": str(block.get("heading_path") or ""),
                "block_id": atom.block_id,
                "quote": atom.exact_quote,
                "evidence_type": _evidence_type(block),
            }
        )
    return references


def _answer_units(
    *,
    configuration: dict[str, Any],
    atoms: list[EvidenceAtom],
) -> list[tuple[str, list[str]]]:
    """Build aligned answer units and their Reference IDs."""

    reference_ids = [f"R{index}" for index in range(1, len(atoms) + 1)]
    if configuration["answerability"] == "answerable":
        return [
            (atom.keypoint_text, [reference_id])
            for atom, reference_id in zip(atoms, reference_ids, strict=True)
        ]

    assumptions = {
        item["json_pointer"]: item["value"]
        for item in configuration["protocol_assumptions"]
    }
    missing_facts = str(
        assumptions.get(
            "/question_generation_constraints/missing_facts",
            "entscheidende fallbezogene Angaben",
        )
    )
    units = [
        (
            "Mit den angegebenen Falldaten ist keine abschließende Entscheidung "
            "möglich.",
            reference_ids,
        ),
        (f"Für eine belastbare Entscheidung fehlen: {missing_facts}.", reference_ids),
        (
            "Die herangezogenen Quellen nennen nur allgemeine Prüfkriterien; "
            "sie ersetzen nicht die fehlenden fallbezogenen Angaben.",
            reference_ids,
        ),
    ]
    return units


def _build_qra(
    *,
    configuration: dict[str, Any],
    configuration_path: Path,
    documents: dict[str, dict[str, Any]],
    atoms: list[EvidenceAtom],
    record_schema_hash: str,
    prompt_hash: str,
    generated_at: str,
    qra_version: str,
    repo_root: Path,
    curated: CuratedQRA | None = None,
) -> dict[str, Any]:
    """Assemble one versioned formal-test QRA review candidate."""

    references = _references(atoms=atoms, documents=documents)
    units = (
        [(text, list(reference_ids)) for text, reference_ids in curated.units]
        if curated is not None
        else _answer_units(configuration=configuration, atoms=atoms)
    )
    answer = "\n\n".join(text for text, _ in units)
    qra_id = _qra_id(configuration["configuration_id"])
    source_count = len(configuration["source_documents"])
    difficulty_type = re.sub(
        r"[^a-z0-9]+",
        "_",
        configuration["target_question_type"].casefold(),
    ).strip("_")
    return {
        "id": qra_id,
        "version": qra_version,
        "status": "development_candidate",
        "record_schema_id": "qra_record_schema_v0.1",
        "record_schema_sha256": record_schema_hash,
        "schema_id": configuration["schema_id"],
        "schema_sha256": configuration["schema_sha256"],
        "configuration_id": configuration["configuration_id"],
        "configuration_version": configuration["version"],
        "configuration_path": configuration_path.relative_to(repo_root).as_posix(),
        "configuration_sha256": sha256_file(configuration_path),
        "domain": configuration["domain"],
        "question_type": configuration["target_question_type"],
        "language": configuration["language"],
        "question": curated.question if curated is not None else _question(configuration),
        "answerable": configuration["answerability"] == "answerable",
        "answer": answer,
        "keypoints": [
            {
                "id": f"K{index}",
                "text": text,
                "required": True,
                "reference_ids": reference_ids,
            }
            for index, (text, reference_ids) in enumerate(units, start=1)
        ],
        "references": references,
        "answer_reference_alignment": [
            {
                "claim_id": f"A{index}",
                "answer_span": text,
                "reference_ids": reference_ids,
            }
            for index, (text, reference_ids) in enumerate(units, start=1)
        ],
        "document_ids": [
            source["document_id"] for source in configuration["source_documents"]
        ],
        "generation_basis": configuration["source_scope"],
        "question_generation_method": "configuration_and_document_grounded",
        "jurisdiction": configuration["jurisdiction"],
        "valid_at": generated_at,
        "difficulty": f"formal_{difficulty_type}_{source_count}_source",
        "split": "test",
        "review_status": "pending",
        "quality_checks": {
            "copied_source_heading": False,
            "external_knowledge_added": False,
            "reference_alignment": (
                f"passed mechanically: {len(units)}/{len(units)} answer spans map "
                "to exact allowed References"
            ),
            "keypoint_reference_coverage": (
                f"passed mechanically: {len(units)}/{len(units)} Keypoints map "
                "to exact allowed References"
            ),
        },
        "generation_record": {
            "method": (
                "Versioned semantic curation over exact frozen evidence"
                if curated is not None
                else "Codex-authored question templates with deterministic atomic "
                "evidence assembly from frozen Configuration C and Document D"
            ),
            "generator": "backend.apps.evaluations.formal_qra_generation",
            "generated_at": generated_at,
            "prompt_path": QRA_PROMPT_PATH,
            "prompt_sha256": prompt_hash,
            "reproducibility_note": (
                "High-risk question semantics are read from the versioned curation "
                "artifact and resolved to exact frozen allowed spans; all other "
                "records use deterministic provenance-first evidence selection. "
                "Final sufficiency remains pending human review."
            ),
            "mechanical_validation": (
                "passed: JSON Schema, frozen Configuration identity and hash, "
                "Document scope and hashes, exact Reference quotes, answer spans, "
                "and Keypoint mappings"
            ),
        },
    }


def _schema_validation_errors(
    payload: dict[str, Any],
    record_schema: dict[str, Any],
) -> list[str]:
    """Return compact QRA JSON Schema errors."""

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


def _write_review_document(
    *,
    path: Path,
    records: list[tuple[dict[str, Any], Path]],
    repo_root: Path,
    qra_version: str,
    dataset_label: str,
) -> None:
    """Write one readable audit surface for all generated formal QRA candidates."""

    lines = [
        f"# {dataset_label} QRA candidate review v{qra_version}",
        "",
        "These 60 `test`-split records are review candidates, not frozen test "
        "items. Review question naturalness, semantic answer sufficiency, each "
        "Keypoint, and every exact source quote before issuing a QRA freeze.",
        "",
    ]
    for qra, output_path in records:
        lines.extend(
            [
                f"## {qra['id']} - {qra['question_type']}",
                "",
                f"- Domain: `{qra['domain']}`",
                (
                    f"- Configuration: `{qra['configuration_id']} "
                    f"v{qra['configuration_version']}`"
                ),
                f"- Candidate: `{output_path.relative_to(repo_root).as_posix()}`",
                f"- Answerable: `{str(qra['answerable']).lower()}`",
                "",
                "**Question**",
                "",
                qra["question"],
                "",
                "**Reference answer**",
                "",
                qra["answer"],
                "",
                "**Keypoints and evidence**",
                "",
            ]
        )
        references = {
            reference["reference_id"]: reference for reference in qra["references"]
        }
        for keypoint in qra["keypoints"]:
            lines.append(f"{keypoint['id']}. {keypoint['text']}")
            for reference_id in keypoint["reference_ids"]:
                reference = references[reference_id]
                lines.append(
                    f"   - `{reference_id}` "
                    f"`{reference['document_id']}/{reference['block_id']}`: "
                    f"{_compact(reference['quote'])}"
                )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def generate_formal_qras(
    *,
    repo_root: Path,
    configuration_manifest_path: Path,
    generated_at: str,
    qra_version: str = "0.1",
    semantic_curation_path: Path | None = None,
    replace_existing: bool = False,
    output_dir: Path = Path(FORMAL_QRA_OUTPUT_DIR),
    manifest_id: str | None = None,
    semantic_curation_id: str | None = None,
    dataset_label: str = "Formal",
) -> dict[str, Any]:
    """Generate all 60 formal QRA review candidates from frozen Configurations."""

    repo_root = repo_root.resolve()
    if not configuration_manifest_path.is_absolute():
        configuration_manifest_path = repo_root / configuration_manifest_path
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    if re.fullmatch(r"[0-9]+\.[0-9]+", qra_version) is None:
        raise ArtifactValidationError(f"Invalid QRA version: {qra_version!r}")
    configuration_manifest = _read_json_object(configuration_manifest_path)
    if (
        configuration_manifest.get("status") != "frozen"
        or configuration_manifest.get("configuration_count") != 60
    ):
        raise ArtifactValidationError(
            f"Expected a frozen 60-item Configuration manifest: "
            f"{configuration_manifest_path}"
        )

    document_manifest_path = (
        repo_root / configuration_manifest["document_manifest"]["path"]
    )
    if (
        sha256_file(document_manifest_path)
        != configuration_manifest["document_manifest"]["sha256"]
    ):
        raise ArtifactValidationError("Frozen Document manifest hash mismatch")
    document_manifest = _read_json_object(document_manifest_path)

    documents: dict[str, dict[str, Any]] = {}
    for declaration in document_manifest["documents"]:
        document_path = repo_root / declaration["path"]
        document = _read_json_object(document_path)
        if sha256_file(document_path) != declaration["file_sha256"]:
            raise ArtifactValidationError(
                f"Frozen Document hash mismatch: {document_path}"
            )
        validate_document_record(
            document=document,
            source_path=resolve_corpus_path(repo_root, document["source"]["path"]),
        )
        documents[document["document_id"]] = document

    configuration_record_path = (
        repo_root / configuration_manifest["record_schema"]["path"]
    )
    configuration_record_schema = _read_json_object(configuration_record_path)
    domain_schemas = {
        domain: _read_json_object(repo_root / path)
        for domain, path in DOMAIN_SCHEMA_PATHS.items()
    }

    qra_record_path = repo_root / QRA_RECORD_SCHEMA_PATH
    qra_record_schema = _read_json_object(qra_record_path)
    qra_record_hash = sha256_file(qra_record_path)
    prompt_path = repo_root / QRA_PROMPT_PATH
    prompt_hash = sha256_file(prompt_path)
    if semantic_curation_path is None:
        semantic_curation_path = Path(QRA_SEMANTIC_CURATION_PATH)
    if not semantic_curation_path.is_absolute():
        semantic_curation_path = repo_root / semantic_curation_path
    semantic_curation = _read_json_object(semantic_curation_path)
    expected_curation_id = (
        semantic_curation_id
        if semantic_curation_id is not None
        else f"formal_qra_semantic_curation_v{qra_version}"
    )
    if (
        semantic_curation.get("curation_id") != expected_curation_id
        or semantic_curation.get("version") != qra_version
        or semantic_curation.get("status") != "development_candidate"
        or not isinstance(semantic_curation.get("entries"), dict)
    ):
        raise ArtifactValidationError(
            f"Invalid semantic curation artifact: {semantic_curation_path}"
        )
    curation_entries = semantic_curation["entries"]
    output_root = output_dir

    generated: list[dict[str, Any]] = []
    review_records: list[tuple[dict[str, Any], Path]] = []
    write_states: list[str] = []
    high_risk_configuration_ids: set[str] = set()
    configuration_ids: set[str] = set()
    for declaration in configuration_manifest["configurations"]:
        configuration_path = repo_root / declaration["path"]
        if sha256_file(configuration_path) != declaration["file_sha256"]:
            raise ArtifactValidationError(
                f"Frozen Configuration hash mismatch: {configuration_path}"
            )
        configuration = _read_json_object(configuration_path)
        configuration_schema_errors = _schema_validation_errors(
            configuration,
            configuration_record_schema,
        )
        if configuration_schema_errors:
            raise ArtifactValidationError(
                f"Invalid frozen Configuration {configuration_path}: "
                f"{'; '.join(configuration_schema_errors[:5])}"
            )
        validate_configuration_record(
            configuration=configuration,
            schema=domain_schemas[configuration["domain"]],
            documents=documents,
        )
        configuration_ids.add(configuration["configuration_id"])

        if configuration["target_question_type"] in {
            "Numerical Comparison",
            "Temporal Sequence",
        }:
            high_risk_configuration_ids.add(configuration["configuration_id"])
        curation_entry = curation_entries.get(configuration["configuration_id"])
        curated = (
            _resolve_curated_qra(
                configuration=configuration,
                documents=documents,
                entry=curation_entry,
            )
            if curation_entry is not None
            else None
        )
        atoms = (
            list(curated.atoms)
            if curated is not None
            else _select_atoms(configuration=configuration, documents=documents)
        )
        qra = _build_qra(
            configuration=configuration,
            configuration_path=configuration_path,
            documents=documents,
            atoms=atoms,
            record_schema_hash=qra_record_hash,
            prompt_hash=prompt_hash,
            generated_at=generated_at,
            qra_version=qra_version,
            repo_root=repo_root,
            curated=curated,
        )
        qra_schema_errors = _schema_validation_errors(qra, qra_record_schema)
        if qra_schema_errors:
            raise ArtifactValidationError(
                f"Invalid QRA candidate {qra['id']}: "
                f"{'; '.join(qra_schema_errors[:5])}"
            )
        validate_qra_record(
            qra=qra,
            configuration=configuration,
            documents=documents,
        )

        output_path = (
            output_root / qra["domain"] / f"{qra['id']}_v{qra_version}.json"
        )
        if output_path.exists():
            existing = _read_json_object(output_path)
            if existing != qra and not replace_existing:
                raise ArtifactValidationError(
                    "QRA candidate differs from existing file; review the change or "
                    f"rerun with replacement enabled: {output_path}"
                )
            if existing == qra:
                write_state = "unchanged"
            else:
                write_json_artifact(output_path, qra)
                write_state = "replaced"
        else:
            write_json_artifact(output_path, qra)
            write_state = "created"
        write_states.append(write_state)

        generated.append(
            {
                "qra_id": qra["id"],
                "configuration_id": qra["configuration_id"],
                "domain": qra["domain"],
                "question_type": qra["question_type"],
                "answerable": qra["answerable"],
                "path": output_path.relative_to(repo_root).as_posix(),
                "file_sha256": sha256_file(output_path),
                "keypoint_count": len(qra["keypoints"]),
                "reference_count": len(qra["references"]),
                "document_ids": qra["document_ids"],
            }
        )
        review_records.append((qra, output_path))

    if len(generated) != 60 or len({item["qra_id"] for item in generated}) != 60:
        raise ArtifactValidationError("Formal QRA generation did not produce 60 unique IDs")
    if not high_risk_configuration_ids.issubset(curation_entries):
        missing = sorted(high_risk_configuration_ids - set(curation_entries))
        raise ArtifactValidationError(
            "Semantic curation must cover all Numerical Comparison and Temporal "
            f"Sequence records; missing={missing}"
        )
    unexpected = sorted(set(curation_entries) - configuration_ids)
    if unexpected:
        raise ArtifactValidationError(
            f"Semantic curation contains unknown Configuration IDs: {unexpected}"
        )
    if Counter(item["answerable"] for item in generated) != {True: 54, False: 6}:
        raise ArtifactValidationError("Formal QRA answerability allocation changed")
    for domain in DOMAIN_SCHEMA_PATHS:
        domain_items = [item for item in generated if item["domain"] == domain]
        if Counter(item["question_type"] for item in domain_items) != EXPECTED_TYPE_COUNTS:
            raise ArtifactValidationError(
                f"Formal QRA type allocation changed for {domain}"
            )

    resolved_manifest_id = (
        manifest_id
        if manifest_id is not None
        else f"formal_qra_generation_manifest_v{qra_version}"
    )
    manifest = {
        "manifest_id": resolved_manifest_id,
        "version": qra_version,
        "status": "review_candidate",
        "generated_at": generated_at,
        "split": "test",
        "review_status": "pending",
        "configuration_manifest": {
            "path": configuration_manifest_path.relative_to(repo_root).as_posix(),
            "sha256": sha256_file(configuration_manifest_path),
        },
        "document_manifest": {
            "path": document_manifest_path.relative_to(repo_root).as_posix(),
            "sha256": sha256_file(document_manifest_path),
        },
        "record_schema": {
            "path": QRA_RECORD_SCHEMA_PATH,
            "sha256": qra_record_hash,
        },
        "generation_prompt": {
            "path": QRA_PROMPT_PATH,
            "sha256": prompt_hash,
        },
        "semantic_curation": {
            "path": semantic_curation_path.relative_to(repo_root).as_posix(),
            "sha256": sha256_file(semantic_curation_path),
            "entry_count": len(curation_entries),
            "scope": semantic_curation.get(
                "scope",
                "Mandatory high-risk coverage plus review-driven repairs",
            ),
        },
        "qra_count": len(generated),
        "answerable_count": 54,
        "unanswerable_count": 6,
        "qras": generated,
        "freeze_note": (
            "These QRA files are not frozen. Complete semantic review and issue an "
            "explicit QRA freeze before any locked formal benchmark run."
        ),
    }
    if qra_version != "0.1":
        manifest["supersedes"] = semantic_curation.get(
            "supersedes_generation_manifest",
            "formal_qra_generation_manifest_v0.1",
        )
    manifest_path = output_root / f"generation_manifest_v{qra_version}.json"
    write_json_artifact(manifest_path, manifest)
    review_path = output_root / f"qra_candidate_review_v{qra_version}.md"
    _write_review_document(
        path=review_path,
        records=review_records,
        repo_root=repo_root,
        qra_version=qra_version,
        dataset_label=dataset_label,
    )

    return {
        "qra_count": len(generated),
        "manifest_path": manifest_path.relative_to(repo_root).as_posix(),
        "review_path": review_path.relative_to(repo_root).as_posix(),
        "created_count": write_states.count("created"),
        "replaced_count": write_states.count("replaced"),
        "unchanged_count": write_states.count("unchanged"),
    }
