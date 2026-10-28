"""Build semantic recipes for high-risk optimization QRA candidates."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

from apps.evaluations.rageval_artifacts import (
    ArtifactValidationError,
    sha256_file,
    write_json_artifact,
)


HIGH_RISK_TYPES = {"Numerical Comparison", "Temporal Sequence"}

QUESTION_TEMPLATES = {
    "Factual": "Welche konkreten Angaben nennt das Material zu „{focus}“?",
    "Summarization": (
        "Wie lassen sich die zentralen Aussagen zu „{focus}“ zusammenfassen?"
    ),
    "Multi-hop Reasoning": (
        "Wie hängen die dokumentierten Aspekte zu „{focus}“ zusammen?"
    ),
    "Information Integration": (
        "Wie lassen sich die Angaben aus den Materialien zu „{focus}“ verbinden?"
    ),
}

TWO_SPAN_SYNTHESIS = {
    "C-OPT-FP-F-001": (
        "Der Break-even-Point bezeichnet damit die Schwelle, ab deren Überschreitung "
        "Gewinn entsteht."
    ),
    "C-OPT-FP-F-002": (
        "Der Liquiditätsplan beurteilt die künftige Zahlungsfähigkeit durch den "
        "termingenauen Vergleich erwarteter Ein- und Auszahlungen."
    ),
    "C-OPT-FP-F-003": (
        "Die Förderung wird somit nicht automatisch bewilligt: Neben der "
        "Ermessensentscheidung werden Fähigkeiten und Kenntnisse beurteilt."
    ),
    "C-OPT-FP-F-004": (
        "Der Businessplan übersetzt die Geschäftsidee in Ziele, Strategien und "
        "Aktivitäten für die zentralen Unternehmensbereiche."
    ),
    "C-OPT-FP-F-005": (
        "Der Finanzplan verbindet den Nachweis künftiger Zahlungsfähigkeit mit der "
        "Darstellung gegenüber möglichen Geldgebern."
    ),
    "C-OPT-LFS-F-001": (
        "Wer die genannten Merkmale einer gewerblichen Tätigkeit erfüllt, muss den "
        "Gewerbeschein vor Aufnahme der Tätigkeit beantragen."
    ),
    "C-OPT-LFS-F-002": (
        "Beim Einzelunternehmen führt eine Person das Unternehmen allein und haftet "
        "persönlich."
    ),
    "C-OPT-LFS-F-003": (
        "Freiberufler arbeiten ebenfalls selbstständig auf eigene Rechnung, werden "
        "aber bei Gewerbesteuer und Buchführung anders behandelt."
    ),
    "C-OPT-LFS-F-004": (
        "Als Kapitalgesellschaft begrenzt die GmbH die Haftung der Gesellschafter auf "
        "das Gesellschaftsvermögen."
    ),
    "C-OPT-LFS-F-005": (
        "Die zuständige Anmeldestelle hängt davon ab, ob eine freiberufliche oder "
        "gewerbliche Tätigkeit aufgenommen wird."
    ),
    "C-OPT-MV-F-001": (
        "Sowohl Fach- und Publikumsmagazine als auch Marktstudien können als Quellen "
        "für neue Geschäftsideen dienen."
    ),
    "C-OPT-MV-F-002": (
        "Direkte Kundenansprache liefert Informationen und trainiert zugleich eine "
        "später im Vertrieb wichtige Fähigkeit."
    ),
    "C-OPT-MV-F-003": (
        "Fehlendes Kundenwissen erhöht damit das Risiko, bei Produktentwicklung, "
        "Vermarktung und Preis am Kundennutzen vorbeizuplanen."
    ),
    "C-OPT-MV-F-004": (
        "Ein MVP ist also auf Kernfunktionen reduziert, muss aber anders als ein "
        "Prototyp bereits funktionieren und Kundennutzen bieten."
    ),
    "C-OPT-MV-F-005": (
        "Die Nischensuche soll eine klare Shop-Positionierung ermöglichen und kann "
        "bei Offline-Konzepten ohne genügend Laufkundschaft ansetzen."
    ),
}

# These recipes intentionally spell out every numerical derivation and sequence.
# All reference keys resolve to exact provenance spans in the frozen Configuration.
RECIPES: dict[str, dict[str, Any]] = {
    "C-OPT-FP-NC-001": {
        "question": (
            "Bei 10.000 EUR Investition, 5.000 EUR monatlichem Umsatz und "
            "4.500 EUR monatlichen Kosten: Wie hoch ist der monatliche Überschuss "
            "und wann wird der Break-even erreicht?"
        ),
        "units": [
            ("Die anfängliche Investition beträgt 10.000 EUR.", ["E1"]),
            (
                "Aus 5.000 EUR Umsatz minus 4.500 EUR Kosten ergibt sich ein "
                "monatlicher Überschuss von 500 EUR.",
                ["E1"],
            ),
            ("Der dokumentierte Break-even liegt deshalb in Monat 20.", ["E1"]),
        ],
    },
    "C-OPT-FP-NC-002": {
        "question": (
            "Wie unterscheiden sich Dauer und Leistung der beiden Phasen des "
            "Gründungszuschusses?"
        ),
        "units": [
            (
                "Die Förderung dauert höchstens 15 Monate und besteht aus zwei "
                "Phasen.",
                ["E1"],
            ),
            (
                "In den ersten sechs Monaten werden das individuelle monatliche "
                "ALG I und zusätzlich 300 EUR zur sozialen Absicherung gezahlt.",
                ["E2"],
            ),
            (
                "Danach können für weitere neun Monate 300 EUR pro Monat gezahlt "
                "werden; diese zweite Phase ist drei Monate länger als die erste.",
                ["E2", "E3"],
            ),
        ],
    },
    "C-OPT-LFS-NC-001": {
        "question": (
            "Welche Umsatz- und Gewinngrenzen gelten laut Material für einfache "
            "Buchführung und EÜR, und wann entsteht Bilanzierungspflicht?"
        ),
        "units": [
            ("Die jährliche Umsatzgrenze liegt bei 800.000 EUR.", ["E1"]),
            ("Die jährliche Gewinngrenze liegt bei 80.000 EUR.", ["E1"]),
            (
                "Wird mindestens eine der beiden Grenzen überschritten, greift "
                "die Bilanzierungspflicht.",
                ["E1", "E2"],
            ),
        ],
    },
    "C-OPT-LFS-NC-002": {
        "question": (
            "Wie verhalten sich Mindeststammkapital und zusätzliche "
            "Gründungskosten einer GmbH zueinander?"
        ),
        "units": [
            ("Das Mindeststammkapital beträgt 25.000 EUR.", ["E1"]),
            (
                "Für Notariat, Gewerbeanmeldung und Handelsregistereintrag fallen "
                "zusätzlich ungefähr 1.000 EUR Gründungskosten an.",
                ["E2", "E3"],
            ),
            (
                "Zusammen sind damit rund 26.000 EUR aus Stammkapital und den "
                "genannten zusätzlichen Kosten zu berücksichtigen.",
                ["E1", "E3"],
            ),
        ],
    },
    "C-OPT-MV-NC-001": {
        "question": (
            "Wie viele Arbeitsschritte und wie viele zentrale Ergebnispunkte nennt "
            "das Material für die Zielgruppenanalyse?"
        ),
        "units": [
            ("Für die Zielgruppenanalyse werden drei Schritte genannt.", ["E1"]),
            ("Das Ergebnis wird anhand von vier zentralen Punkten gebündelt.", ["E2"]),
            (
                "Damit gibt es einen zentralen Ergebnispunkt mehr als "
                "Arbeitsschritte.",
                ["E1", "E2"],
            ),
        ],
    },
    "C-OPT-MV-NC-002": {
        "question": (
            "Welche drei Besonderheiten von B2B-Kaufentscheidungen und welche drei "
            "MAN-Kriterien nennt das Material?"
        ),
        "units": [
            (
                "An B2B-Kaufentscheidungen sind häufig mehrere Personen mit "
                "unterschiedlichen Interessen beteiligt.",
                ["E1"],
            ),
            (
                "Die Entscheidungsprozesse dauern meist länger als im "
                "Endkundensegment, und die Leistungen sind häufig komplex.",
                ["E2", "E3"],
            ),
            (
                "Die drei MAN-Kriterien sind ausreichende Mittel, Autorität für die "
                "Kaufentscheidung und erkannte Notwendigkeit des Kaufs.",
                ["E4", "E5", "E6"],
            ),
        ],
    },
    "C-OPT-FP-TS-001": {
        "question": "In welcher Reihenfolge wird ein einfacher Liquiditätsplan erstellt?",
        "units": [
            (
                "Zuerst werden die voraussichtlichen Zahlungseingänge und "
                "Zahlungsausgänge gegenübergestellt.",
                ["E1"],
            ),
            ("Die Zahlungseingänge werden aus den erwarteten Umsätzen abgeleitet.", ["E2"]),
            (
                "Anschließend werden die verschiedenen Positionen der "
                "Zahlungsausgänge erfasst.",
                ["E3"],
            ),
            (
                "Für den Einstieg werden beide Seiten für einige Monate im Voraus "
                "in einer einfachen Tabelle festgehalten.",
                ["E4"],
            ),
        ],
    },
    "C-OPT-FP-TS-002": {
        "question": (
            "Welche Reihenfolge zentraler Businessplan-Abschnitte bis zum "
            "Finanzteil ergibt sich aus dem Material?"
        ),
        "units": [
            ("Am Anfang steht eine sorgfältige Zusammenfassung.", ["E1"]),
            ("Danach wird die Geschäftsidee beschrieben und getestet.", ["E2"]),
            (
                "Im folgenden Abschnitt wird das Gründungsumfeld beziehungsweise "
                "der Markt behandelt.",
                ["E3"],
            ),
            (
                "Im zweiten Hauptteil folgen die Zahlen der Gründung und die "
                "Ertragsquellen.",
                ["E4"],
            ),
        ],
    },
    "C-OPT-LFS-TS-001": {
        "question": (
            "Welche behördlichen Schritte folgen von der Vorbereitung der "
            "Gewerbeanmeldung bis zu den nachgelagerten Meldungen?"
        ),
        "units": [
            ("Zuerst werden alle erforderlichen Unterlagen gesammelt.", ["E1"]),
            (
                "Danach erfolgt die Gewerbeanmeldung persönlich, schriftlich oder "
                "online.",
                ["E2"],
            ),
            (
                "Nach Erhalt des Gewerbescheins ist der Fragebogen zur steuerlichen "
                "Erfassung innerhalb eines Monats an das Finanzamt zu übermitteln.",
                ["E3"],
            ),
            (
                "Je nach Branche ist außerdem die zuständige Berufsgenossenschaft "
                "zu berücksichtigen.",
                ["E4"],
            ),
        ],
    },
    "C-OPT-LFS-TS-002": {
        "question": (
            "Wie läuft die Anmeldung und steuerliche Erfassung einer "
            "freiberuflichen Tätigkeit zeitlich ab?"
        ),
        "units": [
            ("Erste Anlaufstelle für die Anmeldung ist das Finanzamt.", ["E1"]),
            (
                "Die Meldung muss spätestens vier Wochen nach Aufnahme der "
                "Tätigkeit erfolgen.",
                ["E2"],
            ),
            (
                "Nach Übermittlung der erforderlichen Angaben vergibt das Finanzamt "
                "eine Steuernummer.",
                ["E3"],
            ),
        ],
    },
    "C-OPT-MV-TS-001": {
        "question": "In welcher Reihenfolge durchläuft ein MVP den Lernzyklus?",
        "units": [
            ("Der Zyklus beginnt mit Build beziehungsweise Machen.", ["E1"]),
            ("Danach folgt Measure beziehungsweise Messen.", ["E2"]),
            ("Den Abschluss bildet Learn beziehungsweise Lernen.", ["E3"]),
        ],
    },
    "C-OPT-MV-TS-002": {
        "question": (
            "Welche Abfolge führt bei einem Foodtruck von der Ideensammlung zur "
            "Markt- und Zielgruppenprüfung?"
        ),
        "units": [
            ("Zuerst werden mögliche Gerichte und Küchen als Ideen gesammelt.", ["E1"]),
            (
                "Danach kann das Konzept zeitlich begrenzt, etwa in einem "
                "Pop-up-Restaurant, praktisch erprobt werden.",
                ["E2"],
            ),
            (
                "Anschließend werden Marktlage und mögliche Schwankungen mit einer "
                "Marktanalyse geprüft.",
                ["E3"],
            ),
            (
                "Schließlich wird die Zielgruppe anhand des vorgesehenen Einsatzorts "
                "und Nutzungskontexts eingegrenzt.",
                ["E4"],
            ),
        ],
    },
}


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(f"Cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ArtifactValidationError(f"Expected a JSON object: {path}")
    return payload


def _assumptions(configuration: dict[str, Any]) -> dict[str, Any]:
    return {
        item["json_pointer"]: item["value"]
        for item in configuration["protocol_assumptions"]
    }


def _clean_quote(value: str) -> str:
    value = re.sub(r"\s+", " ", value).strip()
    value = re.sub(r"(?<=\w)\u00ad\s*(?=\w)", "", value)
    return re.sub(r"^(?:[-•]\s*)", "", value)


def _generic_recipe(configuration: dict[str, Any]) -> dict[str, Any]:
    """Create a provenance-complete recipe for non-numerical/non-sequence items."""

    question_type = configuration["target_question_type"]
    focus = configuration["question_generation_constraints"]["focus"]
    provenance = configuration["provenance"]
    reference_keys = [f"E{index}" for index in range(1, len(provenance) + 1)]

    if question_type == "Unanswerable":
        assumptions = _assumptions(configuration)
        scenario = str(
            assumptions[
                "/question_generation_constraints/hypothetical_scenario"
            ]
        ).rstrip(".")
        missing_facts = str(
            assumptions["/question_generation_constraints/missing_facts"]
        )
        return {
            "question": (
                f"{scenario}. Lässt sich die verlangte Entscheidung mit den "
                "vorliegenden Angaben belastbar treffen?"
            ),
            "units": [
                (
                    "Nein. Die Quellen liefern allgemeine Kriterien, aber keine "
                    "hinreichenden Falldaten für die verlangte definitive Aussage.",
                    reference_keys,
                ),
                (
                    f"Für eine belastbare Beurteilung fehlen: {missing_facts}.",
                    reference_keys,
                ),
                (
                    "Die fehlenden individuellen Angaben müssen erhoben und anhand "
                    "der dokumentierten Kriterien geprüft werden; eine Garantie oder "
                    "verbindliche Einzelfallentscheidung ist so nicht ableitbar.",
                    reference_keys,
                ),
            ],
        }

    if question_type not in QUESTION_TEMPLATES:
        raise ArtifactValidationError(
            f"No generic question template for {configuration['configuration_id']}"
        )
    question = QUESTION_TEMPLATES[question_type].format(focus=focus)
    quoted_units = [
        (_clean_quote(str(item["exact_quote"])), [reference_key])
        for item, reference_key in zip(provenance, reference_keys, strict=True)
    ]
    if len(quoted_units) >= 3:
        units = quoted_units[:5]
    elif len(quoted_units) == 2:
        synthesis = TWO_SPAN_SYNTHESIS.get(configuration["configuration_id"])
        if synthesis is None:
            raise ArtifactValidationError(
                f"Missing two-span synthesis for {configuration['configuration_id']}"
            )
        units = [
            *quoted_units,
            (synthesis, reference_keys),
        ]
    else:
        raise ArtifactValidationError(
            f"Generic recipe has fewer than two provenance spans for "
            f"{configuration['configuration_id']}"
        )
    return {"question": question, "units": units}


def build_optimization_qra_curation(
    *,
    repo_root: Path,
    configuration_manifest_path: Path,
    output_path: Path,
    generated_at: str,
) -> dict[str, Any]:
    """Write provenance-bound semantic recipes for all 60 QRA candidates."""

    repo_root = repo_root.resolve()
    if not configuration_manifest_path.is_absolute():
        configuration_manifest_path = repo_root / configuration_manifest_path
    if not output_path.is_absolute():
        output_path = repo_root / output_path

    manifest = _read_json_object(configuration_manifest_path)
    if (
        manifest.get("manifest_id") != "optimization_configuration_manifest_v1.0"
        or manifest.get("status") != "frozen"
        or manifest.get("configuration_count") != 60
    ):
        raise ArtifactValidationError(
            f"Unexpected optimization Configuration manifest: {configuration_manifest_path}"
        )

    entries: dict[str, Any] = {}
    high_risk_ids: set[str] = set()
    for declaration in manifest["configurations"]:
        configuration_path = repo_root / declaration["path"]
        if sha256_file(configuration_path) != declaration["file_sha256"]:
            raise ArtifactValidationError(
                f"Frozen Configuration hash mismatch: {configuration_path}"
            )
        configuration = _read_json_object(configuration_path)
        configuration_id = configuration["configuration_id"]
        if configuration["target_question_type"] in HIGH_RISK_TYPES:
            high_risk_ids.add(configuration_id)
            recipe = RECIPES.get(configuration_id)
            if recipe is None:
                raise ArtifactValidationError(
                    f"Missing high-risk QRA recipe for {configuration_id}"
                )
        else:
            recipe = _generic_recipe(configuration)

        provenance_ids = {
            provenance["provenance_id"] for provenance in configuration["provenance"]
        }
        references = [
            {
                "reference_key": f"E{index}",
                "provenance_id": provenance["provenance_id"],
            }
            for index, provenance in enumerate(configuration["provenance"], start=1)
        ]
        reference_keys = {reference["reference_key"] for reference in references}
        used_keys = {key for _, keys in recipe["units"] for key in keys}
        if len(provenance_ids) != len(references) or used_keys != reference_keys:
            raise ArtifactValidationError(
                f"Recipe reference coverage mismatch for {configuration_id}"
            )

        entries[configuration_id] = {
            "question_type": configuration["target_question_type"],
            "question": recipe["question"],
            "references": references,
            "answer_units": [
                {"text": text, "reference_keys": keys}
                for text, keys in recipe["units"]
            ],
        }

    if high_risk_ids != set(RECIPES):
        raise ArtifactValidationError(
            "High-risk recipe set does not match the frozen Configuration set"
        )

    payload = {
        "curation_id": "optimization_qra_semantic_curation_v0.1",
        "version": "0.1",
        "status": "development_candidate",
        "generated_at": generated_at,
        "scope": (
            "All 60 frozen optimization Configurations. Numerical derivations and "
            "temporal ordering use explicit authored recipes; the remaining items "
            "use provenance-complete deterministic recipes. All remain pending "
            "semantic review."
        ),
        "methodology_note": (
            "Each recipe is bound to exact provenance IDs in the frozen "
            "optimization Configuration manifest. The curation layer supplies "
            "question semantics and 3-5 answer units; the QRA generator resolves "
            "every Reference back to an exact frozen source span."
        ),
        "configuration_manifest": {
            "path": configuration_manifest_path.relative_to(repo_root).as_posix(),
            "sha256": sha256_file(configuration_manifest_path),
        },
        "entries": entries,
    }
    write_json_artifact(output_path, payload)
    return {
        "entry_count": len(entries),
        "path": output_path.relative_to(repo_root).as_posix(),
        "sha256": sha256_file(output_path),
    }
