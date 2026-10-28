"""Build the source-grounded allocation for optimization Configurations."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

from apps.evaluations.rageval_artifacts import (
    ArtifactValidationError,
    sha256_file,
    write_json_artifact,
)


DEFAULT_OUTPUT_PATH = Path(
    "docs/evaluation/optimization_dataset/configurations/"
    "optimization_configuration_allocation_v0.1.json"
)
DOCUMENT_MANIFEST_PATH = Path(
    "docs/evaluation/optimization_dataset/documents/"
    "optimization_document_manifest_v1.0.json"
)
SOURCE_SELECTION_PATH = Path(
    "docs/evaluation/optimization_dataset/source_selection/"
    "optimization_source_selection_v1.0.json"
)
EXCLUSION_REGISTRY_PATH = Path(
    "docs/evaluation/optimization_dataset/documents/"
    "document_candidate_audit_v0.1.json"
)
MARKET_SCOPE_PATH = Path(
    "docs/evaluation/optimization_dataset/documents/"
    "market_case_scope_review_v0.1.json"
)
CONFIGURATION_PROMPT_PATH = Path(
    "docs/evaluation/formal_dataset/prompts/"
    "configuration_generation_prompt_v1.1.md"
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


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(f"Cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ArtifactValidationError(f"Expected a JSON object: {path}")
    return payload


def _evidence(
    document_id: str,
    excerpt_count: int,
    *keywords: str,
) -> dict[str, Any]:
    return {
        "document_id": document_id,
        "excerpt_count": excerpt_count,
        "keywords": list(keywords),
        "preferred_block_ids": [],
    }


def _focused_evidence(
    document_id: str,
    excerpt_count: int,
    block_ids: list[str],
    *keywords: str,
) -> dict[str, Any]:
    """Require extraction from a semantically reviewed block subset."""

    return {
        "document_id": document_id,
        "excerpt_count": excerpt_count,
        "keywords": list(keywords),
        "preferred_block_ids": block_ids,
        "preferred_only": True,
    }


def _answerable(
    configuration_id: str,
    domain: str,
    question_type: str,
    primary_document_id: str,
    topic: str,
    value_slot: str,
    *evidence_plan: dict[str, Any],
) -> dict[str, Any]:
    return {
        "configuration_id": configuration_id,
        "domain": domain,
        "target_question_type": question_type,
        "answerability": "answerable",
        "primary_document_id": primary_document_id,
        "source_scope": (
            "single_document"
            if len({plan["document_id"] for plan in evidence_plan}) == 1
            else "multiple_documents"
        ),
        "topic": topic,
        "value_slot": value_slot,
        "evidence_plan": list(evidence_plan),
    }


def _unanswerable(
    configuration_id: str,
    domain: str,
    topic: str,
    value_slot: str,
    hypothetical_scenario: str,
    missing_facts: str,
    corpus_document_ids: list[str],
    *evidence_plan: dict[str, Any],
) -> dict[str, Any]:
    return {
        "configuration_id": configuration_id,
        "domain": domain,
        "target_question_type": "Unanswerable",
        "answerability": "unanswerable",
        "primary_document_id": None,
        "source_scope": "multiple_documents",
        "topic": topic,
        "value_slot": value_slot,
        "hypothetical_scenario": hypothetical_scenario,
        "missing_facts": missing_facts,
        "corpus_document_ids": corpus_document_ids,
        "evidence_plan": list(evidence_plan),
    }


def _legal_items() -> list[dict[str, Any]]:
    domain = "legal_form_selection"
    return [
        _answerable(
            "C-OPT-LFS-F-001", domain, "Factual", "OD-LFS-001",
            "Pflicht zur Gewerbeanmeldung und Ausnahmen", "legal_rule",
            _focused_evidence("OD-LFS-001", 2, ["OD-LFS-001-B007", "OD-LFS-001-B015"], "Gewerbe anmelden", "Tätigkeit"),
        ),
        _answerable(
            "C-OPT-LFS-F-002", domain, "Factual", "OD-LFS-002",
            "Merkmale und persönliche Haftung eines Einzelunternehmens", "legal_rule",
            _evidence("OD-LFS-002", 2, "Einzelunternehmen", "Haftung", "Privatvermögen"),
        ),
        _answerable(
            "C-OPT-LFS-F-003", domain, "Factual", "OD-LFS-003",
            "Abgrenzung freiberuflicher und gewerblicher Tätigkeit", "legal_rule",
            _evidence("OD-LFS-003", 2, "freiberufliche Tätigkeit", "Gewerbe", "Freiberufler"),
        ),
        _answerable(
            "C-OPT-LFS-F-004", domain, "Factual", "OD-LFS-004",
            "GmbH als Kapitalgesellschaft mit beschränkter Haftung", "legal_rule",
            _focused_evidence("OD-LFS-004", 2, ["OD-LFS-004-B005", "OD-LFS-004-B006"], "GmbH", "Stammkapital", "Privatvermögen"),
        ),
        _answerable(
            "C-OPT-LFS-F-005", domain, "Factual", "OD-LFS-005",
            "Zuständige Stellen für Gewerbe und freie Berufe", "legal_process",
            _focused_evidence("OD-LFS-005", 1, ["OD-LFS-005-B012"], "Finanzamt", "Freiberufler"),
            _focused_evidence("OD-LFS-005", 1, ["OD-LFS-005-B023"], "Gewerbeamt", "Gewerbe anmelden"),
        ),
        _answerable(
            "C-OPT-LFS-S-001", domain, "Summarization", "OD-LFS-002",
            "Vor- und Nachteile des Einzelunternehmens", "legal_decision",
            _evidence("OD-LFS-002", 3, "Einzelunternehmen", "Vorteil", "Nachteil", "Haftung"),
        ),
        _answerable(
            "C-OPT-LFS-S-002", domain, "Summarization", "OD-LFS-006",
            "Typische Haftungs- und Einlagefehler vor Registereintragung", "legal_decision",
            _focused_evidence("OD-LFS-006", 3, ["OD-LFS-006-B008"], "Haftung", "Handelsregister", "Einlage", "Stammkapital"),
        ),
        _answerable(
            "C-OPT-LFS-S-003", domain, "Summarization", "OD-LFS-007",
            "Besonderheiten einer Gründung in den freien Berufen", "legal_rule",
            _evidence("OD-LFS-007", 3, "freie Berufe", "Finanzamt", "Rechtsform", "Gewerbe"),
        ),
        _answerable(
            "C-OPT-LFS-MHR-001", domain, "Multi-hop Reasoning", "OD-LFS-001",
            "Folgen der Gewerbeanmeldung für weitere Behördenkontakte", "legal_process",
            _evidence("OD-LFS-001", 3, "Gewerbeamt", "Finanzamt", "IHK", "Berufsgenossenschaft"),
        ),
        _answerable(
            "C-OPT-LFS-MHR-002", domain, "Multi-hop Reasoning", "OD-LFS-008",
            "Rechtsform, Geschäftsführung und Haftung in Personengesellschaften", "legal_decision",
            _evidence("OD-LFS-008", 3, "Personengesellschaft", "Gesellschafter", "Haftung", "Geschäftsführung"),
        ),
        _answerable(
            "C-OPT-LFS-MHR-003", domain, "Multi-hop Reasoning", "OD-LFS-006",
            "Haftungsrisiken vor Registereintragung und bei fehlerhaften Einlagen", "legal_decision",
            _focused_evidence("OD-LFS-006", 3, ["OD-LFS-006-B008"], "Handelsregister", "haftet", "Einlage", "Haftung"),
        ),
        _answerable(
            "C-OPT-LFS-II-001", domain, "Information Integration", "OD-LFS-005",
            "Registrierungspflichten für Gewerbetreibende und Freiberufler", "legal_process",
            _evidence("OD-LFS-005", 2, "Gewerbeamt", "Finanzamt", "anmelden"),
            _focused_evidence("OD-LFS-001", 2, ["OD-LFS-001-B007", "OD-LFS-001-B015"], "Gewerbe anmelden", "Ausnahmen", "Tätigkeit"),
        ),
        _answerable(
            "C-OPT-LFS-II-002", domain, "Information Integration", "OD-LFS-008",
            "Haftung und Kapital bei Personen- und Kapitalgesellschaften", "legal_decision",
            _evidence("OD-LFS-008", 2, "Personengesellschaft", "Kapitalgesellschaft", "Haftung"),
            _evidence("OD-LFS-004", 2, "GmbH", "Stammkapital", "Privatvermögen"),
        ),
        _answerable(
            "C-OPT-LFS-II-003", domain, "Information Integration", "OD-LFS-007",
            "Kriterien und Registrierungsfolgen einer freiberuflichen Tätigkeit", "legal_rule",
            _evidence("OD-LFS-007", 2, "freie Berufe", "Gewerbe", "Finanzamt"),
            _evidence("OD-LFS-003", 2, "freiberufliche Tätigkeit", "Gewerbe", "anmelden"),
        ),
        _answerable(
            "C-OPT-LFS-NC-001", domain, "Numerical Comparison", "OD-LFS-002",
            "Umsatz- und Gewinngrenzen für einfache Buchführung", "legal_obligation",
            _evidence("OD-LFS-002", 2, "800.000 EUR", "80.000 EUR", "Bilanzierungspflicht"),
        ),
        _answerable(
            "C-OPT-LFS-NC-002", domain, "Numerical Comparison", "OD-LFS-004",
            "Stammkapital und ausgewählte Gründungskosten einer GmbH", "legal_decision",
            _focused_evidence("OD-LFS-004", 3, ["OD-LFS-004-B092", "OD-LFS-004-B093", "OD-LFS-004-B094"], "25.000 EUR", "Handelsregistereintrag", "Notarkosten", "Gründungskosten"),
        ),
        _answerable(
            "C-OPT-LFS-TS-001", domain, "Temporal Sequence", "OD-LFS-001",
            "Behördliche Schritte nach Vorbereitung der Gewerbeanmeldung", "legal_process",
            _focused_evidence("OD-LFS-001", 1, ["OD-LFS-001-B035"], "Unterlagen"),
            _focused_evidence("OD-LFS-001", 1, ["OD-LFS-001-B037"], "eigentliche Gewerbeanmeldung"),
            _focused_evidence("OD-LFS-001", 1, ["OD-LFS-001-B041"], "Finanzamt", "Fragebogen"),
            _focused_evidence("OD-LFS-001", 1, ["OD-LFS-001-B047"], "Berufsgenossenschaft"),
        ),
        _answerable(
            "C-OPT-LFS-TS-002", domain, "Temporal Sequence", "OD-LFS-003",
            "Anmeldung und steuerliche Erfassung einer freiberuflichen Tätigkeit", "legal_process",
            _focused_evidence("OD-LFS-003", 3, ["OD-LFS-003-B053"], "Finanzamt", "Fragebogen", "Steuernummer", "vier Wochen"),
        ),
        _unanswerable(
            "C-OPT-LFS-U-001", domain,
            "Verbindliche Rechtsformwahl ohne Kapital- und Haftungspräferenzen", "legal_decision",
            "Zwei Personen wollen gemeinsam ein digitales Beratungsunternehmen gründen und verlangen eine verbindliche Rechtsformempfehlung.",
            "Kapital, Beteiligungsverhältnis, Haftungsbereitschaft, Vertretungsregeln und Steuerlage",
            [f"OD-LFS-{index:03d}" for index in range(1, 9)],
            _evidence("OD-LFS-004", 2, "GmbH", "Stammkapital", "Haftung"),
            _evidence("OD-LFS-008", 2, "Personengesellschaft", "Gesellschafter", "Haftung"),
        ),
        _unanswerable(
            "C-OPT-LFS-U-002", domain,
            "Definitive Einordnung einer gemischten Tätigkeit als freier Beruf", "legal_rule",
            "Eine Person kombiniert individuelle Designleistungen, standardisierte Software-Abonnements und Warenverkauf, nennt aber keine Umsatz- oder Tätigkeitsschwerpunkte.",
            "Qualifikation, konkrete persönliche Leistung, Trennbarkeit, Umsatzanteile und prägender Tätigkeitsschwerpunkt",
            [f"OD-LFS-{index:03d}" for index in range(1, 9)],
            _evidence("OD-LFS-003", 2, "freiberufliche Tätigkeit", "Gewerbe", "Qualifikation"),
            _evidence("OD-LFS-007", 2, "freie Berufe", "Gewerbe", "Tätigkeit"),
        ),
    ]


def _funding_items() -> list[dict[str, Any]]:
    domain = "funding_preparation"
    return [
        _answerable("C-OPT-FP-F-001", domain, "Factual", "OD-FP-001", "Definition und Aussage des Break-even-Points", "funding_assessment", _focused_evidence("OD-FP-001", 2, ["OD-FP-001-B004", "OD-FP-001-B015"], "Break-even", "Gewinnschwelle", "Gewinne", "Verluste")),
        _answerable("C-OPT-FP-F-002", domain, "Factual", "OD-FP-002", "Liquidität und Zweck eines Liquiditätsplans", "funding_assessment", _evidence("OD-FP-002", 2, "Liquidität", "Liquiditätsplan", "Zahlungsfähigkeit")),
        _answerable("C-OPT-FP-F-003", domain, "Factual", "OD-FP-003", "Prüfung und Bewilligung des Gründungszuschusses", "funding_rule", _focused_evidence("OD-FP-003", 2, ["OD-FP-003-B017"], "Ermessen", "Fähigkeiten", "Kenntnisse", "bewilligt")),
        _answerable("C-OPT-FP-F-004", domain, "Factual", "OD-FP-004", "Zweck und Adressaten eines Businessplans", "funding_rule", _evidence("OD-FP-004", 2, "Businessplan", "Geschäftsidee", "Finanzierung")),
        _answerable("C-OPT-FP-F-005", domain, "Factual", "OD-FP-006", "Funktionen und Bestandteile eines Finanzplans", "funding_process", _focused_evidence("OD-FP-006", 1, ["OD-FP-006-B006"], "Geldgeberinnen", "Zahlen", "Risiken"), _focused_evidence("OD-FP-006", 1, ["OD-FP-006-B012"], "Zahlungsfähigkeit", "Liquiditätsplan")),
        _answerable("C-OPT-FP-S-001", domain, "Summarization", "OD-FP-002", "Eingaben und Unsicherheiten der Liquiditätsplanung", "funding_assessment", _focused_evidence("OD-FP-002", 1, ["OD-FP-002-B015"], "Zahlungseingänge", "Zahlungsausgänge"), _focused_evidence("OD-FP-002", 1, ["OD-FP-002-B025"], "nicht zahlungswirksam"), _focused_evidence("OD-FP-002", 1, ["OD-FP-002-B028"], "Annahmen", "Zahlungszeitpunkt")),
        _answerable("C-OPT-FP-S-002", domain, "Summarization", "OD-FP-005", "Preis-, Personal- und Kapitalannahmen im Businessplan", "funding_assessment", _focused_evidence("OD-FP-005", 3, ["OD-FP-005-B006", "OD-FP-005-B007", "OD-FP-005-B008"], "Preis", "Mitarbeiter", "Kapitalbedarf", "Kosten")),
        _answerable("C-OPT-FP-S-003", domain, "Summarization", "OD-FP-007", "Kreditarten und Auswahlkriterien für Selbstständige", "funding_comparison", _evidence("OD-FP-007", 3, "Kredit", "Förderkredit", "Mikrokredit", "Voraussetzungen")),
        _answerable("C-OPT-FP-MHR-001", domain, "Multi-hop Reasoning", "OD-FP-001", "Zusammenhang von Investition, Umsatz, Kosten und Break-even", "funding_assessment", _focused_evidence("OD-FP-001", 3, ["OD-FP-001-B004", "OD-FP-001-B010", "OD-FP-001-B013"], "Erstinvestitionen", "Umsatz", "Kosten", "Break-even")),
        _answerable("C-OPT-FP-MHR-002", domain, "Multi-hop Reasoning", "OD-FP-005", "Vom Markttest zu belastbaren Businessplan-Annahmen", "funding_assessment", _evidence("OD-FP-005", 3, "Tests", "Befragungen", "Feedback", "Annahmen")),
        _answerable("C-OPT-FP-MHR-003", domain, "Multi-hop Reasoning", "OD-FP-008", "Zusammenhang von Umsatz, Kosten, Kapitalbedarf und Liquidität", "funding_assessment", _focused_evidence("OD-FP-008", 3, ["OD-FP-008-B081", "OD-FP-008-B082", "OD-FP-008-B101", "OD-FP-008-B108"], "Umsatz", "Kosten", "Kapitalbedarf", "Liquidität")),
        _answerable("C-OPT-FP-II-001", domain, "Information Integration", "OD-FP-004", "Text- und Finanzteil eines prüfbaren Businessplans", "funding_process", _focused_evidence("OD-FP-004", 1, ["OD-FP-004-B030"], "Text", "Finanzteil"), _focused_evidence("OD-FP-008", 3, ["OD-FP-008-B017", "OD-FP-008-B043", "OD-FP-008-B081", "OD-FP-008-B101"], "Geschäftsidee", "Markt", "Zahlen", "Kapitalbedarf")),
        _answerable("C-OPT-FP-II-002", domain, "Information Integration", "OD-FP-006", "Finanzplan und Kreditprüfung als Vorbereitung einer Finanzierung", "funding_assessment", _evidence("OD-FP-006", 2, "Finanzplan", "Rentabilität", "Liquidität"), _evidence("OD-FP-007", 2, "Kredit", "Bonität", "Sicherheiten")),
        _answerable("C-OPT-FP-II-003", domain, "Information Integration", "OD-FP-007", "Förderentscheidung zwischen Zuschuss und Kredit", "funding_comparison", _focused_evidence("OD-FP-007", 2, ["OD-FP-007-B124", "OD-FP-007-B126"], "Förderkredit", "Hausbank"), _focused_evidence("OD-FP-003", 2, ["OD-FP-003-B005", "OD-FP-003-B017"], "nicht zurückzahlen", "Ermessen", "bewilligt")),
        _answerable("C-OPT-FP-NC-001", domain, "Numerical Comparison", "OD-FP-001", "Investition, monatlicher Überschuss und Break-even-Zeitpunkt", "funding_assessment", _evidence("OD-FP-001", 1, "10.000 EUR", "5.000 EUR", "4.500 EUR", "Monat 20")),
        _answerable("C-OPT-FP-NC-002", domain, "Numerical Comparison", "OD-FP-003", "Dauer und Leistungen der zwei Förderphasen", "funding_comparison", _evidence("OD-FP-003", 3, "15 Monate", "6 Monate", "9 Monate", "300 EUR")),
        _answerable("C-OPT-FP-TS-001", domain, "Temporal Sequence", "OD-FP-002", "Schrittweise Erstellung eines Liquiditätsplans", "funding_process", _focused_evidence("OD-FP-002", 4, ["OD-FP-002-B015", "OD-FP-002-B016", "OD-FP-002-B017", "OD-FP-002-B028", "OD-FP-002-B029", "OD-FP-002-B032"], "Zahlungseingänge", "Zahlungsausgänge", "Annahmen", "Monate")),
        _answerable("C-OPT-FP-TS-002", domain, "Temporal Sequence", "OD-FP-008", "Reihenfolge zentraler Businessplan-Kapitel bis zur Finanzplanung", "funding_process", _focused_evidence("OD-FP-008", 1, ["OD-FP-008-B013"], "Zusammenfassung"), _focused_evidence("OD-FP-008", 1, ["OD-FP-008-B017"], "Geschäftsidee"), _focused_evidence("OD-FP-008", 1, ["OD-FP-008-B043"], "Umfeld", "Businessplan"), _focused_evidence("OD-FP-008", 1, ["OD-FP-008-B081"], "zweiten Teil", "Zahlen")),
        _unanswerable("C-OPT-FP-U-001", domain, "Exakte Kreditzusage ohne Kapital- und Bonitätsdaten", "funding_assessment", "Ein Gründer verlangt den exakt bewilligten Kreditbetrag, nennt aber weder Kapitalbedarf noch Eigenmittel oder Bonitätsdaten.", "Kapitalbedarf, Eigenmittel, Sicherheiten, Bonität, Kapitaldienstfähigkeit und Bankentscheidung", [f"OD-FP-{index:03d}" for index in range(1, 9)], _focused_evidence("OD-FP-002", 2, ["OD-FP-002-B004", "OD-FP-002-B028"], "Zahlungsfähigkeit", "Annahmen", "Zahlungszeitpunkt"), _evidence("OD-FP-007", 2, "Kredit", "Bonität", "Sicherheiten")),
        _unanswerable("C-OPT-FP-U-002", domain, "Garantierter Gründungszuschuss ohne individuellen Prüfstand", "funding_rule", "Eine Person verlangt vor Antragstellung eine Garantie über Bewilligung, Gesamtdauer und Auszahlung, ohne Restanspruch oder Tragfähigkeitsnachweis zu nennen.", "Arbeitslosengeld-Restanspruch, hauptberufliche Selbstständigkeit, Tragfähigkeitsprüfung, Antragstermin und Ermessensentscheidung", [f"OD-FP-{index:03d}" for index in range(1, 9)], _evidence("OD-FP-003", 2, "Gründungszuschuss", "Voraussetzung", "Arbeitslosengeld"), _evidence("OD-FP-006", 2, "Gründungszuschuss", "Agentur für Arbeit", "selbstständig")),
    ]


def _market_items() -> list[dict[str, Any]]:
    domain = "market_validation"
    return [
        _answerable("C-OPT-MV-F-001", domain, "Factual", "OD-MV-001", "Informationsquellen für neue Geschäftsideen", "market_hypothesis", _focused_evidence("OD-MV-001", 1, ["OD-MV-001-B022"], "Magazinen"), _focused_evidence("OD-MV-001", 1, ["OD-MV-001-B024"], "Marktstudien")),
        _answerable("C-OPT-MV-F-002", domain, "Factual", "OD-MV-002", "Direkte Datenerhebung bei potenziellen Kunden", "market_finding", _focused_evidence("OD-MV-002", 2, ["OD-MV-002-B023", "OD-MV-002-B024", "OD-MV-002-B025"], "Daten erheben", "direkte", "Kunden", "Erkenntnisse")),
        _answerable("C-OPT-MV-F-003", domain, "Factual", "OD-MV-003", "Rolle einer Value Proposition für Kundenorientierung", "market_hypothesis", _focused_evidence("OD-MV-003", 2, ["OD-MV-003-B010", "OD-MV-003-B011"], "Value Proposition", "Nutzen", "Kunden")),
        _answerable("C-OPT-MV-F-004", domain, "Factual", "OD-MV-004", "Merkmale eines Minimum Viable Product", "market_process", _evidence("OD-MV-004", 2, "Minimum Viable Product", "Kernfunktionen", "Nutzen")),
        _answerable("C-OPT-MV-F-005", domain, "Factual", "OD-MV-005", "Nische und Positionierung eines Onlineshops", "market_decision", _focused_evidence("OD-MV-005", 2, ["OD-MV-005-B063", "OD-MV-005-B064", "OD-MV-005-B144"], "positionierst", "Nische", "Zielgruppe", "Wettbewerb")),
        _answerable("C-OPT-MV-S-001", domain, "Summarization", "OD-MV-001", "Nachfrage, Wettbewerb und persönliche Voraussetzungen einer Idee", "market_decision", _evidence("OD-MV-001", 3, "Nachfrage", "Wettbewerber", "Stärken", "Risiken")),
        _answerable("C-OPT-MV-S-002", domain, "Summarization", "OD-MV-006", "Differenzierung, Standort und Zielgruppe eines Second-Hand-Ladens", "market_decision", _evidence("OD-MV-006", 3, "Konkurrenz", "Zielgruppe", "Lage", "Angebot")),
        _answerable("C-OPT-MV-S-003", domain, "Summarization", "OD-MV-008", "Problem, Test, Markt und Zielgruppe eines Foodtruck-Konzepts", "market_process", _focused_evidence("OD-MV-008", 3, ["OD-MV-008-B025", "OD-MV-008-B028", "OD-MV-008-B038", "OD-MV-008-B047"], "Fastfood-Markt", "auszuprobieren", "Marktsituation", "Zielgruppe")),
        _answerable("C-OPT-MV-MHR-001", domain, "Multi-hop Reasoning", "OD-MV-001", "Vom Gründerprofil zur wettbewerbsfähigen Geschäftsidee", "market_decision", _focused_evidence("OD-MV-001", 3, ["OD-MV-001-B002", "OD-MV-001-B030", "OD-MV-001-B031", "OD-MV-001-B032"], "Stärken", "wettbewerbsfähig", "Nachfrage", "Wettbewerb")),
        _answerable("C-OPT-MV-MHR-002", domain, "Multi-hop Reasoning", "OD-MV-003", "Von Kundengesprächen zum belastbaren Nutzenversprechen", "market_process", _evidence("OD-MV-003", 3, "Kunden", "Prototypen", "Feedback", "Value Proposition")),
        _answerable("C-OPT-MV-MHR-003", domain, "Multi-hop Reasoning", "OD-MV-007", "Von Spezialisierung und Alleinstellungsmerkmal zur Zielgruppe", "market_decision", _evidence("OD-MV-007", 3, "Spezialisierung", "Alleinstellungsmerkmal", "Zielgruppe", "Kundschaft")),
        _answerable("C-OPT-MV-II-001", domain, "Information Integration", "OD-MV-005", "Verbindung von Nutzenversprechen und Nischenpositionierung", "market_decision", _focused_evidence("OD-MV-005", 2, ["OD-MV-005-B063", "OD-MV-005-B064", "OD-MV-005-B065"], "Nische", "Positionierung", "Zielgruppe"), _focused_evidence("OD-MV-003", 2, ["OD-MV-003-B036", "OD-MV-003-B043"], "Value Proposition", "Kunden", "Bedürfnisse")),
        _answerable("C-OPT-MV-II-002", domain, "Information Integration", "OD-MV-006", "Zielgruppenanalyse für Standort- und Kanalentscheidungen", "market_decision", _evidence("OD-MV-006", 2, "Lage", "Zielgruppe", "Internet", "Kunden"), _evidence("OD-MV-002", 2, "Zielgruppe", "Kaufverhalten", "Vertriebskanal")),
        _answerable("C-OPT-MV-II-003", domain, "Information Integration", "OD-MV-007", "Wettbewerbsnische und kundenorientiertes Leistungsversprechen", "market_decision", _evidence("OD-MV-007", 2, "Nische", "Wettbewerber", "Kundschaft"), _evidence("OD-MV-003", 2, "Kundschaft", "Nutzenversprechen", "Konkurrenz")),
        _answerable("C-OPT-MV-NC-001", domain, "Numerical Comparison", "OD-MV-002", "Drei Analyseschritte und vier zentrale Ergebnispunkte", "market_finding", _focused_evidence("OD-MV-002", 2, ["OD-MV-002-B029", "OD-MV-002-B038"], "drei Schritten", "vier zentralen Punkte")),
        _answerable("C-OPT-MV-NC-002", domain, "Numerical Comparison", "OD-MV-002", "Drei B2B-Besonderheiten und drei MAN-Kriterien", "market_finding", _focused_evidence("OD-MV-002", 6, ["OD-MV-002-B043", "OD-MV-002-B044", "OD-MV-002-B045", "OD-MV-002-B047", "OD-MV-002-B048", "OD-MV-002-B049"], "Kaufentscheidung", "Entscheidungsprozesse", "komplex", "Mittel", "Autorität", "Notwendigkeit")),
        _answerable("C-OPT-MV-TS-001", domain, "Temporal Sequence", "OD-MV-004", "Build-measure-learn-Zyklus eines MVP", "market_process", _focused_evidence("OD-MV-004", 3, ["OD-MV-004-B018", "OD-MV-004-B019", "OD-MV-004-B020"], "build", "measure", "learn")),
        _answerable("C-OPT-MV-TS-002", domain, "Temporal Sequence", "OD-MV-008", "Von Ideensammlung und Test zur Markt- und Zielgruppenprüfung", "market_process", _focused_evidence("OD-MV-008", 1, ["OD-MV-008-B024"], "Ideensammlung"), _focused_evidence("OD-MV-008", 1, ["OD-MV-008-B028"], "auszuprobieren"), _focused_evidence("OD-MV-008", 1, ["OD-MV-008-B038"], "Marksituation", "Marktanalyse"), _focused_evidence("OD-MV-008", 1, ["OD-MV-008-B047"], "Zielgruppe")),
        _unanswerable("C-OPT-MV-U-001", domain, "Exakte Nachfrage ohne Segmentgröße oder Markttest", "market_finding", "Ein lokales Abo-Angebot soll eine exakte Zahl zahlender Kunden erreichen, ohne Segmentgröße, Reichweite oder Testergebnisse zu kennen.", "Zielgruppengröße, erreichbare Nachfrage, Conversion, Preisakzeptanz und beobachtete Testergebnisse", [f"OD-MV-{index:03d}" for index in range(1, 9)], _evidence("OD-MV-002", 2, "Daten", "Kunden", "Zielgruppe"), _evidence("OD-MV-004", 2, "Feedback", "testen", "Zielgruppe")),
        _unanswerable("C-OPT-MV-U-002", domain, "Definitiv beste Positionierung ohne Kunden- und Wettbewerbsdaten", "market_decision", "Eine neue Agentur verlangt die garantiert beste Nische und Positionierung, hat aber weder Kundengespräche noch Wettbewerbsanalyse durchgeführt.", "Zielsegment, Kundenprobleme, beobachtetes Feedback, Wettbewerber und getestete Nutzenhypothese", [f"OD-MV-{index:03d}" for index in range(1, 9)], _focused_evidence("OD-MV-003", 2, ["OD-MV-003-B036", "OD-MV-003-B038"], "Kunden", "Feedback", "Bedürfnisse"), _evidence("OD-MV-007", 2, "Nische", "Wettbewerber", "Zielgruppe")),
    ]


def prepare_optimization_configuration_allocation(
    *,
    repo_root: Path,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create the reviewed 60-item allocation with hash-pinned evidence policy."""

    repo_root = repo_root.resolve()
    if not output_path.is_absolute():
        output_path = repo_root / output_path

    manifest_path = repo_root / DOCUMENT_MANIFEST_PATH
    selection_path = repo_root / SOURCE_SELECTION_PATH
    manifest = _read_json_object(manifest_path)
    selection = _read_json_object(selection_path)
    if manifest.get("status") != "frozen" or selection.get("status") != "frozen":
        raise ArtifactValidationError(
            "Optimization Documents and source selection must be frozen first."
        )

    items = _legal_items() + _funding_items() + _market_items()
    if len(items) != 60 or len({item["configuration_id"] for item in items}) != 60:
        raise ArtifactValidationError("Optimization allocation must contain 60 unique items.")
    manifest_document_ids = {
        declaration["document_id"] for declaration in manifest["documents"]
    }
    referenced_document_ids = {
        plan["document_id"]
        for item in items
        for plan in item["evidence_plan"]
    } | {
        document_id
        for item in items
        for document_id in item.get("corpus_document_ids", [])
    }
    if referenced_document_ids != manifest_document_ids:
        raise ArtifactValidationError(
            "Optimization allocation does not resolve exactly the frozen Documents."
        )

    for domain in ("legal_form_selection", "funding_preparation", "market_validation"):
        domain_items = [item for item in items if item["domain"] == domain]
        if len(domain_items) != 20:
            raise ArtifactValidationError(f"{domain} must contain 20 items.")
        counts = Counter(item["target_question_type"] for item in domain_items)
        if dict(counts) != EXPECTED_TYPE_COUNTS:
            raise ArtifactValidationError(
                f"Question-type allocation mismatch for {domain}: {counts}"
            )
        primary_counts = Counter(
            item["primary_document_id"]
            for item in domain_items
            if item["primary_document_id"] is not None
        )
        if max(primary_counts.values()) > 3 or len(primary_counts) != 8:
            raise ArtifactValidationError(
                f"Primary-document allocation is unbalanced for {domain}: "
                f"{primary_counts}"
            )

    allocation = {
        "allocation_id": "optimization_configuration_allocation_v0.1",
        "version": "0.1",
        "status": "review_candidate",
        "intended_split": "optimization",
        "created_at": "2026-08-17",
        "output_directory": (
            "docs/evaluation/optimization_dataset/configurations/candidates"
        ),
        "generation_manifest_id": (
            "optimization_configuration_generation_manifest_v0.1"
        ),
        "unanswerable_audit_id": (
            "optimization_unanswerable_configuration_scope_v0.1"
        ),
        "review_title": "Optimization Configuration evidence review v0.1",
        "generation_method": (
            "Codex-authored optimization allocation with deterministic exact-source "
            "evidence extraction and pre-selection leakage/domain-scope gates"
        ),
        "dependencies": {
            "source_selection_path": SOURCE_SELECTION_PATH.as_posix(),
            "source_selection_sha256": sha256_file(selection_path),
            "document_manifest_path": DOCUMENT_MANIFEST_PATH.as_posix(),
            "document_manifest_sha256": sha256_file(manifest_path),
            "configuration_prompt_path": CONFIGURATION_PROMPT_PATH.as_posix(),
            "configuration_prompt_sha256": sha256_file(
                repo_root / CONFIGURATION_PROMPT_PATH
            ),
            "evidence_exclusion_registry_path": EXCLUSION_REGISTRY_PATH.as_posix(),
            "evidence_exclusion_registry_sha256": sha256_file(
                repo_root / EXCLUSION_REGISTRY_PATH
            ),
            "market_case_scope_review_path": MARKET_SCOPE_PATH.as_posix(),
            "market_case_scope_review_sha256": sha256_file(
                repo_root / MARKET_SCOPE_PATH
            ),
        },
        "evidence_policy": {
            "prior_formal_evidence": (
                "Exclude all 55 mapped blocks before evidence ranking and from "
                "unanswerable corpus declarations."
            ),
            "market_sector_cases": (
                "Intersect every sector-case Document with its reviewed market-only "
                "allowed_block_ids before evidence ranking."
            ),
            "configuration_status": (
                "Generated records remain development_draft review candidates; this "
                "allocation does not authorize Configuration freeze."
            ),
        },
        "targets": {
            "configuration_count": 60,
            "per_domain_count": 20,
            "question_type_per_domain": EXPECTED_TYPE_COUNTS,
            "maximum_primary_configuration_per_document": 3,
        },
        "items": items,
    }

    if output_path.exists():
        existing = _read_json_object(output_path)
        if existing == allocation:
            write_state = "unchanged"
        elif not overwrite:
            raise ArtifactValidationError(
                f"Allocation differs from existing output; use --overwrite: {output_path}"
            )
        else:
            write_json_artifact(output_path, allocation)
            write_state = "overwritten"
    else:
        write_json_artifact(output_path, allocation)
        write_state = "created"

    return {
        "allocation_path": output_path.relative_to(repo_root).as_posix(),
        "configuration_count": len(items),
        "write_state": write_state,
    }
