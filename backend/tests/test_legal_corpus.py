from pathlib import Path

from apps.advisors.legal_corpus import (
    _LineRecord,
    _best_occurrence,
    _find_occurrences,
    extract_legal_pdf,
    get_legal_source_profile,
    load_legal_scope,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPOSITORY_ROOT / "knowledge_base"


def test_manifest_identifies_correct_partnership_and_founder_grant_laws():
    _, profiles = load_legal_scope()

    assert "legal_form_selection/primary_law/company_and_partnership_law/PartGG.pdf" in profiles
    assert "funding_preparation/primary_law/social_law/SGB_III.pdf" in profiles
    assert all("PartG_Parteiengesetz" not in source_path for source_path in profiles)
    assert all("SGB_I_Allgemeiner_Teil" not in source_path for source_path in profiles)


def test_partgg_extraction_drops_front_matter_and_repealed_section():
    relative_path = "legal_form_selection/primary_law/company_and_partnership_law/PartGG.pdf"
    profile = get_legal_source_profile(relative_path)
    assert profile is not None

    result = extract_legal_pdf(
        DATA_DIR / relative_path,
        profile,
        chunk_size=1200,
        chunk_overlap=180,
    )

    assert "Vollzitat" not in result.text
    assert "Ausfertigungsdatum" not in result.text
    assert result.audit["skipped_repealed_sections"] == ["3"]
    assert result.audit["front_matter_or_toc_indexed"] is False
    assert {chunk.metadata["law_section"] for chunk in result.chunks} == {
        "§ 1",
        "§ 2",
        "§ 4",
        "§ 5",
        "§ 6",
        "§ 7",
        "§ 8",
        "§ 9",
        "§ 10",
    }
    first = next(chunk for chunk in result.chunks if chunk.metadata["law_section"] == "§ 1")
    assert "Gesellschaft bürgerlichen Rechts" in first.section_title
    assert first.metadata["source_quality"] == "primary_law"
    assert first.metadata["canonical_url"].endswith("/partgg/PartGG.pdf")


def test_sgb_iii_scope_emits_only_founder_grant_sections():
    relative_path = "funding_preparation/primary_law/social_law/SGB_III.pdf"
    profile = get_legal_source_profile(relative_path)
    assert profile is not None

    result = extract_legal_pdf(
        DATA_DIR / relative_path,
        profile,
        chunk_size=1200,
        chunk_overlap=180,
    )

    assert {chunk.metadata["law_section"] for chunk in result.chunks} == {"§ 93", "§ 94"}
    assert "§ 92" not in result.text
    assert "§ 95" not in result.text
    assert "Sechster Abschnitt" not in result.text


def test_hwo_appendices_keep_trade_lists_without_publication_note():
    relative_path = (
        "legal_form_selection/primary_law/"
        "business_registration_and_professional_law/HwO.pdf"
    )
    profile = get_legal_source_profile(relative_path)
    assert profile is not None

    result = extract_legal_pdf(
        DATA_DIR / relative_path,
        profile,
        chunk_size=1200,
        chunk_overlap=180,
    )

    appendices = {
        unit.unit_id: unit.text
        for unit in result.units
        if unit.unit_type == "appendix"
    }
    assert "Friseure" in appendices["A"]
    assert "Kosmetiker" in appendices["B"]
    assert "Handwerksähnliche Gewerbe" in appendices["B"]
    assert "Fundstelle:" not in appendices["A"]
    assert "Fundstelle:" not in appendices["B"]


def test_body_occurrence_wins_over_short_table_of_contents_entry():
    profile = get_legal_source_profile(
        "legal_form_selection/primary_law/company_and_partnership_law/PartGG.pdf"
    )
    assert profile is not None
    records = [
        _LineRecord("§ 1 Example title", 1),
        _LineRecord("§ 2 Next title", 1),
        _LineRecord("§ 1 Example title", 2),
        _LineRecord("(1) This is the complete substantive rule and not a contents entry.", 2),
        _LineRecord("(2) It contains a second statutory paragraph.", 2),
        _LineRecord("§ 2 Next title", 2),
        _LineRecord("The next rule begins here.", 2),
    ]

    sections, _ = _find_occurrences(records, profile)
    selected = _best_occurrence(sections["1"])

    assert selected.page_start == 2
    assert "complete substantive rule" in selected.text
