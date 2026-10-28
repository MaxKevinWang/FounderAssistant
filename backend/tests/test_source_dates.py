"""Collection and import dates are not publication dates."""
import json
from types import SimpleNamespace
from apps.advisors import rag_pipeline as rag


def test_crawler_dates(tmp_path):
    source = tmp_path / "page.html"
    stamp = "2026-06-17T12:00:00Z"
    source.write_text(f"fetched_at_utc: {stamp}\nText", encoding="utf-8")
    assert rag._extract_publication_date(source) == ""
    assert rag._extract_collection_date(source) == stamp
    assert rag._extract_collection_date(tmp_path / "guide.pdf") == ""
    metadata = {"data_dir": str(tmp_path), "chunks": [
        {"source_path": "page.html", "publication_date": stamp},
        {"source_path": "page.html", "publication_date": "2025-03"}]}
    items = rag._normalize_index_metadata(metadata)["chunks"]
    assert items[0]["publication_date"] == ""
    assert items[0]["retrieved_at"] == stamp
    assert items[1]["publication_date"] == "2025-03"


def test_legacy_compact_crawler_date_is_migrated_without_original_data_dir():
    stamp = "20260622T133205Z"
    source_path = (
        "market_validation/gruenderplattform_startup_strategy/"
        f"gruenderplattform.de__design-thinking__{stamp}.html"
    )
    metadata = {
        "data_dir": "C:/obsolete/data/location",
        "chunks": [
            {
                "source_path": source_path,
                "source_name": f"gruenderplattform.de__design-thinking__{stamp}",
                "publication_date": stamp,
            }
        ],
    }

    item = rag._normalize_index_metadata(metadata)["chunks"][0]

    assert item["publication_date"] == ""
    assert item["retrieved_at"] == "2026-06-22T13:32:05Z"


def test_historical_pdf_dates():
    metadata = {"generated_at": "2026-07-01", "chunks": [
        {"source_path": "a.pdf"},
        {"source_path": "b.pdf", "imported_at": "2026-01-01", "retrieved_at": "2025-12-01"}]}
    items = rag._normalize_index_metadata(metadata)["chunks"]
    assert items[0]["imported_at"] == "2026-07-01"
    assert items[0]["retrieved_at"] == items[0]["publication_date"] == ""
    assert items[1]["imported_at"] == "2026-01-01"
    assert items[1]["retrieved_at"] == "2025-12-01"
    unknown = rag._normalize_index_metadata({"chunks": [{"source_path": "a.pdf"}]})
    assert unknown["chunks"][0]["imported_at"] == ""


def test_pdf_import_date_survives_rebuild(tmp_path, monkeypatch):
    index = tmp_path / "index"
    index.mkdir()
    source = tmp_path / "guide.pdf"
    source.touch()
    config = SimpleNamespace(data_dir=tmp_path, index_dir=index, chunk_size=1000, chunk_overlap=50, min_chunk_chars=10)
    monkeypatch.setattr(rag, "_load_reviewed_figure_descriptions_by_source", lambda config: {})
    monkeypatch.setattr(rag, "_extract_corpus_document", lambda *a, **kw: SimpleNamespace(
        text="A useful paragraph about starting a company and preparing a business plan.",
        page_spans=[], structured_chunks=[], metadata={}))
    chunks, documents = rag._build_chunks_for_source_files(config, [source])
    date = documents[0]["imported_at"]
    assert date and chunks[0].imported_at == date
    assert chunks[0].publication_date == chunks[0].retrieved_at == ""
    (index / rag.INDEX_METADATA_FILENAME).write_text(json.dumps({"documents": documents}), encoding="utf-8")
    rebuilt, _ = rag._build_chunks_for_source_files(config, [source])
    assert rebuilt[0].imported_at == date


def test_citation_dates():
    hit = rag._chunk_hit_payload(1, 0.9, {"chunk_id": "x", "domain": "funding", "source_path": "a.pdf",
        "source_name": "a", "chunk_index": 0, "text": "sample", "publication_date": "2025",
        "retrieved_at": "2026-01-01", "imported_at": "2026-02-01"})
    citation = rag._citation_payload([hit])[0]
    assert citation["publication_date"] == "2025"
    assert citation["retrieved_at"] == "2026-01-01"
    assert citation["imported_at"] == "2026-02-01"
