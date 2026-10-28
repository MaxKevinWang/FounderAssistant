import json
from pathlib import Path

from apps.advisors.rag_pipeline import RAGConfig, _save_current_source_manifest
from apps.advisors.rag_web_ingestion_utils import _detect_manifest_delta


def test_stale_manifest_does_not_mark_already_indexed_sources_as_new():
    current = {
        "already-indexed.pdf": {"size_bytes": 10, "mtime_ns": 100},
        "new-law.pdf": {"size_bytes": 20, "mtime_ns": 200},
    }
    recorded = {}

    added, removed, changed = _detect_manifest_delta(
        current,
        recorded,
        indexed_paths={"already-indexed.pdf"},
    )

    assert added == ["new-law.pdf"]
    assert removed == []
    assert changed == []


def test_recorded_signature_change_is_still_detected_for_indexed_source():
    current = {"source.html": {"size_bytes": 11, "mtime_ns": 200}}
    recorded = {"source.html": {"size_bytes": 10, "mtime_ns": 100}}

    added, removed, changed = _detect_manifest_delta(
        current,
        recorded,
        indexed_paths={"source.html"},
    )

    assert added == []
    assert removed == []
    assert changed == ["source.html"]


def test_full_build_manifest_writer_records_current_non_archive_sources(tmp_path: Path):
    data_dir = tmp_path / "knowledge_base"
    index_dir = tmp_path / "index"
    source = data_dir / "legal_form_selection" / "source.txt"
    archived = data_dir / "archive" / "excluded.txt"
    source.parent.mkdir(parents=True)
    archived.parent.mkdir(parents=True)
    index_dir.mkdir()
    source.write_text("current", encoding="utf-8")
    archived.write_text("excluded", encoding="utf-8")
    config = RAGConfig(
        data_dir=data_dir,
        index_dir=index_dir,
        embedding_model="test-embedding",
        llm_provider="ollama",
        llm_base_url="http://localhost",
        llm_api_key="",
        verify_ssl=True,
        timeout_seconds=1,
        chunk_size=100,
        chunk_overlap=10,
        min_chunk_chars=10,
    )

    _save_current_source_manifest(config)

    payload = json.loads((index_dir / "source_manifest.json").read_text(encoding="utf-8"))
    assert payload["source_count"] == 1
    assert set(payload["sources"]) == {"legal_form_selection/source.txt"}
