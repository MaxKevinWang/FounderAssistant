"""CLI command for incremental RAG indexing (new data only)."""

from __future__ import annotations

from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from apps.advisors.rag_pipeline import RAGError
from apps.advisors.rag_web_ingestion_utils import build_rag_index_for_new_documents


class Command(BaseCommand):
    """Django management wrapper around incremental index utility."""

    help = "Index only new source files; auto full-rebuild when required for consistency."

    def add_arguments(self, parser):
        parser.add_argument("--data-dir", type=str, default=None, help="Source data directory path.")
        parser.add_argument("--index-dir", type=str, default=None, help="Output index directory path.")
        parser.add_argument("--embedding-model", type=str, default=None, help="Local embedding model name.")
        parser.add_argument("--chunk-size", type=int, default=None, help="Chunk size in characters.")
        parser.add_argument("--chunk-overlap", type=int, default=None, help="Chunk overlap in characters.")
        parser.add_argument("--min-chunk-chars", type=int, default=None, help="Minimum chunk characters.")

    def handle(self, *args, **options):
        overrides = {}
        if options.get("data_dir"):
            overrides["data_dir"] = Path(options["data_dir"])
        if options.get("index_dir"):
            overrides["index_dir"] = Path(options["index_dir"])
        if options.get("embedding_model"):
            overrides["embedding_model"] = options["embedding_model"]
        if options.get("chunk_size") is not None:
            overrides["chunk_size"] = options["chunk_size"]
        if options.get("chunk_overlap") is not None:
            overrides["chunk_overlap"] = options["chunk_overlap"]
        if options.get("min_chunk_chars") is not None:
            overrides["min_chunk_chars"] = options["min_chunk_chars"]

        try:
            result = build_rag_index_for_new_documents(overrides if overrides else None)
        except RAGError as exc:
            raise CommandError(str(exc)) from exc

        self.stdout.write(self.style.SUCCESS("Incremental RAG indexing completed."))
        self.stdout.write(f"Mode: {result.get('mode')}")
        self.stdout.write(f"Reason: {result.get('reason', '')}")
        self.stdout.write(f"Documents: {result.get('document_count')}")
        self.stdout.write(f"Chunks: {result.get('chunk_count')}")
        self.stdout.write(f"New documents: {result.get('new_document_count', 0)}")
        self.stdout.write(f"New chunks: {result.get('new_chunk_count', 0)}")
        self.stdout.write(f"Duration: {result.get('duration_ms', 0)} ms")
