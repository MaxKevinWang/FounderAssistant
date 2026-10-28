"""CLI command for offline RAG index construction.

This command is the reproducible entrypoint used in local experiments:
- reads corpus documents from data directory,
- runs chunking + embedding,
- writes `metadata.json` and `embeddings.npy`.
"""

from __future__ import annotations

from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from apps.advisors.rag_pipeline import RAGError, build_rag_index


class Command(BaseCommand):
    """Django management wrapper around `rag_pipeline.build_rag_index`."""

    help = "Extract documents, chunk text, embed chunks, and build local RAG index."

    def add_arguments(self, parser):
        """Expose optional pipeline overrides for quick experimentation."""

        parser.add_argument("--data-dir", type=str, default=None, help="Source data directory path.")
        parser.add_argument("--index-dir", type=str, default=None, help="Output index directory path.")
        parser.add_argument("--embedding-model", type=str, default=None, help="Local embedding model name.")
        parser.add_argument("--chunk-size", type=int, default=None, help="Chunk size in characters.")
        parser.add_argument("--chunk-overlap", type=int, default=None, help="Chunk overlap in characters.")
        parser.add_argument("--min-chunk-chars", type=int, default=None, help="Minimum chunk characters.")

    def handle(self, *args, **options):
        """Translate CLI options to pipeline overrides and print build summary."""

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
            result = build_rag_index(overrides if overrides else None)
        except RAGError as exc:
            raise CommandError(str(exc)) from exc

        self.stdout.write(self.style.SUCCESS("RAG index build completed."))
        self.stdout.write(f"Data dir: {result['data_dir']}")
        self.stdout.write(f"Index dir: {result['index_dir']}")
        self.stdout.write(f"Embedding model: {result['embedding_model']}")
        self.stdout.write(f"Documents: {result['document_count']}")
        self.stdout.write(f"Chunks: {result['chunk_count']}")
        self.stdout.write(f"Duration: {result['duration_ms']} ms")
