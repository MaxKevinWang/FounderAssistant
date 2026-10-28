"""Create a legal-section extraction audit without calling an embedding model."""

from __future__ import annotations

import json
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from apps.advisors.legal_corpus import LegalCorpusError, build_legal_corpus_audit
from apps.advisors.rag_pipeline import _resolve_config


class Command(BaseCommand):
    """Validate selected statutes and report exactly what legal chunks would enter RAG."""

    help = "Audit selected legal sections and chunk counts without generating embeddings."

    def add_arguments(self, parser):
        parser.add_argument("--data-dir", type=str, default=None, help="Corpus data directory.")
        parser.add_argument("--output", type=str, default=None, help="Audit JSON output path.")
        parser.add_argument("--chunk-size", type=int, default=None, help="Legal chunk size in characters.")
        parser.add_argument("--chunk-overlap", type=int, default=None, help="Overlap inside oversized sections.")

    def handle(self, *args, **options):
        config_overrides = {}
        if options.get("data_dir"):
            config_overrides["data_dir"] = Path(options["data_dir"])
        if options.get("chunk_size") is not None:
            config_overrides["chunk_size"] = options["chunk_size"]
        if options.get("chunk_overlap") is not None:
            config_overrides["chunk_overlap"] = options["chunk_overlap"]
        config = _resolve_config(config_overrides or None)

        default_output = (
            Path(settings.BASE_DIR).parent / "docs" / "research" / "legal_corpus_chunk_audit_v1.json"
        )
        output_path = Path(options["output"]) if options.get("output") else default_output
        if not output_path.is_absolute():
            output_path = Path.cwd() / output_path

        try:
            audit = build_legal_corpus_audit(
                config.data_dir,
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
            )
        except LegalCorpusError as exc:
            raise CommandError(str(exc)) from exc

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

        errors = [document for document in audit["documents"] if document.get("error")]
        missing = [
            document
            for document in audit["documents"]
            if document.get("missing_selected_sections") or document.get("missing_selected_appendices")
        ]
        self.stdout.write(self.style.SUCCESS("Legal corpus audit completed."))
        self.stdout.write(f"Output: {output_path}")
        self.stdout.write(f"Documents: {audit['successful_document_count']} / {audit['document_count']}")
        self.stdout.write(f"Selected sections: {audit['selected_section_count']}")
        self.stdout.write(f"Chunks: {audit['chunk_count']}")
        if errors or missing:
            raise CommandError(
                f"Audit found {len(errors)} missing files and {len(missing)} documents with missing selected units."
            )
