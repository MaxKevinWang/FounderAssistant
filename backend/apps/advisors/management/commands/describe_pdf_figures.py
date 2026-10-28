from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from django.core.management.base import BaseCommand, CommandParser

from apps.advisors.figure_description_pipeline import (
    default_figure_description_config,
    run_figure_description_pipeline,
)


class Command(BaseCommand):
    """Generate reviewable multimodal descriptions for informative PDF figures."""

    help = (
        "Render PDF pages, use a multimodal model to identify informative figures, "
        "and save reviewable descriptions under backend/rag_index/figure_descriptions."
    )

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument("--model", default=None, help="Vision-capable model name, e.g. gemma4:e2b.")
        parser.add_argument("--output-dir", default=None, help="Output directory for page results and summaries.")
        parser.add_argument("--include-archive", action="store_true", help="Also process knowledge_base/archive PDFs.")
        parser.add_argument("--force", action="store_true", help="Reprocess pages even if page result files exist.")
        parser.add_argument("--no-save-images", action="store_true", help="Do not persist retained page screenshots.")
        parser.add_argument("--source", default="", help="Only process PDFs whose relative path contains this text.")
        parser.add_argument("--page", type=int, default=None, help="Only process this 1-based page number.")
        parser.add_argument("--max-pages", type=int, default=None, help="Stop after processing this many new pages.")
        parser.add_argument("--zoom", type=float, default=None, help="PDF render zoom. Higher improves OCR but costs more.")
        parser.add_argument("--timeout", type=int, default=None, help="Per-request timeout in seconds.")
        parser.add_argument("--max-tokens", type=int, default=None, help="Maximum response tokens per page.")

    def handle(self, *args, **options) -> None:
        config = default_figure_description_config()
        updates = {
            "include_archive": bool(options["include_archive"]),
            "force": bool(options["force"]),
            "save_images": not bool(options["no_save_images"]),
            "source_filter": options["source"] or "",
            "page_filter": options["page"],
            "max_pages": options["max_pages"],
        }
        if options["model"]:
            updates["model"] = options["model"]
        if options["output_dir"]:
            updates["output_dir"] = Path(options["output_dir"])
        if options["zoom"] is not None:
            updates["zoom"] = float(options["zoom"])
        if options["timeout"] is not None:
            updates["timeout_seconds"] = int(options["timeout"])
        if options["max_tokens"] is not None:
            updates["max_tokens"] = int(options["max_tokens"])

        config = replace(config, **updates)
        summary = run_figure_description_pipeline(config)

        self.stdout.write(self.style.SUCCESS("PDF figure description pipeline completed."))
        self.stdout.write(f"Output dir: {config.output_dir}")
        self.stdout.write(f"Model: {config.model}")
        self.stdout.write(f"Processed pages total: {summary.get('processed_pages')}")
        self.stdout.write(f"Processed now: {summary.get('processed_now')}")
        self.stdout.write(f"Skipped existing: {summary.get('skipped_existing')}")
        self.stdout.write(f"Informative pages: {summary.get('informative_pages')}")
        self.stdout.write(f"Retained figures: {summary.get('retained_figures')}")
        self.stdout.write(f"Parse/errors: {summary.get('parse_errors')} / {len(summary.get('errors', []))}")
