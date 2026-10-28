from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from django.core.management.base import BaseCommand, CommandParser

from apps.advisors.html_image_candidate_pipeline import (
    default_html_image_candidate_config,
    run_html_image_candidate_pipeline,
)


class Command(BaseCommand):
    """Extract reviewable image candidates from saved HTML corpus files."""

    help = (
        "Scan knowledge_base/**/*.html for <img>/<picture> metadata and create review artifacts "
        "under backend/rag_index/figure_descriptions. This does not call an LLM and "
        "does not modify the RAG index."
    )

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument("--output-dir", default=None, help="Defaults to backend/rag_index/figure_descriptions.")
        parser.add_argument("--include-archive", action="store_true", help="Also scan knowledge_base/archive HTML files.")
        parser.add_argument("--source", default="", help="Only scan HTML files whose relative path contains this text.")
        parser.add_argument("--max-files", type=int, default=None, help="Stop after this many HTML files.")

    def handle(self, *args, **options) -> None:
        config = default_html_image_candidate_config()
        updates = {
            "include_archive": bool(options["include_archive"]),
            "source_filter": options["source"] or "",
            "max_files": options["max_files"],
        }
        if options["output_dir"]:
            updates["output_dir"] = Path(options["output_dir"])
        config = replace(config, **updates)

        summary = run_html_image_candidate_pipeline(config)

        self.stdout.write(self.style.SUCCESS("HTML image candidate extraction completed."))
        self.stdout.write(f"Output dir: {config.output_dir}")
        self.stdout.write(f"HTML files scanned: {summary.get('html_files_scanned')}")
        self.stdout.write(f"Image candidates: {summary.get('image_candidates')}")
        self.stdout.write(f"By label: {summary.get('by_heuristic_label')}")
        self.stdout.write(f"By domain: {summary.get('by_domain')}")
        self.stdout.write(f"Parse errors: {len(summary.get('parse_errors', []))}")
