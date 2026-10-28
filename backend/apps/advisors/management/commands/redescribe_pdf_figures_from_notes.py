from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from django.core.management.base import BaseCommand, CommandParser

from apps.advisors.figure_description_pipeline import (
    FIGURE_DESCRIPTION_DIRNAME,
    figure_review_markdown_path,
    build_figure_review_gallery,
    default_figure_description_config,
    run_figure_redescription_from_review_notes,
)


class Command(BaseCommand):
    """Regenerate PDF figure descriptions that were flagged by human review notes."""

    help = (
        "Read manually added correction notes from figure_descriptions.md, rerun only the "
        "affected PDF pages with those notes injected into the multimodal prompt, and save "
        "the regenerated descriptions to a separate output directory."
    )

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument("--model", default=None, help="Vision-capable model name, e.g. gemma4:26b.")
        parser.add_argument(
            "--review-markdown",
            default=None,
            help="Markdown review file containing Correction Note lines. Defaults to the main figure review file.",
        )
        parser.add_argument(
            "--output-dir",
            default=None,
            help="Directory for regenerated descriptions. Defaults to backend/rag_index/figure_descriptions_redescribed.",
        )
        parser.add_argument("--force", action="store_true", help="Reprocess pages even if corrected outputs exist.")
        parser.add_argument("--no-save-images", action="store_true", help="Do not persist retained page screenshots.")
        parser.add_argument("--zoom", type=float, default=None, help="PDF render zoom. Higher improves OCR but costs more.")
        parser.add_argument("--timeout", type=int, default=None, help="Per-request timeout in seconds.")
        parser.add_argument("--max-tokens", type=int, default=None, help="Maximum response tokens per page.")

    def handle(self, *args, **options) -> None:
        base_config = default_figure_description_config()
        default_review_path = figure_review_markdown_path(base_config.output_dir)
        default_output_dir = base_config.output_dir.parent / f"{FIGURE_DESCRIPTION_DIRNAME}_redescribed"

        updates = {
            "output_dir": Path(options["output_dir"]) if options["output_dir"] else default_output_dir,
            "force": bool(options["force"]),
            "save_images": not bool(options["no_save_images"]),
        }
        if options["model"]:
            updates["model"] = options["model"]
        if options["zoom"] is not None:
            updates["zoom"] = float(options["zoom"])
        if options["timeout"] is not None:
            updates["timeout_seconds"] = int(options["timeout"])
        if options["max_tokens"] is not None:
            updates["max_tokens"] = int(options["max_tokens"])

        config = replace(base_config, **updates)
        review_markdown_path = Path(options["review_markdown"]) if options["review_markdown"] else default_review_path

        summary = run_figure_redescription_from_review_notes(
            config=config,
            review_markdown_path=review_markdown_path,
        )
        gallery_path = build_figure_review_gallery(config.output_dir, data_dir=config.data_dir, zoom=config.zoom)

        self.stdout.write(self.style.SUCCESS("PDF figure redescription from review notes completed."))
        self.stdout.write(f"Review Markdown: {review_markdown_path}")
        self.stdout.write(f"Output dir: {config.output_dir}")
        self.stdout.write(f"Review gallery: {gallery_path}")
        self.stdout.write(f"Model: {config.model}")
        self.stdout.write(f"Correction figures found: {summary.get('correction_figures_found')}")
        self.stdout.write(f"Correction pages found: {summary.get('correction_pages_found')}")
        self.stdout.write(f"Processed now: {summary.get('processed_now')}")
        self.stdout.write(f"Skipped existing: {summary.get('skipped_existing')}")
        self.stdout.write(f"Informative pages: {summary.get('informative_pages')}")
        self.stdout.write(f"Retained figures: {summary.get('retained_figures')}")
        self.stdout.write(f"Parse/errors: {summary.get('parse_errors')} / {len(summary.get('errors', []))}")
