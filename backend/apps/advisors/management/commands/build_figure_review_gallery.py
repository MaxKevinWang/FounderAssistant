from __future__ import annotations

from pathlib import Path

from django.core.management.base import BaseCommand, CommandParser

from apps.advisors.figure_description_pipeline import (
    build_figure_review_gallery,
    default_figure_description_config,
)


class Command(BaseCommand):
    """Build a local HTML gallery for auditing generated PDF figure descriptions."""

    help = (
        "Create figure_review_gallery.html from page-level figure description results. "
        "This does not call any LLM; it only combines existing JSON results with page images."
    )

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument(
            "--output-dir",
            default=None,
            help="Figure description output directory. Defaults to backend/rag_index/figure_descriptions.",
        )
        parser.add_argument(
            "--no-render-missing-images",
            action="store_true",
            help="Do not render missing review page images from the source PDFs.",
        )
        parser.add_argument("--zoom", type=float, default=1.6, help="PDF render zoom for missing review images.")

    def handle(self, *args, **options) -> None:
        config = default_figure_description_config()
        output_dir = Path(options["output_dir"]) if options["output_dir"] else config.output_dir
        data_dir = None if options["no_render_missing_images"] else config.data_dir
        gallery_path = build_figure_review_gallery(output_dir, data_dir=data_dir, zoom=float(options["zoom"]))

        self.stdout.write(self.style.SUCCESS("PDF figure review gallery completed."))
        self.stdout.write(f"Output dir: {output_dir}")
        self.stdout.write(f"Review gallery: {gallery_path}")
