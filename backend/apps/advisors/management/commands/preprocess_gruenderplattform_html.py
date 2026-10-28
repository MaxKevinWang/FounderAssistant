"""Preprocess cleaned Gruenderplattform HTML snapshots before RAG indexing."""

from __future__ import annotations

from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand

from apps.advisors.gruenderplattform_html_preprocessing import (
    preprocess_gruenderplattform_html_files,
    write_preprocess_report,
)


def _default_data_dir() -> Path:
    configured = getattr(settings, "LOCAL_RAG_DATA_DIR", None)
    return Path(configured) if configured else Path(settings.BASE_DIR).parent / "knowledge_base"


def _default_report_path() -> Path:
    return Path(settings.BASE_DIR).parent / "docs" / "crawl_states" / "gruenderplattform_html_preprocess_report.json"


class Command(BaseCommand):
    """Django command wrapper for Gruenderplattform HTML preprocessing."""

    help = "Normalize Gruenderplattform HTML snapshots and archive exact duplicate pages."

    def add_arguments(self, parser):
        parser.add_argument(
            "--data-dir",
            type=str,
            default=str(_default_data_dir()),
            help="Corpus data directory.",
        )
        parser.add_argument(
            "--report-path",
            type=str,
            default=str(_default_report_path()),
            help="Machine-readable preprocessing report path.",
        )
        parser.add_argument(
            "--no-archive-duplicates",
            action="store_true",
            help="Do not move duplicate HTML snapshots into knowledge_base/archive.",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Report changes without modifying files.",
        )

    def handle(self, *args, **options):
        result = preprocess_gruenderplattform_html_files(
            data_dir=Path(options["data_dir"]),
            archive_duplicates=not bool(options["no_archive_duplicates"]),
            dry_run=bool(options["dry_run"]),
        )
        if not options["dry_run"]:
            write_preprocess_report(Path(options["report_path"]), result)

        self.stdout.write(self.style.SUCCESS("Gruenderplattform HTML preprocessing finished."))
        self.stdout.write(f"Scanned files: {result.scanned_files}")
        self.stdout.write(f"Changed files: {result.changed_files}")
        self.stdout.write(f"Soft hyphen replacements: {result.soft_hyphen_replacements}")
        self.stdout.write(f"NBSP replacements: {result.nbsp_replacements}")
        self.stdout.write(f"CTA links removed: {result.cta_links_removed}")
        self.stdout.write(f"Terminal blocks removed: {result.terminal_blocks_removed}")
        self.stdout.write(f"Duplicate groups: {len(result.duplicate_groups)}")
        self.stdout.write(f"Archived duplicates: {len(result.archived_duplicates)}")
        if options["dry_run"]:
            self.stdout.write("Dry run only: no files were modified.")
