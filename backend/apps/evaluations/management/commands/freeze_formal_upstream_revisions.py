"""Freeze the approved v1.1 formal-evaluation upstream repairs."""

from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from apps.evaluations.formal_upstream_revisions import (
    freeze_formal_upstream_revisions,
)
from apps.evaluations.rageval_artifacts import ArtifactValidationError


class Command(BaseCommand):
    """Expose the audited revision release as a reproducible command."""

    help = "Freeze the approved Document D and Configuration C v1.1 repairs."

    def add_arguments(self, parser) -> None:
        parser.add_argument("--frozen-at", default="2026-08-02")

    def handle(self, *args, **options) -> None:
        try:
            report = freeze_formal_upstream_revisions(
                repo_root=Path(settings.BASE_DIR).parent,
                frozen_at=str(options["frozen_at"]),
            )
        except ArtifactValidationError as exc:
            raise CommandError(str(exc)) from exc

        self.stdout.write(
            self.style.SUCCESS(
                "Formal upstream revision freeze complete: "
                f"{len(report['revised_document_ids'])} Document and "
                f"{len(report['revised_configuration_ids'])} Configurations; "
                f"{report['created_count']} created, "
                f"{report['unchanged_count']} unchanged."
            )
        )
        self.stdout.write(f"  {report['document_manifest_path']}")
        self.stdout.write(f"  {report['configuration_manifest_path']}")
