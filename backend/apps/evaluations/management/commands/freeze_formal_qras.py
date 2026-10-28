"""Freeze audited formal QRA candidates as immutable test version 1.0."""

from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from apps.evaluations.formal_qra_freezing import freeze_formal_qras
from apps.evaluations.rageval_artifacts import ArtifactValidationError


class Command(BaseCommand):
    """Expose deterministic formal QRA freezing."""

    help = "Freeze the audited 60-item formal QRA set as immutable v1.0."

    def add_arguments(self, parser) -> None:
        parser.add_argument("--frozen-at", default="2026-08-03")

    def handle(self, *args, **options) -> None:
        try:
            report = freeze_formal_qras(
                repo_root=Path(settings.BASE_DIR).parent,
                frozen_at=str(options["frozen_at"]),
            )
        except ArtifactValidationError as exc:
            raise CommandError(str(exc)) from exc

        self.stdout.write(
            self.style.SUCCESS(
                "Formal QRA v1.0 freeze complete: "
                f"{report['qra_count']} records; "
                f"{report['created_count']} created, "
                f"{report['unchanged_count']} unchanged."
            )
        )
        self.stdout.write(f"  {report['manifest_path']}")
