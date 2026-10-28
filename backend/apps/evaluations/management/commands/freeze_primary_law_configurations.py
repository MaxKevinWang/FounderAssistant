"""Freeze semantically curated primary-law Configurations as manifest v1.3."""

from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from apps.evaluations.primary_law_configuration_freezing import (
    freeze_primary_law_configurations,
)
from apps.evaluations.rageval_artifacts import ArtifactValidationError


class Command(BaseCommand):
    """Expose the hash-guarded primary-law Configuration freeze."""

    help = "Freeze 17 primary-law Configuration revisions and manifest v1.3."

    def add_arguments(self, parser) -> None:
        parser.add_argument("--frozen-at", default="2026-08-03")

    def handle(self, *args, **options) -> None:
        try:
            report = freeze_primary_law_configurations(
                repo_root=Path(settings.BASE_DIR).parent,
                frozen_at=str(options["frozen_at"]),
            )
        except ArtifactValidationError as exc:
            raise CommandError(str(exc)) from exc

        self.stdout.write(
            self.style.SUCCESS(
                "Primary-law Configuration freeze complete: "
                f"{report['revised_configuration_count']} revised, "
                f"{report['carried_forward_count']} carried forward."
            )
        )
        self.stdout.write(f"  {report['manifest_path']}")
        self.stdout.write(f"  {report['manifest_freeze_record_path']}")
