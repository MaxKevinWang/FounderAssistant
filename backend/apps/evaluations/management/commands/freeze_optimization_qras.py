"""Freeze the authorized optimization QRA candidate set."""

from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from apps.evaluations.optimization_qra_freezing import freeze_optimization_qras
from apps.evaluations.rageval_artifacts import ArtifactValidationError


class Command(BaseCommand):
    help = "Freeze the authorized 60-item optimization QRA set as v1.0."

    def add_arguments(self, parser) -> None:
        parser.add_argument("--frozen-at", default="2026-08-18")

    def handle(self, *args, **options) -> None:
        try:
            report = freeze_optimization_qras(
                repo_root=Path(settings.BASE_DIR).parent,
                frozen_at=str(options["frozen_at"]),
            )
        except ArtifactValidationError as exc:
            raise CommandError(str(exc)) from exc

        self.stdout.write(
            self.style.SUCCESS(
                "Optimization QRA v1.0 freeze complete: "
                f"{report['qra_count']} records; "
                f"{report['created_count']} created, "
                f"{report['unchanged_count']} unchanged."
            )
        )
        self.stdout.write(f"  {report['manifest_path']}")
