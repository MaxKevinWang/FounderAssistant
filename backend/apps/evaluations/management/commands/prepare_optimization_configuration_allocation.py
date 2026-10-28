"""Prepare the 60-item optimization Configuration allocation."""

from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from apps.evaluations.optimization_configuration_allocation import (
    prepare_optimization_configuration_allocation,
)
from apps.evaluations.rageval_artifacts import ArtifactValidationError


class Command(BaseCommand):
    help = "Prepare the hash-pinned optimization Configuration allocation."

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--output",
            default=(
                "docs/evaluation/optimization_dataset/configurations/"
                "optimization_configuration_allocation_v0.1.json"
            ),
        )
        parser.add_argument("--overwrite", action="store_true")

    def handle(self, *args, **options) -> None:
        try:
            report = prepare_optimization_configuration_allocation(
                repo_root=Path(settings.BASE_DIR).parent,
                output_path=Path(options["output"]),
                overwrite=bool(options["overwrite"]),
            )
        except ArtifactValidationError as exc:
            raise CommandError(str(exc)) from exc

        self.stdout.write(
            self.style.SUCCESS(
                "Optimization Configuration allocation prepared: "
                f"{report['configuration_count']} items ({report['write_state']})."
            )
        )
        self.stdout.write(f"  {report['allocation_path']}")
