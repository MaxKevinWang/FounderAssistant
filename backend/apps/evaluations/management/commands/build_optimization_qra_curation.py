"""Build high-risk semantic curation for the optimization QRA set."""

from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from apps.evaluations.optimization_qra_curation import (
    build_optimization_qra_curation,
)
from apps.evaluations.rageval_artifacts import ArtifactValidationError


class Command(BaseCommand):
    help = "Build semantic recipes for all 60 optimization QRAs."

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--configuration-manifest",
            default=(
                "docs/evaluation/optimization_dataset/configurations/frozen/"
                "optimization_configuration_manifest_v1.0.json"
            ),
        )
        parser.add_argument(
            "--output",
            default=(
                "docs/evaluation/optimization_dataset/qra/"
                "qra_semantic_curation_v0.1.json"
            ),
        )
        parser.add_argument("--generated-at", default="2026-08-17")

    def handle(self, *args, **options) -> None:
        try:
            report = build_optimization_qra_curation(
                repo_root=Path(settings.BASE_DIR).parent,
                configuration_manifest_path=Path(options["configuration_manifest"]),
                output_path=Path(options["output"]),
                generated_at=str(options["generated_at"]),
            )
        except ArtifactValidationError as exc:
            raise CommandError(str(exc)) from exc

        self.stdout.write(
            self.style.SUCCESS(
                f"Optimization QRA curation complete: {report['entry_count']} recipes."
            )
        )
        self.stdout.write(f"  {report['path']}")
