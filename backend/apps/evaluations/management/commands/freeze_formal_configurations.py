"""Freeze the approved 60-item formal Configuration C set."""

from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from apps.evaluations.formal_configuration_freezing import (
    freeze_formal_configurations,
)
from apps.evaluations.rageval_artifacts import ArtifactValidationError


class Command(BaseCommand):
    """Expose immutable Configuration freezing as an explicit command."""

    help = "Freeze all reviewed formal Configuration C candidates as v1.0."

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--candidate-manifest",
            default=(
                "docs/evaluation/formal_dataset/configurations/formal_candidates/"
                "generation_manifest_v0.1.json"
            ),
        )
        parser.add_argument("--frozen-at", default="2026-07-31")
        parser.add_argument(
            "--output-dir",
            default="docs/evaluation/formal_dataset/configurations/formal_test",
        )
        parser.add_argument(
            "--manifest-path",
            default=(
                "docs/evaluation/formal_dataset/configurations/formal_test/"
                "formal_test_configuration_manifest_v1.0.json"
            ),
        )
        parser.add_argument(
            "--manifest-id", default="formal_test_configuration_manifest_v1.0"
        )
        parser.add_argument("--dataset-label", default="Formal test")
        parser.add_argument("--authorization", default=None)

    def handle(self, *args, **options) -> None:
        try:
            report = freeze_formal_configurations(
                repo_root=Path(settings.BASE_DIR).parent,
                candidate_manifest_path=Path(options["candidate_manifest"]),
                frozen_at=str(options["frozen_at"]),
                output_dir=Path(options["output_dir"]),
                manifest_path=Path(options["manifest_path"]),
                manifest_id=str(options["manifest_id"]),
                dataset_label=str(options["dataset_label"]),
                authorization_path=(
                    Path(options["authorization"])
                    if options["authorization"]
                    else None
                ),
            )
        except ArtifactValidationError as exc:
            raise CommandError(str(exc)) from exc

        self.stdout.write(
            self.style.SUCCESS(
                "Formal Configuration C freeze complete: "
                f"{report['configuration_count']} configurations; "
                f"{report['created_count']} artifacts created, "
                f"{report['unchanged_count']} unchanged."
            )
        )
        self.stdout.write(f"  {report['manifest_path']}")
        self.stdout.write(f"  {report['manifest_freeze_record_path']}")
