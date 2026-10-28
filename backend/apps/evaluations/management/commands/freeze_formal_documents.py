"""Freeze formal Document D candidates and their source-selection manifest."""

from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from apps.evaluations.formal_document_freezing import freeze_formal_documents
from apps.evaluations.rageval_artifacts import ArtifactValidationError


class Command(BaseCommand):
    """Expose formal Document D freezing as an explicit one-way operation."""

    help = "Freeze the reviewed formal Document D candidates as v1.0 artifacts."

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--selection",
            default=(
                "docs/evaluation/formal_dataset/source_selection/"
                "formal_test_source_selection_v0.1.json"
            ),
        )
        parser.add_argument("--frozen-at", default="2026-07-30")
        parser.add_argument(
            "--document-manifest-id",
            default="formal_test_document_manifest_v1.0",
        )
        parser.add_argument(
            "--document-manifest-output",
            default=(
                "docs/evaluation/formal_dataset/documents/"
                "formal_test_document_manifest_v1.0.json"
            ),
        )
        parser.add_argument(
            "--frozen-selection-id",
            default="formal_test_source_selection_v1.0",
        )
        parser.add_argument(
            "--frozen-selection-output",
            default=(
                "docs/evaluation/formal_dataset/source_selection/"
                "formal_test_source_selection_v1.0.json"
            ),
        )
        parser.add_argument(
            "--authorization",
            help="Optional hash-pinned owner authorization JSON artifact.",
        )

    def handle(self, *args, **options) -> None:
        try:
            report = freeze_formal_documents(
                repo_root=Path(settings.BASE_DIR).parent,
                selection_path=Path(options["selection"]),
                frozen_at=str(options["frozen_at"]),
                document_manifest_id=str(options["document_manifest_id"]),
                document_manifest_path=Path(options["document_manifest_output"]),
                frozen_selection_id=str(options["frozen_selection_id"]),
                frozen_selection_path=Path(options["frozen_selection_output"]),
                authorization_path=(
                    Path(options["authorization"])
                    if options.get("authorization")
                    else None
                ),
            )
        except ArtifactValidationError as exc:
            raise CommandError(str(exc)) from exc

        self.stdout.write(
            self.style.SUCCESS(
                "Formal Document D freeze complete: "
                f"{report['newly_frozen_count']} newly frozen, "
                f"{report['reused_frozen_count']} reused."
            )
        )
        self.stdout.write(f"  {report['document_manifest_path']}")
        self.stdout.write(f"  {report['source_selection_path']}")
