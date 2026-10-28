"""Freeze the audited primary-law Documents D and additive manifest v1.2."""

from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from apps.evaluations.primary_law_document_freezing import (
    freeze_primary_law_documents,
)
from apps.evaluations.rageval_artifacts import ArtifactValidationError


class Command(BaseCommand):
    """Expose the explicit, hash-guarded primary-law freeze operation."""

    help = (
        "Freeze the 14 audited primary-law Document D candidates and publish "
        "formal Document manifest v1.2."
    )

    def add_arguments(self, parser) -> None:
        parser.add_argument("--frozen-at", default="2026-08-03")
        parser.add_argument(
            "--authorization-instruction",
            required=True,
            help=(
                "Exact project-owner continuation instruction recorded in the "
                "authorization artifact."
            ),
        )

    def handle(self, *args, **options) -> None:
        try:
            report = freeze_primary_law_documents(
                repo_root=Path(settings.BASE_DIR).parent,
                frozen_at=str(options["frozen_at"]),
                authorization_instruction=str(options["authorization_instruction"]),
            )
        except ArtifactValidationError as exc:
            raise CommandError(str(exc)) from exc

        self.stdout.write(
            self.style.SUCCESS(
                "Primary-law Document freeze complete: "
                f"{report['frozen_document_count']} frozen, "
                f"{report['carried_forward_count']} carried forward."
            )
        )
        self.stdout.write(f"  {report['document_manifest_path']}")
        self.stdout.write(f"  {report['authorization_path']}")
