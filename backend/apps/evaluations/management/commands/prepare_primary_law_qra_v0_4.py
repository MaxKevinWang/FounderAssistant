"""Prepare QRA semantic curation v0.4 after primary-law revisions."""

from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from apps.evaluations.primary_law_qra_revisions import (
    prepare_primary_law_qra_revisions,
)
from apps.evaluations.rageval_artifacts import ArtifactValidationError


class Command(BaseCommand):
    """Expose deterministic QRA recipe revision preparation."""

    help = "Prepare the 17 primary-law QRA recipe revisions for candidate v0.4."

    def add_arguments(self, parser) -> None:
        parser.add_argument("--created-at", default="2026-08-03")
        parser.add_argument("--overwrite", action="store_true")

    def handle(self, *args, **options) -> None:
        try:
            report = prepare_primary_law_qra_revisions(
                repo_root=Path(settings.BASE_DIR).parent,
                created_at=str(options["created_at"]),
                overwrite=bool(options["overwrite"]),
            )
        except ArtifactValidationError as exc:
            raise CommandError(str(exc)) from exc

        self.stdout.write(
            self.style.SUCCESS(
                "Primary-law QRA v0.4 preparation complete: "
                f"{report['revised_qra_count']} recipes revised; "
                f"{report['curated_entry_count']} curated recipes total."
            )
        )
        self.stdout.write(f"  {report['curation_path']}")
        self.stdout.write(f"  {report['revision_manifest_path']}")
        self.stdout.write(f"  {report['review_path']}")
