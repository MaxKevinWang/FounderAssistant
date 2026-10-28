"""Prepare reviewed upstream artifacts for formal QRA candidate v0.3."""

from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from apps.evaluations.formal_qra_review_revisions import (
    prepare_formal_qra_review_revision,
)
from apps.evaluations.rageval_artifacts import ArtifactValidationError


class Command(BaseCommand):
    """Expose the append-only review revision as a reproducible command."""

    help = "Prepare frozen Configuration v1.2 and semantic curation v0.3."

    def add_arguments(self, parser) -> None:
        parser.add_argument("--revised-at", default="2026-08-02")

    def handle(self, *args, **options) -> None:
        try:
            report = prepare_formal_qra_review_revision(
                repo_root=Path(settings.BASE_DIR).parent,
                revised_at=str(options["revised_at"]),
            )
        except ArtifactValidationError as exc:
            raise CommandError(str(exc)) from exc

        configuration = report["configuration"]
        curation = report["curation"]
        self.stdout.write(
            self.style.SUCCESS(
                "Formal QRA review revision prepared: "
                f"{len(configuration['revised_configuration_ids'])} revised "
                f"Configurations and {curation['entry_count']} curated QRA recipes."
            )
        )
        self.stdout.write(f"  {configuration['manifest_path']}")
        self.stdout.write(f"  {curation['curation_path']}")
