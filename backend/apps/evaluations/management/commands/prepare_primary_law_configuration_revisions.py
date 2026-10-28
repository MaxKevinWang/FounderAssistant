"""Generate the 17 primary-law Configuration revision candidates."""

from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from apps.evaluations.primary_law_configuration_revisions import (
    prepare_primary_law_configuration_revisions,
)
from apps.evaluations.rageval_artifacts import ArtifactValidationError


class Command(BaseCommand):
    """Expose deterministic candidate generation without a freeze side effect."""

    help = (
        "Prepare and validate the 17 primary-law Configuration revisions for "
        "human review."
    )

    def add_arguments(self, parser) -> None:
        parser.add_argument("--generated-at", default="2026-08-03")
        parser.add_argument("--overwrite", action="store_true")

    def handle(self, *args, **options) -> None:
        try:
            report = prepare_primary_law_configuration_revisions(
                repo_root=Path(settings.BASE_DIR).parent,
                generated_at=str(options["generated_at"]),
                overwrite=bool(options["overwrite"]),
            )
        except ArtifactValidationError as exc:
            raise CommandError(str(exc)) from exc

        self.stdout.write(
            self.style.SUCCESS(
                "Primary-law Configuration preparation complete: "
                f"{report['candidate_count']} candidates, "
                f"{report['legal_value_addition_count']} legal values, "
                f"{report['legal_evidence_count']} exact quotes."
            )
        )
        self.stdout.write(f"  {report['candidate_manifest_path']}")
        self.stdout.write(f"  {report['review_path']}")
