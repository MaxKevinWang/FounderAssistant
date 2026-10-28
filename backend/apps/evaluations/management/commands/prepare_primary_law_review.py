"""Prepare review-only manifests and impact maps for primary-law Documents D."""

from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from apps.evaluations.primary_law_review_preparation import (
    prepare_primary_law_review_artifacts,
)
from apps.evaluations.rageval_artifacts import ArtifactValidationError


class Command(BaseCommand):
    """Expose deterministic primary-law review preparation through Django."""

    help = (
        "Audit primary-law Document D candidates and create a non-frozen "
        "Document manifest candidate plus Configuration impact matrix."
    )

    def handle(self, *args, **options) -> None:
        try:
            report = prepare_primary_law_review_artifacts(
                repo_root=Path(settings.BASE_DIR).parent
            )
        except ArtifactValidationError as exc:
            raise CommandError(str(exc)) from exc

        self.stdout.write(
            self.style.SUCCESS(
                "Primary-law review preparation complete: "
                f"{report['candidate_count']} Documents, "
                f"{report['block_count']} legal blocks, "
                f"{report['key_fact_check_count']} key-fact checks, "
                f"{report['configuration_revision_count']}/"
                f"{report['configuration_count']} Configurations affected."
            )
        )
        for key in (
            "candidate_manifest_path",
            "agent_review_markdown_path",
            "impact_matrix_markdown_path",
        ):
            path = report[key]
            self.stdout.write(f"  {path}")
