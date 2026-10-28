"""Apply the reviewed semantic change set to primary-law Configurations."""

from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from apps.evaluations.primary_law_configuration_curation import (
    curate_primary_law_configurations,
)
from apps.evaluations.rageval_artifacts import ArtifactValidationError


class Command(BaseCommand):
    """Expose append-only semantic curation as a reproducible command."""

    help = "Curate the 17 primary-law Configuration candidates as review v0.2."

    def add_arguments(self, parser) -> None:
        parser.add_argument("--reviewed-at", default="2026-08-03")
        parser.add_argument("--overwrite", action="store_true")

    def handle(self, *args, **options) -> None:
        try:
            report = curate_primary_law_configurations(
                repo_root=Path(settings.BASE_DIR).parent,
                reviewed_at=str(options["reviewed_at"]),
                overwrite=bool(options["overwrite"]),
            )
        except ArtifactValidationError as exc:
            raise CommandError(str(exc)) from exc

        self.stdout.write(
            self.style.SUCCESS(
                "Primary-law Configuration semantic curation complete: "
                f"{report['candidate_count']} candidates, "
                f"{report['removed_value_count']} inherited values removed, "
                f"{report['constraint_update_count']} constraints updated."
            )
        )
        self.stdout.write(f"  {report['curation_manifest_path']}")
        self.stdout.write(f"  {report['review_path']}")
        self.stdout.write(f"  {report['candidate_manifest_path']}")
