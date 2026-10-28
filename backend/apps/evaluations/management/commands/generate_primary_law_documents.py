"""Generate review-candidate Document D files from selected primary laws."""

from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from apps.evaluations.primary_law_document_generation import (
    DEFAULT_SELECTION_PATH,
    generate_primary_law_document_candidates,
)
from apps.evaluations.rageval_artifacts import ArtifactValidationError


class Command(BaseCommand):
    """Expose deterministic primary-law Document D generation through Django."""

    help = (
        "Generate primary-law Document D review candidates without changing "
        "the frozen formal evaluation manifests."
    )

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--selection",
            default=DEFAULT_SELECTION_PATH.as_posix(),
            help="Repository-relative primary-law source-selection manifest.",
        )
        parser.add_argument(
            "--overwrite",
            action="store_true",
            help="Replace candidates only after an intentional deterministic change.",
        )

    def handle(self, *args, **options) -> None:
        try:
            report = generate_primary_law_document_candidates(
                repo_root=Path(settings.BASE_DIR).parent,
                selection_path=Path(options["selection"]),
                overwrite=bool(options["overwrite"]),
            )
        except ArtifactValidationError as exc:
            raise CommandError(str(exc)) from exc

        self.stdout.write(
            self.style.SUCCESS(
                "Primary-law Document D generation complete: "
                f"{report['candidate_count']} candidates, "
                f"{report['block_count']} statutory blocks."
            )
        )
        for item in report["candidates"]:
            self.stdout.write(
                f"  {item['document_id']} ({item['law_abbreviation']}): "
                f"{item['write_state']}; {item['block_count']} blocks; {item['path']}"
            )
