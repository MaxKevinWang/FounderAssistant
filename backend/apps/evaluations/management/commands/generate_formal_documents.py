"""Generate formal-test Document D review candidates from frozen raw sources."""

from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from apps.evaluations.formal_document_generation import (
    generate_formal_document_candidates,
)
from apps.evaluations.rageval_artifacts import ArtifactValidationError


class Command(BaseCommand):
    """Expose deterministic formal Document D generation through Django."""

    help = (
        "Generate review-candidate formal Document D artifacts and validate "
        "explicitly reused frozen D artifacts."
    )

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--selection",
            default=(
                "docs/evaluation/formal_dataset/source_selection/"
                "formal_test_source_selection_v0.1.json"
            ),
            help="Repository-relative source-selection manifest path.",
        )
        parser.add_argument(
            "--overwrite",
            action="store_true",
            help="Replace candidate files only when a deterministic rebuild differs.",
        )
        parser.add_argument(
            "--manifest-output",
            default=None,
            help=(
                "Optional repository-relative generation-manifest output path. "
                "The selection's generation_manifest_path is used when omitted."
            ),
        )
        parser.add_argument(
            "--generation-id",
            default=None,
            help="Optional generation ID override for the manifest.",
        )

    def handle(self, *args, **options) -> None:
        repo_root = Path(settings.BASE_DIR).parent
        try:
            report = generate_formal_document_candidates(
                repo_root=repo_root,
                selection_path=Path(options["selection"]),
                overwrite=bool(options["overwrite"]),
                manifest_output_path=(
                    Path(options["manifest_output"])
                    if options["manifest_output"]
                    else None
                ),
                generation_id=options["generation_id"],
            )
        except ArtifactValidationError as exc:
            raise CommandError(str(exc)) from exc

        self.stdout.write(
            self.style.SUCCESS(
                "Formal Document D generation complete: "
                f"{report['new_candidate_count']} new review candidates, "
                f"{report['reused_frozen_count']} reused frozen documents."
            )
        )
        for item in report["new_candidates"]:
            self.stdout.write(
                f"  {item['document_id']}: {item['write_state']}; "
                f"{item['block_count']} blocks; {item['path']}"
            )
