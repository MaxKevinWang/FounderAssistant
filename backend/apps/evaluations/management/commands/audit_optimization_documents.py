"""Audit optimization Document D candidates."""

from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from apps.evaluations.optimization_document_audit import (
    DEFAULT_OUTPUT_PATH,
    DEFAULT_SCOPE_REVIEW_PATH,
    DEFAULT_SELECTION_PATH,
    audit_optimization_document_candidates,
)
from apps.evaluations.rageval_artifacts import ArtifactValidationError


class Command(BaseCommand):
    """Expose the optimization Document D audit through Django."""

    help = "Validate optimization documents and build prior-QRA leakage exclusions."

    def add_arguments(self, parser) -> None:
        parser.add_argument("--selection", default=DEFAULT_SELECTION_PATH.as_posix())
        parser.add_argument("--scope-review", default=DEFAULT_SCOPE_REVIEW_PATH.as_posix())
        parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH.as_posix())
        parser.add_argument("--overwrite", action="store_true")

    def handle(self, *args, **options) -> None:
        try:
            result = audit_optimization_document_candidates(
                repo_root=Path(settings.BASE_DIR).parent,
                selection_path=Path(options["selection"]),
                scope_review_path=Path(options["scope_review"]),
                output_path=Path(options["output"]),
                overwrite=bool(options["overwrite"]),
            )
        except ArtifactValidationError as exc:
            raise CommandError(str(exc)) from exc
        report = result["report"]
        self.stdout.write(
            self.style.SUCCESS(
                f"Optimization Document audit {result['write_state']}: "
                f"{report['summary']['document_count']} documents, "
                f"{report['summary']['block_count']} blocks, "
                f"all automated gates passed={report['all_automated_gates_passed']}."
            )
        )
        self.stdout.write(f"  {result['output_path']}")
        self.stdout.write(f"  {result['markdown_path']}")
