"""Prepare the hash-pinned optimization Document D source selection."""

from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from apps.evaluations.optimization_document_selection import (
    DEFAULT_PLAN_PATH,
    prepare_optimization_document_selection,
)
from apps.evaluations.rageval_artifacts import ArtifactValidationError


class Command(BaseCommand):
    """Expose deterministic optimization source preparation through Django."""

    help = "Resolve the optimization source plan into a hash-pinned selection."

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--plan",
            default=DEFAULT_PLAN_PATH.as_posix(),
            help="Repository-relative optimization source-plan path.",
        )
        parser.add_argument(
            "--overwrite",
            action="store_true",
            help="Replace the prepared selection only when intentionally changed.",
        )

    def handle(self, *args, **options) -> None:
        repo_root = Path(settings.BASE_DIR).parent
        try:
            report = prepare_optimization_document_selection(
                repo_root=repo_root,
                plan_path=Path(options["plan"]),
                overwrite=bool(options["overwrite"]),
            )
        except ArtifactValidationError as exc:
            raise CommandError(str(exc)) from exc

        selection = report["selection"]
        self.stdout.write(
            self.style.SUCCESS(
                f"Optimization selection {report['write_state']}: "
                f"{selection['document_count']} documents; "
                f"{selection['new_source_count']} sources not used by the prior formal selection; "
                f"{report['selection_path']}"
            )
        )
