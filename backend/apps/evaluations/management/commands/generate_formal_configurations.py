"""Generate all formal-test Configuration C review candidates."""

from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from apps.evaluations.formal_configuration_generation import (
    generate_formal_configurations,
)
from apps.evaluations.rageval_artifacts import ArtifactValidationError


class Command(BaseCommand):
    """Expose source-grounded formal Configuration generation through Django."""

    help = "Generate and mechanically validate all 60 formal Configuration candidates."

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--allocation",
            default=(
                "docs/evaluation/formal_dataset/configurations/"
                "formal_configuration_allocation_v0.1.json"
            ),
        )
        parser.add_argument(
            "--replace",
            action="store_true",
            help=(
                "Replace changed development-draft v0.1 candidates. "
                "Frozen Configuration files are never targeted by this command."
            ),
        )

    def handle(self, *args, **options) -> None:
        try:
            report = generate_formal_configurations(
                repo_root=Path(settings.BASE_DIR).parent,
                allocation_path=Path(options["allocation"]),
                replace_existing=options["replace"],
            )
        except ArtifactValidationError as exc:
            raise CommandError(str(exc)) from exc

        self.stdout.write(
            self.style.SUCCESS(
                "Formal Configuration generation complete: "
                f"{report['configuration_count']} candidates "
                f"({report['created_count']} created, "
                f"{report['replaced_count']} replaced, "
                f"{report['unchanged_count']} unchanged)."
            )
        )
        self.stdout.write(f"  {report['manifest_path']}")
        self.stdout.write(f"  {report['review_path']}")
        self.stdout.write(f"  {report['unanswerable_audit_path']}")
