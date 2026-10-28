"""Run the formal QRA v0.4 semantic and traceability review gate."""

from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from apps.evaluations.formal_qra_semantic_review import (
    audit_formal_qra_candidates,
)
from apps.evaluations.rageval_artifacts import ArtifactValidationError


class Command(BaseCommand):
    """Expose the reproducible pre-freeze QRA audit."""

    help = "Audit all 60 formal QRA v0.4 candidates before freezing."

    def add_arguments(self, parser) -> None:
        parser.add_argument("--owner-instruction", required=True)
        parser.add_argument("--reviewed-at", default="2026-08-03")

    def handle(self, *args, **options) -> None:
        try:
            report = audit_formal_qra_candidates(
                repo_root=Path(settings.BASE_DIR).parent,
                owner_instruction=str(options["owner_instruction"]),
                reviewed_at=str(options["reviewed_at"]),
            )
        except ArtifactValidationError as exc:
            raise CommandError(str(exc)) from exc

        self.stdout.write(
            self.style.SUCCESS(
                "Formal QRA v0.4 audit passed: "
                f"{report['qra_count']} items; "
                f"{report['unresolved_blocker_count']} unresolved blockers."
            )
        )
        self.stdout.write(f"  {report['agent_review_path']}")
        self.stdout.write(f"  {report['owner_confirmation_path']}")
