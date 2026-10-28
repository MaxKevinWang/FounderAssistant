"""Build the formal post-hoc neighbor-expansion ablation."""

from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from apps.evaluations.formal_development_benchmark import (
    default_formal_qra_dir,
    load_frozen_development_qras,
)
from apps.evaluations.formal_neighbor_expansion_ablation import (
    FormalNeighborExpansionAblationError,
    build_formal_neighbor_expansion_ablation,
    default_output_dir,
)


class Command(BaseCommand):
    help = "Expand frozen Top-5 chunks with same-source neighbors under a budget."

    def add_arguments(self, parser) -> None:  # noqa: ANN001
        parser.add_argument(
            "--qra-dir", type=str, default=str(default_formal_qra_dir())
        )
        parser.add_argument("--source-report", type=str, required=True)
        parser.add_argument(
            "--output-dir", type=str, default=str(default_output_dir())
        )
        parser.add_argument(
            "--report-id", default="FORMAL-NEIGHBOR-EXPANSION-ABLATION-001"
        )
        parser.add_argument("--max-context-tokens", type=int, default=3000)

    def handle(self, *args, **options) -> None:  # noqa: ANN002, ANN003
        try:
            qras = load_frozen_development_qras(Path(options["qra_dir"]))
            paths = build_formal_neighbor_expansion_ablation(
                qra_items=qras,
                source_report_path=Path(options["source_report"]).resolve(),
                output_dir=Path(options["output_dir"]).resolve(),
                report_id=str(options["report_id"]),
                max_context_tokens=int(options["max_context_tokens"]),
            )
        except FormalNeighborExpansionAblationError as exc:
            raise CommandError(str(exc)) from exc
        self.stdout.write(self.style.SUCCESS(f"JSON report: {paths['json']}"))
        self.stdout.write(self.style.SUCCESS(f"Markdown report: {paths['markdown']}"))
