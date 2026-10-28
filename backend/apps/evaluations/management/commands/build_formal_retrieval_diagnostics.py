"""Build formal retrieval diagnostics from recorded RAG traces."""

from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from apps.evaluations.formal_retrieval_diagnostics import (
    FormalRetrievalDiagnosticsError,
    build_formal_retrieval_diagnostics,
    default_baseline_scores_path,
    default_output_dir,
    default_rag_scores_path,
    default_source_paths,
)


class Command(BaseCommand):
    help = "Build JSON and Markdown retrieval diagnostics without new API calls."

    def add_arguments(self, parser) -> None:  # noqa: ANN001
        parser.add_argument(
            "--source-run",
            action="append",
            dest="source_runs",
            help="Formal RAG JSON source; repeat for multiple model runs.",
        )
        parser.add_argument(
            "--rag-scores",
            default=str(default_rag_scores_path()),
            help="Completed RAG Judge score JSON; pass an empty string to omit.",
        )
        parser.add_argument(
            "--baseline-scores",
            default=str(default_baseline_scores_path()),
            help="Completed Baseline Judge score JSON; pass an empty string to omit.",
        )
        parser.add_argument(
            "--output-dir",
            default=str(default_output_dir()),
        )
        parser.add_argument(
            "--report-id",
            default="FORMAL-RETRIEVAL-DIAGNOSTICS-001",
        )

    def handle(self, *args, **options) -> None:  # noqa: ANN002, ANN003
        source_values = options.get("source_runs")
        sources = (
            [Path(value).resolve() for value in source_values]
            if source_values
            else default_source_paths()
        )
        rag_value = str(options.get("rag_scores") or "").strip()
        baseline_value = str(options.get("baseline_scores") or "").strip()
        try:
            paths = build_formal_retrieval_diagnostics(
                source_paths=sources,
                rag_scores_path=Path(rag_value).resolve() if rag_value else None,
                baseline_scores_path=(
                    Path(baseline_value).resolve() if baseline_value else None
                ),
                output_dir=Path(options["output_dir"]).resolve(),
                report_id=str(options["report_id"]),
            )
        except FormalRetrievalDiagnosticsError as exc:
            raise CommandError(str(exc)) from exc
        self.stdout.write(self.style.SUCCESS(f"JSON report: {paths['json']}"))
        self.stdout.write(self.style.SUCCESS(f"Markdown report: {paths['markdown']}"))
