"""Build a formal paired Baseline/RAG report from completed Codex Judge runs."""

from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from apps.evaluations.formal_codex_comparison import (
    FormalCodexComparisonError,
    build_formal_codex_comparison,
)
from apps.evaluations.formal_codex_judge import project_root


class Command(BaseCommand):
    help = "Build paired formal Baseline/RAG JSON and Markdown reports."

    def add_arguments(self, parser) -> None:  # noqa: ANN001
        parser.add_argument("--baseline-scores", required=True)
        parser.add_argument("--baseline-manifest", required=True)
        parser.add_argument("--rag-scores", required=True)
        parser.add_argument("--rag-manifest", required=True)
        parser.add_argument("--report-id", required=True)
        parser.add_argument(
            "--output-dir",
            default=str(
                project_root()
                / "docs"
                / "evaluation"
                / "runs"
                / "formal"
                / "judge"
                / "comparison"
            ),
        )
        parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
        parser.add_argument("--seed", type=int, default=20260810)

    def handle(self, *args, **options) -> None:  # noqa: ANN002, ANN003
        try:
            paths = build_formal_codex_comparison(
                baseline_scores_path=Path(options["baseline_scores"]).resolve(),
                baseline_manifest_path=Path(options["baseline_manifest"]).resolve(),
                rag_scores_path=Path(options["rag_scores"]).resolve(),
                rag_manifest_path=Path(options["rag_manifest"]).resolve(),
                output_dir=Path(options["output_dir"]).resolve(),
                report_id=str(options["report_id"]),
                bootstrap_iterations=int(options["bootstrap_iterations"]),
                seed=int(options["seed"]),
            )
        except FormalCodexComparisonError as exc:
            raise CommandError(str(exc)) from exc
        self.stdout.write(self.style.SUCCESS(f"JSON report: {paths['json']}"))
        self.stdout.write(self.style.SUCCESS(f"Markdown report: {paths['markdown']}"))
