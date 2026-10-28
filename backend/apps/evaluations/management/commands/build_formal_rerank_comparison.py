"""Build the formal three-arm reranker comparison report."""

from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from apps.evaluations.formal_codex_judge import project_root
from apps.evaluations.formal_rerank_comparison import (
    FormalRerankComparisonError,
    build_formal_rerank_comparison,
)


class Command(BaseCommand):
    help = "Build paired Baseline/Dense@5/Rerank formal reports."

    def add_arguments(self, parser) -> None:  # noqa: ANN001
        for name in (
            "baseline-scores",
            "baseline-manifest",
            "dense5-scores",
            "dense5-manifest",
            "rerank-scores",
            "rerank-manifest",
        ):
            parser.add_argument(f"--{name}", required=True)
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
        parser.add_argument("--seed", type=int, default=20260811)

    def handle(self, *args, **options) -> None:  # noqa: ANN002, ANN003
        try:
            outputs = build_formal_rerank_comparison(
                baseline_scores_path=Path(options["baseline_scores"]),
                baseline_manifest_path=Path(options["baseline_manifest"]),
                dense5_scores_path=Path(options["dense5_scores"]),
                dense5_manifest_path=Path(options["dense5_manifest"]),
                rerank_scores_path=Path(options["rerank_scores"]),
                rerank_manifest_path=Path(options["rerank_manifest"]),
                output_dir=Path(options["output_dir"]),
                report_id=str(options["report_id"]),
                bootstrap_iterations=int(options["bootstrap_iterations"]),
                seed=int(options["seed"]),
            )
        except FormalRerankComparisonError as exc:
            raise CommandError(str(exc)) from exc
        self.stdout.write(self.style.SUCCESS(f"JSON report: {outputs['json']}"))
        self.stdout.write(self.style.SUCCESS(f"Markdown report: {outputs['markdown']}"))
