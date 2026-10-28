"""Run the formal retrieval-only cross-encoder reranking ablation."""

from pathlib import Path
from typing import Any

from django.core.management.base import BaseCommand, CommandError

from apps.evaluations.formal_development_benchmark import (
    default_formal_qra_dir,
    load_frozen_development_qras,
)
from apps.evaluations.formal_rerank_ablation import (
    DEFAULT_RERANKER_MODEL,
    DEFAULT_RERANKER_REVISION,
    FormalRerankAblationError,
    default_output_dir,
    run_formal_rerank_ablation,
)


class Command(BaseCommand):
    help = "Compare Dense@K and cross-encoder Rerank@5 without generation calls."

    def add_arguments(self, parser) -> None:  # noqa: ANN001
        parser.add_argument(
            "--qra-dir", type=str, default=str(default_formal_qra_dir())
        )
        parser.add_argument(
            "--output-dir", type=str, default=str(default_output_dir())
        )
        parser.add_argument(
            "--report-id", default="FORMAL-RERANK-ABLATION-001"
        )
        parser.add_argument("--candidate-k", type=int, default=20)
        parser.add_argument("--context-k", type=int, default=5)
        parser.add_argument("--max-context-tokens", type=int, default=3000)
        parser.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
        parser.add_argument(
            "--reranker-revision", default=DEFAULT_RERANKER_REVISION
        )
        parser.add_argument(
            "--reranker-device", choices=["auto", "cpu", "cuda"], default="auto"
        )
        parser.add_argument("--reranker-batch-size", type=int, default=8)
        parser.add_argument("--reranker-max-length", type=int, default=512)
        parser.add_argument("--qra-id", action="append", default=[])
        parser.add_argument(
            "--domain",
            action="append",
            default=[],
            choices=[
                "legal_form_selection",
                "funding_preparation",
                "market_validation",
            ],
        )
        parser.add_argument("--limit", type=int, default=None)
        parser.add_argument("--resume", action="store_true")

    def _progress(self, event: dict[str, Any]) -> None:
        prefix = f"[{event['position']}/{event['total']}] {event['qra_id']}"
        if event["type"] == "start":
            self.stdout.write(f"{prefix} started")
        elif event["status"] == "success":
            self.stdout.write(
                self.style.SUCCESS(
                    f"{prefix} reranked in "
                    f"{event['reranker_latency_ms'] / 1000:.3f} s"
                )
            )
        else:
            self.stdout.write(self.style.ERROR(f"{prefix} failed: {event['error']}"))

    def handle(self, *args, **options) -> None:  # noqa: ANN002, ANN003
        try:
            qras = load_frozen_development_qras(
                Path(options["qra_dir"]),
                qra_ids=options["qra_id"],
                domains=options["domain"],
                limit=options["limit"],
            )
            paths = run_formal_rerank_ablation(
                qra_items=qras,
                output_dir=Path(options["output_dir"]).resolve(),
                report_id=str(options["report_id"]),
                candidate_k=int(options["candidate_k"]),
                context_k=int(options["context_k"]),
                max_context_tokens=int(options["max_context_tokens"]),
                reranker_model=str(options["reranker_model"]),
                reranker_revision=str(options["reranker_revision"]),
                reranker_device=str(options["reranker_device"]),
                reranker_batch_size=int(options["reranker_batch_size"]),
                reranker_max_length=int(options["reranker_max_length"]),
                resume=bool(options["resume"]),
                progress_callback=self._progress,
            )
        except FormalRerankAblationError as exc:
            raise CommandError(str(exc)) from exc
        self.stdout.write(self.style.SUCCESS(f"JSON report: {paths['json']}"))
        self.stdout.write(self.style.SUCCESS(f"Markdown report: {paths['markdown']}"))
