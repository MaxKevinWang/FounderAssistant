"""Run the formal retrieval-only multi-query ablation."""

from pathlib import Path
from typing import Any

from django.core.management.base import BaseCommand, CommandError

from apps.evaluations.formal_development_benchmark import (
    default_formal_qra_dir,
    load_frozen_development_qras,
)
from apps.evaluations.formal_multiquery_ablation import (
    DEFAULT_DECOMPOSER_MODEL,
    FormalMultiQueryAblationError,
    default_output_dir,
    run_formal_multiquery_ablation,
)
class Command(BaseCommand):
    help = "Compare original dense and decomposed multi-query retrieval profiles."

    def add_arguments(self, parser) -> None:  # noqa: ANN001
        parser.add_argument(
            "--qra-dir", type=str, default=str(default_formal_qra_dir())
        )
        parser.add_argument(
            "--output-dir", type=str, default=str(default_output_dir())
        )
        parser.add_argument(
            "--report-id", default="FORMAL-MULTIQUERY-ABLATION-001"
        )
        parser.add_argument("--decomposer-model", default=DEFAULT_DECOMPOSER_MODEL)
        parser.add_argument(
            "--decomposer-reasoning-effort",
            choices=["none", "low", "medium", "high"],
            default="low",
        )
        parser.add_argument("--max-subqueries", type=int, default=4)
        parser.add_argument("--decomposer-seed", type=int, default=0)
        parser.add_argument("--decomposition-manifest", type=str, default="")
        parser.add_argument("--per-query-k", type=int, default=20)
        parser.add_argument("--candidate-k", type=int, default=20)
        parser.add_argument("--context-k", type=int, default=5)
        parser.add_argument("--max-context-tokens", type=int, default=3000)
        parser.add_argument("--rrf-k", type=int, default=60)
        parser.add_argument("--reranker-model", default="")
        parser.add_argument("--reranker-revision", default="")
        parser.add_argument(
            "--reranker-device", choices=["auto", "cpu", "cuda"], default="auto"
        )
        parser.add_argument("--reranker-batch-size", type=int, default=16)
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
            runtime_latency_ms = int(event["decomposition_runtime_latency_ms"])
            decomposition_status = (
                f"decomposed in {runtime_latency_ms / 1000:.3f} s"
                if runtime_latency_ms
                else "replayed frozen decomposition"
            )
            self.stdout.write(
                self.style.SUCCESS(
                    f"{prefix} {decomposition_status}; "
                    f"retrieved in {event['retrieval_latency_ms'] / 1000:.3f} s"
                )
            )
        else:
            self.stdout.write(self.style.ERROR(f"{prefix} failed: {event['error']}"))

    def handle(self, *args, **options) -> None:  # noqa: ANN002, ANN003
        reasoning_effort = str(options["decomposer_reasoning_effort"])
        if reasoning_effort == "none":
            reasoning_effort = None
        try:
            qras = load_frozen_development_qras(
                Path(options["qra_dir"]),
                qra_ids=options["qra_id"],
                domains=options["domain"],
                limit=options["limit"],
            )
            paths = run_formal_multiquery_ablation(
                qra_items=qras,
                output_dir=Path(options["output_dir"]).resolve(),
                report_id=str(options["report_id"]),
                decomposer_model=str(options["decomposer_model"]),
                decomposer_reasoning_effort=reasoning_effort,
                decomposer_seed=int(options["decomposer_seed"]),
                max_subqueries=int(options["max_subqueries"]),
                per_query_k=int(options["per_query_k"]),
                candidate_k=int(options["candidate_k"]),
                context_k=int(options["context_k"]),
                max_context_tokens=int(options["max_context_tokens"]),
                rrf_k=int(options["rrf_k"]),
                reranker_model=str(options["reranker_model"]),
                reranker_revision=str(options["reranker_revision"]),
                reranker_device=str(options["reranker_device"]),
                reranker_batch_size=int(options["reranker_batch_size"]),
                reranker_max_length=int(options["reranker_max_length"]),
                decomposition_manifest_path=(
                    Path(options["decomposition_manifest"]).resolve()
                    if options["decomposition_manifest"]
                    else None
                ),
                resume=bool(options["resume"]),
                progress_callback=self._progress,
            )
        except FormalMultiQueryAblationError as exc:
            raise CommandError(str(exc)) from exc
        self.stdout.write(self.style.SUCCESS(f"JSON report: {paths['json']}"))
        self.stdout.write(self.style.SUCCESS(f"Markdown report: {paths['markdown']}"))
