"""Freeze LLM query decompositions for reproducible retrieval experiments."""

from pathlib import Path
from typing import Any

from django.core.management.base import BaseCommand, CommandError

from apps.evaluations.formal_development_benchmark import (
    default_formal_qra_dir,
    load_frozen_development_qras,
)
from apps.evaluations.formal_query_decomposition_manifest import (
    DEFAULT_DECOMPOSER_MODEL,
    FormalQueryDecompositionManifestError,
    default_output_dir,
    run_query_decomposition_freeze,
)


class Command(BaseCommand):
    help = "Generate each formal query decomposition once and freeze a manifest."

    def add_arguments(self, parser) -> None:  # noqa: ANN001
        parser.add_argument(
            "--qra-dir", type=str, default=str(default_formal_qra_dir())
        )
        parser.add_argument(
            "--output-dir", type=str, default=str(default_output_dir())
        )
        parser.add_argument(
            "--manifest-id", default="FORMAL-QUERY-DECOMPOSITIONS-001"
        )
        parser.add_argument("--model", default=DEFAULT_DECOMPOSER_MODEL)
        parser.add_argument(
            "--reasoning-effort",
            choices=["none", "low", "medium", "high"],
            default="low",
        )
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--max-subqueries", type=int, default=4)
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
                    f"{prefix} frozen in {event['latency_ms'] / 1000:.3f} s"
                )
            )
        else:
            self.stdout.write(self.style.ERROR(f"{prefix} failed: {event['error']}"))

    def handle(self, *args, **options) -> None:  # noqa: ANN002, ANN003
        reasoning_effort = str(options["reasoning_effort"])
        if reasoning_effort == "none":
            reasoning_effort = None
        try:
            qras = load_frozen_development_qras(
                Path(options["qra_dir"]),
                qra_ids=options["qra_id"],
                domains=options["domain"],
                limit=options["limit"],
            )
            path = run_query_decomposition_freeze(
                qra_items=qras,
                output_dir=Path(options["output_dir"]).resolve(),
                manifest_id=str(options["manifest_id"]),
                model=str(options["model"]),
                reasoning_effort=reasoning_effort,
                seed=int(options["seed"]),
                max_subqueries=int(options["max_subqueries"]),
                resume=bool(options["resume"]),
                progress_callback=self._progress,
            )
        except FormalQueryDecompositionManifestError as exc:
            raise CommandError(str(exc)) from exc
        self.stdout.write(self.style.SUCCESS(f"Manifest: {path}"))
