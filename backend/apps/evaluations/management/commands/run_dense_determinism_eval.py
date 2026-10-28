"""Freeze formal-query embeddings and repeat Dense retrieval locally."""

from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from apps.evaluations.dense_determinism import (
    DEFAULT_REPORT_ID,
    DenseDeterminismError,
    default_output_dir,
    run_dense_determinism_evaluation,
)
from apps.evaluations.formal_development_benchmark import (
    default_formal_qra_dir,
    load_frozen_development_qras,
)


class Command(BaseCommand):
    help = (
        "Freeze one query embedding per exposed formal QRA and verify Dense@20/5 "
        "ranking determinism over repeated local replay."
    )

    def add_arguments(self, parser) -> None:
        parser.add_argument("--qra-dir", default=str(default_formal_qra_dir()))
        parser.add_argument("--output-dir", default=str(default_output_dir()))
        parser.add_argument("--report-id", default=DEFAULT_REPORT_ID)
        parser.add_argument("--repeats", type=int, default=3)
        parser.add_argument("--candidate-k", type=int, default=20)
        parser.add_argument("--context-k", type=int, default=5)
        parser.add_argument("--query-embedding-manifest", default="")

    def handle(self, *args, **options) -> None:
        manifest_value = str(options["query_embedding_manifest"]).strip()
        try:
            qra_items = load_frozen_development_qras(
                Path(options["qra_dir"]).resolve()
            )
            result = run_dense_determinism_evaluation(
                qra_items=qra_items,
                output_dir=Path(options["output_dir"]),
                report_id=str(options["report_id"]),
                repeats=int(options["repeats"]),
                candidate_k=int(options["candidate_k"]),
                context_k=int(options["context_k"]),
                query_embedding_manifest_path=(
                    Path(manifest_value) if manifest_value else None
                ),
            )
        except (DenseDeterminismError, OSError, ValueError) as exc:
            raise CommandError(str(exc)) from exc

        report = result["report"]
        self.stdout.write(self.style.SUCCESS(f"Status: {report['status']}"))
        self.stdout.write(
            f"All gates passed: {report['acceptance']['all_gates_passed']}"
        )
        self.stdout.write(f"Query embeddings: {result['query_embeddings']}")
        self.stdout.write(f"Manifest: {result['query_embedding_manifest']}")
        self.stdout.write(f"JSON: {result['json']}")
        self.stdout.write(f"Markdown: {result['markdown']}")
