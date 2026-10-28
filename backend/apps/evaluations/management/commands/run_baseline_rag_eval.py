"""CLI command to execute baseline-vs-RAG benchmark runs.

Usage:
- single run (default): exports one JSON + Markdown report pair
- repeated runs (`--repeats N`): exports one aggregate batch report pair
"""

from __future__ import annotations

from pathlib import Path

from django.core.management.base import BaseCommand

from apps.advisors.model_profiles import DEFAULT_ANSWER_MAX_TOKENS
from apps.evaluations.benchmarking import (
    default_output_dir,
    default_test_set_path,
    run_baseline_rag_benchmark,
    run_baseline_rag_benchmark_batch,
)


class Command(BaseCommand):
    """Management command wrapper around benchmarking service functions."""

    help = "Run baseline vs RAG evaluation and export report files."

    def add_arguments(self, parser):
        """Expose benchmark and persistence knobs to the CLI."""

        parser.add_argument("--test-set", type=str, default=str(default_test_set_path()))
        parser.add_argument("--output-dir", type=str, default=str(default_output_dir()))
        parser.add_argument("--mode", type=str, choices=["both", "baseline", "rag"], default="both")
        parser.add_argument("--tf-limit", type=int, default=None, help="Run first N TF questions for quick test.")
        parser.add_argument("--skip-scenario", action="store_true")
        parser.add_argument("--temperature", type=float, default=0.1)
        parser.add_argument("--top-k", type=int, default=5)
        parser.add_argument("--max-tokens-tf", type=int, default=220)
        parser.add_argument("--max-tokens-scenario", type=int, default=DEFAULT_ANSWER_MAX_TOKENS)
        parser.add_argument("--generation-model", type=str, default=None)
        parser.add_argument("--persist-db", action="store_true")
        parser.add_argument("--owner-username", type=str, default="benchmark_runner")
        parser.add_argument("--venture-name", type=str, default="Benchmark Venture")
        parser.add_argument("--repeats", type=int, default=1, help="Number of repeated benchmark runs.")

    def handle(self, *args, **options):
        """Dispatch to single-run or batch-run execution branch."""

        # `repeats=1` -> single report; `repeats>1` -> aggregate batch report.
        repeats = int(options["repeats"])
        if repeats <= 1:
            result = run_baseline_rag_benchmark(
                test_set_path=Path(options["test_set"]).resolve(),
                output_dir=Path(options["output_dir"]).resolve(),
                mode=str(options["mode"]),
                tf_limit=options["tf_limit"],
                skip_scenario=bool(options["skip_scenario"]),
                temperature=float(options["temperature"]),
                top_k=int(options["top_k"]),
                max_tokens_tf=int(options["max_tokens_tf"]),
                max_tokens_scenario=int(options["max_tokens_scenario"]),
                generation_model=options.get("generation_model") or None,
                persist_db=bool(options["persist_db"]),
                owner_username=str(options["owner_username"]),
                venture_name=str(options["venture_name"]),
            )

            report = result["report"]
            baseline_tf = report["summary"]["baseline_tf"]
            rag_tf = report["summary"]["rag_tf"]
            self.stdout.write(self.style.SUCCESS("Evaluation finished."))
            self.stdout.write(f"Report JSON: {result['json_path']}")
            self.stdout.write(f"Report MD:   {result['md_path']}")
            self.stdout.write(f"Generation model: {report['config']['generation_model']}")
            self.stdout.write(
                f"TF Accuracy -> baseline: {baseline_tf['accuracy']:.2%}, rag: {rag_tf['accuracy']:.2%}"
            )
            if result["db_persistence"].get("persisted"):
                self.stdout.write(
                    "DB persisted -> "
                    f"baseline_run={result['db_persistence'].get('baseline_run_id')}, "
                    f"rag_run={result['db_persistence'].get('rag_run_id')}, "
                    f"comparison={result['db_persistence'].get('comparison_id')}"
                )
            return

        batch = run_baseline_rag_benchmark_batch(
            repeats=repeats,
            test_set_path=Path(options["test_set"]).resolve(),
            output_dir=Path(options["output_dir"]).resolve(),
            mode=str(options["mode"]),
            tf_limit=options["tf_limit"],
            skip_scenario=bool(options["skip_scenario"]),
            temperature=float(options["temperature"]),
            top_k=int(options["top_k"]),
            max_tokens_tf=int(options["max_tokens_tf"]),
            max_tokens_scenario=int(options["max_tokens_scenario"]),
            generation_model=options.get("generation_model") or None,
            persist_db=bool(options["persist_db"]),
            owner_username=str(options["owner_username"]),
            venture_name=str(options["venture_name"]),
        )
        aggregate = batch["batch_report"]["aggregate"]
        self.stdout.write(self.style.SUCCESS("Batch evaluation finished."))
        self.stdout.write(f"Batch JSON: {batch['batch_json_path']}")
        self.stdout.write(f"Batch MD:   {batch['batch_md_path']}")
        self.stdout.write(
            f"Generation model: {batch['batch_report']['config']['generation_model']}"
        )
        self.stdout.write(
            "Accuracy mean/std -> "
            f"baseline={aggregate['baseline_accuracy']['mean']:.4f}/{aggregate['baseline_accuracy']['std']:.4f}, "
            f"rag={aggregate['rag_accuracy']['mean']:.4f}/{aggregate['rag_accuracy']['std']:.4f}"
        )
