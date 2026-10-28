"""Generate Baseline/RAG answers for frozen formal-development QRA items."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from apps.advisors.local_llm import LocalLLMError, check_status
from apps.advisors.rag_pipeline import SUPPORTED_RAG_PROMPT_VERSIONS
from apps.advisors.model_profiles import DEFAULT_ANSWER_MAX_TOKENS, classify_generation_model
from apps.evaluations.formal_development_benchmark import (
    FormalBenchmarkError,
    default_output_dir,
    default_qra_dir,
    formal_run_lock,
    load_frozen_development_qras,
    load_generation_prompt,
    run_formal_development_benchmark,
    snapshot_rag_index,
    validate_models,
)


class Command(BaseCommand):
    """CLI wrapper with visible per-request progress for long remote runs."""

    help = (
        "Run unscored Baseline/RAG generations on a frozen development or "
        "formal-test QRA release and export resumable JSON/Markdown artifacts."
    )

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--model",
            action="append",
            dest="models",
            help="Generation model ID. Repeat for a multi-model run.",
        )
        parser.add_argument(
            "--qra-dir",
            type=str,
            default=str(default_qra_dir()),
        )
        parser.add_argument(
            "--output-dir",
            type=str,
            default=str(default_output_dir()),
        )
        parser.add_argument(
            "--prompt-path",
            type=str,
            default=None,
            help="Answer-generation prompt JSON; formal runs should use a frozen prompt.",
        )
        parser.add_argument(
            "--qra-id",
            action="append",
            default=[],
            help="Run one frozen QRA ID. Repeat to select several items.",
        )
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
        parser.add_argument(
            "--question-type",
            action="append",
            default=[],
            choices=[
                "Factual",
                "Summarization",
                "Multi-hop Reasoning",
                "Information Integration",
                "Numerical Comparison",
                "Temporal Sequence",
                "Unanswerable",
            ],
        )
        parser.add_argument("--limit", type=int, default=None)
        parser.add_argument(
            "--mode",
            choices=["both", "baseline", "rag"],
            default="both",
        )
        parser.add_argument("--temperature", type=float, default=0.1)
        parser.add_argument("--top-k", type=int, default=5)
        parser.add_argument("--candidate-k", type=int, default=None)
        parser.add_argument("--reranker-model", type=str, default="")
        parser.add_argument("--reranker-revision", type=str, default="")
        parser.add_argument("--max-context-tokens", type=int, default=None)
        parser.add_argument(
            "--neighbor-expansion-distance", type=int, default=0
        )
        parser.add_argument(
            "--forbid-keyword-fallback", action="store_true"
        )
        parser.add_argument("--use-confidence-gating", action="store_true")
        parser.add_argument(
            "--rag-prompt-version",
            choices=sorted(SUPPORTED_RAG_PROMPT_VERSIONS),
            default="",
        )
        parser.add_argument("--max-tokens", type=int, default=DEFAULT_ANSWER_MAX_TOKENS)
        parser.add_argument(
            "--repeats",
            type=int,
            default=1,
            help="Independent generations per model/QRA/condition.",
        )
        parser.add_argument(
            "--schedule-seed",
            type=int,
            default=20260807,
            help="Seed for reproducible QRA scheduling within each model block.",
        )
        parser.add_argument(
            "--run-id",
            type=str,
            default=None,
            help="Stable run ID; required with --resume.",
        )
        parser.add_argument(
            "--resume",
            action="store_true",
            help="Resume a partial run and retry only unsuccessful combinations.",
        )
        parser.add_argument(
            "--reasoning-effort",
            choices=["low", "medium", "high", "max"],
            default=None,
            help="Optional reasoning effort forwarded to OpenAI-compatible providers.",
        )
        parser.add_argument(
            "--skip-model-check",
            action="store_true",
            help="Allow model IDs absent from the current service inventory.",
        )
        parser.add_argument(
            "--fail-on-error",
            action="store_true",
            help="Stop after the first generation error; retain partial JSON.",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Validate inputs and print the call plan without generation.",
        )

    def _progress(self, event: dict[str, Any]) -> None:
        prefix = (
            f"[{event['position']}/{event['total']}] "
            f"{event['model']} ({event['model_type']}) / "
            f"{event['qra_id']} / {event['mode']}"
        )
        if event["type"] == "start":
            self.stdout.write(f"{prefix} started")
            return
        if event["status"] == "success":
            self.stdout.write(
                self.style.SUCCESS(
                    f"{prefix} completed in {event['latency_ms'] / 1000:.3f} s"
                )
            )
        else:
            self.stdout.write(self.style.ERROR(f"{prefix} failed: {event['error']}"))

    def handle(self, *args, **options) -> None:
        try:
            if int(options["repeats"]) < 1:
                raise FormalBenchmarkError("repeats must be at least 1")
            if options["resume"] and not options["run_id"]:
                raise FormalBenchmarkError("--resume requires --run-id")
            qra_items = load_frozen_development_qras(
                Path(options["qra_dir"]),
                qra_ids=options["qra_id"],
                domains=options["domain"],
                question_types=options["question_type"],
                limit=options["limit"],
            )
            requested_models = options["models"] or [
                str(getattr(settings, "LOCAL_LLM_MODEL", "")).strip()
            ]
            service = check_status()
            models = (
                [model.strip() for model in requested_models if model.strip()]
                if options["skip_model_check"]
                else validate_models(
                    requested_models,
                    service.get("generation_models", []),
                )
            )
            if not models:
                raise FormalBenchmarkError("At least one model is required")
            modes = (
                ["baseline", "rag"] if options["mode"] == "both" else [options["mode"]]
            )
            rag_options = {
                key: value
                for key, value in {
                    "candidate_k": options["candidate_k"],
                    "reranker_model": options["reranker_model"],
                    "reranker_revision": options["reranker_revision"],
                    "max_context_tokens": options["max_context_tokens"],
                    "neighbor_expansion_distance": options[
                        "neighbor_expansion_distance"
                    ],
                    "allow_keyword_fallback": not options[
                        "forbid_keyword_fallback"
                    ],
                    "use_confidence_gating": options[
                        "use_confidence_gating"
                    ],
                    "rag_prompt_version": options["rag_prompt_version"],
                }.items()
                if value not in (None, "")
            }

            split = str(qra_items[0]["split"])
            self.stdout.write(f"Frozen {split} QRA items: {len(qra_items)}")
            for model in models:
                self.stdout.write(
                    f"Model: {model} ({classify_generation_model(model)})"
                )
            self.stdout.write(f"Modes: {', '.join(modes)}")
            self.stdout.write(f"Repeats: {options['repeats']}")
            self.stdout.write(
                "Scheduled calls: "
                f"{len(qra_items) * len(models) * len(modes) * int(options['repeats'])}"
            )

            if options["dry_run"]:
                prompt = load_generation_prompt(
                    Path(options["prompt_path"]) if options["prompt_path"] else None
                )
                index = snapshot_rag_index()
                self.stdout.write(
                    f"Prompt: {prompt['prompt_id']} ({prompt['_artifact']['sha256']})"
                )
                self.stdout.write(
                    "RAG index: "
                    f"{index['document_count']} documents, "
                    f"{index['chunk_count']} chunks, "
                    f"embedding={index['embedding_model']}"
                )
                self.stdout.write(self.style.SUCCESS("Dry run validation passed."))
                return

            output_dir = Path(options["output_dir"]).resolve()
            with formal_run_lock(output_dir, options["run_id"]):
                result = run_formal_development_benchmark(
                    models=models,
                    qra_items=qra_items,
                    output_dir=output_dir,
                    modes=modes,
                    temperature=float(options["temperature"]),
                    top_k=int(options["top_k"]),
                    max_tokens=int(options["max_tokens"]),
                    reasoning_effort=options["reasoning_effort"],
                    prompt_path=(
                        Path(options["prompt_path"]).resolve()
                        if options["prompt_path"]
                        else None
                    ),
                    repeats=int(options["repeats"]),
                    schedule_seed=int(options["schedule_seed"]),
                    run_id=options["run_id"],
                    resume=bool(options["resume"]),
                    skip_model_check=bool(options["skip_model_check"]),
                    fail_on_error=bool(options["fail_on_error"]),
                    progress_callback=self._progress,
                    service_status=service,
                    rag_options=rag_options,
                )
        except (FormalBenchmarkError, LocalLLMError) as exc:
            raise CommandError(str(exc)) from exc

        report = result["report"]
        self.stdout.write(self.style.SUCCESS(f"Run status: {report['status']}"))
        self.stdout.write(f"JSON: {result['json_path']}")
        self.stdout.write(f"Markdown: {result['markdown_path']}")
