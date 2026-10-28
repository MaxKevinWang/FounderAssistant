"""Score completed development answers with a blind Responses API Judge."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from apps.evaluations.openai_judge import (
    OpenAIJudgeError,
    default_output_root,
    load_generation_run,
    prepare_blinded_candidates,
    run_openai_judge,
)


class Command(BaseCommand):
    """CLI wrapper with visible progress and explicit cost-related settings."""

    help = (
        "Blindly score a completed development generation artifact with the "
        "OpenAI-compatible Responses API and export comparison artifacts."
    )

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--input-run",
            required=True,
            help="Completed formal-development generation JSON artifact.",
        )
        parser.add_argument(
            "--output-dir",
            default=None,
            help=(
                "Parent directory for numbered Judge runs. Defaults to "
                "<source-run-dir>/judge/openai_api."
            ),
        )
        parser.add_argument(
            "--tested-model",
            action="append",
            default=[],
            help="Select one tested model from the source run. Repeat as needed.",
        )
        parser.add_argument(
            "--qra-id",
            action="append",
            default=[],
            help="Select one QRA ID. Repeat as needed.",
        )
        parser.add_argument(
            "--mode",
            choices=["both", "baseline", "rag"],
            default="both",
        )
        parser.add_argument(
            "--provider",
            choices=["openai", "deepseek"],
            default="openai",
            help="Responses API provider. Defaults to openai.",
        )
        parser.add_argument(
            "--base-url",
            default=None,
            help="Optional provider endpoint; DeepSeek defaults to DEEPSEEK_BASE_URL.",
        )
        parser.add_argument(
            "--judge-model",
            default=None,
        )
        parser.add_argument(
            "--reasoning-effort",
            choices=["none", "low", "medium", "high", "xhigh"],
            default=None,
        )
        parser.add_argument(
            "--max-output-tokens",
            type=int,
            default=None,
        )
        parser.add_argument(
            "--timeout-seconds",
            type=int,
            default=None,
        )
        parser.add_argument(
            "--max-retries",
            type=int,
            default=None,
        )
        parser.add_argument("--seed", type=int, default=20260728)
        parser.add_argument(
            "--run-id",
            default=None,
            help="Optional stable run ID for a named experiment.",
        )
        parser.add_argument(
            "--fail-on-error",
            action="store_true",
            help="Stop after the first Judge error and retain the partial JSON.",
        )
        parser.add_argument(
            "--resume",
            action="store_true",
            help=(
                "Resume an interrupted run from its partial JSON. Requires the "
                "same filters, Judge settings, prompt, seed, and --run-id."
            ),
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Validate and show the candidate plan without an API call.",
        )

    def _progress(self, event: dict[str, Any]) -> None:
        prefix = (
            f"[{event['position']}/{event['total']}] "
            f"{event['candidate_id']} / {event['qra_id']} / "
            f"{event['tested_model']} / {event['mode']} / "
            f"repeat-{event.get('repeat', 1)}"
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
        input_path = Path(options["input_run"]).resolve()
        modes = ["baseline", "rag"] if options["mode"] == "both" else [options["mode"]]
        provider = str(options["provider"])
        if provider == "deepseek":
            defaults = {
                "judge_model": settings.DEEPSEEK_JUDGE_MODEL,
                "reasoning_effort": settings.DEEPSEEK_JUDGE_REASONING_EFFORT,
                "max_output_tokens": settings.DEEPSEEK_JUDGE_MAX_OUTPUT_TOKENS,
                "timeout_seconds": settings.DEEPSEEK_JUDGE_TIMEOUT_SECONDS,
                "max_retries": settings.DEEPSEEK_JUDGE_MAX_RETRIES,
                "base_url": settings.DEEPSEEK_BASE_URL,
            }
            api_key_name = "DEEPSEEK_API_KEY"
        else:
            defaults = {
                "judge_model": settings.OPENAI_JUDGE_MODEL,
                "reasoning_effort": settings.OPENAI_JUDGE_REASONING_EFFORT,
                "max_output_tokens": settings.OPENAI_JUDGE_MAX_OUTPUT_TOKENS,
                "timeout_seconds": settings.OPENAI_JUDGE_TIMEOUT_SECONDS,
                "max_retries": settings.OPENAI_JUDGE_MAX_RETRIES,
                "base_url": None,
            }
            api_key_name = "OPENAI_API_KEY"
        for option_name, default_value in defaults.items():
            if options.get(option_name) is None:
                options[option_name] = default_value
        try:
            if options["resume"] and not options["run_id"]:
                raise OpenAIJudgeError("--resume requires --run-id")
            if options["dry_run"]:
                source = load_generation_run(input_path)
                selection = prepare_blinded_candidates(
                    source,
                    tested_models=options["tested_model"],
                    qra_ids=options["qra_id"],
                    modes=modes,
                    seed=int(options["seed"]),
                )
                self.stdout.write(f"Source run: {source['run_id']}")
                self.stdout.write(f"Candidate answers: {selection['candidate_count']}")
                self.stdout.write(
                    f"Blind selection hash: {selection['selection_sha256']}"
                )
                self.stdout.write(
                    f"Judge provider/model: {provider}/{options['judge_model']} "
                    f"(reasoning={options['reasoning_effort']})"
                )
                self.stdout.write(
                    f"Output root: "
                    f"{Path(options['output_dir']).resolve() if options['output_dir'] else default_output_root(input_path)}"
                )
                self.stdout.write(
                    self.style.SUCCESS("Dry-run validation passed; no API call made.")
                )
                return

            api_key = os.getenv(api_key_name, "")
            if not api_key:
                raise OpenAIJudgeError(
                    f"{api_key_name} is missing from backend/.env"
                )
            result = run_openai_judge(
                source_run_path=input_path,
                output_root=(
                    Path(options["output_dir"]).resolve()
                    if options["output_dir"]
                    else None
                ),
                tested_models=options["tested_model"],
                qra_ids=options["qra_id"],
                modes=modes,
                judge_model=str(options["judge_model"]),
                judge_provider=provider,
                api_key=api_key,
                api_base_url=(
                    str(options["base_url"]) if options["base_url"] else None
                ),
                reasoning_effort=str(options["reasoning_effort"]),
                max_output_tokens=int(options["max_output_tokens"]),
                timeout_seconds=int(options["timeout_seconds"]),
                max_retries=int(options["max_retries"]),
                seed=int(options["seed"]),
                run_id=options["run_id"],
                progress_callback=self._progress,
                fail_on_error=bool(options["fail_on_error"]),
                resume=bool(options["resume"]),
            )
        except OpenAIJudgeError as exc:
            raise CommandError(str(exc)) from exc

        report = result["report"]
        self.stdout.write(self.style.SUCCESS(f"Judge status: {report['status']}"))
        self.stdout.write(
            "Token usage: "
            f"input={report['usage'].get('input_tokens', 0)}, "
            f"output={report['usage'].get('output_tokens', 0)}, "
            f"total={report['usage'].get('total_tokens', 0)}"
        )
        self.stdout.write(f"Scores: {result['scores_path']}")
        self.stdout.write(f"Report: {result['markdown_path']}")
        self.stdout.write(f"Manifest: {result['manifest_path']}")
