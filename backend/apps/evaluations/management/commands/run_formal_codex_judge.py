"""Run a resumable, blinded Codex CLI Judge on formal generations."""

from __future__ import annotations

from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from apps.evaluations.formal_codex_judge import (
    FormalCodexJudgeError,
    default_codex_judge_output_root,
    default_formal_source_paths,
    make_codex_cli_batch_judge,
    prepare_formal_codex_judge_run,
    run_formal_codex_judge_phase,
)
from apps.evaluations.formal_development_benchmark import (
    FormalBenchmarkError,
    formal_run_lock,
)


class Command(BaseCommand):
    help = "Run an isolated Codex 5.6 Judge with batch checkpoints and resume logs."

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--source-run",
            action="append",
            default=[],
            help="Formal generation JSON; repeat for multiple source runs.",
        )
        parser.add_argument(
            "--output-root",
            default=str(default_codex_judge_output_root()),
        )
        parser.add_argument("--run-id", required=True)
        parser.add_argument(
            "--condition",
            choices=("baseline", "rag"),
            default="rag",
            help="Generation condition contained in the source runs.",
        )
        parser.add_argument("--phase", choices=("first", "remaining"), default="first")
        parser.add_argument("--first-phase-model-count", type=int, default=4)
        parser.add_argument("--batch-size", type=int, default=10)
        parser.add_argument("--judge-model", default="gpt-5.6-sol")
        parser.add_argument("--reasoning-effort", default="medium")
        parser.add_argument("--seed", type=int, default=20260809)
        parser.add_argument("--timeout-seconds", type=int, default=900)
        parser.add_argument("--max-batch-retries", type=int, default=1)
        parser.add_argument("--max-batches", type=int)
        parser.add_argument("--resume", action="store_true")
        parser.add_argument("--prepare-only", action="store_true")

    def handle(self, *args, **options) -> None:  # noqa: ANN002, ANN003
        run_id = str(options["run_id"])
        run_dir = Path(options["output_root"]).resolve() / run_id
        source_paths = (
            [Path(value).resolve() for value in options["source_run"]]
            if options["source_run"]
            else default_formal_source_paths()
        )
        command_parts = [
            ".\\.venv\\Scripts\\python.exe",
            "manage.py",
            "run_formal_codex_judge",
            "--run-id",
            run_id,
            "--condition",
            str(options["condition"]),
            "--phase",
            str(options["phase"]),
            "--first-phase-model-count",
            str(options["first_phase_model_count"]),
            "--batch-size",
            str(options["batch_size"]),
            "--judge-model",
            str(options["judge_model"]),
            "--reasoning-effort",
            str(options["reasoning_effort"]),
            "--seed",
            str(options["seed"]),
            "--timeout-seconds",
            str(options["timeout_seconds"]),
            "--max-batch-retries",
            str(options["max_batch_retries"]),
            "--resume",
        ]
        resume_command = " ".join(command_parts)
        try:
            with formal_run_lock(Path(options["output_root"]).resolve(), run_id):
                if not options["resume"]:
                    paths = prepare_formal_codex_judge_run(
                        source_paths,
                        run_dir=run_dir,
                        run_id=run_id,
                        seed=int(options["seed"]),
                        first_phase_model_count=int(options["first_phase_model_count"]),
                        judge_model=str(options["judge_model"]),
                        reasoning_effort=str(options["reasoning_effort"]),
                        condition_mode=str(options["condition"]),
                    )
                    self.stdout.write(f"Prepared blinded packets: {paths['packets']}")
                    if options["prepare_only"]:
                        return
                elif not run_dir.is_dir():
                    raise FormalCodexJudgeError(
                        f"Cannot resume missing Judge run directory: {run_dir}"
                    )

                batch_judge = make_codex_cli_batch_judge(
                    model=str(options["judge_model"]),
                    reasoning_effort=str(options["reasoning_effort"]),
                    timeout_seconds=int(options["timeout_seconds"]),
                    raw_batch_dir=run_dir / "raw_batches",
                )

                def progress(payload) -> None:  # noqa: ANN001
                    self.stdout.write(
                        "Codex Judge "
                        f"{payload['phase']}: {payload['completed']}/{payload['total']} "
                        f"(last batch {payload['last_batch_seconds']}s; "
                        f"next={payload['next_candidate_id']})"
                    )

                checkpoint = run_formal_codex_judge_phase(
                    run_dir=run_dir,
                    run_id=run_id,
                    phase=str(options["phase"]),
                    batch_size=int(options["batch_size"]),
                    batch_judge=batch_judge,
                    resume_command=resume_command,
                    max_batches=options["max_batches"],
                    max_batch_retries=int(options["max_batch_retries"]),
                    progress_callback=progress,
                )
        except (FormalCodexJudgeError, FormalBenchmarkError) as exc:
            raise CommandError(str(exc)) from exc
        self.stdout.write(self.style.SUCCESS(f"Judge status: {checkpoint['status']}"))
        self.stdout.write(f"Progress: {run_dir / f'{run_id}_progress.md'}")
