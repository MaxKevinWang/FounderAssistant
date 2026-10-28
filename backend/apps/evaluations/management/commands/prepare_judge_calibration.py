"""Prepare corrected pipeline comparison and blinded human calibration packet."""

from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from apps.evaluations.judge_calibration import (
    JudgeCalibrationError,
    prepare_calibration_artifacts,
)


class Command(BaseCommand):
    help = "Consolidate Judge corrections and prepare a blinded calibration sample."

    def add_arguments(self, parser) -> None:
        parser.add_argument("--primary-scores", type=Path, required=True)
        parser.add_argument(
            "--correction-scores", type=Path, action="append", default=[]
        )
        parser.add_argument("--blinded-inputs", type=Path, required=True)
        parser.add_argument("--output-dir", type=Path, required=True)
        parser.add_argument("--analysis-id", required=True)
        parser.add_argument("--sample-size", type=int, default=24)
        parser.add_argument("--seed", type=int, default=20260807)

    def handle(self, *args, **options) -> None:
        try:
            paths = prepare_calibration_artifacts(
                primary_scores_path=options["primary_scores"],
                correction_scores_paths=options["correction_scores"],
                blinded_inputs_path=options["blinded_inputs"],
                output_dir=options["output_dir"],
                analysis_id=options["analysis_id"],
                sample_size=options["sample_size"],
                seed=options["seed"],
            )
        except (JudgeCalibrationError, OSError, ValueError) as exc:
            raise CommandError(str(exc)) from exc
        for name, path in paths.items():
            self.stdout.write(f"{name}: {path.resolve()}")
