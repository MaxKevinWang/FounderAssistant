"""Run the frozen formal-test QRA release with safe formal defaults."""

from apps.evaluations.formal_development_benchmark import (
    default_formal_output_dir,
    default_formal_prompt_path,
    default_formal_qra_dir,
)
from apps.evaluations.management.commands.run_formal_development_eval import (
    Command as SharedCommand,
)


class Command(SharedCommand):
    """Formal-test entry point backed by the shared resumable runner."""

    help = (
        "Run unscored Baseline/RAG generations on the frozen formal-test QRA "
        "release with the frozen formal prompt and resumable checkpoints."
    )

    def add_arguments(self, parser) -> None:
        super().add_arguments(parser)
        parser.set_defaults(
            qra_dir=str(default_formal_qra_dir()),
            output_dir=str(default_formal_output_dir()),
            prompt_path=str(default_formal_prompt_path()),
        )
