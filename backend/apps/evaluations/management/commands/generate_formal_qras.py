"""Generate formal QRA review candidates from frozen Configurations."""

from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from apps.evaluations.formal_qra_generation import generate_formal_qras
from apps.evaluations.rageval_artifacts import ArtifactValidationError


class Command(BaseCommand):
    """Expose reproducible formal QRA candidate generation."""

    help = "Generate 60 formal-test QRA review candidates from frozen C files."

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--configuration-manifest",
            default=(
                "docs/evaluation/formal_dataset/configurations/formal_test/"
                "formal_test_configuration_manifest_v1.0.json"
            ),
        )
        parser.add_argument("--generated-at", default="2026-07-31")
        parser.add_argument("--qra-version", default="0.1")
        parser.add_argument(
            "--semantic-curation",
            default=(
                "docs/evaluation/formal_dataset/qra/"
                "qra_semantic_curation_v0.1.json"
            ),
        )
        parser.add_argument("--replace-existing", action="store_true")
        parser.add_argument(
            "--output-dir",
            default="docs/evaluation/formal_dataset/qra/formal_candidates",
        )
        parser.add_argument("--manifest-id", default=None)
        parser.add_argument("--semantic-curation-id", default=None)
        parser.add_argument("--dataset-label", default="Formal")

    def handle(self, *args, **options) -> None:
        try:
            report = generate_formal_qras(
                repo_root=Path(settings.BASE_DIR).parent,
                configuration_manifest_path=Path(
                    options["configuration_manifest"]
                ),
                generated_at=str(options["generated_at"]),
                qra_version=str(options["qra_version"]),
                semantic_curation_path=Path(options["semantic_curation"]),
                replace_existing=bool(options["replace_existing"]),
                output_dir=Path(options["output_dir"]),
                manifest_id=options["manifest_id"],
                semantic_curation_id=options["semantic_curation_id"],
                dataset_label=str(options["dataset_label"]),
            )
        except ArtifactValidationError as exc:
            raise CommandError(str(exc)) from exc

        self.stdout.write(
            self.style.SUCCESS(
                "Formal QRA candidate generation complete: "
                f"{report['qra_count']} records; "
                f"{report['created_count']} created, "
                f"{report['replaced_count']} replaced, "
                f"{report['unchanged_count']} unchanged."
            )
        )
        self.stdout.write(f"  {report['manifest_path']}")
        self.stdout.write(f"  {report['review_path']}")
