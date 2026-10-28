"""Deduplicate the FAQ problem dataset with exact + lexical + semantic stages."""

from __future__ import annotations

from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from apps.advisors.faq_dedup import default_faq_dedup_config, deduplicate_faq_dataset
from apps.advisors.rag_pipeline import RAGError


class Command(BaseCommand):
    help = (
        "Deduplicate FAQ problem dataset using exact URL/hash, lexical similarity, "
        "and semantic embedding similarity."
    )

    def add_arguments(self, parser):
        defaults = default_faq_dedup_config()

        parser.add_argument("--input-json", type=str, default=str(defaults.input_json_path))
        parser.add_argument("--output-clean-json", type=str, default=str(defaults.output_clean_json_path))
        parser.add_argument("--output-map-json", type=str, default=str(defaults.output_map_json_path))
        parser.add_argument("--output-review-csv", type=str, default=str(defaults.output_review_csv_path))
        parser.add_argument("--output-summary-md", type=str, default=str(defaults.output_summary_md_path))

        parser.add_argument("--embedding-model", type=str, default=defaults.embedding_model)
        parser.add_argument("--llm-base-url", type=str, default=defaults.llm_base_url)
        parser.add_argument("--timeout", type=int, default=defaults.timeout_seconds)

        parser.add_argument("--lexical-threshold", type=float, default=defaults.lexical_threshold)
        parser.add_argument("--lexical-review-margin", type=float, default=defaults.lexical_review_margin)
        parser.add_argument("--semantic-threshold", type=float, default=defaults.semantic_threshold)
        parser.add_argument("--semantic-review-margin", type=float, default=defaults.semantic_review_margin)
        parser.add_argument("--min-shared-tokens", type=int, default=defaults.min_shared_tokens)
        parser.add_argument("--max-token-doc-frequency", type=int, default=defaults.max_token_doc_frequency)
        parser.add_argument("--max-length-delta-ratio", type=float, default=defaults.max_length_delta_ratio)
        parser.add_argument(
            "--allow-cross-domain-semantic",
            action="store_true",
            help="Allow semantic merges across different decision domains (default: disabled).",
        )

    def handle(self, *args, **options):
        defaults = default_faq_dedup_config()
        config = defaults.__class__(
            input_json_path=Path(options["input_json"]),
            output_clean_json_path=Path(options["output_clean_json"]),
            output_map_json_path=Path(options["output_map_json"]),
            output_review_csv_path=Path(options["output_review_csv"]),
            output_summary_md_path=Path(options["output_summary_md"]),
            embedding_model=str(options["embedding_model"]),
            llm_base_url=str(options["llm_base_url"]),
            timeout_seconds=int(options["timeout"]),
            lexical_threshold=float(options["lexical_threshold"]),
            lexical_review_margin=float(options["lexical_review_margin"]),
            semantic_threshold=float(options["semantic_threshold"]),
            semantic_review_margin=float(options["semantic_review_margin"]),
            min_shared_tokens=int(options["min_shared_tokens"]),
            max_token_doc_frequency=int(options["max_token_doc_frequency"]),
            max_length_delta_ratio=float(options["max_length_delta_ratio"]),
            require_same_domain_for_semantic=not bool(options["allow_cross_domain_semantic"]),
        )

        try:
            result = deduplicate_faq_dataset(config=config)
        except RAGError as exc:
            raise CommandError(str(exc)) from exc

        self.stdout.write(self.style.SUCCESS("FAQ dataset deduplication finished."))
        self.stdout.write(f"Input: {result['input_json_path']}")
        self.stdout.write(f"Clean JSON: {result['output_clean_json_path']}")
        self.stdout.write(f"Dedup map: {result['output_map_json_path']}")
        self.stdout.write(f"Review CSV: {result['output_review_csv_path']}")
        self.stdout.write(f"Summary MD: {result['output_summary_md_path']}")
        self.stdout.write(f"Original items: {result['original_items']}")
        self.stdout.write(f"Cleaned items: {result['cleaned_items']}")
        self.stdout.write(f"Removed items: {result['removed_items']}")
        self.stdout.write(f"Clusters (>1): {result['cluster_count_multi']}")
        self.stdout.write(
            "Merge counts | exact_url: "
            f"{result['exact_url_merges']} | exact_qa: {result['exact_qa_merges']} | "
            f"lexical: {result['lexical_merges']} | semantic: {result['semantic_merges']}"
        )
        self.stdout.write(
            f"Review rows | lexical: {result['lexical_review_rows']} | semantic: {result['semantic_review_rows']}"
        )
        self.stdout.write(f"Duration: {result['duration_ms']} ms")
