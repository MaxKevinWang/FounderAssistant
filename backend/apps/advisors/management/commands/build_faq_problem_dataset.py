"""Build a FAQ-based problem dataset from Existenzgruendungsportal pages.

Use case:
- Crawl the BMWE FAQ pool (paginated list + detail pages).
- Export structured Q/A records for benchmark design and holdout sampling.
"""

from __future__ import annotations

from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from apps.advisors.faq_dataset import (
    ALLOWED_DECISION_DOMAINS,
    DEFAULT_START_URL,
    DEFAULT_USER_AGENT,
    FAQCrawlConfig,
    crawl_existenzgruendung_faq_to_dataset,
)
from apps.advisors.rag_pipeline import RAGError


def _default_output_json_path() -> Path:
    """Resolve default dataset path under project-level docs directory."""

    return Path(settings.BASE_DIR).parent / "docs" / "problem_datasets" / "existenzgruendung_faq_problem_dataset.json"


def _default_output_markdown_path() -> Path:
    """Resolve default markdown summary path under docs/problem_datasets."""

    return Path(settings.BASE_DIR).parent / "docs" / "problem_datasets" / "existenzgruendung_faq_problem_dataset.md"


def _default_split_json_path() -> Path:
    """Resolve default split JSON path for train/dev/test IDs."""

    return (
        Path(settings.BASE_DIR).parent
        / "docs"
        / "problem_datasets"
        / "existenzgruendung_faq_problem_dataset_splits.json"
    )


class Command(BaseCommand):
    """Django wrapper around FAQ crawl-and-export utility."""

    help = "Crawl Existenzgruendungsportal FAQ and export a structured problem dataset (JSON + summary markdown)."

    def add_arguments(self, parser):
        parser.add_argument(
            "--start-url",
            type=str,
            default=DEFAULT_START_URL,
            help="FAQ list URL to start from.",
        )
        parser.add_argument(
            "--output-json",
            type=str,
            default=str(_default_output_json_path()),
            help="Output JSON dataset path.",
        )
        parser.add_argument(
            "--output-md",
            type=str,
            default=str(_default_output_markdown_path()),
            help="Output markdown summary path.",
        )
        parser.add_argument(
            "--state-path",
            type=str,
            default=None,
            help="Optional resume state JSON path (defaults next to output JSON).",
        )
        parser.add_argument(
            "--split-json",
            type=str,
            default=str(_default_split_json_path()),
            help="Output split JSON path (train/dev/test IDs).",
        )
        parser.add_argument(
            "--no-split",
            action="store_true",
            help="Disable split file generation.",
        )
        parser.add_argument(
            "--max-list-pages",
            type=int,
            default=None,
            help="Optional cap for crawled list pages (useful for smoke tests).",
        )
        parser.add_argument(
            "--max-faqs",
            type=int,
            default=None,
            help="Optional cap for crawled FAQ detail items (useful for smoke tests).",
        )
        parser.add_argument(
            "--crawl-delay",
            type=float,
            default=30.0,
            help="Delay between HTTP requests in seconds (robots.txt currently indicates 30s).",
        )
        parser.add_argument(
            "--timeout",
            type=int,
            default=30,
            help="HTTP timeout in seconds.",
        )
        parser.add_argument(
            "--retries",
            type=int,
            default=3,
            help="Retry attempts per request.",
        )
        parser.add_argument(
            "--retry-backoff",
            type=float,
            default=3.0,
            help="Retry backoff base in seconds (exponential).",
        )
        parser.add_argument(
            "--user-agent",
            type=str,
            default=DEFAULT_USER_AGENT,
            help="HTTP User-Agent header.",
        )
        parser.add_argument(
            "--decision-domain",
            action="append",
            choices=list(ALLOWED_DECISION_DOMAINS),
            dest="decision_domains",
            default=None,
            help=(
                "Limit output to selected decision domains. Repeat to include multiple domains. "
                "Default: all thesis domains."
            ),
        )
        parser.add_argument(
            "--min-stand-year",
            type=int,
            default=None,
            help="Keep only records with stand year >= this value.",
        )
        parser.add_argument(
            "--max-stand-year",
            type=int,
            default=None,
            help="Keep only records with stand year <= this value.",
        )
        parser.add_argument(
            "--fail-fast",
            action="store_true",
            help="Stop immediately when one FAQ detail page fails.",
        )
        parser.add_argument(
            "--no-resume",
            action="store_true",
            help="Ignore existing output/state files and start a fresh crawl.",
        )

    def handle(self, *args, **options):
        output_json = Path(options["output_json"])
        output_md = Path(options["output_md"])
        split_json = None if options.get("no_split") else Path(options["split_json"])
        state_path = Path(options["state_path"]) if options.get("state_path") else None
        resume = not bool(options.get("no_resume"))
        selected_domains = tuple(options.get("decision_domains") or ALLOWED_DECISION_DOMAINS)

        config = FAQCrawlConfig(
            start_url=str(options["start_url"]),
            crawl_delay_seconds=float(options["crawl_delay"]),
            timeout_seconds=int(options["timeout"]),
            max_list_pages=options.get("max_list_pages"),
            max_faqs=options.get("max_faqs"),
            retries=int(options["retries"]),
            retry_backoff_seconds=float(options["retry_backoff"]),
            user_agent=str(options["user_agent"]),
            decision_domains=selected_domains,
            min_stand_year=options.get("min_stand_year"),
            max_stand_year=options.get("max_stand_year"),
            continue_on_error=not bool(options.get("fail_fast")),
        )

        try:
            result = crawl_existenzgruendung_faq_to_dataset(
                config=config,
                output_json_path=output_json,
                output_markdown_path=output_md,
                split_output_path=split_json,
                state_path=state_path,
                resume=resume,
            )
        except RAGError as exc:
            raise CommandError(str(exc)) from exc

        status_style = self.style.WARNING if result.get("status") == "partial" else self.style.SUCCESS
        self.stdout.write(status_style(f"FAQ problem dataset build {result.get('status')}."))
        self.stdout.write(f"JSON: {result['output_json_path']}")
        self.stdout.write(f"Markdown: {result['output_markdown_path']}")
        if result.get("split_output_path"):
            self.stdout.write(f"Splits: {result['split_output_path']}")
        self.stdout.write(f"State: {result['state_path']}")
        self.stdout.write(f"Items: {result['items']}")
        self.stdout.write(f"List pages crawled this run: {result['list_pages_crawled']}")
        self.stdout.write(f"Detail pages crawled this run: {result['detail_pages_crawled']}")
        self.stdout.write(f"Skipped by domain: {result.get('skipped_by_domain', 0)}")
        self.stdout.write(f"Skipped by stand-year: {result.get('skipped_by_year', 0)}")
        self.stdout.write(f"Failed detail pages: {result.get('failed_detail_count', 0)}")
        if result.get("split_counts"):
            self.stdout.write(f"Split counts: {result['split_counts']}")
        self.stdout.write(f"Portal total hint: {result.get('portal_total_entries_hint')}")
        self.stdout.write(f"Crawl delay: {result['crawl_delay_seconds']} s")
        if result.get("last_error_message"):
            self.stdout.write(f"Last error: {result['last_error_message']}")
        self.stdout.write(f"Duration: {result['duration_ms']} ms")
