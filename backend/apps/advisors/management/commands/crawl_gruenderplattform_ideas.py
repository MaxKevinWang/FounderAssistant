"""Crawl Gruenderplattform business idea overview + detail pages."""

from __future__ import annotations

from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from apps.advisors.gruenderplattform_ideas import (
    DEFAULT_DOMAIN,
    DEFAULT_OUTPUT_SUBFOLDER,
    DEFAULT_OVERVIEW_URL,
    DEFAULT_USER_AGENT,
    GruenderplattformCrawlConfig,
    crawl_gruenderplattform_business_ideas,
    extract_business_idea_links,
)
from apps.advisors.rag_pipeline import RAGError


def _default_data_dir() -> Path:
    configured = getattr(settings, "LOCAL_RAG_DATA_DIR", None)
    return Path(configured) if configured else Path(settings.BASE_DIR).parent / "knowledge_base"


def _default_state_path() -> Path:
    return Path(settings.BASE_DIR).parent / "docs" / "crawl_states" / "gruenderplattform_business_ideas_crawl_state.json"


class Command(BaseCommand):
    """Django command wrapper for the Gruenderplattform idea crawler."""

    help = "Crawl Gruenderplattform business idea pages into knowledge_base/market_validation as cleaned HTML."

    def add_arguments(self, parser):
        parser.add_argument(
            "--overview-url",
            type=str,
            default=DEFAULT_OVERVIEW_URL,
            help="Business idea overview page URL.",
        )
        parser.add_argument(
            "--data-dir",
            type=str,
            default=str(_default_data_dir()),
            help="Corpus data directory.",
        )
        parser.add_argument(
            "--domain",
            type=str,
            default=DEFAULT_DOMAIN,
            help="Output top-level data folder.",
        )
        parser.add_argument(
            "--subfolder",
            type=str,
            default=DEFAULT_OUTPUT_SUBFOLDER,
            help="Output subfolder inside the selected domain folder.",
        )
        parser.add_argument(
            "--state-path",
            type=str,
            default=str(_default_state_path()),
            help="Progress/summary JSON path outside the RAG data directory.",
        )
        parser.add_argument(
            "--max-pages",
            type=int,
            default=None,
            help="Optional cap for detail pages. Useful for smoke tests.",
        )
        parser.add_argument(
            "--no-overview",
            action="store_true",
            help="Do not save the overview page itself, only detail pages.",
        )
        parser.add_argument(
            "--force-refresh",
            action="store_true",
            help="Re-fetch pages even when the same source_url already exists.",
        )
        parser.add_argument(
            "--crawl-delay",
            type=float,
            default=1.0,
            help="Delay between HTTP requests in seconds.",
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
            default=2.0,
            help="Retry backoff base in seconds.",
        )
        parser.add_argument(
            "--user-agent",
            type=str,
            default=DEFAULT_USER_AGENT,
            help="HTTP User-Agent header.",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Only fetch the overview page and print discovered detail links.",
        )

    def handle(self, *args, **options):
        data_dir = Path(options["data_dir"])
        state_path = Path(options["state_path"])
        overview_url = str(options["overview_url"])

        if options.get("dry_run"):
            self._run_dry_run(
                overview_url=overview_url,
                timeout=int(options["timeout"]),
                user_agent=str(options["user_agent"]),
            )
            return

        max_pages = options.get("max_pages")
        if max_pages is not None and int(max_pages) < 0:
            raise CommandError("--max-pages must be zero or greater.")

        config = GruenderplattformCrawlConfig(
            overview_url=overview_url,
            data_dir=data_dir,
            output_domain=str(options["domain"]),
            output_subfolder=str(options["subfolder"]),
            crawl_delay_seconds=float(options["crawl_delay"]),
            timeout_seconds=int(options["timeout"]),
            retries=int(options["retries"]),
            retry_backoff_seconds=float(options["retry_backoff"]),
            user_agent=str(options["user_agent"]),
            max_pages=max_pages,
            include_overview=not bool(options["no_overview"]),
            force_refresh=bool(options["force_refresh"]),
            state_path=state_path,
        )

        try:
            result = crawl_gruenderplattform_business_ideas(config)
        except RAGError as exc:
            raise CommandError(str(exc)) from exc

        self.stdout.write(self.style.SUCCESS("Gruenderplattform business idea crawl finished."))
        self.stdout.write(f"Detail links discovered: {result['detail_link_count']}")
        self.stdout.write(f"Detail links selected: {result['selected_detail_count']}")
        self.stdout.write(f"Saved: {result['saved_count']}")
        self.stdout.write(f"Skipped: {result['skipped_count']}")
        self.stdout.write(f"Failed: {result['failed_count']}")
        self.stdout.write(f"Output dir: {result['output_dir']}")
        self.stdout.write(f"State file: {result['state_path']}")
        if result["failed_count"]:
            for item in result["failed"][:10]:
                self.stderr.write(f"FAILED {item['url']}: {item['error']}")

    def _run_dry_run(self, *, overview_url: str, timeout: int, user_agent: str) -> None:
        import urllib.request

        request = urllib.request.Request(
            overview_url,
            headers={
                "User-Agent": user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            },
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            html = response.read().decode(response.headers.get_content_charset() or "utf-8", errors="replace")
            final_url = response.geturl()

        links = extract_business_idea_links(html, final_url)
        self.stdout.write(self.style.SUCCESS(f"Discovered {len(links)} detail link(s)."))
        for link in links:
            self.stdout.write(link)
