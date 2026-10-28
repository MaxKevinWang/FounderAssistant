"""Fetch web pages and store them as local HTML files under `knowledge_base/`.

Purpose:
- Quickly collect startup-relevant web references for the corpus.
- Keep raw HTML snapshots reproducible for later RAG ingestion/chunking.
"""

from __future__ import annotations

import re
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError


def _sanitize_path_fragment(value: str) -> str:
    """Normalize arbitrary text into a filesystem-safe fragment."""

    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip())
    cleaned = cleaned.strip("-._")
    return cleaned or "source"


def _normalize_domain_label(domain: str) -> str:
    """Constrain corpus domain label to avoid unsafe path traversal."""

    safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", domain.strip())
    return safe.strip("_") or "web_sources"


def _default_data_dir() -> Path:
    """Resolve default corpus directory from Django settings."""

    configured = getattr(settings, "LOCAL_RAG_DATA_DIR", None)
    if configured:
        return Path(configured)
    return Path(settings.BASE_DIR).parent / "knowledge_base"


class Command(BaseCommand):
    """Download one or more URLs and save each response as `.html`."""

    help = "Fetch URLs and save HTML snapshots into local knowledge_base/<domain>/web_sources/."

    def add_arguments(self, parser):
        """Expose CLI options for URL, output domain, and fetch behavior."""

        parser.add_argument(
            "urls",
            nargs="*",
            help="One or more URLs to fetch. You can also pass repeated --url options.",
        )
        parser.add_argument(
            "--url",
            action="append",
            dest="url_options",
            default=[],
            help="Additional URL (repeatable).",
        )
        parser.add_argument(
            "--domain",
            type=str,
            default="market_validation",
            help="Top-level data domain folder (default: market_validation).",
        )
        parser.add_argument(
            "--data-dir",
            type=str,
            default=None,
            help="Override corpus data directory path.",
        )
        parser.add_argument(
            "--filename",
            type=str,
            default=None,
            help="Optional fixed file name (single URL only).",
        )
        parser.add_argument(
            "--timeout",
            type=int,
            default=30,
            help="Network timeout seconds (default: 30).",
        )
        parser.add_argument(
            "--user-agent",
            type=str,
            default="FounderPilotCrawler/1.0 (+local thesis demo crawler)",
            help="HTTP User-Agent header.",
        )

    def handle(self, *args, **options):
        """Run download loop and print where files were saved."""

        urls = list(dict.fromkeys([*(options.get("urls") or []), *(options.get("url_options") or [])]))
        if not urls:
            raise CommandError("Please provide at least one URL.")

        if options.get("filename") and len(urls) != 1:
            raise CommandError("--filename can only be used when exactly one URL is provided.")

        timeout = int(options.get("timeout") or 30)
        if timeout <= 0:
            raise CommandError("--timeout must be greater than 0.")

        data_dir = Path(options["data_dir"]) if options.get("data_dir") else _default_data_dir()
        domain = _normalize_domain_label(options.get("domain") or "web_sources")
        target_dir = data_dir / domain / "web_sources"
        target_dir.mkdir(parents=True, exist_ok=True)

        user_agent = options.get("user_agent") or "FounderPilotCrawler/1.0"
        saved_paths: list[Path] = []

        for index, url in enumerate(urls):
            parsed = urlparse(url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise CommandError(f"Invalid URL: {url}")

            filename_override = options.get("filename")
            output_name = filename_override if filename_override and index == 0 else None
            output_path = self._build_output_path(target_dir, url, output_name)

            html_payload = self._fetch_html(url, timeout=timeout, user_agent=user_agent)
            output_path.write_text(html_payload, encoding="utf-8")
            saved_paths.append(output_path)
            self.stdout.write(self.style.SUCCESS(f"Saved: {output_path}"))

        self.stdout.write(self.style.SUCCESS(f"Completed. Saved {len(saved_paths)} file(s)."))

    @staticmethod
    def _build_output_path(target_dir: Path, url: str, fixed_name: str | None) -> Path:
        """Create deterministic output path from URL (or user provided file name)."""

        if fixed_name:
            base_name = fixed_name.strip()
            if not base_name.lower().endswith(".html"):
                base_name = f"{base_name}.html"
            safe_stem = _sanitize_path_fragment(base_name[:-5] if base_name.lower().endswith(".html") else base_name)
            return (target_dir / safe_stem).with_suffix(".html")

        parsed = urlparse(url)
        host = _sanitize_path_fragment(parsed.netloc.replace(":", "_"))
        page_name = _sanitize_path_fragment(Path(parsed.path).stem or "index")
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        return target_dir / f"{host}__{page_name}__{timestamp}.html"

    @staticmethod
    def _fetch_html(url: str, timeout: int, user_agent: str) -> str:
        """Fetch URL response, decode to text, and prepend snapshot metadata."""

        request = urllib.request.Request(
            url=url,
            headers={
                "User-Agent": user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read()
                content_type = response.headers.get("Content-Type", "")
                charset = response.headers.get_content_charset()
                final_url = response.geturl()
        except urllib.error.HTTPError as exc:
            raise CommandError(f"HTTP error while fetching {url}: {exc}") from exc
        except urllib.error.URLError as exc:
            raise CommandError(f"Network error while fetching {url}: {exc}") from exc

        html_text = Command._decode_body(body, charset)
        fetched_at = datetime.now(UTC).isoformat()
        header = (
            "<!--\n"
            f"source_url: {final_url}\n"
            f"fetched_at_utc: {fetched_at}\n"
            f"content_type: {content_type}\n"
            "-->\n"
        )
        return f"{header}{html_text}"

    @staticmethod
    def _decode_body(body: bytes, suggested_charset: str | None) -> str:
        """Decode response bytes with charset fallback strategy."""

        tried: set[str] = set()
        for encoding in [suggested_charset, "utf-8", "cp1252", "latin-1"]:
            if not encoding:
                continue
            if encoding in tried:
                continue
            tried.add(encoding)
            try:
                return body.decode(encoding)
            except UnicodeDecodeError:
                continue
        return body.decode("utf-8", errors="replace")
