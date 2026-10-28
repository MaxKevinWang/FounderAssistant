from __future__ import annotations

"""Minimal notebook API for resumable BMWE FAQ crawling."""

import os
import sys
from pathlib import Path
from typing import Any

import django


def _find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "backend" / "manage.py").exists():
            return candidate
    raise RuntimeError("Repository root not found from current notebook location.")


_REPO_ROOT = _find_repo_root(Path(__file__).resolve())
_BACKEND_DIR = _REPO_ROOT / "backend"
_DOCS_DIR = _REPO_ROOT / "docs" / "problem_datasets"

if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

from apps.advisors.faq_dataset import (  # noqa: E402
    ALLOWED_DECISION_DOMAINS,
    DEFAULT_START_URL,
    FAQCrawlConfig,
    crawl_existenzgruendung_faq_to_dataset,
)


def run_faq_crawl(
    *,
    start_url: str = DEFAULT_START_URL,
    resume: bool = True,
    output_json: str | None = None,
    output_markdown: str | None = None,
    split_json: str | None = None,
    state_path: str | None = None,
    crawl_delay_seconds: float = 30.0,
    timeout_seconds: int = 30,
    retries: int = 3,
    retry_backoff_seconds: float = 3.0,
    decision_domains: tuple[str, ...] = ALLOWED_DECISION_DOMAINS,
    min_stand_year: int | None = None,
    max_stand_year: int | None = None,
    max_list_pages: int | None = None,
    max_faqs: int | None = None,
    continue_on_error: bool = True,
) -> dict[str, Any]:
    """Run resumable crawl and print concise progress summary."""

    resolved_output_json = Path(output_json) if output_json else (_DOCS_DIR / "existenzgruendung_faq_problem_dataset.json")
    resolved_output_md = (
        Path(output_markdown) if output_markdown else (_DOCS_DIR / "existenzgruendung_faq_problem_dataset.md")
    )
    resolved_split_json = (
        Path(split_json) if split_json else (_DOCS_DIR / "existenzgruendung_faq_problem_dataset_splits.json")
    )
    resolved_state_path = Path(state_path) if state_path else None

    config = FAQCrawlConfig(
        start_url=start_url,
        crawl_delay_seconds=crawl_delay_seconds,
        timeout_seconds=timeout_seconds,
        max_list_pages=max_list_pages,
        max_faqs=max_faqs,
        retries=retries,
        retry_backoff_seconds=retry_backoff_seconds,
        user_agent="FounderPilotFAQCrawler/1.0 (+local thesis demo crawler)",
        decision_domains=decision_domains,
        min_stand_year=min_stand_year,
        max_stand_year=max_stand_year,
        continue_on_error=continue_on_error,
    )

    result = crawl_existenzgruendung_faq_to_dataset(
        config=config,
        output_json_path=resolved_output_json,
        output_markdown_path=resolved_output_md,
        split_output_path=resolved_split_json,
        state_path=resolved_state_path,
        resume=resume,
    )

    print(f"Status: {result.get('status')}")
    print(f"Items: {result.get('items')}")
    print(f"List pages crawled (this run): {result.get('list_pages_crawled')}")
    print(f"Detail pages crawled (this run): {result.get('detail_pages_crawled')}")
    print(f"Failed detail pages (this run): {result.get('failed_detail_count')}")
    print(f"Retried failed details: {result.get('retried_failed_recovered')}/{result.get('retried_failed_attempted')}")
    print(f"Next list URL: {result.get('next_list_url') or '<done>'}")
    print(f"State file: {result.get('state_path')}")
    print(f"JSON output: {result.get('output_json_path')}")

    return result
