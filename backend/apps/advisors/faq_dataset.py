"""Crawl and transform BMWE FAQ pages into a reusable problem dataset.

This module intentionally avoids third-party HTML parsers so it can run with
the project's current lightweight dependency set.

Design goals:
1) Crawl paginated FAQ list pages.
2) Visit FAQ detail pages and extract question/answer/date fields.
3) Map entries to thesis decision domains for downstream filtering.
4) Persist JSON/Markdown outputs and optional resume state.
"""

from __future__ import annotations

import json
import hashlib
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from html import unescape
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

from .rag_pipeline import RAGError

DEFAULT_START_URL = (
    "https://www.existenzgruendungsportal.de/Navigation/DE/Infopool/FragenUndAntworten/infopool"
)
DEFAULT_USER_AGENT = "FounderPilotFAQCrawler/1.0 (+local thesis demo crawler)"
ALLOWED_DECISION_DOMAINS = ("legal_form_selection", "funding_preparation", "market_validation")

DETAIL_URL_RE = re.compile(
    r'href="(?P<url>https://www\.existenzgruendungsportal\.de/Redaktion/DE/BMWK-Infopool/Antworten/[^"#?]+)"',
    re.IGNORECASE,
)
NEXT_PAGE_RE = re.compile(
    r'<a[^>]*class="[^"]*pagination-link[^"]*"[^>]*aria-label="Next"[^>]*href="(?P<url>[^"]+)"',
    re.IGNORECASE,
)
TOTAL_ENTRIES_RE = re.compile(r"iNumTotalEntries\s+(\d+)")
QUESTION_HEADER_RE = re.compile(r"<h2>\s*Frage\s*</h2>", re.IGNORECASE)
ANSWER_HEADER_RE = re.compile(r"<h2>\s*Antwort\s*</h2>", re.IGNORECASE)
SECTION_RELATED_MARKER = '<div class="sectionRelated"'
STAND_RE = re.compile(
    r"<strong>\s*Stand:\s*</strong>\s*<br\s*/?>\s*(?P<value>.*?)</p>",
    re.IGNORECASE | re.DOTALL,
)
STAND_TEXT_RE = re.compile(
    r"(?im)^\s*stand\s*:\s*(?P<value>[^\n]+?)\s*$",
)
TITLE_RE = re.compile(
    r'<meta\s+property="og:title"\s+content="(?P<value>.*?)"\s*/?>',
    re.IGNORECASE | re.DOTALL,
)
PARAGRAPH_RE = re.compile(r"<p[^>]*>(?P<body>.*?)</p>", re.IGNORECASE | re.DOTALL)

LEGAL_DOMAIN_KEYWORDS = {
    "recht",
    "steuern",
    "gruendungsplanung",
    "freie-berufe",
    "rechtsformen",
    "gewerbe",
    "genehmigungen",
    "handwerk",
    "versicherungen",
}
FUNDING_DOMAIN_KEYWORDS = {
    "foerderung",
    "finanzierung",
    "arbeitsfoerderung",
    "gruendungszuschuss",
    "darlehen",
    "wagniskapital",
    "kapital",
    "liquiditaet",
    "investor",
}
MARKET_DOMAIN_KEYWORDS = {
    "kunden-gewinnen",
    "vertrieb",
    "werbung",
    "e-business",
    "online-marketing",
    "online-shops",
    "auslandsgeschaefte",
    "export",
    "import",
}


@dataclass(frozen=True)
class FAQCrawlConfig:
    """Runtime options for the FAQ crawler."""

    start_url: str
    crawl_delay_seconds: float
    timeout_seconds: int
    max_list_pages: int | None
    max_faqs: int | None
    retries: int
    retry_backoff_seconds: float
    user_agent: str
    decision_domains: tuple[str, ...]
    min_stand_year: int | None
    max_stand_year: int | None
    continue_on_error: bool


def _decode_body(body: bytes, suggested_charset: str | None) -> str:
    """Decode HTTP response bytes using a conservative fallback chain."""

    tried: set[str] = set()
    for encoding in [suggested_charset, "utf-8", "cp1252", "latin-1"]:
        if not encoding or encoding in tried:
            continue
        tried.add(encoding)
        try:
            return body.decode(encoding)
        except UnicodeDecodeError:
            continue
    return body.decode("utf-8", errors="replace")


def _fetch_html(url: str, *, timeout: int, user_agent: str, retries: int, retry_backoff_seconds: float) -> str:
    """Fetch one URL and return decoded HTML text with retry support."""

    request = urllib.request.Request(
        url=url,
        headers={
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
    )

    last_error: Exception | None = None
    for attempt in range(max(1, retries)):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read()
                charset = response.headers.get_content_charset()
            return _decode_body(body, charset)
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            if attempt + 1 >= max(1, retries):
                break
            sleep_seconds = retry_backoff_seconds * (2**attempt)
            time.sleep(max(0.1, sleep_seconds))

    raise RAGError(f"Failed to fetch URL after retries: {url}. Error: {last_error}")


def _sleep_for_crawl_delay(last_request_timestamp: float | None, crawl_delay_seconds: float) -> None:
    """Enforce a minimum delay between consecutive requests."""

    if last_request_timestamp is None:
        return
    if crawl_delay_seconds <= 0:
        return
    elapsed = time.monotonic() - last_request_timestamp
    remaining = crawl_delay_seconds - elapsed
    if remaining > 0:
        time.sleep(remaining)


def _extract_detail_urls(list_html: str) -> list[str]:
    """Extract FAQ detail URLs from one list page."""

    found = [match.group("url").strip() for match in DETAIL_URL_RE.finditer(list_html)]
    # Preserve order while removing duplicates from repeated overlays.
    return list(dict.fromkeys(found))


def _extract_next_page_url(list_html: str, base_url: str) -> str | None:
    """Extract pagination next URL, if present."""

    match = NEXT_PAGE_RE.search(list_html)
    if not match:
        return None
    return urljoin(base_url, match.group("url").strip())


def _extract_total_entries_hint(list_html: str) -> int | None:
    """Extract the total-entry hint shown on list pages (if present)."""

    match = TOTAL_ENTRIES_RE.search(list_html)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _html_to_text(fragment: str) -> str:
    """Convert an HTML fragment into normalized plain text."""

    text = fragment
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</p>", "\n\n", text)
    text = re.sub(r"(?i)</li>", "\n", text)
    text = re.sub(r"(?i)<li[^>]*>", "- ", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = unescape(text)
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _extract_title(detail_html: str) -> str:
    """Extract FAQ title from OpenGraph meta tag."""

    match = TITLE_RE.search(detail_html)
    if not match:
        return ""
    return _html_to_text(match.group("value"))


def _extract_question_answer_sections(detail_html: str) -> tuple[str, str]:
    """Extract raw HTML fragments for question and answer sections."""

    q_match = QUESTION_HEADER_RE.search(detail_html)
    a_match = ANSWER_HEADER_RE.search(detail_html)
    if not q_match or not a_match:
        raise RAGError("Could not locate 'Frage'/'Antwort' sections in detail page.")
    if a_match.start() <= q_match.end():
        raise RAGError("Invalid FAQ structure: answer section appears before question section.")

    question_fragment = detail_html[q_match.end() : a_match.start()]

    answer_start = a_match.end()
    section_related_index = detail_html.find(SECTION_RELATED_MARKER, answer_start)
    if section_related_index == -1:
        answer_fragment = detail_html[answer_start:]
    else:
        answer_fragment = detail_html[answer_start:section_related_index]

    return question_fragment, answer_fragment


def _extract_stand_value(answer_fragment: str) -> str:
    """Extract 'Stand' (date/version) field from answer fragment."""

    match = STAND_RE.search(answer_fragment)
    if match:
        value = _html_to_text(match.group("value"))
        if value:
            return value

    plain_text = _html_to_text(answer_fragment)
    text_match = STAND_TEXT_RE.search(plain_text)
    if text_match:
        return text_match.group("value").strip()

    # Fallback: "Stand:" followed by value in next line.
    multi_line_match = re.search(
        r"(?is)\bstand\s*:\s*\n+\s*(?P<value>[^\n]+)",
        plain_text,
    )
    if multi_line_match:
        return multi_line_match.group("value").strip()
    return ""


def _strip_trailing_stand_line(answer_text: str) -> str:
    """Remove trailing 'Stand: ...' footer text from answer body."""

    cleaned = re.sub(r"(?is)\n+\s*stand\s*:\s*[^\n]*\s*$", "", answer_text).strip()
    cleaned = re.sub(r"(?is)\n+\s*stand\s*:\s*\n+\s*[^\n]+\s*$", "", cleaned).strip()
    return cleaned


def _extract_paragraph_texts(fragment: str) -> list[str]:
    """Extract text paragraphs from an HTML fragment."""

    paragraphs: list[str] = []
    for match in PARAGRAPH_RE.finditer(fragment):
        value = _html_to_text(match.group("body"))
        if value:
            paragraphs.append(value)
    return paragraphs


def _topic_slug_from_url(url: str) -> str:
    """Extract taxonomy slug from FAQ detail URL."""

    parsed = urlparse(url)
    marker = "/Antworten/"
    if marker not in parsed.path:
        return ""
    suffix = parsed.path.split(marker, 1)[1].strip("/")
    return suffix


def _category_path_from_topic_slug(topic_slug: str) -> str:
    """Convert URL topic slug to a readable category breadcrumb."""

    if not topic_slug:
        return ""
    # Drop terminal FAQ slug, keep category folders only.
    parts = [item for item in topic_slug.split("/") if item]
    if len(parts) <= 1:
        return ""
    category_parts = parts[:-1]
    pretty_parts = [part.replace("-", " ") for part in category_parts]
    return " > ".join(pretty_parts)


def _decision_domain_from_topic_slug(topic_slug: str) -> str:
    """Map a portal taxonomy slug to thesis decision domains."""

    normalized = topic_slug.lower()
    if any(keyword in normalized for keyword in FUNDING_DOMAIN_KEYWORDS):
        return "funding_preparation"
    if any(keyword in normalized for keyword in MARKET_DOMAIN_KEYWORDS):
        return "market_validation"
    if any(keyword in normalized for keyword in LEGAL_DOMAIN_KEYWORDS):
        return "legal_form_selection"
    # Conservative default for entrepreneurial FAQ corpus.
    return "market_validation"


def _extract_year_from_stand(stand_value: str) -> int | None:
    """Extract first 4-digit year from stand string."""

    match = re.search(r"\b(19|20)\d{2}\b", stand_value)
    if not match:
        return None
    try:
        return int(match.group(0))
    except ValueError:
        return None


def _within_year_range(year: int | None, min_year: int | None, max_year: int | None) -> bool:
    """Return True when a stand year satisfies optional bounds."""

    if min_year is None and max_year is None:
        return True
    if year is None:
        return False
    if min_year is not None and year < min_year:
        return False
    if max_year is not None and year > max_year:
        return False
    return True


def _is_allowed_decision_domain(domain: str, allowed_domains: tuple[str, ...]) -> bool:
    """Check whether mapped decision domain is enabled in this crawl."""

    return domain in set(allowed_domains)


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _default_state_path(output_json_path: Path) -> Path:
    return output_json_path.with_name(f"{output_json_path.stem}.state.json")


def _new_dataset_payload(start_url: str) -> dict[str, Any]:
    return {
        "metadata": {
            "name": "existenzgruendung_faq_problem_dataset",
            "version": "1.0",
            "jurisdiction": "Germany",
            "source_portal": "existenzgruendungsportal.de",
            "start_url": start_url,
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "notes": (
                "FAQ-derived open QA dataset. Portal itself states content may include older legal context; "
                "verify time-sensitive legal answers before production usage."
            ),
        },
        "items": [],
    }


def _stable_bucket(value: str) -> int:
    """Map a string to a stable [0, 99] bucket for deterministic splits."""

    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 100


def _build_split_payload(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Build deterministic train/dev/test splits stratified by domain."""

    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        domain = str(item.get("decision_domain", ""))
        grouped.setdefault(domain, []).append(item)

    split_to_ids: dict[str, list[str]] = {"train": [], "dev": [], "test": []}

    for domain in sorted(grouped.keys()):
        domain_items = sorted(grouped[domain], key=lambda row: str(row.get("id", "")))
        for item in domain_items:
            key = f"{item.get('source_url', '')}::{item.get('id', '')}"
            bucket = _stable_bucket(key)
            if bucket < 70:
                split_to_ids["train"].append(str(item.get("id", "")))
            elif bucket < 85:
                split_to_ids["dev"].append(str(item.get("id", "")))
            else:
                split_to_ids["test"].append(str(item.get("id", "")))

    return {
        "metadata": {
            "strategy": "stable_hash_domain_stratified",
            "ratios": {"train": 0.70, "dev": 0.15, "test": 0.15},
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "total_items": len(items),
        },
        "splits": split_to_ids,
        "split_counts": {name: len(ids) for name, ids in split_to_ids.items()},
    }


def _build_markdown_summary(dataset: dict[str, Any]) -> str:
    """Build a compact human-readable summary for quick inspection."""

    items = dataset.get("items", [])
    domain_counts: dict[str, int] = {}
    for item in items:
        domain = str(item.get("decision_domain", ""))
        domain_counts[domain] = domain_counts.get(domain, 0) + 1

    lines = [
        "# Existenzgruendungsportal FAQ Problem Dataset",
        "",
        f"- Generated at (UTC): {dataset.get('metadata', {}).get('generated_at_utc', '')}",
        f"- Source start URL: {dataset.get('metadata', {}).get('start_url', '')}",
        f"- Item count: {len(items)}",
        f"- Decision domains: {dataset.get('metadata', {}).get('decision_domains', [])}",
        f"- Stand year range: {dataset.get('metadata', {}).get('min_stand_year', '')} .. {dataset.get('metadata', {}).get('max_stand_year', '')}",
        f"- Skipped by domain filter: {dataset.get('metadata', {}).get('skipped_by_domain', 0)}",
        f"- Skipped by year filter: {dataset.get('metadata', {}).get('skipped_by_year', 0)}",
        f"- Failed detail pages: {dataset.get('metadata', {}).get('failed_detail_count', 0)}",
        "",
        "## Decision Domain Distribution",
        "",
    ]

    if domain_counts:
        for key in sorted(domain_counts.keys()):
            lines.append(f"- `{key}`: {domain_counts[key]}")
    else:
        lines.append("- No items.")

    lines.extend(
        [
            "",
            "## Sample Items (first 20)",
            "",
            "| ID | Domain | Stand | Title |",
            "| --- | --- | --- | --- |",
        ]
    )

    for item in items[:20]:
        title = str(item.get("title", "")).replace("|", "\\|")
        lines.append(
            f"| {item.get('id', '')} | {item.get('decision_domain', '')} | "
            f"{item.get('stand', '')} | {title} |"
        )

    return "\n".join(lines) + "\n"


def crawl_existenzgruendung_faq_to_dataset(
    *,
    config: FAQCrawlConfig,
    output_json_path: Path,
    output_markdown_path: Path | None = None,
    split_output_path: Path | None = None,
    state_path: Path | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    """Crawl FAQ pages and persist a problem dataset JSON.

    Returns a summary payload suitable for CLI reporting.
    """

    if config.max_list_pages is not None and config.max_list_pages <= 0:
        raise RAGError("max_list_pages must be > 0 when provided.")
    if config.max_faqs is not None and config.max_faqs <= 0:
        raise RAGError("max_faqs must be > 0 when provided.")
    if config.timeout_seconds <= 0:
        raise RAGError("timeout_seconds must be > 0.")
    if config.retries <= 0:
        raise RAGError("retries must be > 0.")
    if config.retry_backoff_seconds < 0:
        raise RAGError("retry_backoff_seconds must be >= 0.")
    if not config.decision_domains:
        raise RAGError("decision_domains must not be empty.")
    for domain in config.decision_domains:
        if domain not in ALLOWED_DECISION_DOMAINS:
            allowed = ", ".join(ALLOWED_DECISION_DOMAINS)
            raise RAGError(f"Invalid decision domain: {domain}. Allowed: {allowed}")
    if (
        config.min_stand_year is not None
        and config.max_stand_year is not None
        and config.min_stand_year > config.max_stand_year
    ):
        raise RAGError("min_stand_year must be <= max_stand_year.")

    resolved_state_path = state_path or _default_state_path(output_json_path)
    dataset = _new_dataset_payload(config.start_url)
    state: dict[str, Any] = {
        "start_url": config.start_url,
        "next_list_url": config.start_url,
        "visited_list_urls": [],
        "seen_detail_urls": [],
        "failed_detail_urls": [],
        "updated_at_utc": datetime.now(UTC).isoformat(),
    }

    if resume:
        loaded_dataset = _load_json(output_json_path)
        loaded_state = _load_json(resolved_state_path)
        if loaded_dataset and isinstance(loaded_dataset.get("items"), list):
            dataset = loaded_dataset
        if loaded_state and isinstance(loaded_state.get("seen_detail_urls"), list):
            if loaded_state.get("start_url") == config.start_url:
                state = loaded_state

    items: list[dict[str, Any]] = list(dataset.get("items", []))
    seen_detail_urls: set[str] = set(str(url) for url in state.get("seen_detail_urls", []))
    visited_list_urls: set[str] = set(str(url) for url in state.get("visited_list_urls", []))
    failed_detail_urls: set[str] = set(str(url) for url in state.get("failed_detail_urls", []))
    next_list_url = str(state.get("next_list_url") or config.start_url)
    next_id_num = len(items) + 1

    started = time.monotonic()
    list_pages_crawled = 0
    detail_pages_crawled = 0
    total_entries_hint: int | None = None
    last_request_timestamp: float | None = None
    skipped_by_domain = 0
    skipped_by_year = 0
    failed_detail_count = 0
    last_error_message = ""
    retried_failed_attempted = 0
    retried_failed_recovered = 0

    # Fallback: if state is missing/partial but dataset already exists, recover seen URLs.
    if not seen_detail_urls and items:
        for row in items:
            source_url = str(row.get("source_url", "")).strip()
            if source_url:
                seen_detail_urls.add(source_url)

    def _persist_progress(current_list_url: str) -> None:
        dataset["items"] = items
        dataset["metadata"]["generated_at_utc"] = datetime.now(UTC).isoformat()
        dataset["metadata"]["record_count"] = len(items)
        _write_json(output_json_path, dataset)

        state["start_url"] = config.start_url
        state["next_list_url"] = current_list_url
        state["visited_list_urls"] = sorted(visited_list_urls)
        state["seen_detail_urls"] = sorted(seen_detail_urls)
        state["failed_detail_urls"] = sorted(failed_detail_urls)
        state["updated_at_utc"] = datetime.now(UTC).isoformat()
        state["last_error_message"] = last_error_message
        _write_json(resolved_state_path, state)

    def _process_detail_url(detail_url: str, current_list_url: str) -> None:
        nonlocal last_request_timestamp
        nonlocal detail_pages_crawled
        nonlocal skipped_by_domain
        nonlocal skipped_by_year
        nonlocal failed_detail_count
        nonlocal last_error_message
        nonlocal next_id_num

        try:
            _sleep_for_crawl_delay(last_request_timestamp, config.crawl_delay_seconds)
            detail_html = _fetch_html(
                detail_url,
                timeout=config.timeout_seconds,
                user_agent=config.user_agent,
                retries=config.retries,
                retry_backoff_seconds=config.retry_backoff_seconds,
            )
            last_request_timestamp = time.monotonic()
            detail_pages_crawled += 1

            title = _extract_title(detail_html)
            question_fragment, answer_fragment = _extract_question_answer_sections(detail_html)
            stand_value = _extract_stand_value(answer_fragment)
            question_text = "\n\n".join(_extract_paragraph_texts(question_fragment)).strip()
            answer_paragraphs = []
            for paragraph in _extract_paragraph_texts(answer_fragment):
                lower = paragraph.lower()
                if lower.startswith("stand:"):
                    continue
                answer_paragraphs.append(paragraph)
            answer_text = _strip_trailing_stand_line("\n\n".join(answer_paragraphs).strip())

            topic_slug = _topic_slug_from_url(detail_url)
            category_path = _category_path_from_topic_slug(topic_slug)
            decision_domain = _decision_domain_from_topic_slug(topic_slug)
            stand_year = _extract_year_from_stand(stand_value)

            if not _is_allowed_decision_domain(decision_domain, config.decision_domains):
                skipped_by_domain += 1
                seen_detail_urls.add(detail_url)
            elif not _within_year_range(stand_year, config.min_stand_year, config.max_stand_year):
                skipped_by_year += 1
                seen_detail_urls.add(detail_url)
            else:
                items.append(
                    {
                        "id": f"FAQ-{next_id_num:04d}",
                        "title": title,
                        "question": question_text,
                        "reference_answer": answer_text,
                        "stand": stand_value,
                        "stand_year": stand_year,
                        "category_path": category_path,
                        "topic_slug": topic_slug,
                        "decision_domain": decision_domain,
                        "source_url": detail_url,
                    }
                )
                next_id_num += 1
                seen_detail_urls.add(detail_url)

            if detail_url in failed_detail_urls:
                failed_detail_urls.discard(detail_url)
        except Exception as exc:
            failed_detail_count += 1
            last_error_message = f"{detail_url}: {exc}"
            failed_detail_urls.add(detail_url)
            if not config.continue_on_error:
                raise RAGError(f"Failed to process detail page {detail_url}: {exc}") from exc

        _persist_progress(current_list_url=current_list_url)

    # Resume safety: retry previously failed detail pages before advancing pagination.
    if failed_detail_urls:
        for failed_url in sorted(failed_detail_urls):
            if config.max_faqs is not None and len(items) >= config.max_faqs:
                break
            retried_failed_attempted += 1
            before_failed = failed_url in failed_detail_urls
            _process_detail_url(failed_url, current_list_url=next_list_url)
            if before_failed and failed_url not in failed_detail_urls and failed_url in seen_detail_urls:
                retried_failed_recovered += 1

    while next_list_url:
        if config.max_list_pages is not None and list_pages_crawled >= config.max_list_pages:
            break
        if config.max_faqs is not None and len(items) >= config.max_faqs:
            break

        _sleep_for_crawl_delay(last_request_timestamp, config.crawl_delay_seconds)
        try:
            list_html = _fetch_html(
                next_list_url,
                timeout=config.timeout_seconds,
                user_agent=config.user_agent,
                retries=config.retries,
                retry_backoff_seconds=config.retry_backoff_seconds,
            )
        except RAGError as exc:
            last_error_message = str(exc)
            if config.continue_on_error:
                break
            raise
        last_request_timestamp = time.monotonic()
        list_pages_crawled += 1

        if total_entries_hint is None:
            total_entries_hint = _extract_total_entries_hint(list_html)

        detail_urls = _extract_detail_urls(list_html)
        detail_urls = [url for url in detail_urls if url not in seen_detail_urls]

        for detail_url in detail_urls:
            if config.max_faqs is not None and len(items) >= config.max_faqs:
                break
            _process_detail_url(detail_url, current_list_url=next_list_url)

        visited_list_urls.add(next_list_url)
        next_candidate = _extract_next_page_url(list_html, base_url=next_list_url)
        if not next_candidate or next_candidate in visited_list_urls:
            next_list_url = ""
        else:
            next_list_url = next_candidate

        state["start_url"] = config.start_url
        state["next_list_url"] = next_list_url
        state["visited_list_urls"] = sorted(visited_list_urls)
        state["seen_detail_urls"] = sorted(seen_detail_urls)
        state["failed_detail_urls"] = sorted(failed_detail_urls)
        state["updated_at_utc"] = datetime.now(UTC).isoformat()
        state["last_error_message"] = last_error_message
        _write_json(resolved_state_path, state)

    dataset["items"] = items
    dataset["metadata"]["generated_at_utc"] = datetime.now(UTC).isoformat()
    dataset["metadata"]["record_count"] = len(items)
    dataset["metadata"]["decision_domains"] = list(config.decision_domains)
    dataset["metadata"]["min_stand_year"] = config.min_stand_year
    dataset["metadata"]["max_stand_year"] = config.max_stand_year
    dataset["metadata"]["skipped_by_domain"] = skipped_by_domain
    dataset["metadata"]["skipped_by_year"] = skipped_by_year
    dataset["metadata"]["failed_detail_count"] = failed_detail_count
    dataset["metadata"]["retried_failed_attempted"] = retried_failed_attempted
    dataset["metadata"]["retried_failed_recovered"] = retried_failed_recovered
    if total_entries_hint is not None:
        dataset["metadata"]["portal_total_entries_hint"] = total_entries_hint
    _write_json(output_json_path, dataset)

    if output_markdown_path is not None:
        output_markdown_path.parent.mkdir(parents=True, exist_ok=True)
        output_markdown_path.write_text(_build_markdown_summary(dataset), encoding="utf-8")

    split_payload: dict[str, Any] | None = None
    if split_output_path is not None:
        split_payload = _build_split_payload(items)
        _write_json(split_output_path, split_payload)

    elapsed_ms = int((time.monotonic() - started) * 1000)
    return {
        "status": "partial" if last_error_message else "ok",
        "output_json_path": str(output_json_path),
        "output_markdown_path": str(output_markdown_path) if output_markdown_path else "",
        "split_output_path": str(split_output_path) if split_output_path else "",
        "state_path": str(resolved_state_path),
        "items": len(items),
        "list_pages_crawled": list_pages_crawled,
        "detail_pages_crawled": detail_pages_crawled,
        "skipped_by_domain": skipped_by_domain,
        "skipped_by_year": skipped_by_year,
        "failed_detail_count": failed_detail_count,
        "retried_failed_attempted": retried_failed_attempted,
        "retried_failed_recovered": retried_failed_recovered,
        "split_counts": (split_payload.get("split_counts", {}) if split_payload else {}),
        "portal_total_entries_hint": total_entries_hint,
        "duration_ms": elapsed_ms,
        "next_list_url": next_list_url,
        "crawl_delay_seconds": config.crawl_delay_seconds,
        "last_error_message": last_error_message,
    }
