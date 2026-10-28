"""Crawler for Gruenderplattform business idea pages.

The crawler starts from the business-idea overview page, extracts all linked
detail pages under `/geschaeftsideen/<slug>`, cleans each page down to readable
main-body HTML, and stores the result in the local RAG corpus.
"""

from __future__ import annotations

import html
import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

from .gruenderplattform_html_preprocessing import preprocess_gruenderplattform_snapshot_html
from .rag_pipeline import RAGError

DEFAULT_OVERVIEW_URL = "https://gruenderplattform.de/geschaeftsideen"
DEFAULT_DOMAIN = "market_validation"
DEFAULT_OUTPUT_SUBFOLDER = "gruenderplattform_business_ideas"
DEFAULT_USER_AGENT = "FounderPilotCrawler/1.0 (+local thesis demo crawler)"

TRACKING_QUERY_KEYS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "utm_id",
    "gclid",
    "fbclid",
    "mc_cid",
    "mc_eid",
}

VOID_TAGS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}

SKIP_TAGS = {
    "button",
    "footer",
    "form",
    "header",
    "nav",
    "noscript",
    "script",
    "style",
    "svg",
    "template",
}

ALLOWED_TAGS = {
    "a",
    "abbr",
    "blockquote",
    "b",
    "body",
    "div",
    "em",
    "figcaption",
    "figure",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "i",
    "img",
    "li",
    "main",
    "ol",
    "p",
    "picture",
    "section",
    "source",
    "span",
    "strong",
    "table",
    "tbody",
    "td",
    "th",
    "thead",
    "tr",
    "ul",
}

ALLOWED_ATTRS = {
    "a": {"class", "href", "rel", "target", "title"},
    "abbr": {"class", "title"},
    "div": {"class", "id"},
    "figure": {"class"},
    "img": {"alt", "class", "src", "title"},
    "main": {"class", "id"},
    "p": {"class"},
    "section": {"class", "id"},
    "source": {"media", "srcset"},
    "span": {"aria-hidden", "class", "lang"},
}


@dataclass(frozen=True)
class GruenderplattformCrawlConfig:
    """Runtime settings for one Gruenderplattform crawl run."""

    overview_url: str = DEFAULT_OVERVIEW_URL
    data_dir: Path = Path("knowledge_base")
    output_domain: str = DEFAULT_DOMAIN
    output_subfolder: str = DEFAULT_OUTPUT_SUBFOLDER
    crawl_delay_seconds: float = 1.0
    timeout_seconds: int = 30
    retries: int = 3
    retry_backoff_seconds: float = 2.0
    user_agent: str = DEFAULT_USER_AGENT
    max_pages: int | None = None
    include_overview: bool = True
    force_refresh: bool = False
    state_path: Path | None = None


def canonicalize_url(url: str) -> str:
    """Normalize URL for duplicate detection."""

    parsed = urlparse(url)
    query_pairs = []
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        if key.lower() in TRACKING_QUERY_KEYS:
            continue
        query_pairs.append((key, value))
    query = urlencode(sorted(query_pairs)) if query_pairs else ""
    path = parsed.path.rstrip("/") or "/"
    return urlunparse((parsed.scheme.lower(), parsed.netloc.lower(), path, "", query, ""))


def extract_business_idea_links(overview_html: str, overview_url: str = DEFAULT_OVERVIEW_URL) -> list[str]:
    """Extract unique detail-page URLs from the overview HTML."""

    overview_canonical = canonicalize_url(overview_url)
    links: set[str] = set()
    for match in re.finditer(r"""href=["']([^"']+)["']""", overview_html, flags=re.IGNORECASE):
        href = html.unescape(match.group(1))
        full_url = canonicalize_url(urljoin(overview_url, href))
        parsed = urlparse(full_url)
        if parsed.netloc != "gruenderplattform.de":
            continue
        if not parsed.path.startswith("/geschaeftsideen/"):
            continue
        if full_url == overview_canonical:
            continue
        links.add(full_url)
    return sorted(links)


def crawl_gruenderplattform_business_ideas(config: GruenderplattformCrawlConfig) -> dict[str, Any]:
    """Crawl overview + detail pages and save cleaned HTML snapshots."""

    output_dir = _resolve_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)

    overview_url = canonicalize_url(config.overview_url)
    last_request_at: float | None = None

    overview_html, overview_final_url, overview_content_type, last_request_at = _fetch_with_delay(
        url=overview_url,
        config=config,
        last_request_at=last_request_at,
    )
    detail_links = extract_business_idea_links(overview_html, overview_final_url)
    selected_detail_links = detail_links[: config.max_pages] if config.max_pages is not None else detail_links
    urls_to_process = [overview_final_url, *selected_detail_links] if config.include_overview else selected_detail_links

    saved: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []

    for index, url in enumerate(urls_to_process, start=1):
        canonical_url = canonicalize_url(url)
        existing = _find_existing_source_file(output_dir, canonical_url)
        if existing is not None and not config.force_refresh:
            skipped.append(
                {
                    "url": url,
                    "reason": "already_exists",
                    "path": str(existing),
                }
            )
            continue

        try:
            if canonical_url == canonicalize_url(overview_final_url):
                page_html = overview_html
                final_url = overview_final_url
                content_type = overview_content_type
            else:
                page_html, final_url, content_type, last_request_at = _fetch_with_delay(
                    url=url,
                    config=config,
                    last_request_at=last_request_at,
                )

            cleaned_html, _plain_text, title = clean_gruenderplattform_page_html(page_html)
            cleaned_html, _preprocess_stats = preprocess_gruenderplattform_snapshot_html(cleaned_html)
            plain_text = _normalize_whitespace(_strip_tags(cleaned_html))
            if not plain_text:
                raise RAGError("No readable main content extracted.")

            output_path = _unique_output_path(output_dir, final_url)
            output_path.write_text(
                _render_snapshot_html(
                    source_url=final_url,
                    content_type=content_type,
                    title=title or _title_from_url(final_url),
                    cleaned_html=cleaned_html,
                    classification_domain=config.output_domain,
                ),
                encoding="utf-8",
            )
            saved.append(
                {
                    "url": final_url,
                    "path": str(output_path),
                    "bytes": output_path.stat().st_size,
                    "chars": len(plain_text),
                    "title": title,
                    "index": index,
                }
            )
        except Exception as exc:  # noqa: BLE001
            failed.append({"url": url, "error": str(exc), "index": index})

        _persist_state(
            config=config,
            overview_url=overview_final_url,
            detail_links=detail_links,
            saved=saved,
            skipped=skipped,
            failed=failed,
        )

    result = {
        "status": "ok" if not failed else "partial",
        "overview_url": overview_final_url,
        "detail_link_count": len(detail_links),
        "selected_detail_count": len(selected_detail_links),
        "include_overview": config.include_overview,
        "saved_count": len(saved),
        "skipped_count": len(skipped),
        "failed_count": len(failed),
        "output_dir": str(output_dir),
        "state_path": str(_resolve_state_path(config)),
        "saved": saved,
        "skipped": skipped,
        "failed": failed,
        "detail_links": detail_links,
    }
    _persist_state(
        config=config,
        overview_url=overview_final_url,
        detail_links=detail_links,
        saved=saved,
        skipped=skipped,
        failed=failed,
        final_result=result,
    )
    return result


def clean_gruenderplattform_page_html(raw_html: str) -> tuple[str, str, str]:
    """Return cleaned main HTML, plain text, and title."""

    title_match = re.search(r"<title[^>]*>(.*?)</title>", raw_html, flags=re.IGNORECASE | re.DOTALL)
    title = _normalize_whitespace(_strip_tags(title_match.group(1))) if title_match else ""

    parser = _MainContentHTMLParser()
    parser.feed(raw_html)
    cleaned = parser.cleaned_html().strip()
    text = parser.cleaned_text()
    return cleaned, text, title


def _fetch_with_delay(
    *,
    url: str,
    config: GruenderplattformCrawlConfig,
    last_request_at: float | None,
) -> tuple[str, str, str, float]:
    if last_request_at is not None and config.crawl_delay_seconds > 0:
        elapsed = time.monotonic() - last_request_at
        remaining = config.crawl_delay_seconds - elapsed
        if remaining > 0:
            time.sleep(remaining)
    html_text, final_url, content_type = _fetch_html(
        url=url,
        timeout=config.timeout_seconds,
        user_agent=config.user_agent,
        retries=config.retries,
        retry_backoff_seconds=config.retry_backoff_seconds,
    )
    return html_text, canonicalize_url(final_url), content_type, time.monotonic()


def _fetch_html(
    *,
    url: str,
    timeout: int,
    user_agent: str,
    retries: int,
    retry_backoff_seconds: float,
) -> tuple[str, str, str]:
    last_error: Exception | None = None
    for attempt in range(max(1, retries)):
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
                charset = response.headers.get_content_charset() or "utf-8"
                content_type = response.headers.get("Content-Type", "")
                final_url = response.geturl()
            return body.decode(charset, errors="replace"), final_url, content_type
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            if attempt < retries - 1:
                time.sleep(retry_backoff_seconds * (2**attempt))
    raise RAGError(f"Failed to fetch {url}: {last_error}")


class _MainContentHTMLParser(HTMLParser):
    """Sanitize the page to visible main content while dropping chrome."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self._inside_main = False
        self._main_depth = 0
        self._skip_depth = 0
        self._skip_tag_stack: list[str] = []
        self._parts: list[str] = []
        self._text_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag_name = tag.lower()
        if not self._inside_main:
            if tag_name == "main":
                self._inside_main = True
                self._main_depth = 1
                self._parts.append("<main>\n")
            return

        if tag_name not in VOID_TAGS:
            self._main_depth += 1

        if tag_name in SKIP_TAGS or _looks_like_breadcrumb(attrs):
            self._skip_depth += 1
            self._skip_tag_stack.append(tag_name)
            return
        if self._skip_depth:
            return

        if tag_name in ALLOWED_TAGS:
            attr_html = _attrs_to_html(tag_name, attrs)
            if tag_name in VOID_TAGS:
                self._parts.append(f"<{tag_name}{attr_html}>\n")
            else:
                self._parts.append(f"<{tag_name}{attr_html}>")

    def handle_endtag(self, tag: str) -> None:
        tag_name = tag.lower()
        if not self._inside_main:
            return

        if self._skip_depth:
            if self._skip_tag_stack and self._skip_tag_stack[-1] == tag_name:
                self._skip_tag_stack.pop()
                self._skip_depth = max(0, self._skip_depth - 1)
            if tag_name not in VOID_TAGS:
                self._main_depth = max(0, self._main_depth - 1)
            return

        if tag_name == "main" and self._main_depth <= 1:
            self._parts.append("\n</main>\n")
            self._inside_main = False
            self._main_depth = 0
            return

        if tag_name in ALLOWED_TAGS and tag_name not in VOID_TAGS:
            self._parts.append(f"</{tag_name}>\n")
        if tag_name not in VOID_TAGS:
            self._main_depth = max(0, self._main_depth - 1)

    def handle_data(self, data: str) -> None:
        if not self._inside_main or self._skip_depth:
            return
        if data.strip():
            self._parts.append(html.escape(data))
            self._text_parts.append(data)

    def handle_entityref(self, name: str) -> None:
        if self._inside_main and not self._skip_depth:
            self._parts.append(f"&{name};")
            self._text_parts.append(html.unescape(f"&{name};"))

    def handle_charref(self, name: str) -> None:
        if self._inside_main and not self._skip_depth:
            self._parts.append(f"&#{name};")
            self._text_parts.append(html.unescape(f"&#{name};"))

    def cleaned_html(self) -> str:
        return "".join(self._parts)

    def cleaned_text(self) -> str:
        return _normalize_whitespace(" ".join(self._text_parts))


def _looks_like_breadcrumb(attrs: list[tuple[str, str | None]]) -> bool:
    attr_map = {key.lower(): value or "" for key, value in attrs}
    class_value = attr_map.get("class", "").casefold()
    aria_value = attr_map.get("aria-label", "").casefold()
    return "breadcrumb" in class_value or "breadcrumb" in aria_value


def _attrs_to_html(tag: str, attrs: list[tuple[str, str | None]]) -> str:
    allowed = ALLOWED_ATTRS.get(tag, set())
    parts: list[str] = []
    for key, value in attrs:
        key_l = key.lower()
        if key_l not in allowed or value is None:
            continue
        if key_l in {"href", "src", "srcset"} and value.strip().lower().startswith("javascript:"):
            continue
        parts.append(f'{key_l}="{html.escape(value, quote=True)}"')
    return (" " + " ".join(parts)) if parts else ""


def _render_snapshot_html(
    *,
    source_url: str,
    content_type: str,
    title: str,
    cleaned_html: str,
    classification_domain: str,
) -> str:
    fetched_at = datetime.now(UTC).isoformat()
    return f"""<!--
source_url: {source_url}
fetched_at_utc: {fetched_at}
content_type: {content_type}
extraction: cleaned main content; header, breadcrumbs, navigation, scripts, forms, buttons, and footer removed
classification_domain: {classification_domain}
classification_reason: Gruenderplattform business-idea pages describe startup ideas, customer demand, business models, industry context, and idea validation.
-->
<!doctype html>
<html lang="de">
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
</head>
<body>
{cleaned_html}
</body>
</html>
"""


def _find_existing_source_file(output_dir: Path, canonical_source_url: str) -> Path | None:
    matches: list[tuple[Path, float]] = []
    if not output_dir.exists():
        return None
    for html_file in output_dir.glob("*.html"):
        source_url = _extract_source_url(html_file)
        if source_url and canonicalize_url(source_url) == canonical_source_url:
            matches.append((html_file, html_file.stat().st_mtime))
    if not matches:
        return None
    matches.sort(key=lambda item: item[1], reverse=True)
    return matches[0][0]


def _resolve_output_dir(config: GruenderplattformCrawlConfig) -> Path:
    subfolder = _safe_fragment(config.output_subfolder)
    if not subfolder:
        return config.data_dir / config.output_domain
    return config.data_dir / config.output_domain / subfolder


def _extract_source_url(html_path: Path) -> str | None:
    head = html_path.read_text(encoding="utf-8", errors="ignore")[:6000]
    match = re.search(r"source_url:\s*(\S+)", head)
    return match.group(1).strip() if match else None


def _unique_output_path(output_dir: Path, source_url: str) -> Path:
    parsed = urlparse(source_url)
    host = _safe_fragment(parsed.netloc.replace(":", "_"))
    page = _safe_fragment(Path(parsed.path).name or "index")
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    target = output_dir / f"{host}__{page}__{timestamp}.html"
    if not target.exists():
        return target
    index = 2
    while True:
        candidate = output_dir / f"{host}__{page}__{timestamp}__{index}.html"
        if not candidate.exists():
            return candidate
        index += 1


def _safe_fragment(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-._")
    return cleaned or "source"


def _title_from_url(url: str) -> str:
    return Path(urlparse(url).path).name.replace("-", " ").strip().title() or "Gruenderplattform Business Idea"


def _normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(value or "")).strip()


def _strip_tags(value: str) -> str:
    return re.sub(r"<[^>]+>", " ", value or "")


def _resolve_state_path(config: GruenderplattformCrawlConfig) -> Path:
    if config.state_path is not None:
        return config.state_path
    return Path("docs") / "crawl_states" / "gruenderplattform_business_ideas_crawl_state.json"


def _persist_state(
    *,
    config: GruenderplattformCrawlConfig,
    overview_url: str,
    detail_links: list[str],
    saved: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    failed: list[dict[str, Any]],
    final_result: dict[str, Any] | None = None,
) -> None:
    state_path = _resolve_state_path(config)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "updated_at_utc": datetime.now(UTC).isoformat(),
        "overview_url": overview_url,
        "detail_link_count": len(detail_links),
        "detail_links": detail_links,
        "output_domain": config.output_domain,
        "output_subfolder": config.output_subfolder,
        "output_dir": str(_resolve_output_dir(config)),
        "saved_count": len(saved),
        "skipped_count": len(skipped),
        "failed_count": len(failed),
        "saved": saved,
        "skipped": skipped,
        "failed": failed,
    }
    if final_result is not None:
        payload["final_result"] = {
            key: value
            for key, value in final_result.items()
            if key not in {"saved", "skipped", "failed", "detail_links"}
        }
    state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
