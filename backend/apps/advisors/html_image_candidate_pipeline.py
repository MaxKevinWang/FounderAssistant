"""Extract reviewable image candidates from saved HTML corpus files.

This module is intentionally lightweight. It does not call a multimodal model
and it does not change the RAG index. The goal is to help a human reviewer see
which HTML images might be worth describing later.
"""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

from django.conf import settings

from .figure_description_pipeline import FIGURE_DESCRIPTION_DIRNAME

HTML_IMAGE_CANDIDATES_JSON_FILENAME = "html_image_candidates.json"
HTML_IMAGE_CANDIDATES_MARKDOWN_FILENAME = "html_image_candidates.md"
HTML_IMAGE_REVIEW_HTML_FILENAME = "html_image_review_gallery.html"
HTML_IMAGE_SUMMARY_FILENAME = "html_image_candidate_summary.json"


@dataclass(frozen=True)
class HtmlImageCandidateConfig:
    """Runtime options for HTML image candidate extraction."""

    data_dir: Path
    output_dir: Path
    include_archive: bool = False
    source_filter: str = ""
    max_files: int | None = None


@dataclass
class _FigureContext:
    """Track images and caption text while parsing one <figure> element."""

    image_indices: list[int] = field(default_factory=list)
    caption_parts: list[str] = field(default_factory=list)


class _HtmlImageCandidateParser(HTMLParser):
    """Collect image candidates plus nearby textual context from one HTML page."""

    _SKIP_TAGS = {"script", "style", "noscript", "template", "svg"}
    _TEXT_TAGS = {"p", "li", "blockquote", "figcaption"}
    _HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}

    def __init__(self, source_path: str, source_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.source_path = source_path
        self.source_url = source_url
        self.title = ""
        self.candidates: list[dict[str, Any]] = []
        self._skip_depth = 0
        self._tag_stack: list[str] = []
        self._heading_stack: list[tuple[int, str]] = []
        self._text_buffer_tag = ""
        self._text_buffer: list[str] = []
        self._recent_text_blocks: list[str] = []
        self._picture_sources_stack: list[list[dict[str, str]]] = []
        self._figure_stack: list[_FigureContext] = []
        self._in_title = False

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ANN001
        tag_name = tag.casefold()
        attr_map = {str(key).casefold(): str(value or "") for key, value in attrs}
        self._tag_stack.append(tag_name)

        if tag_name in self._SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return

        if tag_name == "title":
            self._in_title = True
            self._start_text_buffer("title")
            return
        if tag_name == "figure":
            self._figure_stack.append(_FigureContext())
            return
        if tag_name == "picture":
            self._picture_sources_stack.append([])
            return
        if tag_name == "source" and self._picture_sources_stack:
            self._picture_sources_stack[-1].append(
                {
                    "srcset": attr_map.get("srcset", ""),
                    "media": attr_map.get("media", ""),
                    "type": attr_map.get("type", ""),
                }
            )
            return
        if tag_name in self._HEADING_TAGS:
            self._start_text_buffer(tag_name)
            return
        if tag_name in self._TEXT_TAGS:
            self._start_text_buffer(tag_name)
            return
        if tag_name == "img":
            self._record_image(attr_map)

    def handle_endtag(self, tag: str) -> None:
        tag_name = tag.casefold()

        if self._skip_depth:
            if tag_name in self._SKIP_TAGS:
                self._skip_depth = max(0, self._skip_depth - 1)
            self._pop_tag(tag_name)
            return

        if tag_name == "title" and self._text_buffer_tag == "title":
            self.title = _normalize_inline_text(" ".join(self._text_buffer))
            self._reset_text_buffer()
            self._in_title = False
        elif tag_name in self._HEADING_TAGS and self._text_buffer_tag == tag_name:
            heading_text = _normalize_inline_text(" ".join(self._text_buffer))
            if heading_text:
                level = int(tag_name[1])
                self._heading_stack = [(item_level, text) for item_level, text in self._heading_stack if item_level < level]
                self._heading_stack.append((level, heading_text))
                self._remember_text(heading_text)
            self._reset_text_buffer()
        elif tag_name in self._TEXT_TAGS and self._text_buffer_tag == tag_name:
            text = _normalize_inline_text(" ".join(self._text_buffer))
            if text:
                self._remember_text(text)
                if tag_name == "figcaption" and self._figure_stack:
                    self._figure_stack[-1].caption_parts.append(text)
            self._reset_text_buffer()
        elif tag_name == "figure" and self._figure_stack:
            context = self._figure_stack.pop()
            caption = _normalize_inline_text(" ".join(context.caption_parts))
            for image_index in context.image_indices:
                self.candidates[image_index]["figcaption"] = caption
        elif tag_name == "picture" and self._picture_sources_stack:
            self._picture_sources_stack.pop()

        self._pop_tag(tag_name)

    def handle_data(self, data: str) -> None:
        if self._skip_depth or not data.strip():
            return
        if self._text_buffer_tag:
            self._text_buffer.append(data)

    def _record_image(self, attrs: dict[str, str]) -> None:
        raw_url = _best_image_url(attrs, self._picture_sources_stack[-1] if self._picture_sources_stack else [])
        image_url = urljoin(self.source_url, raw_url) if raw_url else ""
        srcset_candidates = _extract_srcset_candidates(attrs, self._picture_sources_stack[-1] if self._picture_sources_stack else [])
        normalized_srcset = [urljoin(self.source_url, item) for item in srcset_candidates]

        candidate = {
            "candidate_id": f"{self.source_path}#img-{len(self.candidates) + 1:03d}",
            "source_path": self.source_path,
            "source_url": self.source_url,
            "html_title": self.title,
            "image_index": len(self.candidates) + 1,
            "image_url": image_url,
            "raw_src": raw_url,
            "srcset_candidates": normalized_srcset,
            "alt": _normalize_inline_text(attrs.get("alt", "")),
            "title": _normalize_inline_text(attrs.get("title", "")),
            "class": _normalize_inline_text(attrs.get("class", "")),
            "width": attrs.get("width", ""),
            "height": attrs.get("height", ""),
            "loading": attrs.get("loading", ""),
            "decoding": attrs.get("decoding", ""),
            "heading_path": self._heading_path(),
            "figcaption": "",
            "surrounding_text_before": self._recent_text_blocks[-3:],
            "inside_figure": bool(self._figure_stack),
            "inside_picture": bool(self._picture_sources_stack),
        }
        candidate.update(_classify_candidate(candidate))
        self.candidates.append(candidate)

        if self._figure_stack:
            self._figure_stack[-1].image_indices.append(len(self.candidates) - 1)

    def _start_text_buffer(self, tag_name: str) -> None:
        self._reset_text_buffer()
        self._text_buffer_tag = tag_name
        self._text_buffer = []

    def _reset_text_buffer(self) -> None:
        self._text_buffer_tag = ""
        self._text_buffer = []

    def _heading_path(self) -> str:
        return " > ".join(text for _, text in self._heading_stack)

    def _remember_text(self, text: str) -> None:
        cleaned = _normalize_inline_text(text)
        if len(cleaned) < 20:
            return
        self._recent_text_blocks.append(cleaned)
        self._recent_text_blocks = self._recent_text_blocks[-8:]

    def _pop_tag(self, tag_name: str) -> None:
        if not self._tag_stack:
            return
        if self._tag_stack[-1] == tag_name:
            self._tag_stack.pop()
            return
        try:
            index = len(self._tag_stack) - 1 - self._tag_stack[::-1].index(tag_name)
        except ValueError:
            return
        del self._tag_stack[index:]


def default_html_image_candidate_config() -> HtmlImageCandidateConfig:
    """Return default paths for HTML image candidate extraction."""

    backend_dir = Path(settings.BASE_DIR)
    return HtmlImageCandidateConfig(
        data_dir=Path(getattr(settings, "LOCAL_RAG_DATA_DIR", backend_dir.parent / "knowledge_base")),
        output_dir=backend_dir / "rag_index" / FIGURE_DESCRIPTION_DIRNAME,
    )


def run_html_image_candidate_pipeline(config: HtmlImageCandidateConfig) -> dict[str, Any]:
    """Extract candidates and write JSON, Markdown, and HTML review artifacts."""

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    html_files = _collect_html_files(config)
    candidates: list[dict[str, Any]] = []
    parse_errors: list[dict[str, str]] = []

    for html_path in html_files:
        relative_path = html_path.relative_to(config.data_dir)
        text = html_path.read_text(encoding="utf-8", errors="replace")
        source_url = _extract_source_url(text) or html_path.as_uri()
        parser = _HtmlImageCandidateParser(source_path=relative_path.as_posix(), source_url=source_url)
        try:
            parser.feed(text)
            parser.close()
        except Exception as exc:  # HTML from crawled sites can be imperfect.
            parse_errors.append({"source_path": relative_path.as_posix(), "error": str(exc)})

        for candidate in parser.candidates:
            candidate["domain"] = relative_path.parts[0] if relative_path.parts else ""
            candidate["document_stem"] = html_path.stem
            candidate["relative_parent"] = relative_path.parent.as_posix()
            candidate["extracted_at_utc"] = datetime.now(UTC).isoformat()
            candidates.append(candidate)

    candidates = _sort_candidates_for_review(candidates)
    summary = _build_summary(config, html_files, candidates, parse_errors)
    _write_json(output_dir / HTML_IMAGE_CANDIDATES_JSON_FILENAME, candidates)
    _write_json(output_dir / HTML_IMAGE_SUMMARY_FILENAME, summary)
    _write_markdown(output_dir / HTML_IMAGE_CANDIDATES_MARKDOWN_FILENAME, candidates, summary)
    _write_review_html(output_dir / HTML_IMAGE_REVIEW_HTML_FILENAME, candidates, summary)
    return summary


def _sort_candidates_for_review(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Put likely useful candidates first in review artifacts."""

    priority = {
        "potentially_informative": 0,
        "needs_manual_review": 1,
        "likely_decorative": 2,
    }
    return sorted(
        candidates,
        key=lambda candidate: (
            priority.get(str(candidate.get("heuristic_label") or ""), 9),
            str(candidate.get("domain") or ""),
            str(candidate.get("source_path") or ""),
            int(candidate.get("image_index") or 0),
        ),
    )


def _collect_html_files(config: HtmlImageCandidateConfig) -> list[Path]:
    """Collect in-scope HTML files from the corpus data directory."""

    files: list[Path] = []
    for path in sorted(config.data_dir.rglob("*.html")):
        relative_path = path.relative_to(config.data_dir)
        if not config.include_archive and relative_path.parts and relative_path.parts[0].casefold() == "archive":
            continue
        if config.source_filter and config.source_filter.casefold() not in relative_path.as_posix().casefold():
            continue
        files.append(path)
        if config.max_files is not None and len(files) >= config.max_files:
            break
    return files


def _extract_source_url(text: str) -> str:
    """Extract crawler source_url metadata from the HTML comment header."""

    match = re.search(r"(?mi)^source_url:\s*(\S+)\s*$", text[:3000])
    return html.unescape(match.group(1).strip()) if match else ""


def _best_image_url(attrs: dict[str, str], picture_sources: list[dict[str, str]]) -> str:
    """Pick the most useful URL for review display from img/srcset metadata."""

    for key in ("src", "data-src", "data-lazy-src", "data-original", "data-url"):
        value = attrs.get(key, "").strip()
        if value:
            return html.unescape(value)
    candidates = _extract_srcset_candidates(attrs, picture_sources)
    return candidates[-1] if candidates else ""


def _extract_srcset_candidates(attrs: dict[str, str], picture_sources: list[dict[str, str]]) -> list[str]:
    """Return all URLs found in img srcset and enclosing picture source tags."""

    raw_srcsets = [attrs.get("srcset", "")]
    raw_srcsets.extend(source.get("srcset", "") for source in picture_sources)
    candidates: list[str] = []
    for raw_srcset in raw_srcsets:
        for entry in raw_srcset.split(","):
            url = entry.strip().split(" ")[0].strip()
            if url and url not in candidates:
                candidates.append(html.unescape(url))
    return candidates


def _classify_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    """Assign a review-oriented heuristic label before any vision model call."""

    metadata_pool = " ".join(
        str(candidate.get(key) or "")
        for key in (
            "image_url",
            "raw_src",
            "alt",
            "title",
            "class",
            "heading_path",
            "figcaption",
            "html_title",
        )
    ).casefold()
    context_pool = f"{candidate.get('heading_path', '')} {' '.join(candidate.get('surrounding_text_before') or [])}".casefold()
    combined_pool = f"{metadata_pool} {context_pool}"

    informative_terms = (
        "chart",
        "diagram",
        "diagramm",
        "grafik",
        "graph",
        "infografik",
        "infographic",
        "canvas",
        "matrix",
        "tabelle",
        "table",
        "prozess",
        "process",
        "flow",
        "ablauf",
        "schema",
        "modell",
        "model",
        "rechner",
        "calculator",
        "break-even",
        "value-proposition",
        "business-model",
        "finanzplan",
        "checkliste",
        "checklist",
        "mvp",
    )
    decorative_terms = (
        "getty",
        "istock",
        "shutterstock",
        "adobe stock",
        "unsplash",
        "pexels",
        "fotolia",
        "stock",
        "copyright",
        "©",
        "video thumbnail",
        "logo",
        "icon",
        "sprite",
        "avatar",
        "portrait",
        "hero",
        "banner",
        "teaser",
        "social",
        "facebook",
        "linkedin",
        "twitter",
        "tracking",
        "pixel",
    )

    informative_hits = [term for term in informative_terms if term in metadata_pool]
    context_keyword_hits = [term for term in informative_terms if term in context_pool and term not in informative_hits]
    decorative_hits = [term for term in decorative_terms if term in combined_pool]

    width = _safe_int(candidate.get("width"))
    height = _safe_int(candidate.get("height"))
    is_tiny = bool(width and height and (width < 120 or height < 120))
    image_url = str(candidate.get("image_url") or "").strip().casefold()
    is_data_image = image_url.startswith("data:image")

    if is_data_image:
        label = "likely_decorative"
        action = "filter_placeholder_or_inline_svg"
        reason = "Image is an inline data URI, often a placeholder SVG or video thumbnail in the saved HTML."
    elif is_tiny or (decorative_hits and not informative_hits):
        label = "likely_decorative"
        action = "filter_decorative"
        reason = "Heuristic matched decorative, stock, logo, icon, tracking, or tiny-image signals."
    elif informative_hits:
        label = "potentially_informative"
        action = "review_for_multimodal_description"
        reason = "Image metadata matched chart, table, diagram, canvas, checklist, or business-model signals."
    elif context_keyword_hits and not decorative_hits:
        label = "needs_manual_review"
        action = "manual_review_before_model_call"
        reason = "Nearby text matched informative concepts, but the image metadata itself did not."
    else:
        label = "needs_manual_review"
        action = "manual_review_before_model_call"
        reason = "No strong decorative or informative signal was found."

    return {
        "heuristic_label": label,
        "recommended_action": action,
        "heuristic_reason": reason,
        "informative_keyword_hits": informative_hits,
        "context_keyword_hits": context_keyword_hits,
        "decorative_keyword_hits": decorative_hits,
        "is_tiny_image": is_tiny,
        "is_data_image": is_data_image,
    }


def _safe_int(value: Any) -> int:
    try:
        return int(str(value or "").strip())
    except ValueError:
        return 0


def _normalize_inline_text(text: str) -> str:
    """Collapse HTML text into a compact single-line value."""

    unescaped = html.unescape(text or "").replace("\x00", " ")
    return re.sub(r"\s+", " ", unescaped).strip()


def _build_summary(
    config: HtmlImageCandidateConfig,
    html_files: list[Path],
    candidates: list[dict[str, Any]],
    parse_errors: list[dict[str, str]],
) -> dict[str, Any]:
    """Build a compact summary for command output and review headers."""

    by_label: dict[str, int] = {}
    by_domain: dict[str, int] = {}
    for candidate in candidates:
        by_label[str(candidate.get("heuristic_label") or "")] = by_label.get(str(candidate.get("heuristic_label") or ""), 0) + 1
        by_domain[str(candidate.get("domain") or "")] = by_domain.get(str(candidate.get("domain") or ""), 0) + 1
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "data_dir": str(config.data_dir),
        "output_dir": str(config.output_dir),
        "include_archive": config.include_archive,
        "source_filter": config.source_filter,
        "html_files_scanned": len(html_files),
        "image_candidates": len(candidates),
        "by_heuristic_label": by_label,
        "by_domain": by_domain,
        "parse_errors": parse_errors,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_markdown(path: Path, candidates: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    """Write a compact Markdown review list."""

    lines = [
        "# HTML Image Candidate Review",
        "",
        "This file lists image candidates extracted from saved HTML corpus files. It does not contain multimodal descriptions yet.",
        "",
        "## Summary",
        "",
        f"- HTML files scanned: {summary['html_files_scanned']}",
        f"- Image candidates: {summary['image_candidates']}",
        f"- By heuristic label: `{json.dumps(summary['by_heuristic_label'], ensure_ascii=False)}`",
        f"- By domain: `{json.dumps(summary['by_domain'], ensure_ascii=False)}`",
        "",
    ]

    for candidate in candidates:
        lines.extend(
            [
                f"## {candidate['candidate_id']}",
                "",
                f"- Label: `{candidate.get('heuristic_label', '')}`",
                f"- Recommended action: `{candidate.get('recommended_action', '')}`",
                f"- Source: `{candidate.get('source_path', '')}`",
                f"- Source URL: {candidate.get('source_url', '')}",
                f"- Image URL: {candidate.get('image_url', '')}",
                f"- Alt: {candidate.get('alt', '')}",
                f"- Title: {candidate.get('title', '')}",
                f"- Figure caption: {candidate.get('figcaption', '')}",
                f"- Heading path: {candidate.get('heading_path', '')}",
                f"- Reason: {candidate.get('heuristic_reason', '')}",
                "",
                "### Surrounding Text Before",
                "",
            ]
        )
        for text in candidate.get("surrounding_text_before") or []:
            lines.append(f"- {text}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def _write_review_html(path: Path, candidates: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    """Write a browser-friendly gallery for human review."""

    cards = "\n".join(_candidate_card(index, candidate) for index, candidate in enumerate(candidates, start=1))
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>HTML Image Candidate Review</title>
  <style>
    :root {{
      --bg: #f6f3eb;
      --paper: #fffdf8;
      --ink: #151922;
      --muted: #667085;
      --line: #ded7ca;
      --green: #157347;
      --orange: #a15c00;
      --red: #b42318;
      --blue: #1f5eff;
      --shadow: 0 18px 56px rgba(34, 28, 15, 0.12);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: radial-gradient(circle at top left, #fff6d8 0, var(--bg) 32%, #ebe6dc 100%);
      color: var(--ink);
      font-family: Georgia, "Times New Roman", serif;
      line-height: 1.55;
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 20;
      padding: 22px 34px;
      background: rgba(255, 253, 248, 0.94);
      border-bottom: 1px solid var(--line);
      backdrop-filter: blur(14px);
    }}
    header h1 {{ margin: 0 0 6px; font-size: 28px; }}
    header p {{ margin: 0; color: var(--muted); }}
    main {{ padding: 28px 34px 70px; }}
    .card {{
      display: grid;
      grid-template-columns: minmax(320px, 40vw) minmax(420px, 1fr);
      gap: 24px;
      max-width: 1680px;
      margin: 0 auto 28px;
      padding: 20px;
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 22px;
      box-shadow: var(--shadow);
    }}
    .image-panel {{
      position: sticky;
      top: 112px;
      align-self: start;
      display: grid;
      min-height: 260px;
      place-items: center;
      padding: 12px;
      background: #181b20;
      border-radius: 16px;
    }}
    .image-panel img {{
      max-width: 100%;
      max-height: calc(100vh - 160px);
      border-radius: 10px;
      background: white;
    }}
    .image-missing {{
      color: #c8ccd4;
      text-align: center;
      padding: 30px;
      border: 1px dashed #59606d;
      border-radius: 12px;
    }}
    .badge {{
      display: inline-flex;
      padding: 5px 10px;
      border-radius: 999px;
      font: 700 12px/1.2 Verdana, sans-serif;
      letter-spacing: 0.02em;
      text-transform: uppercase;
    }}
    .badge.potentially_informative {{ background: #e8f5ee; color: var(--green); }}
    .badge.needs_manual_review {{ background: #fff3dc; color: var(--orange); }}
    .badge.likely_decorative {{ background: #fff1f0; color: var(--red); }}
    h2 {{ margin: 10px 0 14px; font-size: 22px; overflow-wrap: anywhere; }}
    h3 {{ margin: 18px 0 8px; font-size: 16px; }}
    code {{
      padding: 2px 5px;
      border-radius: 5px;
      background: #f0ece3;
      font-family: "Cascadia Mono", Consolas, monospace;
      font-size: 12px;
      overflow-wrap: anywhere;
    }}
    a {{ color: var(--blue); overflow-wrap: anywhere; }}
    .meta-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
      margin: 14px 0;
      padding: 14px;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: #faf7f0;
    }}
    .text-box {{
      margin: 10px 0;
      padding: 12px 14px;
      border-left: 4px solid var(--blue);
      background: #f8fbff;
      border-radius: 0 12px 12px 0;
      white-space: pre-wrap;
    }}
    .muted {{ color: var(--muted); }}
    ul {{ margin-top: 6px; padding-left: 20px; }}
    @media (max-width: 1050px) {{
      header, main {{ padding-left: 18px; padding-right: 18px; }}
      .card {{ grid-template-columns: 1fr; }}
      .image-panel {{ position: relative; top: auto; }}
      .meta-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>HTML Image Candidate Review</h1>
    <p>{_e(summary['image_candidates'])} image candidates from {_e(summary['html_files_scanned'])} HTML files. Labels are heuristic only; use this page to decide which images deserve multimodal descriptions.</p>
  </header>
  <main>
    {cards or '<p>No HTML image candidates were found.</p>'}
  </main>
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")


def _candidate_card(index: int, candidate: dict[str, Any]) -> str:
    image_url = str(candidate.get("image_url") or "").strip()
    image_html = (
        f'<a href="{_e(image_url)}" target="_blank" rel="noreferrer"><img src="{_e(image_url)}" alt="{_e(candidate.get("alt"))}"></a>'
        if image_url
        else '<div class="image-missing">No image URL found</div>'
    )
    surrounding = "\n".join(f"<li>{_e(text)}</li>" for text in candidate.get("surrounding_text_before") or [])
    if not surrounding:
        surrounding = '<li class="muted">No surrounding text captured before this image.</li>'
    label = str(candidate.get("heuristic_label") or "needs_manual_review")
    return f"""
    <article class="card" id="candidate-{index}">
      <div class="image-panel">
        {image_html}
      </div>
      <div class="text-panel">
        <span class="badge {label}">{_e(label)}</span>
        <h2>{_e(candidate.get("candidate_id"))}</h2>
        <div class="meta-grid">
          <div><strong>Recommended action</strong><br><code>{_e(candidate.get("recommended_action"))}</code></div>
          <div><strong>Domain</strong><br><code>{_e(candidate.get("domain"))}</code></div>
          <div><strong>Source file</strong><br><code>{_e(candidate.get("source_path"))}</code></div>
          <div><strong>Heading path</strong><br>{_e(candidate.get("heading_path")) or '<span class="muted">None</span>'}</div>
        </div>
        <p><strong>Source URL:</strong><br><a href="{_e(candidate.get("source_url"))}" target="_blank" rel="noreferrer">{_e(candidate.get("source_url"))}</a></p>
        <p><strong>Image URL:</strong><br><a href="{_e(image_url)}" target="_blank" rel="noreferrer">{_e(image_url)}</a></p>
        <h3>Available HTML metadata</h3>
        <div class="text-box"><strong>Alt:</strong> {_e(candidate.get("alt")) or '<span class="muted">None</span>'}<br><strong>Title:</strong> {_e(candidate.get("title")) or '<span class="muted">None</span>'}<br><strong>Figcaption:</strong> {_e(candidate.get("figcaption")) or '<span class="muted">None</span>'}</div>
        <h3>Heuristic reason</h3>
        <div class="text-box">{_e(candidate.get("heuristic_reason"))}</div>
        <h3>Keyword hits</h3>
        <div class="meta-grid">
          <div><strong>Image metadata</strong><br><code>{_e(', '.join(candidate.get("informative_keyword_hits") or [])) or 'none'}</code></div>
          <div><strong>Context only</strong><br><code>{_e(', '.join(candidate.get("context_keyword_hits") or [])) or 'none'}</code></div>
          <div><strong>Decorative</strong><br><code>{_e(', '.join(candidate.get("decorative_keyword_hits") or [])) or 'none'}</code></div>
        </div>
        <h3>Surrounding text before image</h3>
        <ul>{surrounding}</ul>
      </div>
    </article>
    """


def _e(value: Any) -> str:
    return html.escape(str(value or "").strip())
