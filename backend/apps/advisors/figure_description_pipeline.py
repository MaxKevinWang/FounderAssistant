"""Generate reviewable descriptions for informative PDF figures.

The regular RAG pipeline is intentionally text-first. This module adds a
separate, reviewable visual layer: render PDF pages, ask a multimodal model
whether the page contains an informative figure, and persist only useful figure
descriptions. The generated descriptions are not automatically merged into the
main corpus; they are stored as review artifacts first.
"""

from __future__ import annotations

import base64
import html
import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import fitz
from django.conf import settings
from pypdf import PdfReader

from .local_llm import DEFAULT_CONFIG, _normalize_openwebui_base_url, _request_headers, _ssl_context

FIGURE_DESCRIPTION_DIRNAME = "figure_descriptions"
FIGURE_PAGE_RESULT_DIRNAME = "page_results"
FIGURE_IMAGE_DIRNAME = "retained_page_images"
FIGURE_JSON_FILENAME = "figure_descriptions.json"
FIGURE_MARKDOWN_FILENAME = "figure_descriptions.md"
FIGURE_SUMMARY_FILENAME = "figure_description_summary.json"
FIGURE_REDESCRIPTION_NOTES_FILENAME = "figure_redescription_notes.json"
FIGURE_REVIEW_HTML_FILENAME = "figure_review_gallery.html"
FIGURE_REVIEW_IMAGE_DIRNAME = "review_page_images"


def figure_review_markdown_path(output_dir: Path) -> Path:
    """Keep the default review document with the knowledge-base documentation."""
    backend_dir = Path(settings.BASE_DIR)
    default_dir = backend_dir / "rag_index" / FIGURE_DESCRIPTION_DIRNAME
    if output_dir.resolve() == default_dir.resolve():
        return backend_dir.parent / "docs" / "knowledge_base" / FIGURE_MARKDOWN_FILENAME
    return output_dir / FIGURE_MARKDOWN_FILENAME


@dataclass(frozen=True)
class FigureDescriptionConfig:
    """Runtime options for the PDF figure-description pipeline."""

    data_dir: Path
    output_dir: Path
    model: str
    timeout_seconds: int
    temperature: float
    max_tokens: int
    zoom: float
    include_archive: bool
    force: bool
    save_images: bool
    source_filter: str
    page_filter: int | None
    max_pages: int | None


def default_figure_description_config() -> FigureDescriptionConfig:
    """Return default paths and model settings from the Django project."""

    backend_dir = Path(settings.BASE_DIR)
    return FigureDescriptionConfig(
        data_dir=Path(getattr(settings, "LOCAL_RAG_DATA_DIR", backend_dir.parent / "knowledge_base")),
        output_dir=backend_dir / "rag_index" / FIGURE_DESCRIPTION_DIRNAME,
        model="gemma4:e2b",
        timeout_seconds=max(int(getattr(settings, "LOCAL_LLM_TIMEOUT_SECONDS", 360)), 240),
        temperature=0.0,
        max_tokens=1100,
        zoom=1.6,
        include_archive=False,
        force=False,
        save_images=True,
        source_filter="",
        page_filter=None,
        max_pages=None,
    )


def _safe_artifact_stem(relative_path: Path, page_number: int) -> str:
    """Create a filesystem-safe, stable artifact stem for one PDF page."""

    raw = f"{relative_path.as_posix()}__page_{page_number:03d}"
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", raw)
    return safe.strip("_")


def _strip_markdown_value(value: str) -> str:
    """Remove lightweight Markdown wrappers from a review-line value."""

    cleaned = value.strip()
    if cleaned.startswith(":"):
        cleaned = cleaned[1:].strip()
    cleaned = cleaned.strip("`").strip()
    return cleaned


def _extract_review_bullet(block: str, label: str) -> str:
    """Extract a metadata bullet value from one Markdown review block."""

    pattern = re.compile(rf"^-\s*{re.escape(label)}\s*:\s*(.+)$", flags=re.IGNORECASE | re.MULTILINE)
    match = pattern.search(block)
    return _strip_markdown_value(match.group(1)) if match else ""


def _extract_correction_lines(block: str) -> list[str]:
    """Collect human correction notes from a Markdown review block.

    The review file was edited manually, so this parser accepts small spelling
    differences such as "Correction" and "Corretion". It intentionally ignores
    the generic "Needs review" field, because every generated figure has that
    flag and it does not mean the user found a concrete error.
    """

    notes: list[str] = []
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped.startswith("-"):
            continue
        lowered = stripped.casefold()
        if "needs review" in lowered:
            continue
        if "correction" not in lowered and "corretion" not in lowered:
            continue

        note = re.sub(
            r"^-\s*(needs\s+)?corr(?:ec|e)?tion(?:\s+note)?\s*:?\s*",
            "",
            stripped,
            flags=re.IGNORECASE,
        )
        note = _strip_markdown_value(note)
        if note and note.casefold() not in {"true", "false"}:
            notes.append(note)
    return notes


def parse_correction_notes_from_markdown(review_markdown_path: Path) -> list[dict[str, Any]]:
    """Parse manually marked correction notes from the review Markdown file."""

    if not review_markdown_path.exists():
        raise FileNotFoundError(f"Review Markdown does not exist: {review_markdown_path}")

    text = review_markdown_path.read_text(encoding="utf-8")
    blocks = re.split(r"(?m)^##\s+", text)
    corrections: list[dict[str, Any]] = []

    for raw_block in blocks[1:]:
        block = raw_block.strip()
        if not block:
            continue
        title, _, remainder = block.partition("\n")
        source_path = _extract_review_bullet(remainder, "Source")
        page_raw = _extract_review_bullet(remainder, "Page")
        figure_type = _extract_review_bullet(remainder, "Type")
        notes = _extract_correction_lines(remainder)
        if not source_path or not page_raw or not notes:
            continue
        try:
            page_number = int(re.search(r"\d+", page_raw).group(0))  # type: ignore[union-attr]
        except Exception:
            continue

        corrections.append(
            {
                "title": title.strip(),
                "source_path": source_path,
                "page_number": page_number,
                "previous_type": figure_type,
                "correction_notes": notes,
            }
        )

    return corrections


def _collect_pdf_files(config: FigureDescriptionConfig) -> list[Path]:
    """Collect in-scope PDF files from the project data directory."""

    files: list[Path] = []
    for path in sorted(config.data_dir.rglob("*.pdf")):
        relative_path = path.relative_to(config.data_dir)
        if not config.include_archive and relative_path.parts[0].casefold() == "archive":
            continue
        if config.source_filter and config.source_filter.casefold() not in relative_path.as_posix().casefold():
            continue
        files.append(path)
    return files


def _extract_page_text(pdf_path: Path, page_index: int) -> str:
    """Extract text from one PDF page for visual model cross-checking."""

    try:
        reader = PdfReader(str(pdf_path))
        raw = reader.pages[page_index].extract_text() or ""
    except Exception:
        return ""
    text = raw.replace("\x00", " ").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _render_page_png(pdf_path: Path, page_index: int, zoom: float) -> bytes:
    """Render one PDF page to PNG bytes for multimodal analysis."""

    with fitz.open(pdf_path) as document:
        page = document[page_index]
        pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        return pixmap.tobytes("png")


def _chat_completion_url() -> str:
    """Return the OpenWebUI OpenAI-compatible chat completion URL."""

    base_url = _normalize_openwebui_base_url(DEFAULT_CONFIG.base_url)
    return f"{base_url}/api/chat/completions"


def _vision_prompt(
    source_path: str,
    page_number: int,
    page_text: str,
    correction_guidance: str = "",
) -> str:
    """Build the multimodal prompt used for figure filtering and description."""

    if len(page_text) > 9000:
        clipped_text = f"{page_text[:4200]}\n\n[... middle of page text omitted ...]\n\n{page_text[-4200:]}"
    else:
        clipped_text = page_text
    correction_section = ""
    if correction_guidance.strip():
        correction_section = f"""

Human reviewer correction notes:
{correction_guidance.strip()}

Use these notes as high-priority guidance:
- Re-check the visual type. Do not label a chart as a table unless it is truly a structured table.
- Return only the reviewed figure(s) named in the correction notes; do not add unrelated figures from the same page.
- Do not mix values from different figures or tables on the same page.
- Preserve row/column or axis/legend relationships exactly.
- If a note says a retained item is not a useful figure, filter it out unless another informative figure is present on the same page.
- If exact values are unreadable, write "unsicher" instead of guessing.
"""
    return f"""You are preparing reviewable RAG figure descriptions for a German entrepreneurship assistant.

Task:
Inspect the provided PDF page image and decide whether it contains at least one informative visual figure worth keeping for RAG.

Keep only visuals with real information value:
- charts, statistical graphics, diagrams, flowcharts, process models, matrices, canvases, structured templates, decision trees, visual tables, annotated maps, or comparable business/legal/finance visuals.

Filter out:
- decorative photos, portraits, cover images, logos, stock photos, icons, seals, publisher imprints, pure page layout, and ordinary text columns.

Important:
- Use the page image for visual structure.
- Use the extracted page text below as the authoritative source for exact labels, percentages, and German wording when it contains the figure text.
- All generated description text must be in German.
- Do not invent values. If a value is uncertain, mark it as "unsicher".
- Return JSON only. No Markdown fence.

Source path: {source_path}
Page number: {page_number}
{correction_section}

Extracted page text for cross-checking:
{clipped_text}

Return this JSON schema:
{{
  "has_informative_figure": true,
  "irrelevance_reason": "",
  "figures": [
    {{
      "figure_type": "chart | table | flowchart | canvas | matrix | process_diagram | infographic | other",
      "title": "exact or inferred German title",
      "description_markdown": "German Markdown. Include visual structure, exact labels/values when readable, and interpretation for founders.",
      "extracted_values": ["label: value", "label: value"],
      "source_note": "source path and page number",
      "confidence": "high | medium | low",
      "needs_review": true
    }}
  ]
}}

If no informative figure should be kept, return:
{{
  "has_informative_figure": false,
  "irrelevance_reason": "short German reason",
  "figures": []
}}
"""


def _parse_json_response(content: str) -> dict[str, Any]:
    """Parse JSON from model output, accepting accidental fenced responses."""

    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)


def _normalise_model_result(result: dict[str, Any], source_path: str, page_number: int) -> dict[str, Any]:
    """Normalize model JSON so downstream review files have stable fields."""

    has_figure = bool(result.get("has_informative_figure"))
    figures = result.get("figures") if isinstance(result.get("figures"), list) else []
    normalized_figures: list[dict[str, Any]] = []

    for index, figure in enumerate(figures, start=1):
        if not isinstance(figure, dict):
            continue
        description = str(figure.get("description_markdown", "")).strip()
        title = str(figure.get("title", "")).strip()
        figure_type = str(figure.get("figure_type", "other")).strip() or "other"
        if not description and not title:
            continue
        normalized_figures.append(
            {
                "figure_id": f"{source_path}::page_{page_number}::figure_{index}",
                "figure_type": figure_type,
                "title": title,
                "description_markdown": description,
                "extracted_values": figure.get("extracted_values", [])
                if isinstance(figure.get("extracted_values"), list)
                else [],
                "source_note": str(figure.get("source_note", f"{source_path}, page {page_number}")).strip(),
                "confidence": str(figure.get("confidence", "low")).strip() or "low",
                "needs_review": True,
            }
        )

    return {
        "has_informative_figure": has_figure and bool(normalized_figures),
        "irrelevance_reason": str(result.get("irrelevance_reason", "")).strip(),
        "figures": normalized_figures,
    }


def describe_page(
    *,
    source_path: str,
    page_number: int,
    page_text: str,
    image_bytes: bytes,
    config: FigureDescriptionConfig,
    correction_guidance: str = "",
) -> dict[str, Any]:
    """Call the configured multimodal model for one rendered page."""

    payload = {
        "model": config.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": _vision_prompt(
                            source_path,
                            page_number,
                            page_text,
                            correction_guidance=correction_guidance,
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,"
                            + base64.b64encode(image_bytes).decode("ascii")
                        },
                    },
                ],
            }
        ],
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "stream": False,
    }

    request = urllib.request.Request(
        _chat_completion_url(),
        data=json.dumps(payload).encode("utf-8"),
        headers=_request_headers(DEFAULT_CONFIG),
        method="POST",
    )

    started = time.perf_counter()
    try:
        with urllib.request.urlopen(
            request,
            timeout=config.timeout_seconds,
            context=_ssl_context(DEFAULT_CONFIG),
        ) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:1200]
        raise RuntimeError(f"Vision model returned HTTP {exc.code}: {body}") from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to call vision model: {exc}") from exc

    choices = response_payload.get("choices", [])
    content = choices[0].get("message", {}).get("content", "") if choices else ""
    try:
        parsed = _parse_json_response(content)
        normalized = _normalise_model_result(parsed, source_path=source_path, page_number=page_number)
        parse_error = ""
    except Exception as exc:
        normalized = {
            "has_informative_figure": False,
            "irrelevance_reason": "Model response could not be parsed as JSON.",
            "figures": [],
        }
        parse_error = str(exc)

    return {
        "source_path": source_path,
        "page_number": page_number,
        "model": response_payload.get("model", config.model),
        "latency_ms": int((time.perf_counter() - started) * 1000),
        "created_at": datetime.now(UTC).isoformat(),
        "raw_response": content,
        "parse_error": parse_error,
        **normalized,
    }


def _write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    """Write UTF-8 JSON with stable formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_markdown(path: Path, figures: list[dict[str, Any]]) -> None:
    """Write a human-readable review document for retained figures."""

    lines = [
        "# PDF Figure Descriptions",
        "",
        "Generated descriptions are review artifacts. They are not automatically trusted as legal facts.",
        "",
    ]
    for figure in figures:
        lines.extend(
            [
                f"## {figure.get('title') or figure.get('figure_id')}",
                "",
                f"- Source: `{figure.get('source_path')}`",
                f"- Page: `{figure.get('page_number')}`",
                f"- Type: `{figure.get('figure_type')}`",
                f"- Confidence: `{figure.get('confidence')}`",
                f"- Needs review: `{figure.get('needs_review')}`",
                "",
                figure.get("description_markdown", "").strip(),
                "",
            ]
        )
        values = figure.get("extracted_values") or []
        if values:
            lines.append("Extracted values:")
            lines.extend(f"- {value}" for value in values)
            lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def _page_result_filename(output_dir: Path, result_path: Path) -> str:
    """Return a stable, slash-separated page result path for review output."""

    return str(result_path.relative_to(output_dir)).replace("\\", "/")


def _load_page_results(output_dir: Path) -> list[dict[str, Any]]:
    """Load page-level figure description results for review output."""

    page_dir = output_dir / FIGURE_PAGE_RESULT_DIRNAME
    pages: list[dict[str, Any]] = []
    for result_path in sorted(page_dir.glob("*.json")):
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        payload["page_result_file"] = _page_result_filename(output_dir, result_path)
        pages.append(payload)
    return pages


def _ensure_review_page_image(
    *,
    output_dir: Path,
    data_dir: Path | None,
    payload: dict[str, Any],
    zoom: float,
) -> str:
    """Return an image path for one page, rendering a review-only image if needed."""

    retained_image = str(payload.get("retained_page_image") or "").strip()
    if retained_image and (output_dir / retained_image).exists():
        return retained_image

    if data_dir is None:
        return retained_image

    source_path = str(payload.get("source_path") or "").strip()
    page_number = int(payload.get("page_number") or 0)
    if not source_path or page_number <= 0:
        return retained_image

    image_dir = output_dir / FIGURE_REVIEW_IMAGE_DIRNAME
    image_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / f"{_safe_artifact_stem(Path(source_path), page_number)}.png"
    if not image_path.exists():
        image_path.write_bytes(_render_page_png(data_dir / source_path, page_index=page_number - 1, zoom=zoom))
    relative_image = str(image_path.relative_to(output_dir)).replace("\\", "/")
    payload["review_page_image"] = relative_image
    return relative_image


def _html_text(value: Any) -> str:
    """Escape text for HTML while preserving line breaks through CSS."""

    return html.escape(str(value or "").strip())


def _html_list(values: Any) -> str:
    """Render a simple escaped HTML list."""

    if not isinstance(values, list) or not values:
        return '<p class="muted">None</p>'
    items = "\n".join(f"<li>{_html_text(value)}</li>" for value in values)
    return f"<ul>{items}</ul>"


def _format_used_correction_notes(figures: list[dict[str, Any]]) -> list[str]:
    """Flatten correction notes attached to regenerated figures."""

    notes: list[str] = []
    seen: set[str] = set()
    for figure in figures:
        for correction in figure.get("correction_notes_used", []) or []:
            title = str(correction.get("title") or "").strip()
            previous_type = str(correction.get("previous_type") or "").strip()
            for note in correction.get("correction_notes", []) or []:
                label = f"{title} [{previous_type}]: {note}".strip()
                if label not in seen:
                    notes.append(label)
                    seen.add(label)
    return notes


def _write_review_html(path: Path, pages: list[dict[str, Any]]) -> None:
    """Write an image-and-text review gallery for human audit."""

    cards: list[str] = []
    retained_count = sum(1 for page in pages if page.get("has_informative_figure"))
    figure_count = sum(len(page.get("figures", []) or []) for page in pages)

    for page_index, page in enumerate(pages, start=1):
        source_path = _html_text(page.get("source_path"))
        page_number = _html_text(page.get("page_number"))
        page_result_file = _html_text(page.get("page_result_file"))
        image_path = str(page.get("review_page_image") or page.get("retained_page_image") or "").strip()
        image_filename = _html_text(image_path or "No page image available")
        image_html = (
            f'<a href="{html.escape(image_path)}" target="_blank">'
            f'<img src="{html.escape(image_path)}" alt="{source_path}, page {page_number}">'
            "</a>"
            if image_path
            else '<div class="image-missing">No page image available</div>'
        )
        status_class = "kept" if page.get("has_informative_figure") else "filtered"
        status_text = "Retained" if page.get("has_informative_figure") else "Filtered out"
        parse_error = str(page.get("parse_error") or "").strip()
        correction_notes = _format_used_correction_notes(page.get("figures", []) or [])
        if not correction_notes:
            for correction in page.get("correction_notes_used", []) or []:
                for note in correction.get("correction_notes", []) or []:
                    correction_notes.append(str(note))

        figure_sections: list[str] = []
        for figure_index, figure in enumerate(page.get("figures", []) or [], start=1):
            description = _html_text(figure.get("description_markdown")) or '<span class="muted">No description returned.</span>'
            figure_sections.append(
                f"""
                <section class="figure">
                  <h3>{figure_index}. {_html_text(figure.get("title") or figure.get("figure_id"))}</h3>
                  <div class="meta-row">
                    <span>Type: <code>{_html_text(figure.get("figure_type"))}</code></span>
                    <span>Confidence: <code>{_html_text(figure.get("confidence"))}</code></span>
                    <span>Needs review: <code>{_html_text(figure.get("needs_review"))}</code></span>
                  </div>
                  <h4>Description</h4>
                  <div class="description">{description}</div>
                  <h4>Extracted values</h4>
                  {_html_list(figure.get("extracted_values"))}
                </section>
                """
            )
        if not figure_sections:
            figure_sections.append(
                f"""
                <section class="figure filtered-note">
                  <h3>No retained figure</h3>
                  <p>{_html_text(page.get("irrelevance_reason")) or "The page was filtered out by the model."}</p>
                </section>
                """
            )

        cards.append(
            f"""
            <article class="card" id="page-{page_index}">
              <div class="image-panel">
                {image_html}
              </div>
              <div class="text-panel">
                <div class="card-header">
                  <span class="badge {status_class}">{status_text}</span>
                  <span class="muted">Review item {page_index}</span>
                </div>
                <h2>{source_path} - page {page_number}</h2>
                <div class="meta-grid">
                  <div><strong>Page image</strong><br><code>{image_filename}</code></div>
                  <div><strong>Page result JSON</strong><br><code>{page_result_file}</code></div>
                  <div><strong>Model</strong><br><code>{_html_text(page.get("model"))}</code></div>
                  <div><strong>Latency</strong><br><code>{_html_text(page.get("latency_ms"))} ms</code></div>
                </div>
                <h4>Human correction notes used</h4>
                {_html_list(correction_notes)}
                {"<h4>Parse error</h4><pre>" + _html_text(parse_error) + "</pre>" if parse_error else ""}
                {''.join(figure_sections)}
              </div>
            </article>
            """
        )

    html_document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PDF Figure Review Gallery</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f4f1ea;
      --paper: #fffdf8;
      --ink: #151922;
      --muted: #667085;
      --line: #ddd5c7;
      --blue: #1f5eff;
      --green: #147a46;
      --red: #b42318;
      --shadow: 0 18px 60px rgba(28, 25, 18, 0.12);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: radial-gradient(circle at top left, #fff8df 0, var(--bg) 34%, #ebe7df 100%);
      color: var(--ink);
      font-family: Georgia, "Times New Roman", serif;
      line-height: 1.55;
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 10;
      padding: 22px 36px;
      background: rgba(255, 253, 248, 0.92);
      border-bottom: 1px solid var(--line);
      backdrop-filter: blur(16px);
    }}
    header h1 {{ margin: 0 0 6px; font-size: 28px; }}
    header p {{ margin: 0; color: var(--muted); }}
    main {{ padding: 28px 36px 64px; }}
    .card {{
      display: grid;
      grid-template-columns: minmax(360px, 46vw) minmax(420px, 1fr);
      gap: 24px;
      margin: 0 auto 28px;
      max-width: 1720px;
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
      max-height: calc(100vh - 140px);
      overflow: auto;
      padding: 12px;
      background: #191b20;
      border-radius: 16px;
    }}
    .image-panel img {{
      display: block;
      width: 100%;
      height: auto;
      border-radius: 8px;
      background: white;
    }}
    .image-missing {{
      display: grid;
      min-height: 320px;
      place-items: center;
      color: #c7cbd1;
      border: 1px dashed #555b66;
      border-radius: 12px;
    }}
    .card-header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 10px;
    }}
    .badge {{
      display: inline-flex;
      padding: 5px 10px;
      border-radius: 999px;
      font: 700 12px/1.2 Verdana, sans-serif;
      letter-spacing: 0.02em;
      text-transform: uppercase;
    }}
    .badge.kept {{ background: #e7f6ee; color: var(--green); }}
    .badge.filtered {{ background: #fff1f0; color: var(--red); }}
    h2 {{ margin: 0 0 14px; font-size: 22px; overflow-wrap: anywhere; }}
    h3 {{ margin: 18px 0 8px; font-size: 19px; }}
    h4 {{ margin: 18px 0 6px; font: 700 13px/1.2 Verdana, sans-serif; text-transform: uppercase; letter-spacing: 0.04em; }}
    code {{
      padding: 2px 5px;
      border-radius: 5px;
      background: #f1ede4;
      font-family: "Cascadia Mono", Consolas, monospace;
      font-size: 12px;
      overflow-wrap: anywhere;
    }}
    pre {{
      overflow: auto;
      padding: 12px;
      border-radius: 12px;
      background: #191b20;
      color: #f4f6fb;
    }}
    ul {{ margin-top: 6px; padding-left: 22px; }}
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
    .meta-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      color: var(--muted);
    }}
    .description {{
      white-space: pre-wrap;
      padding: 12px 14px;
      border-left: 4px solid var(--blue);
      background: #f8fbff;
      border-radius: 0 12px 12px 0;
    }}
    .figure {{
      margin-top: 18px;
      padding-top: 14px;
      border-top: 1px solid var(--line);
    }}
    .filtered-note {{
      border-top-color: #f2b8b5;
    }}
    .muted {{ color: var(--muted); }}
    @media (max-width: 1050px) {{
      header, main {{ padding-left: 18px; padding-right: 18px; }}
      .card {{ grid-template-columns: 1fr; }}
      .image-panel {{ position: relative; top: auto; max-height: none; }}
      .meta-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>PDF Figure Review Gallery</h1>
    <p>{len(pages)} reviewed pages, {retained_count} retained pages, {figure_count} retained figures. Images and generated descriptions are shown together for audit.</p>
  </header>
  <main>
    {''.join(cards)}
  </main>
</body>
</html>
"""
    path.write_text(html_document, encoding="utf-8")


def build_figure_review_gallery(output_dir: Path, data_dir: Path | None = None, zoom: float = 1.6) -> Path:
    """Build an HTML gallery that places rendered page images beside descriptions."""

    pages = _load_page_results(output_dir)
    for page in pages:
        _ensure_review_page_image(output_dir=output_dir, data_dir=data_dir, payload=page, zoom=zoom)
    html_path = output_dir / FIGURE_REVIEW_HTML_FILENAME
    _write_review_html(html_path, pages)
    return html_path


def rebuild_figure_outputs(output_dir: Path) -> dict[str, Any]:
    """Rebuild consolidated JSON/Markdown files from page-level results."""

    page_dir = output_dir / FIGURE_PAGE_RESULT_DIRNAME
    pages: list[dict[str, Any]] = []
    figures: list[dict[str, Any]] = []
    processed_pages = 0
    informative_pages = 0
    parse_errors = 0

    for result_path in sorted(page_dir.glob("*.json")):
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        payload["page_result_file"] = _page_result_filename(output_dir, result_path)
        pages.append(payload)
        processed_pages += 1
        if payload.get("parse_error"):
            parse_errors += 1
        if payload.get("has_informative_figure"):
            informative_pages += 1
        for figure in payload.get("figures", []):
            if not isinstance(figure, dict):
                continue
            figures.append(
                {
                    "source_path": payload.get("source_path"),
                    "page_number": payload.get("page_number"),
                    "model": payload.get("model"),
                    "page_result_file": str(result_path.relative_to(output_dir)).replace("\\", "/"),
                    **figure,
                }
            )

    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "processed_pages": processed_pages,
        "informative_pages": informative_pages,
        "retained_figures": len(figures),
        "parse_errors": parse_errors,
        "review_required": True,
    }
    _write_json(output_dir / FIGURE_JSON_FILENAME, figures)
    _write_json(output_dir / FIGURE_SUMMARY_FILENAME, summary)
    _write_markdown(figure_review_markdown_path(output_dir), figures)
    _write_review_html(output_dir / FIGURE_REVIEW_HTML_FILENAME, pages)
    return summary


def _group_corrections_by_page(corrections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group figure-level correction notes into page-level redescription jobs."""

    grouped: dict[tuple[str, int], dict[str, Any]] = {}
    for correction in corrections:
        key = (str(correction["source_path"]), int(correction["page_number"]))
        group = grouped.setdefault(
            key,
            {
                "source_path": key[0],
                "page_number": key[1],
                "figures": [],
            },
        )
        group["figures"].append(correction)
    return list(grouped.values())


def _format_correction_guidance(page_corrections: dict[str, Any]) -> str:
    """Create prompt guidance from reviewer notes for all figures on one page."""

    lines: list[str] = []
    for index, figure in enumerate(page_corrections.get("figures", []), start=1):
        lines.append(f"{index}. Reviewed figure title: {figure.get('title')}")
        if figure.get("previous_type"):
            lines.append(f"   Previous generated type: {figure.get('previous_type')}")
        for note in figure.get("correction_notes", []):
            lines.append(f"   Correction note: {note}")
    return "\n".join(lines)


def run_figure_redescription_from_review_notes(
    *,
    config: FigureDescriptionConfig,
    review_markdown_path: Path,
) -> dict[str, Any]:
    """Regenerate only the pages with human correction notes.

    The output should normally point to a separate directory, because the input
    Markdown contains the user's manual review notes. Writing corrected results
    elsewhere keeps the review trail intact.
    """

    page_dir = config.output_dir / FIGURE_PAGE_RESULT_DIRNAME
    image_dir = config.output_dir / FIGURE_IMAGE_DIRNAME
    page_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    corrections = parse_correction_notes_from_markdown(review_markdown_path)
    page_jobs = _group_corrections_by_page(corrections)
    _write_json(config.output_dir / FIGURE_REDESCRIPTION_NOTES_FILENAME, page_jobs)

    processed_now = 0
    skipped_existing = 0
    errors: list[dict[str, Any]] = []

    for page_job in page_jobs:
        source_path = str(page_job["source_path"])
        page_number = int(page_job["page_number"])
        pdf_path = config.data_dir / source_path
        result_path = page_dir / f"{_safe_artifact_stem(Path(source_path), page_number)}.json"
        if result_path.exists() and not config.force:
            skipped_existing += 1
            continue

        try:
            page_index = page_number - 1
            image_bytes = _render_page_png(pdf_path, page_index=page_index, zoom=config.zoom)
            page_text = _extract_page_text(pdf_path, page_index=page_index)
            correction_guidance = _format_correction_guidance(page_job)
            result = describe_page(
                source_path=source_path,
                page_number=page_number,
                page_text=page_text,
                image_bytes=image_bytes,
                config=config,
                correction_guidance=correction_guidance,
            )
            result["redescribed_from_review"] = True
            result["correction_notes_used"] = page_job.get("figures", [])
            for figure in result.get("figures", []):
                if isinstance(figure, dict):
                    figure["redescribed_from_review"] = True
                    figure["correction_notes_used"] = page_job.get("figures", [])
            if config.save_images and result.get("has_informative_figure"):
                image_path = image_dir / f"{_safe_artifact_stem(Path(source_path), page_number)}.png"
                image_path.write_bytes(image_bytes)
                result["retained_page_image"] = str(image_path.relative_to(config.output_dir)).replace("\\", "/")
            _write_json(result_path, result)
            processed_now += 1
        except Exception as exc:
            errors.append({"source_path": source_path, "page_number": page_number, "error": str(exc)})
            _write_json(
                result_path,
                {
                    "source_path": source_path,
                    "page_number": page_number,
                    "created_at": datetime.now(UTC).isoformat(),
                    "has_informative_figure": False,
                    "irrelevance_reason": "Processing error.",
                    "figures": [],
                    "redescribed_from_review": True,
                    "correction_notes_used": page_job.get("figures", []),
                    "parse_error": str(exc),
                },
            )
            processed_now += 1

    summary = rebuild_figure_outputs(config.output_dir)
    summary.update(
        {
            "review_markdown_path": str(review_markdown_path),
            "correction_figures_found": len(corrections),
            "correction_pages_found": len(page_jobs),
            "processed_now": processed_now,
            "skipped_existing": skipped_existing,
            "errors": errors,
            "redescribed_from_review": True,
        }
    )
    _write_json(config.output_dir / FIGURE_SUMMARY_FILENAME, summary)
    return summary


def run_figure_description_pipeline(config: FigureDescriptionConfig) -> dict[str, Any]:
    """Render, classify, and describe PDF figure pages with resume support."""

    page_dir = config.output_dir / FIGURE_PAGE_RESULT_DIRNAME
    image_dir = config.output_dir / FIGURE_IMAGE_DIRNAME
    page_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    processed_now = 0
    skipped_existing = 0
    errors: list[dict[str, Any]] = []

    for pdf_path in _collect_pdf_files(config):
        relative_path = pdf_path.relative_to(config.data_dir)
        source_path = relative_path.as_posix()
        try:
            page_count = len(fitz.open(pdf_path))
        except Exception as exc:
            errors.append({"source_path": source_path, "error": f"Failed to open PDF: {exc}"})
            continue

        for page_index in range(page_count):
            page_number = page_index + 1
            if config.page_filter is not None and page_number != config.page_filter:
                continue

            stem = _safe_artifact_stem(relative_path, page_number)
            result_path = page_dir / f"{stem}.json"
            if result_path.exists() and not config.force:
                skipped_existing += 1
                continue
            if config.max_pages is not None and processed_now >= config.max_pages:
                summary = rebuild_figure_outputs(config.output_dir)
                summary.update(
                    {
                        "processed_now": processed_now,
                        "skipped_existing": skipped_existing,
                        "errors": errors,
                        "stopped_after_max_pages": True,
                    }
                )
                return summary

            try:
                image_bytes = _render_page_png(pdf_path, page_index=page_index, zoom=config.zoom)
                page_text = _extract_page_text(pdf_path, page_index=page_index)
                result = describe_page(
                    source_path=source_path,
                    page_number=page_number,
                    page_text=page_text,
                    image_bytes=image_bytes,
                    config=config,
                )
                if config.save_images and result.get("has_informative_figure"):
                    image_path = image_dir / f"{stem}.png"
                    image_path.write_bytes(image_bytes)
                    result["retained_page_image"] = str(image_path.relative_to(config.output_dir)).replace("\\", "/")
                _write_json(result_path, result)
                processed_now += 1
            except Exception as exc:
                errors.append({"source_path": source_path, "page_number": page_number, "error": str(exc)})
                _write_json(
                    result_path,
                    {
                        "source_path": source_path,
                        "page_number": page_number,
                        "created_at": datetime.now(UTC).isoformat(),
                        "has_informative_figure": False,
                        "irrelevance_reason": "Processing error.",
                        "figures": [],
                        "parse_error": str(exc),
                    },
                )
                processed_now += 1

    summary = rebuild_figure_outputs(config.output_dir)
    summary.update({"processed_now": processed_now, "skipped_existing": skipped_existing, "errors": errors})
    return summary
