"""Preprocess cleaned Gruenderplattform HTML snapshots for RAG ingestion."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any


GRUENDERPLATTFORM_FOLDER_PREFIX = "gruenderplattform_"
DEFAULT_DUPLICATE_ARCHIVE_DIR = Path("archive") / "duplicate_corpus" / "gruenderplattform"

SOFT_HYPHEN_PATTERNS = ("\u00ad", "&shy;", "&#173;", "&#xAD;", "&#xad;")
NBSP_PATTERNS = ("\u00a0", "&nbsp;", "&#160;", "&#xA0;", "&#xa0;")

TAIL_MARKERS = (
    "Genug gelesen?Dann leg los!",
    "Wie hat dir diese Seite gefallen?",
    "Einloggen oder Registrieren um selbst eine Bewertung abzugeben.",
)

CTA_LINK_RE = re.compile(
    r"<a\b(?=[^>]*\bclass\s*=\s*['\"][^'\"]*\bbtn\b)[^>]*>.*?</a>",
    flags=re.IGNORECASE | re.DOTALL,
)


@dataclass
class HtmlPreprocessStats:
    """Mutation counters for one HTML snapshot."""

    soft_hyphen_replacements: int = 0
    nbsp_replacements: int = 0
    cta_links_removed: int = 0
    terminal_blocks_removed: int = 0

    @property
    def changed(self) -> bool:
        return any(
            (
                self.soft_hyphen_replacements,
                self.nbsp_replacements,
                self.cta_links_removed,
                self.terminal_blocks_removed,
            )
        )


@dataclass
class GruenderplattformPreprocessResult:
    """Summary of a preprocessing run."""

    scanned_files: int = 0
    changed_files: int = 0
    archived_duplicates: list[dict[str, str]] = field(default_factory=list)
    duplicate_groups: list[list[str]] = field(default_factory=list)
    soft_hyphen_replacements: int = 0
    nbsp_replacements: int = 0
    cta_links_removed: int = 0
    terminal_blocks_removed: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "scanned_files": self.scanned_files,
            "changed_files": self.changed_files,
            "archived_duplicate_count": len(self.archived_duplicates),
            "archived_duplicates": self.archived_duplicates,
            "duplicate_group_count": len(self.duplicate_groups),
            "duplicate_groups": self.duplicate_groups,
            "soft_hyphen_replacements": self.soft_hyphen_replacements,
            "nbsp_replacements": self.nbsp_replacements,
            "cta_links_removed": self.cta_links_removed,
            "terminal_blocks_removed": self.terminal_blocks_removed,
        }


class _PlainTextHTMLParser(HTMLParser):
    """Extract plain text for duplicate detection."""

    _SKIP_TAGS = {"script", "style", "noscript", "template", "svg"}
    _BLOCK_TAGS = {
        "article",
        "aside",
        "blockquote",
        "br",
        "dd",
        "div",
        "dl",
        "dt",
        "figcaption",
        "footer",
        "form",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hr",
        "li",
        "main",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "td",
        "th",
        "tr",
        "ul",
    }

    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ANN001
        tag_name = tag.lower()
        if tag_name in self._SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag_name in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag_name = tag.lower()
        if tag_name in self._SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth:
            return
        if tag_name in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = unescape(data or "")
        if text.strip():
            self._parts.append(text)

    def text(self) -> str:
        return "".join(self._parts)


def iter_gruenderplattform_html_files(data_dir: Path) -> list[Path]:
    """Return active Gruenderplattform HTML snapshots under the RAG data directory."""

    files: list[Path] = []
    for domain_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        if domain_dir.name == "archive" or domain_dir.name.startswith("."):
            continue
        for folder in sorted(path for path in domain_dir.iterdir() if path.is_dir()):
            if folder.name.startswith(GRUENDERPLATTFORM_FOLDER_PREFIX):
                files.extend(sorted(folder.glob("*.html")))
    return files


def preprocess_gruenderplattform_snapshot_html(html: str) -> tuple[str, HtmlPreprocessStats]:
    """Normalize one cleaned snapshot and remove residual non-content chrome."""

    stats = HtmlPreprocessStats()
    processed = html

    for pattern in SOFT_HYPHEN_PATTERNS:
        count = processed.count(pattern)
        if count:
            processed = processed.replace(pattern, "")
            stats.soft_hyphen_replacements += count

    for pattern in NBSP_PATTERNS:
        count = processed.count(pattern)
        if count:
            processed = processed.replace(pattern, " ")
            stats.nbsp_replacements += count

    processed, cta_count = CTA_LINK_RE.subn("", processed)
    stats.cta_links_removed += cta_count

    processed, terminal_count = _remove_terminal_non_content_block(processed)
    stats.terminal_blocks_removed += terminal_count

    return processed, stats


def preprocess_gruenderplattform_html_files(
    *,
    data_dir: Path,
    archive_duplicates: bool = True,
    dry_run: bool = False,
) -> GruenderplattformPreprocessResult:
    """Preprocess active Gruenderplattform snapshots and archive exact duplicates."""

    data_dir = data_dir.resolve()
    files = iter_gruenderplattform_html_files(data_dir)
    result = GruenderplattformPreprocessResult(scanned_files=len(files))

    for html_file in files:
        original = html_file.read_text(encoding="utf-8-sig", errors="replace")
        processed, stats = preprocess_gruenderplattform_snapshot_html(original)
        if stats.changed:
            result.changed_files += 1
            result.soft_hyphen_replacements += stats.soft_hyphen_replacements
            result.nbsp_replacements += stats.nbsp_replacements
            result.cta_links_removed += stats.cta_links_removed
            result.terminal_blocks_removed += stats.terminal_blocks_removed
            if not dry_run:
                html_file.write_text(processed, encoding="utf-8")

    active_files = iter_gruenderplattform_html_files(data_dir)
    duplicate_groups = _find_exact_duplicate_groups(active_files)
    result.duplicate_groups = [[str(path.relative_to(data_dir)) for path in group] for group in duplicate_groups]

    if archive_duplicates:
        archive_root = data_dir / DEFAULT_DUPLICATE_ARCHIVE_DIR
        for group in duplicate_groups:
            keep_file = _select_duplicate_to_keep(group)
            for duplicate_file in group:
                if duplicate_file == keep_file:
                    continue
                destination = archive_root / duplicate_file.relative_to(data_dir)
                result.archived_duplicates.append(
                    {
                        "kept": str(keep_file.relative_to(data_dir)),
                        "archived": str(duplicate_file.relative_to(data_dir)),
                        "destination": str(destination.relative_to(data_dir)),
                    }
                )
                if not dry_run:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(duplicate_file), str(destination))

    return result


def write_preprocess_report(report_path: Path, result: GruenderplattformPreprocessResult) -> None:
    """Persist a machine-readable preprocessing report."""

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        **result.as_dict(),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _remove_terminal_non_content_block(html: str) -> tuple[str, int]:
    for marker in TAIL_MARKERS:
        marker_index = html.find(marker)
        if marker_index == -1:
            continue
        main_close_index = html.find("</main>", marker_index)
        if main_close_index == -1:
            continue
        block_start_index = html.rfind("<div", 0, marker_index)
        if block_start_index == -1:
            block_start_index = marker_index
        processed = html[:block_start_index].rstrip() + "\n</main>" + html[main_close_index + len("</main>") :]
        return processed, 1
    return html, 0


def _find_exact_duplicate_groups(files: list[Path]) -> list[list[Path]]:
    grouped: dict[str, list[Path]] = {}
    for html_file in files:
        text = _normalized_duplicate_text(html_file.read_text(encoding="utf-8-sig", errors="replace"))
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        grouped.setdefault(digest, []).append(html_file)
    return [sorted(paths) for paths in grouped.values() if len(paths) > 1]


def _normalized_duplicate_text(html: str) -> str:
    parser = _PlainTextHTMLParser()
    parser.feed(html)
    parser.close()
    text = parser.text().replace("\u00ad", "").replace("\u00a0", " ")
    return re.sub(r"\s+", " ", text).casefold().strip()


def _select_duplicate_to_keep(group: list[Path]) -> Path:
    def sort_key(path: Path) -> tuple[int, str]:
        name = path.name.casefold()
        duplicate_suffix_penalty = 1 if "-b__" in name or "_b__" in name else 0
        return duplicate_suffix_penalty, name

    return sorted(group, key=sort_key)[0]
