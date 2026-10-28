from __future__ import annotations

"""Notebook-facing minimal API for web crawl + incremental RAG indexing.

This module keeps notebook cells minimal by hiding Django bootstrap details.
"""

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

if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

from apps.advisors.rag_web_ingestion_utils import (  # noqa: E402
    build_rag_index_for_new_documents,
    crawl_and_auto_classify_url,
)


def crawl_url(url: str) -> dict[str, Any]:
    """Crawl and auto-classify one URL into `knowledge_base/`.

    Prints a concise bilingual completion summary for notebook users.
    """

    result = crawl_and_auto_classify_url(url)
    saved_path = Path(result.get("saved_path", ""))
    folder_path = saved_path.parent if saved_path else Path(".")
    domain = result.get("inferred_domain", "")
    subfolder = result.get("inferred_subfolder", "")
    already_exists = bool(result.get("already_exists", False))
    classification = f"knowledge_base/{domain}" if not subfolder else f"knowledge_base/{domain}/{subfolder}"

    if already_exists:
        print("Crawl skipped: this URL already exists in knowledge_base/ (tracking params ignored).")
        print("已跳过爬取：该 URL（去追踪参数后）已存在于 data/。")
    else:
        print("Crawl completed.")
        print("爬取完成。")

    print(f"Classified folder: {classification}")
    print(f"分类目录：{classification}")
    print(f"Saved file: {saved_path}")
    print(f"保存文件：{saved_path}")
    print(f"Folder path: {folder_path}")
    print(f"文件夹路径：{folder_path}")
    return result


def run_rag_for_new_data() -> dict[str, Any]:
    """Run incremental RAG indexing for documents not indexed yet."""

    return build_rag_index_for_new_documents()
