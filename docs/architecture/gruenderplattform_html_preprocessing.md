# Gründerplattform HTML Preprocessing

This note documents how the local Gründerplattform HTML snapshots are prepared
before they are used as RAG source documents.

## Purpose

The Gründerplattform pages are saved as cleaned HTML snapshots, not as raw web
pages. The crawler already keeps the `<main>` content and removes global
website chrome such as headers, navigation, breadcrumbs, forms, scripts, and
footers.

However, the cleaned snapshots still contained several recurring non-content
elements that would be visible to the generic RAG HTML extractor:

1. soft hyphens inside German compound words;
2. non-breaking spaces in button and text fragments;
3. call-to-action buttons;
4. terminal page-rating and registration blocks;
5. exact duplicate pages created by alternative URLs.

The preprocessing step removes these artifacts while preserving the article
body, headings, lists, tables, source URL comments, and classification metadata.

## Command

Run from the backend directory:

```powershell
.\.venv\Scripts\python.exe manage.py preprocess_gruenderplattform_html
```

Dry run:

```powershell
.\.venv\Scripts\python.exe manage.py preprocess_gruenderplattform_html --dry-run
```

The command is implemented in:

- `backend/apps/advisors/gruenderplattform_html_preprocessing.py`
- `backend/apps/advisors/management/commands/preprocess_gruenderplattform_html.py`

## Processing Rules

### 1. File Selection

Only active Gründerplattform HTML snapshots are processed:

```text
knowledge_base/<domain>/gruenderplattform_*/*.html
```

The command ignores:

- `knowledge_base/archive/**`
- hidden directories
- non-HTML files
- non-Gründerplattform source folders

### 2. Character Normalization

The command removes soft hyphen variants:

```text
\u00ad
&shy;
&#173;
&#xAD;
```

It converts non-breaking space variants to normal spaces:

```text
\u00a0
&nbsp;
&#160;
&#xA0;
```

This prevents words such as `Liquiditätsplanung` from being split by invisible
characters in chunks and embeddings.

### 3. CTA Link Removal

HTML anchor elements styled as buttons are removed:

```html
<a class="... btn ...">...</a>
```

These links usually contain interface actions such as starting a tool, logging
in, opening a wizard, or clicking a generic call to action. They do not add
stable domain evidence for RAG.

### 4. Terminal Block Removal

The command removes the recurring terminal page block starting from one of
these markers through the closing `</main>` tag:

```text
Genug gelesen?Dann leg los!
Wie hat dir diese Seite gefallen?
Einloggen oder Registrieren um selbst eine Bewertung abzugeben.
```

This removes the page rating widget, login/register prompt, generic wizard
container, and final site-level conversion prompts.

### 5. Exact Duplicate Handling

After text normalization, the command computes a SHA-256 hash over extracted
plain text. If multiple active pages have the same normalized text, one page is
kept and the duplicate is moved to:

```text
knowledge_base/archive/duplicate_corpus/gruenderplattform/
```

The RAG source collector excludes `knowledge_base/archive/`, so archived duplicates are
not indexed.

## Current Run Summary

The preprocessing run on 2026-06-22 produced:

| Metric | Value |
| --- | ---: |
| Scanned HTML snapshots | 143 |
| Changed HTML snapshots | 143 |
| Soft hyphen replacements | 220 |
| NBSP replacements | 4,923 |
| CTA links removed | 692 |
| Terminal blocks removed | 143 |
| Exact duplicate groups | 2 |
| Archived duplicate files | 2 |

Archived duplicates:

| Kept page | Archived duplicate |
| --- | --- |
| `knowledge_base/market_validation/gruenderplattform_business_ideas/gruenderplattform.de__restaurant-eroeffnen__20260617T135409Z.html` | `knowledge_base/archive/duplicate_corpus/gruenderplattform/market_validation/gruenderplattform_business_ideas/gruenderplattform.de__restaurant-eroeffnen-b__20260617T135411Z.html` |
| `knowledge_base/market_validation/gruenderplattform_business_ideas/gruenderplattform.de__virtuelle-assistenz__20260617T135447Z.html` | `knowledge_base/archive/duplicate_corpus/gruenderplattform/market_validation/gruenderplattform_business_ideas/gruenderplattform.de__virtuelle-assistenz-b__20260617T135448Z.html` |

The machine-readable report is stored at:

```text
docs/crawl_states/gruenderplattform_html_preprocess_report.json
```

## Validation Checks

After preprocessing:

1. all active Gründerplattform HTML files still produce non-empty extracted
   text with the RAG extractor;
2. the duplicate pages are no longer under active `knowledge_base/<domain>/...` folders;
3. `knowledge_base/archive/duplicate_corpus/` is excluded from RAG collection because the
   whole `knowledge_base/archive/` tree is skipped;
4. source comments such as `source_url`, `fetched_at_utc`, `content_type`, and
   `classification_domain` remain in the HTML snapshots.

## Thesis Note

For the thesis methodology, this preprocessing can be described as a local web
corpus curation step: official or semi-official web pages are first captured as
main-content HTML snapshots, then normalized to remove web-interface artifacts,
deduplicated, and only then considered for RAG indexing. This keeps the corpus
traceable while reducing navigation, conversion, and rating noise in retrieval
chunks.
