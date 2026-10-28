# Knowledge base and index

Place local documents under `knowledge_base/legal_form_selection/`, `knowledge_base/funding_preparation/`, and `knowledge_base/market_validation/`. Directory names determine the retrieval domain. No source-document corpus or generated index is distributed here.

Start the configured embedding service and run from `backend/`:

```powershell
.\.venv\Scripts\python.exe manage.py build_rag_index
```

For newly added files with unchanged index settings, use `manage.py build_rag_index_incremental`. Rebuild after changing the embedding model, chunking configuration or existing documents.

Generated `backend/rag_index/` files contain embeddings, chunk metadata, extraction snapshots and a source manifest. They are ignored by Git.

German statutes under `knowledge_base/**/primary_law/` use dedicated parsing and section scope from `backend/apps/advisors/legal_corpus_scope_v1.json`. Supply the corresponding PDFs before running `manage.py audit_legal_corpus`.

See [HTML preprocessing](../architecture/gruenderplattform_html_preprocessing.md) and [notebook tools](../../notebooks/README.md) for web ingestion.
