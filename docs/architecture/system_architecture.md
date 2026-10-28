# System Architecture: Founder Assistant Demo

Last updated: 2026-07-14

## 1. Purpose

This document describes the current frontend-backend architecture, RAG data flow, API boundaries, runtime components, offline data pipelines, and file responsibilities of the Founder Assistant demo. It replaces the older Week 2 RAG-only architecture documents and the outdated file responsibility map.

The current system includes OpenWebUI integration, streaming responses, consumer/developer UI modes, metadata review, benchmark tooling, structure-aware chunking, and PDF figure enrichment.

## 2. High-Level Architecture

The system is a locally run web demo with four layers:

1. Frontend presentation layer: Vue 3 + TypeScript + Vite.
2. Backend API layer: Django + Django REST Framework.
3. LLM / embedding service layer: local Ollama or institute OpenWebUI.
4. Local RAG data layer: `knowledge_base/`, `backend/rag_index/`, and `docs/evaluation/`.

```text
User Browser
  |
  | Vue SPA
  | - Advisor chat UI
  | - Metadata Review UI
  | - Benchmark UI
  | - Test Set UI
  v
Django REST API
  |
  | apps.advisors
  | - baseline LLM endpoints
  | - RAG chat endpoints
  | - SSE streaming endpoint
  | - chunk metadata review endpoints
  | - index build endpoint
  |
  | apps.evaluations
  | - benchmark test-set endpoint
  | - benchmark run endpoint
  v
RAG Runtime
  |
  | Query time
  | - embed user query
  | - retrieve top-k chunks
  | - build grounded prompt
  | - call generation model
  | - return answer + citations
  |
  | Build time
  | - read knowledge_base/
  | - extract PDF/HTML/text
  | - enrich PDF pages with reviewed figure descriptions
  | - chunk text
  | - infer metadata
  | - embed chunks
  | - save local index
  v
Model Provider
  |
  | Ollama or OpenWebUI
  | - generation models
  | - embedding model
  v
Local Artifacts
  |
  | knowledge_base/
  | backend/rag_index/metadata.json
  | backend/rag_index/embeddings.npy
  | backend/rag_index/extracted_texts/*.txt
  | backend/rag_index/figure_descriptions/
  | docs/evaluation/test_question_set.json
```

Current rebuilt index status:

| Item | Current value |
|---|---:|
| Embedding model | `qwen3-embedding:8b` |
| Embedding provider | `openwebui` |
| Documents | 165 |
| Chunks | 4377 |
| Chunk size | 1200 |
| Chunk overlap | 180 |
| Min chunk chars | 180 |

## 3. Technology Stack

| Layer | Technology | Role |
|---|---|---|
| Frontend framework | Vue 3 | Single-page application and reactive UI |
| Frontend language | TypeScript | Typed API contracts and UI state |
| Frontend build tool | Vite | Development server and production build |
| Frontend state | Pinia | UI mode and consumer chat persistence |
| Frontend routing | Vue Router | Page-level route control |
| Markdown rendering | marked + DOMPurify | Safe rendering of model Markdown |
| Backend framework | Django | Project configuration and HTTP server |
| API layer | Django REST Framework | Request validation and JSON/SSE responses |
| LLM provider | Ollama or OpenWebUI | Text generation and streaming generation |
| Embedding provider | Ollama/OpenWebUI endpoint | Query and chunk embeddings |
| Vector storage | local NumPy file | `embeddings.npy` stores dense vectors |
| Metadata storage | local JSON file | `metadata.json` stores chunk text and metadata |
| PDF text extraction | pypdf | Extract page text from PDFs |
| PDF page rendering | PyMuPDF / fitz | Render PDF pages for multimodal figure description |
| Text splitter | LangChain text splitters | Recursive character splitting with offsets |

## 4. Frontend Runtime Modes

The frontend has two modes, stored by `frontend/src/stores/uiMode.ts` in browser `localStorage`.

### 4.1 Consumer Mode

Consumer mode is the simplified user-facing chat UI.

Consumer mode defaults to asking the user to select one of the three supported
decision domains immediately after they send a question:

```text
legal_form_selection
funding_preparation
market_validation
```

The chat writes the question locally and immediately shows an inline
confirmation card; it does not call an LLM or retrieve chunks until the user
chooses. Besides the three domains, the card and Topic selector offer an
all-topic option for users who are unsure. A selected domain restricts Dense@5
retrieval to its curated collection. The all-topic option omits the domain and
lets chunks from all three collections compete directly for the five places.
It uses no separate LLM classifier.

The benchmark remains separate: it supplies each item's declared gold domain
directly, so the consumer topic-selection interaction is not included in the
reported RAG scores.

Responsibilities:

1. Show only the main Founder Assistant chat.
2. Hide system prompt controls, benchmark pages, and metadata review.
3. Persist chat sessions locally.
4. Support new chat, rename, and delete.
5. Stream model output token by token.
6. Render Markdown responses.
7. Show citations and distinguish source publication/update dates from collection timestamps.
8. Check structured sensitive identifiers in the browser before sending.
9. Route assessed requests to answer, clarify, warn, or refuse before retrieval.
10. Support copy and thumbs-up/thumbs-down feedback.

Main files:

| File | Responsibility |
|---|---|
| `frontend/src/App.vue` | App shell, sidebar, sessions, settings drawer, mode switching |
| `frontend/src/views/AdvisorView.vue` | Consumer chat UI and streaming RAG flow |
| `frontend/src/stores/consumerChat.ts` | Local sessions, messages, feedback, persistence |
| `frontend/src/stores/uiMode.ts` | Consumer/developer mode state |
| `frontend/src/utils/markdown.ts` | Markdown rendering and sanitization |
| `frontend/src/utils/sensitiveInput.ts` | Browser-local structured sensitive-data detection |

### 4.2 Developer Mode

Developer mode is the research/debugging workspace.

Responsibilities:

1. Inspect provider and model status.
2. Test baseline and RAG responses.
3. Switch and compare generation models.
4. Rebuild the RAG index.
5. Run benchmarks.
6. Browse the test set.
7. Review and edit chunk metadata.
8. Inspect system prompt and RAG parameters.

Developer routes:

| Route | Component | Purpose |
|---|---|---|
| `/advisor` | `AdvisorView.vue` | Main LLM/RAG interaction |
| `/metadata-review` | `MetadataReviewView.vue` | Expert chunk metadata review |
| `/benchmark` | `BenchmarkView.vue` | Baseline/RAG benchmark |
| `/test-set` | `TestSetView.vue` | Benchmark question browser |

Developer-only routes are guarded in `frontend/src/router/index.ts`. This is a UI-level guard, not production authentication.

## 5. Frontend Data Flow

### 5.1 Application Bootstrap

`frontend/src/main.ts`:

```text
create Vue app
-> install Pinia
-> install Vue Router
-> import global CSS
-> mount App.vue
```

`App.vue` owns the global shell and renders page components through `<RouterView />`. It does not call the RAG API directly.

### 5.2 API Services

Frontend API access is centralized in `frontend/src/services/`.

| Service | Responsibility |
|---|---|
| `localLlm.ts` | Advisor API client, RAG API client, chunk metadata API client, SSE parsing |
| `benchmark.ts` | Benchmark runner API client |
| `testSet.ts` | Test-set API client |

Important advisor functions:

| Function | Endpoint | Purpose |
|---|---|---|
| `fetchLocalLLMStatus()` | `GET /api/advisors/local-llm/status/` | Provider/model status |
| `chatWithLocalLLM()` | `POST /api/advisors/local-llm/chat/` | Baseline answer |
| `assessAdvisorRequest()` | `POST /api/advisors/request-assessment/` | Clarification, risk, shortcut, and sensitive-data preflight |
| `fetchRAGStatus()` | `GET /api/advisors/rag/status/` | RAG readiness |
| `buildRAGIndex()` | `POST /api/advisors/rag/build-index/` | Full index build |
| `chatWithRAG()` | `POST /api/advisors/rag/chat/` | Non-streaming RAG answer |
| `streamChatWithRAG()` | `POST /api/advisors/rag/chat/stream/` | Streaming RAG answer |
| `fetchRAGChunks()` | `GET /api/advisors/rag/chunks/` | Chunk metadata list |
| `updateRAGChunkMetadata()` | `PATCH /api/advisors/rag/chunks/metaknowledge_base/` | Save metadata edits |

### 5.3 Consumer Chat Store

`frontend/src/stores/consumerChat.ts` stores:

```text
sessions
activeSessionId
messagesBySession
```

Each session can also store its selected RAG domain or all-topic mode. Each message stores content,
role, model, latency, retrieval latency, citations, feedback, runtime profile,
error flag, and streaming flag. Chat history is stored only in browser
`localStorage` and is not persisted by the backend. Deleting a session removes
its messages and retrieval-scope selection.

Before a consumer request, the frontend selects at most the newest three turns
(six messages) from that session. Failed or still-streaming assistant messages
are excluded. A conservative Unicode-aware estimator caps the history at 1,500
tokens and each message at 600 tokens.

### 5.4 AdvisorView Flow

Developer flow:

```text
prompt/settings
-> submitPrompt()
-> requestOne(primary)
-> optional requestOne(comparison)
-> Promise.all()
-> render result cards
```

Consumer streaming flow:

```text
user sends message
-> browser checks structured identifiers; require explicit confirmation when found
-> preselected domain/all-topic mode, or show inline scope confirmation immediately
-> do not retrieve until a scope is selected
-> store user message
-> select bounded history from the same session
-> backend request assessment returns answer | clarify | warn | refuse
-> clarify/refuse: show the controlled response and stop
-> answer/warn: continue, attaching the assessed safety instruction
-> create empty streaming assistant message
-> call streamChatWithRAG(question, optional domain, history)
-> receive profile event
-> receive metadata event
-> receive token events
-> receive done event
-> finalize and persist message
```

The frontend shows a live generation timer and renders the final response with `marked` + `DOMPurify`.

### 5.5 Metadata Review Flow

`MetadataReviewView.vue`:

```text
open /metadata-review
-> GET /api/advisors/rag/chunks/
-> filter by query/domain/reviewed
-> edit metadata fields
-> PATCH /api/advisors/rag/chunks/metaknowledge_base/
-> backend updates metadata.json only
```

Embeddings are not recalculated.

## 6. Backend Architecture

### 6.1 Django Project

| File | Responsibility |
|---|---|
| `backend/manage.py` | Django CLI entrypoint |
| `backend/config/settings.py` | Apps, database, CORS, REST framework, LLM/RAG environment settings |
| `backend/config/urls.py` | Root URL routing |
| `backend/config/asgi.py` / `wsgi.py` | Server entrypoints |
| `backend/requirements/base.txt` | Runtime dependencies |

Root routes:

```text
/admin/
/api/schema/
/api/docs/swagger/
/api/docs/redoc/
/api/auth/
/api/advisors/
/api/evaluations/
```

Runtime apps:

| App | Status | Role |
|---|---|---|
| `apps.advisors` | active | LLM, RAG, chunks, index build, corpus utilities |
| `apps.evaluations` | active | benchmark and test-set API |
| `apps.users` | legacy/support | custom user model |
| `apps.ventures` | legacy/support | earlier venture model |
| `apps.knowledge` | legacy/support | earlier knowledge models |

### 6.2 Environment Configuration

Important variables:

| Variable | Purpose |
|---|---|
| `LOCAL_LLM_PROVIDER` | `ollama` or `openwebui` |
| `LOCAL_LLM_BASE_URL` | provider base URL |
| `LOCAL_LLM_MODEL` | default generation model |
| `LOCAL_LLM_API_KEY` | OpenWebUI API key |
| `LOCAL_LLM_TIMEOUT_SECONDS` | backend LLM timeout |
| `LOCAL_LLM_VERIFY_SSL` | certificate verification toggle |
| `ADVISOR_GUARD_MODEL` | dedicated request-assessment model; currently `ministral-3:14b` |
| `LOCAL_RAG_EMBED_MODEL` | embedding model |
| `LOCAL_RAG_DATA_DIR` | corpus directory |
| `LOCAL_RAG_INDEX_DIR` | index directory |
| `LOCAL_RAG_CHUNK_SIZE` | chunk size |
| `LOCAL_RAG_CHUNK_OVERLAP` | chunk overlap |
| `LOCAL_RAG_MIN_CHUNK_CHARS` | minimum chunk length |

## 7. Advisor Backend Components

### 7.1 API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/advisors/local-llm/status/` | GET | Provider and model status |
| `/api/advisors/local-llm/chat/` | POST | Baseline generation |
| `/api/advisors/request-assessment/` | POST | Structured preflight assessment and program-controlled action |
| `/api/advisors/rag/status/` | GET | RAG readiness |
| `/api/advisors/rag/build-index/` | POST | Full RAG index build |
| `/api/advisors/rag/chunks/` | GET | Paginated chunk metadata |
| `/api/advisors/rag/chunks/metaknowledge_base/` | PATCH | Manual metadata update |
| `/api/advisors/rag/chat/` | POST | Non-streaming, domain-scoped RAG answer |
| `/api/advisors/rag/chat/stream/` | POST | Streaming, domain-scoped RAG answer |

### 7.2 LLM Adapter

`backend/apps/advisors/local_llm.py` hides provider differences between Ollama and OpenWebUI.

Responsibilities:

1. Load provider configuration.
2. Add Bearer authentication when needed.
3. Normalize OpenWebUI URLs.
4. Handle SSL verification settings.
5. List and classify available models.
6. Run synchronous generation.
7. Run streaming generation.

| Provider | Status endpoint | Generation endpoint | Stream format |
|---|---|---|---|
| `ollama` | `/api/tags` | `/api/generate` | line-delimited JSON |
| `openwebui` | `/api/models` | `/api/chat/completions` | OpenAI-compatible stream |

### 7.3 Runtime Profiles

`backend/apps/advisors/model_profiles.py` provides default RAG runtime parameters:

| Model type | `top_k` | `max_tokens` | Rationale |
|---|---:|---:|---|
| dense | 5 | 900 | Evaluated Dense@5 context |
| expert / MoE | 5 | 900 | Same context width and requested output budget |

Both profiles and the answer-generation benchmark defaults use
`DEFAULT_ANSWER_MAX_TOKENS` from `model_profiles.py`. Explicit request or run
settings can override this budget, including the larger budgets in the Kimi
evaluation. Model-family labels remain available in runtime metadata.

The public answer endpoints accept an optional supported `domain` and only
`top_k=5`. With a domain, the application uses
`selected domain filter -> dense retrieval -> Top-5 context`. With the explicit
all-topic option, it uses `all indexed chunks -> dense retrieval -> Top-5 context`.
This global path is the earlier unfiltered retrieval behavior and is not an LLM
domain classifier. Benchmark runs continue to supply their declared domain:
`declared domain filter -> dense retrieval -> Top-5 context`.

Conversation history has an independent budget: newest three turns, at most
1,500 estimated tokens, and at most 600 tokens per message. Both frontend and
backend enforce the policy. It does not reduce the configured output limit;
instead, it bounds additional input context.

### 7.4 RAG Core

`backend/apps/advisors/rag_pipeline.py` owns:

1. Configuration resolution.
2. Corpus file collection.
3. PDF/HTML/text extraction.
4. PDF cleanup and boilerplate removal.
5. PDF page enrichment with reviewed figure descriptions.
6. Chunking and metadata inference.
7. Embedding and index persistence.
8. Query-time retrieval.
9. Prompt construction with `[S1]` markers.
10. LLM generation / streaming.
11. Citation payload generation.
12. Chunk metadata list/update for expert review.

### 7.5 Request Assessment Guard

`backend/apps/advisors/request_guard.py` runs before consumer retrieval. Obvious
requests to conceal, evade, or falsify are matched by deterministic multilingual
rules. Other requests are classified by an LLM at temperature zero into a
validated JSON schema covering risk level, risk categories, questionable
shortcuts, sensitive-data categories, and clarification need. Application code,
not free-form model prose, derives one of four actions:

```text
answer -> continue normally
clarify -> ask one focused question and stop
warn -> show a fixed high-risk/sensitive notice, then continue with constraints
refuse -> show a fixed refusal and safe redirection, then stop
```

Invalid classifier output falls back to deterministic rules. The guard is a
prototype decision-support control, not a substitute for authentication,
professional review, or a dedicated safety classifier.

The Guard uses `ministral-3:14b` independently of the answer model. It receives
at most the newest two conversation turns, capped at 2,400 characters in total,
and can generate at most 200 tokens. The API response records the resolved Guard
model, Guard latency, and provider token-usage fields. The main RAG path retains
its larger three-turn, 1,500-token history budget.

## 8. RAG Index Build Flow

Entrypoints:

```powershell
cd backend
python manage.py build_rag_index
```

or:

```text
POST /api/advisors/rag/build-index/
```

Flow:

```text
resolve config
-> collect source files from knowledge_base/
-> load reviewed figure descriptions
-> extract document content by type
-> enrich PDF pages with matching figure descriptions
-> chunk text
-> infer metadata
-> embed chunk texts
-> L2-normalize vectors
-> write metadata.json, embeddings.npy, extracted_texts/
```

Document extraction:

| Type | Path | Notes |
|---|---|---|
| PDF | `_extract_pdf_document()` | page-aware pypdf extraction |
| HTML | `_extract_html_document()` | structure-aware parser |
| Plain text | direct read | normalized text |

Figure descriptions are loaded from `backend/rag_index/figure_descriptions/figure_descriptions.json` and aligned by `source_path + page_number`. They are inserted into the matching PDF page text before chunking, not indexed as standalone documents.

Chunk parameters:

```text
chunk_size = 1200
chunk_overlap = 180
min_chunk_chars = 180
```

Current chunk type distribution:

| Chunk type | Count |
|---|---:|
| `section_text` | 3451 |
| `list` | 848 |
| `figure_enriched_text` | 62 |
| `table_like_text` | 16 |

Index artifacts:

| Artifact | Purpose |
|---|---|
| `backend/rag_index/metadata.json` | Chunk text and metadata |
| `backend/rag_index/embeddings.npy` | Dense vector matrix |
| `backend/rag_index/extracted_texts/*.txt` | Extracted text snapshots |
| `backend/rag_index/source_manifest.json` | Source signatures for incremental indexing |

## 9. RAG Query Flow

```text
frontend receives a domain or all-topic choice from the selector/confirmation card
-> frontend selects up to three recent turns from the active session
-> frontend sends question + optional domain + bounded history
-> Django validates request
-> backend validates roles and reapplies the 1,500-token history budget
-> backend resolves runtime profile
-> backend limits eligible chunks to the selected domain, or keeps all chunks
-> backend combines recent history with the current question for retrieval
-> backend embeds the history-aware query
-> backend loads embeddings.npy and metadata.json
-> backend computes dense similarities
-> backend selects five chunks
-> backend builds citation-marked context
-> backend builds grounded prompt with history marked as context, not evidence
-> backend calls generation model
-> backend returns answer + citations + latency
-> frontend renders Markdown and source list
```

Dense retrieval uses:

```text
similarities = embeddings @ query_vector
```

Because vectors are normalized, this behaves like cosine similarity. If embedding fails, the system falls back to keyword-overlap retrieval and reports `keyword_fallback`.

## 10. Streaming RAG Flow

Endpoint:

```text
POST /api/advisors/rag/chat/stream/
```

SSE event types:

| Event | Meaning |
|---|---|
| `profile` | resolved model type, `top_k`, and `max_tokens` |
| `metadata` | retrieval results, citations, retrieval latency |
| `token` | generated text delta |
| `done` | final model/provider/latency |
| `error` | error message |

The frontend parses frames manually from the browser stream and updates the active assistant message incrementally.

## 11. Baseline Flow

Endpoint:

```text
POST /api/advisors/local-llm/chat/
```

Flow:

```text
frontend sends prompt/system_prompt/model
-> backend validates request
-> local_llm.generate_text()
-> provider call to Ollama/OpenWebUI
-> backend returns response/model/latency
```

No retrieval and no citations are used.

## 12. Metadata Review Flow

```text
GET /api/advisors/rag/chunks/
-> load metadata.json only
-> filter chunks
-> return paginated rows
-> PATCH /api/advisors/rag/chunks/metaknowledge_base/
-> update one chunk in metadata.json
```

The vector matrix is unchanged.

## 13. Benchmark Flow

`apps.evaluations` exposes:

```text
GET /api/evaluations/benchmark/test-set/
POST /api/evaluations/benchmark/run/
```

Benchmark execution:

```text
load test set
-> run baseline and/or RAG
-> parse TRUE/FALSE prediction
-> compute accuracy and citation metrics
-> optionally run scenario question
-> write JSON/Markdown report
-> return summary
```

If `repeats > 1`, the backend returns batch aggregate metrics.

## 14. Offline Pipelines

### 14.1 Web Ingestion

```text
URL
-> crawl page
-> save cleaned HTML
-> extract visible text
-> infer corpus domain
-> move into knowledge_base/
-> remove older duplicates
```

### 14.2 Incremental Indexing

`build_rag_index_incremental` uses `source_manifest.json`. It appends only new files when safe and falls back to full rebuild when sources, embedding model, or chunk parameters changed.

### 14.3 FAQ Dataset

FAQ crawler and deduplication outputs are currently used mainly for benchmark/problem-set construction, not automatically inserted into the RAG corpus.

### 14.4 PDF Figure Pipeline

```text
PDF page
-> render PNG with PyMuPDF
-> extract page text with pypdf
-> send image + text to multimodal model
-> filter decorative visuals
-> save review artifacts
-> allow correction-note reruns
-> merge final reviewed descriptions
-> RAG build inserts descriptions into matching PDF page text
```

## 15. Storage Map

| Location | Producer | Consumer | Purpose |
|---|---|---|---|
| `knowledge_base/` | manual files/crawlers | RAG builder | Raw corpus |
| `backend/rag_index/metadata.json` | RAG builder | retrieval, metadata review | Chunk text and metadata |
| `backend/rag_index/embeddings.npy` | RAG builder | retrieval | Dense vectors |
| `backend/rag_index/extracted_texts/` | RAG builder | developer/researcher | Extracted text snapshots |
| `backend/rag_index/source_manifest.json` | incremental indexer | incremental indexer | Source change tracking |
| `backend/rag_index/figure_descriptions/` | figure pipeline | RAG builder/reviewer | Reviewed visual descriptions |
| `docs/evaluation/test_question_set.json` | curated/generated data | benchmark/test-set UI | Evaluation questions |
| `docs/evaluation/runs/` | benchmark runner | researcher | Generated reports |
| browser `localStorage` | Vue stores | Vue frontend | UI mode and chat history |
| `backend/.env` | developer | Django/model adapter | Runtime config and secrets |

## 16. End-to-End Example

For a consumer RAG question:

```text
Ich möchte einen Friseursalon in Deutschland eröffnen. Was muss ich anmelden?
```

Flow:

1. User submits the question and confirms a domain or all-topic search when the session has no scope.
2. The browser checks for structured sensitive identifiers and asks for confirmation before transmission when detected.
3. Frontend stores the selected scope on that session and selects its newest three turns under the 1,500-token budget.
4. The request-assessment endpoint returns `answer`, `clarify`, `warn`, or `refuse`; only answer/warn proceeds to retrieval.
5. `streamChatWithRAG()` sends the current question, optional domain, bounded history, and any guard instruction.
6. Backend validates roles and reapplies the history limits.
7. Backend creates a history-aware retrieval query without an extra LLM call.
8. Backend filters to the selected domain when present, embeds the query, and retrieves five chunks.
9. Backend builds a grounded prompt containing current evidence and the bounded history.
10. The prompt labels history as conversational context rather than factual evidence.
11. Backend emits metadata and calls OpenWebUI/Ollama through `stream_generate_text()`.
12. Provider tokens are forwarded to the frontend and appended to the assistant message.
13. Frontend finalizes and persists the message; later follow-ups use it only within the same session.

## 17. Security and Privacy Notes

1. API endpoints use `AllowAny`; this is suitable for local demo, not production.
2. Developer mode is a UI mode, not authentication.
3. OpenWebUI API key is stored only in `backend/.env`.
4. Consumer chats are stored in browser localStorage. Only the bounded recent
   history is transmitted with a request; the backend does not persist it, but
   upstream OpenWebUI/provider logging follows that service's retention policy.
5. CORS is open by default for local development.
6. `LOCAL_LLM_VERIFY_SSL=false` is a local workaround; long term, the institution CA chain should be installed.
7. Markdown is sanitized with DOMPurify.
8. Structured identifiers are detected in the browser before transmission. If the user continues, the full message is still sent to the backend and configured provider.
9. Semantic sensitive-data and risk assessment necessarily processes the prompt through the configured guard model; no detected values are copied into the UI warning.

## 18. Current Limitations

1. Retrieval is dense-similarity first; metadata-aware reranking is not implemented yet.
2. Metadata review changes labels only; embeddings are unchanged.
3. Consumer feedback is local only.
4. Figure descriptions are appended to page text, not inserted at exact visual coordinates.
5. HTML images are not yet processed by the multimodal figure pipeline.
6. Incremental indexing should be used carefully after major pipeline changes.
7. Legacy apps remain in Django config but are not central to the current UI.
8. Request assessment is model-assisted and can misclassify ambiguous requests; deterministic rules cover only explicit shortcut patterns.
9. Browser-local sensitive-data detection covers common structured identifiers, not every possible confidential fact.

## 19. Summary

The system has two connected loops:

```text
Vue chat UI -> Django RAG API -> Ollama/OpenWebUI -> streamed answer with citations -> Vue UI
```

and:

```text
Corpus collection -> chunking/embedding -> metadata review -> benchmark/test-set evaluation -> thesis documentation
```

The frontend remains lightweight and typed. The backend owns model access, RAG logic, corpus processing, and evaluation. Data exchange between frontend and backend uses explicit JSON contracts or SSE events. Internal research artifacts are persisted as local files, making the system inspectable, reproducible, and suitable for thesis implementation documentation.
