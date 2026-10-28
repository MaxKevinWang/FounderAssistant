# System Overview

## 1. Project Overview

Founder Assistant is a web assistant for entrepreneurial questions in Germany, powered by open-source LLMs and retrieval-augmented generation.

The system covers three domains:

1. `legal_form_selection`: choosing a legal form.
2. `funding_preparation`: preparing for funding.
3. `market_validation`: validating a market and business idea.

Development covers knowledge ingestion, retrieval, source transparency, model integration, and request-risk controls.

## 2. Current Features

### 2.1 Consumer application

- Multi-session chat with create, rename, and delete operations.
- Separate domain selection and bounded conversation history for each session.
- Three explicit domains plus “Not sure · Search all topics”.
- Streaming RAG responses with model, latency, citations, and source dates.
- Markdown rendering, copying, thumbs-up, and thumbs-down feedback.
- Local browser checks for common structured sensitive information.
- Pre-answer assessment for clarification needs, high-risk requests, questionable shortcuts, and sensitive information.

The frontend contains the consumer chat interface. The backend also retains evaluation and metadata APIs. Historical development interfaces and research datasets are not included here.

### 2.2 Default retrieval configuration

When the user selects a specific domain, the product uses:

```text
Domain filter
→ Query embedding
→ Dense similarity search
→ Top-5 chunks
→ Grounded prompt
→ Streaming response
```

When all topics are selected, no domain filter is applied. Chunks from all three collections compete for the five retrieval positions.

Reranking, multi-query retrieval, neighbour expansion, and confidence gating remain explicit experimental options. They are not web-application defaults.

## 3. End-to-End Request Flow

```text
User input
→ Browser-side sensitive-information check
→ Explicit domain or all-topic selection
→ Bounded history from the current session
→ Backend request assessment
→ answer / clarify / warn / refuse
→ Domain-filtered Dense@5 retrieval
→ Grounded prompt
→ Streaming LLM generation
→ Risk state, citations, and dates displayed
→ Response stored in the browser session
```

Deterministic multilingual rules handle clear requests involving concealment, evasion, or fabrication. A fixed Guard model returns a validated structured classification for the remaining requests.

The example configuration uses `ministral-3:14b`. The Guard receives the two most recent turns capped at 2,400 characters and has a 30-second timeout before deterministic fallback. Its model, output-token limit, and optional reasoning effort are configurable for provider compatibility. The main RAG answer uses up to three recent turns and approximately 1,500 history tokens.

The Guard only classifies the request. Backend code decides whether to clarify, warn, refuse, or continue.

## 4. Architecture and Technology Stack

```text
Vue 3 / TypeScript / Vite
        ↓ HTTP + SSE
Django / Django REST Framework
        ├─ Request Guard
        ├─ RAG retrieval and prompt construction
        ├─ Evaluation runners
        └─ Metadata review API
        ↓
Ollama or OpenWebUI model service
        ↓
Local RAG index: NumPy embeddings + JSON metadata
```

| Layer | Technology | Responsibility |
|---|---|---|
| Frontend | Vue 3, TypeScript, Pinia, Vite | Consumer chat, sessions, and streaming UI |
| Backend | Django, Django REST Framework | API, Guard, RAG, evaluation, and metadata management |
| Model service | Ollama or OpenWebUI | LLM generation and embeddings |
| Retrieval | NumPy dense index | Top-k similarity search |
| Document processing | pypdf, PyMuPDF, LangChain splitters | Extraction, figure handling, and chunking |
| Database | PostgreSQL/pgvector or local configuration | Django data; the RAG index is stored separately |
| Async infrastructure | Redis, Celery | Supporting infrastructure, not the current chat path |

Architecture documentation:

- [English architecture](system_architecture.md)

## 5. Repository Structure

| Path | Contents |
|---|---|
| `backend/` | Django API, RAG, Guard, evaluation commands, and tests |
| `frontend/` | Vue web application |
| `knowledge_base/` | Source documents for the three domains |
| `backend/rag_index/` | Embeddings, chunk metadata, and index manifest |
| `docs/architecture/` | Architecture and preprocessing documentation |
| `docs/deployment/` | Configuration and validation |
| `docs/knowledge_base/` | Local corpus and index maintenance |
| `notebooks/` | Ingestion and FAQ crawling tools |

Document collections, indices and experimental datasets are supplied locally and ignored by Git.

## 11. Privacy, Safety, and Scope

- Chat history is stored in browser `localStorage`; deleting a session removes its local history.
- Each request sends only bounded history from the active session. The backend does not persist chat history.
- OpenWebUI, Ollama, or upstream providers may still apply their own logging and retention policies.
- The API currently uses `AllowAny` and is suitable only for a local research demo.
- The Guard is a model-assisted prototype control and may misclassify ambiguous requests.
- Browser-side sensitive-information checks cover only common structured identifiers.
- The system does not provide professional legal, tax, financial, or compliance advice.

## 12. Known Limitations

1. Most knowledge sources are German; smaller LLMs may be influenced by the evidence language.
2. Output language is mainly prompt-controlled, without complete post-generation validation and retry.
3. Not every source has a reliable publication or last-updated date.
4. Citation markers are generated by the LLM; the backend does not yet verify each claim against its citation.
5. Metadata changes made through the metadata API do not automatically recompute embeddings.
6. Consumer feedback remains local and is not yet part of formal evaluation logging.
7. Experimental CPU rerankers add substantial latency and are not production defaults.

## 15. Maintenance Principles

- Never commit `.env`, API keys, or real sensitive data.
- Never overwrite a frozen evaluation artefact in place; create a versioned revision.
- Every report must identify itself as development, exploratory, or locked-test evidence.
- Do not present retrieval or prompt tuning on exposed test results as independent validation.
- Revalidate index and evaluation hashes after changing the corpus, embedding model, or chunking configuration.
