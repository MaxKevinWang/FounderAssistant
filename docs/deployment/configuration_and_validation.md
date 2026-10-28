# Configuration and Validation

## 7. Model-Service Configuration

### 7.1 Ollama

```env
LOCAL_LLM_PROVIDER=ollama
LOCAL_LLM_BASE_URL=http://127.0.0.1:11434
LOCAL_LLM_MODEL=qwen2.5:7b
LOCAL_LLM_API_KEY=
LOCAL_LLM_VERIFY_SSL=true
LOCAL_LLM_TIMEOUT_SECONDS=360
LOCAL_RAG_EMBED_MODEL=bge-m3
ADVISOR_GUARD_MODEL=ministral-3:14b
ADVISOR_GUARD_TIMEOUT_SECONDS=30
ADVISOR_GUARD_MAX_TOKENS=200
ADVISOR_GUARD_REASONING_EFFORT=
```

Ollama must have the configured answer, embedding, and Guard models installed. Leave `ADVISOR_GUARD_MODEL` empty to reuse the answer model when no separate Guard is available.

### 7.2 OpenWebUI

```env
LOCAL_LLM_PROVIDER=openwebui
LOCAL_LLM_BASE_URL=<openwebui-base-url>
LOCAL_LLM_MODEL=<answer-model>
LOCAL_LLM_API_KEY=<api-key>
LOCAL_LLM_VERIFY_SSL=true
LOCAL_LLM_TIMEOUT_SECONDS=360
LOCAL_RAG_EMBED_MODEL=<embedding-model>
ADVISOR_GUARD_MODEL=ministral-3:14b
ADVISOR_GUARD_TIMEOUT_SECONDS=30
ADVISOR_GUARD_MAX_TOKENS=200
ADVISOR_GUARD_REASONING_EFFORT=
```

Model names must exactly match the values returned by OpenWebUI `/api/models`. Restart the backend after changing `.env`.

Use `LOCAL_LLM_VERIFY_SSL=false` only as a temporary workaround for a local certificate-chain problem. Restore verification after installing the correct CA chain.

### 7.3 Main environment variables

| Variable | Purpose |
|---|---|
| `LOCAL_LLM_PROVIDER` | `ollama` or `openwebui` |
| `LOCAL_LLM_BASE_URL` | Model-service URL |
| `LOCAL_LLM_MODEL` | Default answer model |
| `LOCAL_LLM_API_KEY` | OpenWebUI API key |
| `LOCAL_LLM_TIMEOUT_SECONDS` | Model-call timeout |
| `LOCAL_RAG_EMBED_MODEL` | Embedding model used to build and query the index |
| `LOCAL_RAG_DATA_DIR` | Source-document path |
| `LOCAL_RAG_INDEX_DIR` | RAG-index path |
| `ADVISOR_GUARD_MODEL` | Fixed request-assessment model |
| `ADVISOR_GUARD_TIMEOUT_SECONDS` | Guard timeout before deterministic fallback |
| `ADVISOR_GUARD_MAX_TOKENS` | Guard output-token budget; reasoning models may require a larger value |
| `ADVISOR_GUARD_REASONING_EFFORT` | Optional `low`, `medium`, or `high` reasoning setting |
| `OPENAI_API_KEY` | OpenAI Judge API key |
| `DEEPSEEK_API_KEY` | DeepSeek Judge API key |

See [`backend/.env.example`](../../backend/.env.example) for the complete example.

## 10. Testing and Validation

### 10.1 Backend

```powershell
cd backend
.\.venv\Scripts\python.exe manage.py check
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
```

### 10.2 Frontend

```powershell
cd frontend
npm.cmd run build
```

`npm.cmd run build` performs TypeScript/Vue type checking and the Vite production build.

## 13. Troubleshooting

### Backend reports that Django is missing

Use the project virtual environment explicitly:

```powershell
.\.venv\Scripts\python.exe manage.py check
```

### RAG index not ready

```powershell
.\.venv\Scripts\python.exe manage.py build_rag_index
```

### OpenWebUI model not found

Confirm that the model name exactly matches `/api/models`, then restart the backend after changing `.env`.

### Windows blocks npm.ps1

Use `npm.cmd`:

```powershell
npm.cmd run dev
```

### SSL certificate verify failed

Install the organisation's CA chain in the Python environment. Use `LOCAL_LLM_VERIFY_SSL=false` only for temporary local debugging.
