# Local Open-Source LLM + RAG Deployment (Ollama)

## 1. Goal

Deploy a fully local demo with:

1. Generation models: `qwen2.5:7b`, `llama3.1:8b`, `gemma3:12b`
2. Embedding model: `bge-m3`
3. Backend: Django API
4. Frontend: Vue benchmark/advisor pages

## 2. Install and start Ollama (Windows)

```powershell
winget install Ollama.Ollama -e
ollama serve
```

## 3. Pull required models

```powershell
ollama pull qwen2.5:7b
ollama pull llama3.1:8b
ollama pull gemma3:12b
ollama pull bge-m3
```

Optional checks:

```powershell
ollama run qwen2.5:7b "hello"
ollama run llama3.1:8b "hello"
ollama run gemma3:12b "hello"
ollama embed bge-m3 "test sentence"
```

## 4. Backend environment

`backend/.env` example:

```env
LOCAL_LLM_BASE_URL=http://127.0.0.1:11434
LOCAL_LLM_MODEL=qwen2.5:7b
LOCAL_LLM_TIMEOUT_SECONDS=360
LOCAL_RAG_EMBED_MODEL=bge-m3
LOCAL_RAG_DATA_DIR=../data
LOCAL_RAG_INDEX_DIR=rag_index
LOCAL_RAG_CHUNK_SIZE=1200
LOCAL_RAG_CHUNK_OVERLAP=180
LOCAL_RAG_MIN_CHUNK_CHARS=180
```

Notes:

1. `LOCAL_LLM_MODEL` is only default model.
2. `/advisor` and `/benchmark` can override generation model per request.
3. Embedding model remains fixed for fair RAG comparison unless you intentionally change it.

## 5. Build index and run services

```powershell
cd backend
.\.venv\Scripts\python.exe manage.py build_rag_index
.\.venv\Scripts\python.exe manage.py runserver
```

```powershell
cd frontend
npm.cmd run dev
```

## 6. Verify status endpoints

```text
GET http://127.0.0.1:8000/api/advisors/local-llm/status/
GET http://127.0.0.1:8000/api/advisors/rag/status/
```

Expected local-llm status fields:

1. `available_models`
2. `generation_models`
3. `embedding_models`

## 7. Advisor usage

`/advisor` supports:

1. Pure LLM mode
2. RAG mode
3. Primary + comparison model selection
4. Side-by-side output comparison

## 8. Benchmark usage

CLI:

```powershell
cd backend
.\.venv\Scripts\python.exe manage.py run_baseline_rag_eval --generation-model qwen2.5:7b
.\.venv\Scripts\python.exe manage.py run_baseline_rag_eval --generation-model llama3.1:8b
.\.venv\Scripts\python.exe manage.py run_baseline_rag_eval --generation-model gemma3:12b
.\.venv\Scripts\python.exe manage.py run_baseline_rag_eval --generation-model qwen2.5:7b --repeats 3
```

API:

```text
POST http://127.0.0.1:8000/api/evaluations/benchmark/run/
```

Payload example:

```json
{
  "mode": "both",
  "generation_model": "gemma3:12b",
  "repeats": 1,
  "skip_scenario": false
}
```

Outputs:

- `docs/evaluation/runs/benchmark_*.json`
- `docs/evaluation/runs/benchmark_*.md`

## 9. Troubleshooting

1. If model not found, run `ollama list` and pull missing model.
2. If RAG status shows index not ready, run `python manage.py build_rag_index`.
3. If responses are slow, reduce `top_k` and `max_tokens` first.
