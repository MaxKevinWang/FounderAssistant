# Founder Assistant

A web assistant for questions about starting a business in Germany.

## Prerequisites

- Python 3.12 and Node.js 22 (including npm).
- Docker Compose for the PostgreSQL and Redis services used below.
- Access to a running Ollama or OpenWebUI service with an answer model and an
  embedding model available.

The commands below use Windows PowerShell and start from the repository root.

## Configure `backend/.env`

For first-time setup, copy the template:

```powershell
Copy-Item backend/.env.example backend/.env
```

Edit `backend/.env` with your own service settings. The backend reads this file
at startup; restart it after changing values. Keep real keys in this ignored
file. Other users need their own credentials for the service they connect to.

Choose one of the following model configurations.

### Local Ollama

```env
LOCAL_LLM_PROVIDER=ollama
LOCAL_LLM_BASE_URL=http://127.0.0.1:11434
LOCAL_LLM_MODEL=qwen2.5:7b
LOCAL_LLM_API_KEY=
LOCAL_LLM_VERIFY_SSL=true
LOCAL_RAG_EMBED_MODEL=bge-m3
ADVISOR_GUARD_MODEL=ministral-3:14b
```

For a local Ollama service without authentication, leave the API key empty.
Install the models and keep Ollama running:

```powershell
ollama pull qwen2.5:7b
ollama pull bge-m3
ollama pull ministral-3:14b
```

### OpenWebUI

```env
LOCAL_LLM_PROVIDER=openwebui
LOCAL_LLM_BASE_URL=https://your-openwebui-server.example
LOCAL_LLM_MODEL=your-answer-model-id
LOCAL_LLM_API_KEY=your-own-openwebui-api-key
LOCAL_LLM_VERIFY_SSL=true
LOCAL_RAG_EMBED_MODEL=your-embedding-model-id
ADVISOR_GUARD_MODEL=
```

Replace the placeholders with your server URL, your OpenWebUI account's API key,
and model IDs available on that server. The service must support both generation
and embedding requests. An empty `ADVISOR_GUARD_MODEL` reuses the answer model;
set it to another available model if you want a separate request-assessment model.

### Database and file paths

For the local database started by `docker compose` below, keep these template values:

```env
DB_ENGINE=django.db.backends.postgresql
DB_NAME=founder_assistant
DB_USER=founder
DB_PASSWORD=founder
DB_HOST=localhost
DB_PORT=5432
LOCAL_RAG_DATA_DIR=../knowledge_base
LOCAL_RAG_INDEX_DIR=rag_index
```

Replace `DJANGO_SECRET_KEY=change-me` with a generated value. For example, run
`py -3.12 -c "import secrets; print(secrets.token_urlsafe(50))"` and paste its
output into that setting. The template's `DJANGO_DEBUG=true` is for local use.

## Start the application

1. Complete the `.env` configuration above.

2. Start the database and Redis from the repository root:

   ```powershell
   docker compose up -d db redis
   ```

3. Install the backend, prepare the database and knowledge index, then start it:

   ```powershell
   cd backend
   py -3.12 -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements/base.txt
   .\.venv\Scripts\python.exe manage.py migrate
   .\.venv\Scripts\python.exe manage.py build_rag_index
   .\.venv\Scripts\python.exe manage.py runserver 127.0.0.1:8000
   ```

   `requirements/base.txt` installs the dependencies needed to run the system.
   For development and tests, install `requirements/dev.txt` instead; it includes
   the base dependencies plus pytest, Ruff, Black, and mypy. The optional
   `requirements/reranker.txt` is only needed for reranking experiments.

   First add your own documents to the three domain directories under
   `knowledge_base/`, as described in the [knowledge-base guide](docs/knowledge_base/README.md).
   No document corpus or prebuilt index is included in this repository.

   The index build uses the configured embedding service; keep it running. Rebuild
   the index if you change the embedding model.

4. In another PowerShell window, start the frontend from the repository root:

   ```powershell
   cd frontend
   npm.cmd ci
   npm.cmd run dev -- --host 127.0.0.1
   ```

   `npm.cmd ci` installs the frontend dependencies from `package-lock.json`.

5. Open <http://127.0.0.1:5173/advisor>. Keep both terminals and the model service running.

## Use the frontend

1. Click **+ New chat**, or select an existing chat in the sidebar.
2. Use the topic selector below the message box to choose legal form selection,
   funding preparation, or market validation. Choose **Not sure · Search all topics**
   to search across topics. With **Choose topic after sending**, the assistant asks
   you to select a topic before answering.
3. Select an available model, or keep **Auto model**.
4. Type a question and click Send or press **Enter**. Use **Shift+Enter** for a new line.
5. Read the streamed answer and the **Sources** listed beneath it. Open source links
   to inspect the supporting material. Answer any clarification questions to continue.
6. Use the copy button to copy an answer, or the thumbs buttons to record feedback.
7. Open a chat's sidebar menu to **Rename** or **Delete** it.

Chats are saved in this browser. Deleting a chat removes its local history.
If a sensitive-information prompt appears, edit the message to remove personal details.

For additional configuration and troubleshooting, see the
[configuration guide](docs/deployment/configuration_and_validation.md). Architecture,
knowledge-base maintenance, and developer instructions are listed in the
[documentation index](docs/README.md).
