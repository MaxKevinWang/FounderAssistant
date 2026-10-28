# Development

Install `backend/requirements/dev.txt` in a local virtual environment. From `backend/`, run:

```powershell
.\.venv\Scripts\python.exe manage.py check
.\.venv\Scripts\python.exe -m pytest -q
```

From `frontend/`, run `npm.cmd ci` followed by `npm.cmd run build`.

The source distribution retains backend evaluation tooling and tests, but excludes the academic datasets, reports and document corpus. `backend/tests/conftest.py` lists optional repository-integrity test modules and omits them from collection when their external artifact directory is absent. Supplying the matching historical artifacts enables those modules again. Source-level evaluation and application tests remain enabled.

The two ingestion notebooks contain source cells with execution output cleared. FAQ crawls, checkpoints, knowledge documents and RAG indices are generated locally and ignored by Git.

This repository starts with a fresh Git history. It includes no original repository history, thesis, literature, questionnaires, survey analysis, presentations, progress reports or experimental outputs. The original research repository remains separate.

Versioned JSON judge prompts are retained as executable evaluation configuration, without evaluation inputs or results.
