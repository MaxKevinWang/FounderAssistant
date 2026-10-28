"""Run auditable Baseline/RAG generations on frozen development QRA items.

This module deliberately does not score answers. It creates judge-ready raw
artifacts while the closed-model judge protocol is still being finalized.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import random
import re
import statistics
import time
import uuid
from contextlib import contextmanager
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

from django.conf import settings

from apps.advisors.local_llm import check_status, generate_text
from apps.advisors.model_profiles import DEFAULT_ANSWER_MAX_TOKENS, classify_generation_model
from apps.advisors.rag_pipeline import DEFAULT_RAG_CONFIG, rag_answer


class FormalBenchmarkError(RuntimeError):
    """Raised when frozen inputs or benchmark configuration are invalid."""


ProgressCallback = Callable[[dict[str, Any]], None]
GenerationFunction = Callable[..., dict[str, Any]]

ALLOWED_MODES = {"baseline", "rag"}
ALLOWED_SPLITS = {"development", "test"}
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]+")
FORMAL_TEST_MANIFEST_NAME = "formal_test_qra_manifest_v1.0.json"
REQUIRED_QRA_FIELDS = {
    "id",
    "version",
    "status",
    "split",
    "review_status",
    "domain",
    "question_type",
    "language",
    "question",
    "answerable",
    "answer",
    "keypoints",
    "references",
    "configuration_path",
    "configuration_sha256",
}


def _process_is_running(pid: int) -> bool:
    """Return whether ``pid`` still identifies a live process."""

    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        process_query_limited_information = 0x1000
        still_active = 259
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.OpenProcess(
            process_query_limited_information,
            False,
            pid,
        )
        if not handle:
            return ctypes.get_last_error() == 5  # Access denied means it exists.
        try:
            exit_code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return True
            return exit_code.value == still_active
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:
        return exc.errno != errno.ESRCH
    return True


@contextmanager
def formal_run_lock(output_dir: Path, run_id: str | None) -> Iterator[None]:
    """Ensure one CLI process writes a named formal run at a time.

    Explicit run IDs are resumable and therefore vulnerable to two processes
    updating the same checkpoint. The lock records its owner PID and safely
    reclaims files left behind by a process that no longer exists.
    """

    if run_id is None:
        yield
        return
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise FormalBenchmarkError(
            "run_id must contain only letters, digits, dots, underscores, and hyphens"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / f"{run_id}.lock"
    owner = {
        "pid": os.getpid(),
        "token": uuid.uuid4().hex,
        "created_at": datetime.now(UTC).isoformat(),
    }

    for _ in range(2):
        try:
            descriptor = os.open(
                lock_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            )
        except FileExistsError:
            try:
                existing = json.loads(lock_path.read_text(encoding="utf-8"))
                existing_pid = int(existing.get("pid", 0))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                existing_pid = 0
                try:
                    age_seconds = time.time() - lock_path.stat().st_mtime
                except OSError:
                    age_seconds = 0
                if age_seconds < 30:
                    raise FormalBenchmarkError(
                        f"Formal run lock is being initialized: {lock_path}"
                    ) from None
            if _process_is_running(existing_pid):
                raise FormalBenchmarkError(
                    f"Formal run is already active: {run_id} (pid {existing_pid})"
                ) from None
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass
            continue
        else:
            with os.fdopen(descriptor, "w", encoding="utf-8") as target:
                json.dump(owner, target, sort_keys=True)
                target.flush()
                os.fsync(target.fileno())
            break
    else:
        raise FormalBenchmarkError(f"Could not acquire formal run lock: {lock_path}")

    try:
        yield
    finally:
        try:
            current = json.loads(lock_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            current = {}
        if current.get("token") == owner["token"]:
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass


def project_root() -> Path:
    """Return the repository root."""

    return Path(settings.BASE_DIR).parent


def default_qra_dir() -> Path:
    """Return the versioned formal QRA artifact directory."""

    return project_root() / "docs" / "evaluation" / "formal_dataset" / "qra"


def default_formal_qra_dir() -> Path:
    """Return the frozen formal-test QRA release directory."""

    return default_qra_dir() / "formal_test"


def default_prompt_path() -> Path:
    """Return the development answer-generation prompt artifact."""

    return (
        project_root()
        / "docs"
        / "evaluation"
        / "formal_dataset"
        / "prompts"
        / "development_answer_generation_prompt_v0.1.json"
    )


def default_formal_prompt_path() -> Path:
    """Return the frozen formal-test answer-generation prompt."""

    return (
        project_root()
        / "docs"
        / "evaluation"
        / "formal_dataset"
        / "prompts"
        / "formal_answer_generation_prompt_v1.0.json"
    )


def default_output_dir() -> Path:
    """Return the development-run output directory."""

    return project_root() / "docs" / "evaluation" / "runs" / "development"


def default_formal_output_dir() -> Path:
    """Return the formal-test generation output directory."""

    return project_root() / "docs" / "evaluation" / "runs" / "formal"


def sha256_file(path: Path) -> str:
    """Hash one artifact without loading the whole file into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FormalBenchmarkError(f"Required JSON artifact not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FormalBenchmarkError(f"Invalid JSON artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise FormalBenchmarkError(f"Expected a JSON object in {path}")
    return payload


def _resolve_artifact_path(raw_path: str, root: Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else root / path


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _validate_frozen_qra(
    path: Path,
    root: Path,
    *,
    expected_split: str | None = None,
    freeze_record_path: Path | None = None,
) -> dict[str, Any]:
    qra = _load_json(path)
    missing = sorted(REQUIRED_QRA_FIELDS - qra.keys())
    if missing:
        raise FormalBenchmarkError(f"{path.name} is missing fields: {missing}")
    split = str(qra["split"])
    if split not in ALLOWED_SPLITS:
        raise FormalBenchmarkError(f"{path.name} has unsupported split: {split}")
    if expected_split is not None and split != expected_split:
        raise FormalBenchmarkError(
            f"{path.name} split mismatch: expected {expected_split}, found {split}"
        )
    if (
        qra["version"] != "1.0"
        or qra["status"] != "frozen"
        or qra["review_status"] != "single_checked"
    ):
        raise FormalBenchmarkError(
            f"{path.name} is not a frozen v1.0/single_checked QRA"
        )
    if not str(qra["question"]).strip() or not str(qra["answer"]).strip():
        raise FormalBenchmarkError(f"{path.name} has an empty question or answer")
    if not 3 <= len(qra["keypoints"]) <= 5:
        raise FormalBenchmarkError(f"{path.name} must contain 3-5 Keypoints")

    freeze_record_path = freeze_record_path or path.with_name(
        f"{path.stem}_freeze_record.json"
    )
    freeze_record = _load_json(freeze_record_path)
    artifacts = freeze_record.get("artifacts", {})
    qra_artifact = artifacts.get("frozen_qra") or artifacts.get("qra") or {}
    recorded_path = _resolve_artifact_path(str(qra_artifact.get("path", "")), root)
    if recorded_path.resolve() != path.resolve():
        raise FormalBenchmarkError(
            f"Freeze record path mismatch for {qra['id']}: {recorded_path}"
        )
    if qra_artifact.get("sha256") != sha256_file(path):
        raise FormalBenchmarkError(f"Freeze record hash mismatch for {qra['id']}")

    configuration_path = _resolve_artifact_path(
        str(qra["configuration_path"]),
        root,
    )
    if not configuration_path.exists():
        raise FormalBenchmarkError(
            f"Configuration missing for {qra['id']}: {configuration_path}"
        )
    if qra["configuration_sha256"] != sha256_file(configuration_path):
        raise FormalBenchmarkError(f"Configuration hash mismatch for {qra['id']}")

    return {
        **qra,
        "_artifact": {
            "path": _display_path(path, root),
            "sha256": sha256_file(path),
            "freeze_record_path": _display_path(freeze_record_path, root),
            "freeze_record_sha256": sha256_file(freeze_record_path),
        },
    }


def _validate_declared_artifact(
    declaration: dict[str, Any],
    *,
    root: Path,
    label: str,
) -> dict[str, Any]:
    raw_path = str(declaration.get("path", ""))
    expected_sha256 = str(declaration.get("sha256", ""))
    if not raw_path or not expected_sha256:
        raise FormalBenchmarkError(f"{label} declaration is incomplete")
    path = _resolve_artifact_path(raw_path, root).resolve()
    if not path.exists():
        raise FormalBenchmarkError(f"{label} does not exist: {path}")
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise FormalBenchmarkError(f"{label} hash mismatch: {path}")
    return {
        "path": _display_path(path, root),
        "sha256": actual_sha256,
    }


def _load_frozen_release_manifest(
    manifest_path: Path,
    *,
    root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = manifest_path.resolve()
    manifest = _load_json(manifest_path)
    if (
        manifest.get("version") != "1.0"
        or manifest.get("status") != "frozen"
        or manifest.get("split") not in ALLOWED_SPLITS
        or manifest.get("review_status") != "single_checked"
    ):
        raise FormalBenchmarkError(
            f"{manifest_path.name} is not a frozen v1.0/single_checked release"
        )
    declarations = manifest.get("qras")
    if not isinstance(declarations, list) or not declarations:
        raise FormalBenchmarkError(f"{manifest_path.name} has no QRA declarations")
    if manifest.get("qra_count") != len(declarations):
        raise FormalBenchmarkError(
            f"{manifest_path.name} qra_count does not match its declarations"
        )

    manifest_freeze_path = manifest_path.with_name(
        f"{manifest_path.stem}_freeze_record.json"
    )
    manifest_freeze = _load_json(manifest_freeze_path)
    recorded_manifest_path = _resolve_artifact_path(
        str(manifest_freeze.get("file", "")),
        root,
    )
    if recorded_manifest_path.resolve() != manifest_path:
        raise FormalBenchmarkError(
            f"Formal release manifest freeze path mismatch: {recorded_manifest_path}"
        )
    if manifest_freeze.get("file_sha256") != sha256_file(manifest_path):
        raise FormalBenchmarkError(
            f"Formal release manifest hash mismatch: {manifest_path}"
        )

    dependencies: dict[str, dict[str, Any]] = {}
    for name in (
        "candidate_manifest",
        "configuration_manifest",
        "document_manifest",
    ):
        declaration = manifest.get(name)
        if isinstance(declaration, dict):
            dependencies[name] = _validate_declared_artifact(
                declaration,
                root=root,
                label=f"Formal release {name}",
            )

    release_artifact = {
        "manifest_id": str(manifest.get("manifest_id", manifest_path.stem)),
        "version": str(manifest["version"]),
        "split": str(manifest["split"]),
        "review_status": str(manifest["review_status"]),
        "path": _display_path(manifest_path, root),
        "sha256": sha256_file(manifest_path),
        "freeze_record_path": _display_path(manifest_freeze_path, root),
        "freeze_record_sha256": sha256_file(manifest_freeze_path),
        "dependencies": dependencies,
    }

    items: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for declaration in declarations:
        if not isinstance(declaration, dict):
            raise FormalBenchmarkError("Formal release contains a non-object QRA entry")
        qra_id = str(declaration.get("qra_id", ""))
        if not qra_id or qra_id in seen_ids:
            raise FormalBenchmarkError(f"Invalid or duplicate formal QRA ID: {qra_id}")
        seen_ids.add(qra_id)
        qra_path = _resolve_artifact_path(str(declaration.get("path", "")), root)
        freeze_path = _resolve_artifact_path(
            str(declaration.get("freeze_record_path", "")),
            root,
        )
        if declaration.get("file_sha256") != sha256_file(qra_path):
            raise FormalBenchmarkError(f"Manifest QRA hash mismatch for {qra_id}")
        if declaration.get("freeze_record_sha256") != sha256_file(freeze_path):
            raise FormalBenchmarkError(
                f"Manifest QRA freeze-record hash mismatch for {qra_id}"
            )
        item = _validate_frozen_qra(
            path=qra_path,
            root=root,
            expected_split=str(manifest["split"]),
            freeze_record_path=freeze_path,
        )
        declared_values = {
            "id": qra_id,
            "version": str(declaration.get("version", "")),
            "domain": str(declaration.get("domain", "")),
            "question_type": str(declaration.get("question_type", "")),
            "review_status": str(declaration.get("review_status", "")),
        }
        mismatches = [
            field
            for field, expected in declared_values.items()
            if str(item.get(field, "")) != expected
        ]
        if mismatches:
            raise FormalBenchmarkError(
                f"Manifest declaration mismatch for {qra_id}: {mismatches}"
            )
        item["_release_manifest"] = release_artifact
        items.append(item)
    return release_artifact, items


def load_frozen_development_qras(
    qra_dir: Path | None = None,
    *,
    qra_ids: Iterable[str] | None = None,
    domains: Iterable[str] | None = None,
    question_types: Iterable[str] | None = None,
    limit: int | None = None,
    root: Path | None = None,
) -> list[dict[str, Any]]:
    """Load and verify frozen development or formal-test QRA items."""

    root = (root or project_root()).resolve()
    qra_dir = (qra_dir or default_qra_dir()).resolve()
    requested_ids = {value.strip() for value in qra_ids or [] if value.strip()}
    domain_filter = {value.strip() for value in domains or [] if value.strip()}
    type_filter = {value.strip() for value in question_types or [] if value.strip()}
    if limit is not None and limit < 1:
        raise FormalBenchmarkError("limit must be at least 1")
    if not qra_dir.exists():
        raise FormalBenchmarkError(f"QRA directory not found: {qra_dir}")

    manifest_path = qra_dir / FORMAL_TEST_MANIFEST_NAME
    if not manifest_path.exists():
        release_manifests = sorted(qra_dir.glob("*_qra_manifest_v1.0.json"))
        if len(release_manifests) > 1:
            raise FormalBenchmarkError(
                f"QRA directory contains multiple v1.0 release manifests: {qra_dir}"
            )
        if release_manifests:
            manifest_path = release_manifests[0]
    if manifest_path.exists():
        _, candidates = _load_frozen_release_manifest(manifest_path, root=root)
    else:
        candidates = [
            _validate_frozen_qra(path=path, root=root)
            for path in sorted(qra_dir.glob("*_v1.0.json"))
        ]

    items: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for item in candidates:
        qra_id = str(item["id"])
        if qra_id in seen_ids:
            raise FormalBenchmarkError(f"Duplicate frozen QRA ID: {qra_id}")
        seen_ids.add(qra_id)
        if requested_ids and qra_id not in requested_ids:
            continue
        if domain_filter and item["domain"] not in domain_filter:
            continue
        if type_filter and item["question_type"] not in type_filter:
            continue
        items.append(item)

    missing_ids = sorted(requested_ids - seen_ids)
    if missing_ids:
        raise FormalBenchmarkError(f"Requested frozen QRA IDs not found: {missing_ids}")
    if limit is not None:
        items = items[:limit]
    if not items:
        raise FormalBenchmarkError("No frozen QRA items matched")
    return items


def load_generation_prompt(path: Path | None = None) -> dict[str, Any]:
    """Load the versioned prompt and verify its template placeholder."""

    path = (path or default_prompt_path()).resolve()
    prompt = _load_json(path)
    required = {
        "prompt_id",
        "version",
        "status",
        "system_prompt",
        "baseline_prompt_template",
    }
    missing = sorted(required - prompt.keys())
    if missing:
        raise FormalBenchmarkError(f"Prompt artifact is missing fields: {missing}")
    template = str(prompt["baseline_prompt_template"])
    if template.count("{question}") != 1:
        raise FormalBenchmarkError(
            "Baseline prompt template must contain exactly one {question}"
        )
    status = str(prompt["status"])
    if status not in {"development", "frozen"}:
        raise FormalBenchmarkError(f"Unsupported prompt status: {status}")
    artifact = {
        "path": _display_path(path, project_root()),
        "sha256": sha256_file(path),
    }
    if status == "frozen":
        freeze_path = path.with_name(f"{path.stem}_freeze_record.json")
        freeze_record = _load_json(freeze_path)
        recorded_path = _resolve_artifact_path(
            str(freeze_record.get("file", "")),
            project_root(),
        )
        if recorded_path.resolve() != path:
            raise FormalBenchmarkError(f"Prompt freeze path mismatch: {recorded_path}")
        if freeze_record.get("file_sha256") != artifact["sha256"]:
            raise FormalBenchmarkError(f"Prompt freeze hash mismatch: {path}")
        artifact.update(
            freeze_record_path=_display_path(freeze_path, project_root()),
            freeze_record_sha256=sha256_file(freeze_path),
        )
    return {
        **prompt,
        "_artifact": artifact,
    }


def build_dataset_manifest(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Create a standalone, hashable snapshot for later judge execution."""

    manifest_items = []
    for item in items:
        manifest_items.append(
            {
                "id": item["id"],
                "domain": item["domain"],
                "question_type": item["question_type"],
                "language": item["language"],
                "question": item["question"],
                "answerable": item["answerable"],
                "reference_answer": item["answer"],
                "keypoints": item["keypoints"],
                "references": item["references"],
                "qra_artifact": item["_artifact"],
                "configuration_path": item["configuration_path"],
                "configuration_sha256": item["configuration_sha256"],
            }
        )
    splits = {str(item["split"]) for item in items}
    review_statuses = {str(item["review_status"]) for item in items}
    if len(splits) != 1 or len(review_statuses) != 1:
        raise FormalBenchmarkError("A generation run cannot mix QRA releases")
    split = next(iter(splits))
    release_manifests = {
        _canonical_sha256(item["_release_manifest"]): item["_release_manifest"]
        for item in items
        if "_release_manifest" in item
    }
    if release_manifests and len(release_manifests) != 1:
        raise FormalBenchmarkError("QRA items refer to different release manifests")
    release_manifest = (
        next(iter(release_manifests.values())) if release_manifests else None
    )
    dataset = {
        "dataset_id": (
            release_manifest["manifest_id"]
            if release_manifest
            else f"frozen_{split}_qra_v1.0"
        ),
        "split": split,
        "review_status": next(iter(review_statuses)),
        "item_count": len(manifest_items),
        "items": manifest_items,
        "manifest_sha256": _canonical_sha256(manifest_items),
    }
    if release_manifest:
        dataset["release_manifest"] = release_manifest
    return dataset


def snapshot_rag_index(root: Path | None = None) -> dict[str, Any]:
    """Capture exact index files and non-secret retrieval configuration."""

    root = (root or project_root()).resolve()
    index_dir = DEFAULT_RAG_CONFIG.index_dir.resolve()
    metadata_path = index_dir / "metadata.json"
    embeddings_path = index_dir / "embeddings.npy"
    source_manifest_path = index_dir / "source_manifest.json"
    if (
        not metadata_path.exists()
        or not embeddings_path.exists()
        or not source_manifest_path.exists()
    ):
        raise FormalBenchmarkError(
            f"RAG index is incomplete in {index_dir}; rebuild it before evaluation"
        )
    metadata = _load_json(metadata_path)
    return {
        "index_dir": _display_path(index_dir, root),
        "metadata_path": _display_path(metadata_path, root),
        "metadata_sha256": sha256_file(metadata_path),
        "embeddings_path": _display_path(embeddings_path, root),
        "embeddings_sha256": sha256_file(embeddings_path),
        "source_manifest_path": _display_path(source_manifest_path, root),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "embedding_model": metadata.get("embedding_model"),
        "embedding_provider": metadata.get("embedding_provider"),
        "generated_at": metadata.get("generated_at"),
        "document_count": metadata.get("document_count"),
        "chunk_count": metadata.get("chunk_count"),
        "chunk_size": metadata.get("chunk_size"),
        "chunk_overlap": metadata.get("chunk_overlap"),
        "min_chunk_chars": metadata.get("min_chunk_chars"),
    }


def validate_models(
    requested_models: Iterable[str],
    available_models: Iterable[str],
) -> list[str]:
    """Normalize model IDs and reject duplicates or unavailable models."""

    models = [value.strip() for value in requested_models if value.strip()]
    if not models:
        raise FormalBenchmarkError("At least one generation model is required")
    if len(models) != len(set(models)):
        raise FormalBenchmarkError("Duplicate generation model IDs are not allowed")
    unavailable = sorted(set(models) - set(available_models))
    if unavailable:
        raise FormalBenchmarkError(
            f"Generation models are unavailable from the configured service: {unavailable}"
        )
    return models


def _safe_generation_call(
    *,
    mode: str,
    question: str,
    model: str,
    system_prompt: str,
    baseline_prompt_template: str,
    temperature: float,
    max_tokens: int,
    top_k: int,
    baseline_fn: GenerationFunction,
    rag_fn: GenerationFunction,
    reasoning_effort: str | None = None,
    domain: str = "",
    rag_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    started_at = datetime.now(UTC).isoformat()
    try:
        if mode == "baseline":
            payload = baseline_fn(
                prompt=baseline_prompt_template.format(question=question),
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                model=model,
                reasoning_effort=reasoning_effort,
            )
            citations: list[dict[str, Any]] = []
            retrieved_chunks: list[dict[str, Any]] = []
            retrieval_latency_ms = 0
            retrieval_mode = "none"
        else:
            payload = rag_fn(
                question=question,
                domain=domain,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                top_k=top_k,
                model=model,
                reasoning_effort=reasoning_effort,
                include_retrieved_text=True,
                **dict(rag_options or {}),
            )
            citations = list(payload.get("citations", []))
            retrieved_chunks = list(payload.get("retrieved_chunks", []))
            retrieval_latency_ms = int(payload.get("retrieval_latency_ms", 0))
            retrieval_mode = str(payload.get("retrieval_mode", ""))
        raw_response = payload.get("response", "")
        response = raw_response if isinstance(raw_response, str) else ""
        finish_reason = str(payload.get("finish_reason", "") or "")
        raw_usage = payload.get("usage", {})
        usage = raw_usage if isinstance(raw_usage, dict) else {}
        output_token_count = usage.get(
            "output_tokens",
            usage.get("completion_tokens", 0),
        )
        output_token_count = (
            int(output_token_count)
            if isinstance(output_token_count, (int, float))
            else 0
        )
        reasoning_char_count = int(payload.get("reasoning_content_char_count", 0) or 0)
        result = {
            "status": "success",
            "started_at": started_at,
            "completed_at": datetime.now(UTC).isoformat(),
            "provider": str(payload.get("provider", "")),
            "response_model": str(payload.get("model", model)),
            "done": bool(payload.get("done", False)),
            "response": response,
            "response_char_count": len(response),
            "latency_ms": int(payload.get("latency_ms", 0)),
            "finish_reason": finish_reason,
            "usage": usage,
            "reasoning_content_present": bool(
                payload.get("reasoning_content_present", False)
            ),
            "reasoning_content_char_count": reasoning_char_count,
            "retrieval_latency_ms": retrieval_latency_ms,
            "retrieval_mode": retrieval_mode,
            "citation_count": len(citations),
            "citations": citations,
            "retrieved_chunks": retrieved_chunks,
            "retrieval_details": {
                key: payload.get(key)
                for key in (
                    "candidate_retrieval_latency_ms",
                    "reranker_latency_ms",
                    "selection_latency_ms",
                    "expansion_latency_ms",
                    "candidate_k",
                    "candidate_count",
                    "context_k",
                    "selected_chunk_count",
                    "estimated_context_tokens",
                    "token_estimator",
                    "reranker_model",
                    "reranker_revision",
                    "expanded_neighbor_count",
                    "rag_prompt_version",
                    "confidence_gating_applied",
                    "retrieval_confidence",
                )
                if key in payload
            },
            "error_type": "",
            "error": "",
        }
        # Some OpenAI-compatible gateways include hidden reasoning or use a
        # different accounting boundary in `output_tokens`, so totals can be
        # greater than the requested visible-output limit even when the model
        # returns a complete response with finish_reason=stop. Equality remains
        # a useful fallback for providers that misleadingly report `stop` when
        # they terminate exactly at the configured cap (observed with Kimi K3).
        reached_reported_limit = output_token_count == max_tokens
        if finish_reason.lower() == "length" or reached_reported_limit:
            result.update(
                status="error",
                error_type="FormalBenchmarkError",
                error=(
                    "Model output reached the token limit and may be truncated "
                    f"(reported output tokens: {output_token_count}, "
                    f"configured maximum: {max_tokens})"
                ),
            )
        elif not response.strip():
            result.update(
                status="error",
                error_type="FormalBenchmarkError",
                error="Model service returned an empty response",
            )
        return result
    except Exception as exc:  # noqa: BLE001 - preserve long batch progress
        return {
            "status": "error",
            "started_at": started_at,
            "completed_at": datetime.now(UTC).isoformat(),
            "provider": "",
            "response_model": model,
            "done": False,
            "response": "",
            "response_char_count": 0,
            "latency_ms": 0,
            "finish_reason": "",
            "usage": {},
            "reasoning_content_present": False,
            "reasoning_content_char_count": 0,
            "retrieval_latency_ms": 0,
            "retrieval_mode": "error" if mode == "rag" else "none",
            "citation_count": 0,
            "citations": [],
            "retrieved_chunks": [],
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def aggregate_generation_results(
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize operational metrics without pretending to score correctness."""

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for item in results:
        groups.setdefault((item["model"], item["mode"]), []).append(item)

    summaries = []
    for (model, mode), group in sorted(groups.items()):
        successful = [
            item for item in group if item["generation"]["status"] == "success"
        ]
        latency = [
            item["generation"]["latency_ms"]
            for item in successful
            if item["generation"]["latency_ms"] > 0
        ]
        retrieval_latency = [
            item["generation"]["retrieval_latency_ms"]
            for item in successful
            if item["generation"]["retrieval_latency_ms"] > 0
        ]
        response_chars = [
            item["generation"]["response_char_count"] for item in successful
        ]
        retrieval_modes: dict[str, int] = {}
        for item in successful:
            value = item["generation"]["retrieval_mode"]
            retrieval_modes[value] = retrieval_modes.get(value, 0) + 1
        summaries.append(
            {
                "model": model,
                "model_type": classify_generation_model(model),
                "mode": mode,
                "scheduled": len(group),
                "succeeded": len(successful),
                "errors": len(group) - len(successful),
                "mean_latency_seconds": (
                    round(statistics.mean(latency) / 1000, 3) if latency else 0.0
                ),
                "mean_retrieval_seconds": (
                    round(statistics.mean(retrieval_latency) / 1000, 3)
                    if retrieval_latency
                    else 0.0
                ),
                "mean_response_chars": (
                    round(statistics.mean(response_chars), 1) if response_chars else 0.0
                ),
                "retrieval_modes": retrieval_modes,
            }
        )
    return summaries


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def build_markdown_report(report: dict[str, Any]) -> str:
    """Render a review-oriented companion to the complete JSON artifact."""

    split = str(report["dataset"]["split"])
    run_label = "formal test" if split == "test" else "formal development"
    lines = [
        f"# {run_label.title()} Baseline/RAG generation report",
        "",
        f"- Run ID: `{report['run_id']}`",
        f"- Status: `{report['status']}`",
        f"- Dataset: `{report['dataset']['dataset_id']}`",
        f"- Dataset hash: `{report['dataset']['manifest_sha256']}`",
        f"- QRA items: `{report['dataset']['item_count']}`",
        f"- Conditions: `{', '.join(report['config']['modes'])}`",
        f"- Temperature: `{report['config']['temperature']}`",
        f"- Top-K: `{report['config']['top_k']}`",
        f"- Max tokens: `{report['config']['max_tokens']}`",
        f"- Repeats: `{report['config'].get('repeats', 1)}`",
        f"- Schedule seed: `{report['config'].get('schedule_seed', '')}`",
        "",
        f"This artifact records unscored {run_label} generations. Correctness, "
        "Keypoint coverage, faithfulness, and irrelevance still require the "
        "frozen Judge protocol.",
        "",
        "## Operational summary",
        "",
        "| Model | Type | Mode | Success | Errors | Mean latency (s) | "
        "Mean retrieval (s) | Mean response chars |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for item in report["summary"]:
        lines.append(
            f"| `{item['model']}` | {item['model_type']} | {item['mode']} | "
            f"{item['succeeded']}/{item['scheduled']} | {item['errors']} | "
            f"{item['mean_latency_seconds']:.3f} | "
            f"{item['mean_retrieval_seconds']:.3f} | "
            f"{item['mean_response_chars']:.1f} |"
        )

    lines.extend(["", "## Responses", ""])
    questions = {item["id"]: item for item in report["dataset"]["items"]}
    for result in report["results"]:
        generation = result["generation"]
        question = questions[result["qra_id"]]["question"]
        lines.extend(
            [
                f"### {result['qra_id']} / {result['model']} / "
                f"{result['mode']} / repeat {result.get('repeat', 1)}",
                "",
                f"**Question:** {question}",
                "",
                f"- Status: `{generation['status']}`",
                f"- Latency: `{generation['latency_ms'] / 1000:.3f} s`",
                f"- Retrieval: `{generation['retrieval_latency_ms'] / 1000:.3f} s`",
                f"- Retrieval mode: `{generation['retrieval_mode']}`",
                f"- Citations: `{generation['citation_count']}`",
                "",
            ]
        )
        if generation["status"] == "success":
            lines.extend([generation["response"], ""])
        else:
            lines.extend(
                [
                    f"**Error:** `{generation['error_type']}` {generation['error']}",
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def _build_generation_schedule(
    *,
    models: list[str],
    qra_items: list[dict[str, Any]],
    modes: list[str],
    repeats: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Create a reproducible, model-blocked and condition-interleaved schedule."""

    rng = random.Random(seed)
    schedule: list[dict[str, Any]] = []
    for model in models:
        item_repeats = [
            (item, repeat)
            for item in qra_items
            for repeat in range(1, repeats + 1)
        ]
        rng.shuffle(item_repeats)
        for item, repeat in item_repeats:
            for mode in modes:
                schedule.append(
                    {
                        "model": model,
                        "qra_id": str(item["id"]),
                        "repeat": repeat,
                        "mode": mode,
                        "item": item,
                    }
                )
    for position, entry in enumerate(schedule, start=1):
        entry["sequence"] = position
    return schedule


def _schedule_manifest(schedule: list[dict[str, Any]]) -> dict[str, Any]:
    entries = [
        {
            key: entry[key]
            for key in ("sequence", "model", "qra_id", "repeat", "mode")
        }
        for entry in schedule
    ]
    return {
        "strategy": "model_blocked_shuffled_qra_condition_interleaved",
        "call_count": len(entries),
        "sha256": _canonical_sha256(entries),
        "entries": entries,
    }


def _generation_result_key(item: dict[str, Any]) -> tuple[str, str, int, str]:
    return (
        str(item["model"]),
        str(item["qra_id"]),
        int(item.get("repeat", 1)),
        str(item["mode"]),
    )


def run_formal_development_benchmark(
    *,
    models: list[str],
    qra_items: list[dict[str, Any]],
    output_dir: Path,
    modes: Iterable[str] = ("baseline", "rag"),
    temperature: float = 0.1,
    top_k: int = 5,
    max_tokens: int = DEFAULT_ANSWER_MAX_TOKENS,
    reasoning_effort: str | None = None,
    prompt_path: Path | None = None,
    repeats: int = 1,
    schedule_seed: int = 20260807,
    run_id: str | None = None,
    resume: bool = False,
    skip_model_check: bool = False,
    fail_on_error: bool = False,
    progress_callback: ProgressCallback | None = None,
    baseline_fn: GenerationFunction = generate_text,
    rag_fn: GenerationFunction = rag_answer,
    model_status_fn: Callable[[], dict[str, Any]] = check_status,
    service_status: dict[str, Any] | None = None,
    rag_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate and checkpoint all requested model/QRA/condition combinations."""

    normalized_modes = [value.strip().lower() for value in modes if value.strip()]
    if not normalized_modes or set(normalized_modes) - ALLOWED_MODES:
        raise FormalBenchmarkError(f"Modes must be drawn from {sorted(ALLOWED_MODES)}")
    if len(normalized_modes) != len(set(normalized_modes)):
        raise FormalBenchmarkError("Duplicate modes are not allowed")
    if not 0 <= temperature <= 2:
        raise FormalBenchmarkError("temperature must be between 0 and 2")
    if top_k < 1:
        raise FormalBenchmarkError("top_k must be at least 1")
    if max_tokens < 1:
        raise FormalBenchmarkError("max_tokens must be at least 1")
    if repeats < 1:
        raise FormalBenchmarkError("repeats must be at least 1")
    if reasoning_effort is not None and not reasoning_effort.strip():
        raise FormalBenchmarkError("reasoning_effort must be non-empty when provided")
    if resume and not run_id:
        raise FormalBenchmarkError("resume requires an explicit existing run_id")
    normalized_rag_options = dict(rag_options or {})
    allowed_rag_options = {
        "candidate_k",
        "reranker_model",
        "reranker_revision",
        "max_context_tokens",
        "allow_keyword_fallback",
        "neighbor_expansion_distance",
        "use_confidence_gating",
        "rag_prompt_version",
    }
    unknown_rag_options = set(normalized_rag_options) - allowed_rag_options
    if unknown_rag_options:
        raise FormalBenchmarkError(
            f"Unsupported formal RAG options: {sorted(unknown_rag_options)}"
        )
    candidate_k = normalized_rag_options.get("candidate_k")
    if candidate_k is not None and int(candidate_k) < top_k:
        raise FormalBenchmarkError("candidate_k must be at least top_k")
    reranker_model = str(normalized_rag_options.get("reranker_model", ""))
    reranker_revision = str(normalized_rag_options.get("reranker_revision", ""))
    if bool(reranker_model) != bool(reranker_revision):
        raise FormalBenchmarkError(
            "reranker_model and pinned reranker_revision must be set together"
        )
    neighbor_distance = int(
        normalized_rag_options.get("neighbor_expansion_distance", 0)
    )
    if neighbor_distance < 0:
        raise FormalBenchmarkError("neighbor_expansion_distance cannot be negative")
    if neighbor_distance and not normalized_rag_options.get("max_context_tokens"):
        raise FormalBenchmarkError(
            "Neighbor expansion requires max_context_tokens"
        )

    service = service_status or model_status_fn()
    available_models = list(service.get("generation_models", []))
    normalized_models = (
        [value.strip() for value in models if value.strip()]
        if skip_model_check
        else validate_models(models, available_models)
    )
    if skip_model_check and (
        not normalized_models or len(normalized_models) != len(set(normalized_models))
    ):
        raise FormalBenchmarkError("Model IDs must be non-empty and unique")

    prompt = load_generation_prompt(prompt_path)
    dataset = build_dataset_manifest(qra_items)
    index_snapshot = snapshot_rag_index()
    root = project_root()
    implementation_path = Path(__file__).resolve()
    rag_pipeline_path = root / "backend" / "apps" / "advisors" / "rag_pipeline.py"
    model_profiles_path = root / "backend" / "apps" / "advisors" / "model_profiles.py"
    schedule = _build_generation_schedule(
        models=normalized_models,
        qra_items=qra_items,
        modes=normalized_modes,
        repeats=repeats,
        seed=schedule_seed,
    )
    schedule_manifest = _schedule_manifest(schedule)
    now = datetime.now(UTC)
    run_prefix = "formal_test" if dataset["split"] == "test" else "formal_development"
    run_id = run_id or f"{run_prefix}_{now.strftime('%Y%m%dT%H%M%S%fZ')}"
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise FormalBenchmarkError(
            "run_id must contain only letters, digits, dots, underscores, and hyphens"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"{run_id}.partial.json"
    final_json_path = output_dir / f"{run_id}.json"
    final_md_path = output_dir / f"{run_id}.md"

    sanitized_service = {
        "reachable": bool(service.get("reachable")),
        "provider": str(service.get("provider", "")),
        "configured_model": str(service.get("configured_model", "")),
        "generation_models": available_models,
        "embedding_models": list(service.get("embedding_models", [])),
        "verify_ssl": bool(service.get("verify_ssl", True)),
    }
    new_report: dict[str, Any] = {
        "run_id": run_id,
        "status": "running",
        "evaluation_status": f"{dataset['split']}_unscored_generation_only",
        "started_at": now.isoformat(),
        "completed_at": None,
        "config": {
            "models": [
                {
                    "id": model,
                    "type": classify_generation_model(model),
                    "classification_source": _display_path(
                        model_profiles_path,
                        root,
                    ),
                }
                for model in normalized_models
            ],
            "modes": normalized_modes,
            "temperature": temperature,
            "top_k": top_k,
            "max_tokens": max_tokens,
            "reasoning_effort": reasoning_effort,
            "rag_options": normalized_rag_options,
            "repeats": repeats,
            "schedule_seed": schedule_seed,
            "schedule": schedule_manifest,
            "implementation": {
                "path": _display_path(implementation_path, root),
                "sha256": sha256_file(implementation_path),
            },
            "prompt": prompt["_artifact"],
            "rag_pipeline": {
                "path": _display_path(rag_pipeline_path, root),
                "sha256": sha256_file(rag_pipeline_path),
            },
            "model_profiles": {
                "path": _display_path(model_profiles_path, root),
                "sha256": sha256_file(model_profiles_path),
            },
            "rag_index": index_snapshot,
            "service": sanitized_service,
        },
        "dataset": dataset,
        "summary": [],
        "results": [],
        "prior_attempts": [],
        "resume": {
            "resumed": False,
            "resume_count": 0,
            "preexisting_successful_answers": 0,
        },
    }
    resume_validation = {
        "dataset_manifest_sha256": dataset["manifest_sha256"],
        "schedule_sha256": schedule_manifest["sha256"],
        "models": [item["id"] for item in new_report["config"]["models"]],
        "modes": normalized_modes,
        "temperature": temperature,
        "top_k": top_k,
        "max_tokens": max_tokens,
        "reasoning_effort": reasoning_effort,
        "rag_options": normalized_rag_options,
        "repeats": repeats,
        "schedule_seed": schedule_seed,
        "prompt_sha256": prompt["_artifact"]["sha256"],
        "implementation_sha256": new_report["config"]["implementation"]["sha256"],
        "rag_pipeline_sha256": new_report["config"]["rag_pipeline"]["sha256"],
        "model_profiles_sha256": new_report["config"]["model_profiles"]["sha256"],
        "rag_index_metadata_sha256": index_snapshot.get("metadata_sha256", ""),
        "rag_index_embeddings_sha256": index_snapshot.get("embeddings_sha256", ""),
        "rag_index_source_manifest_sha256": index_snapshot.get(
            "source_manifest_sha256",
            "",
        ),
    }
    new_report["resume_validation_sha256"] = _canonical_sha256(resume_validation)

    if resume:
        if not checkpoint_path.exists():
            raise FormalBenchmarkError(
                f"Resume checkpoint does not exist: {checkpoint_path}"
            )
        report = _load_json(checkpoint_path)
        if (
            report.get("resume_validation_sha256")
            != new_report["resume_validation_sha256"]
        ):
            raise FormalBenchmarkError("Resume configuration or frozen inputs differ")
        previous_results = report.get("results", [])
        if not isinstance(previous_results, list):
            raise FormalBenchmarkError("Resume checkpoint results must be a list")
        successful_results = [
            item
            for item in previous_results
            if item.get("generation", {}).get("status") == "success"
        ]
        prior_attempts = list(report.get("prior_attempts", []))
        prior_attempts.extend(
            item
            for item in previous_results
            if item.get("generation", {}).get("status") != "success"
        )
        previous_resume_count = int(report.get("resume", {}).get("resume_count", 0))
        report["results"] = successful_results
        report["prior_attempts"] = prior_attempts
        report["status"] = "running"
        report["completed_at"] = None
        report["resume"] = {
            "resumed": True,
            "resume_count": previous_resume_count + 1,
            "preexisting_successful_answers": len(successful_results),
        }
        _write_json_atomic(checkpoint_path, report)
    else:
        collisions = [
            path
            for path in (checkpoint_path, final_json_path, final_md_path)
            if path.exists()
        ]
        if collisions:
            raise FormalBenchmarkError(f"Run output already exists: {collisions[0]}")
        report = new_report
        _write_json_atomic(checkpoint_path, report)

    total = len(schedule)
    completed_keys = {
        _generation_result_key(item)
        for item in report["results"]
        if item.get("generation", {}).get("status") == "success"
    }
    if len(completed_keys) != len(report["results"]):
        raise FormalBenchmarkError("Resume checkpoint contains duplicate successes")

    for entry in schedule:
        item = entry["item"]
        model = str(entry["model"])
        mode = str(entry["mode"])
        repeat = int(entry["repeat"])
        position = int(entry["sequence"])
        result_key = (model, str(item["id"]), repeat, mode)
        if result_key in completed_keys:
            continue
        start_event = {
            "type": "start",
            "position": position,
            "total": total,
            "model": model,
            "model_type": classify_generation_model(model),
            "qra_id": item["id"],
            "mode": mode,
            "repeat": repeat,
        }
        if progress_callback:
            progress_callback(start_event)
        generation = _safe_generation_call(
            mode=mode,
            question=str(item["question"]),
            domain=str(item["domain"]),
            model=model,
            system_prompt=str(prompt["system_prompt"]),
            baseline_prompt_template=str(prompt["baseline_prompt_template"]),
            temperature=temperature,
            max_tokens=max_tokens,
            top_k=top_k,
            baseline_fn=baseline_fn,
            rag_fn=rag_fn,
            reasoning_effort=reasoning_effort,
            rag_options=normalized_rag_options,
        )
        result = {
            "sequence": position,
            "qra_id": item["id"],
            "domain": item["domain"],
            "question_type": item["question_type"],
            "model": model,
            "model_type": classify_generation_model(model),
            "mode": mode,
            "repeat": repeat,
            "generation": generation,
        }
        report["results"].append(result)
        report["results"].sort(key=lambda value: int(value["sequence"]))
        report["summary"] = aggregate_generation_results(report["results"])
        _write_json_atomic(checkpoint_path, report)
        if progress_callback:
            progress_callback(
                {
                    **start_event,
                    "type": "complete",
                    "status": generation["status"],
                    "latency_ms": generation["latency_ms"],
                    "error": generation["error"],
                }
            )
        if fail_on_error and generation["status"] == "error":
            report["status"] = "failed"
            report["completed_at"] = datetime.now(UTC).isoformat()
            _write_json_atomic(checkpoint_path, report)
            raise FormalBenchmarkError(
                f"Generation failed for {model}/{item['id']}/repeat-{repeat}/{mode}: "
                f"{generation['error']}"
            )

    report["status"] = (
        "completed_with_errors"
        if any(item["generation"]["status"] == "error" for item in report["results"])
        else "completed"
    )
    report["completed_at"] = datetime.now(UTC).isoformat()
    report["summary"] = aggregate_generation_results(report["results"])
    _write_json_atomic(final_json_path, report)
    final_md_path.write_text(build_markdown_report(report), encoding="utf-8")
    if report["status"] == "completed":
        checkpoint_path.unlink(missing_ok=True)
    return {
        "report": report,
        "json_path": str(final_json_path),
        "markdown_path": str(final_md_path),
    }
