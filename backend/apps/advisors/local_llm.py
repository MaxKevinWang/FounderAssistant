"""Thin adapter around local or institutional LLM HTTP APIs.

Responsibilities:
1) Read local model config from environment.
2) Provide health/status checks (`check_status`).
3) Provide synchronous text generation (`generate_text`).
4) Provide streaming text generation (`stream_generate_text`) for chat UIs.

This module is intentionally lightweight and stateless so that it can be
reused by both regular chat endpoints and benchmark tooling.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Iterator

import requests

EMBEDDING_FAMILIES = {"bert", "nomic-bert", "clip"}
EMBEDDING_NAME_MARKERS = ("embed", "embedding", "bge", "e5", "gte")


@dataclass(frozen=True)
class LocalLLMConfig:
    """Runtime configuration for local LLM access."""

    provider: str
    base_url: str
    model: str
    timeout_seconds: int
    api_key: str
    verify_ssl: bool


DEFAULT_CONFIG = LocalLLMConfig(
    # `ollama` keeps the original local setup. `openwebui` uses Open WebUI's
    # OpenAI-compatible `/api/chat/completions` endpoint.
    provider=os.getenv("LOCAL_LLM_PROVIDER", "ollama").strip().lower(),
    base_url=os.getenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:11434").rstrip("/"),
    # Demo default model; can be overridden via env.
    model=os.getenv("LOCAL_LLM_MODEL", "qwen2.5:7b"),
    timeout_seconds=int(os.getenv("LOCAL_LLM_TIMEOUT_SECONDS", "360")),
    api_key=os.getenv("LOCAL_LLM_API_KEY", ""),
    verify_ssl=os.getenv("LOCAL_LLM_VERIFY_SSL", "true").strip().lower()
    not in {"0", "false", "no", "off"},
)


class LocalLLMError(Exception):
    """Raised when local LLM service is unavailable or returns invalid data."""

    pass


def _resolve_config(
    config: LocalLLMConfig,
    model_override: str | None = None,
    timeout_override: int | None = None,
) -> LocalLLMConfig:
    """Return runtime config with optional per-request overrides.

    This keeps the default `.env`-based configuration intact while allowing
    endpoints to switch generation model at request time for A/B comparisons.
    """

    return LocalLLMConfig(
        provider=config.provider,
        base_url=config.base_url,
        model=(model_override or config.model),
        timeout_seconds=(
            timeout_override if timeout_override is not None else config.timeout_seconds
        ),
        api_key=config.api_key,
        verify_ssl=config.verify_ssl,
    )


def _normalize_openwebui_base_url(base_url: str) -> str:
    """Return the root Open WebUI URL even if `/ollama` was configured."""

    normalized = base_url.rstrip("/")
    if normalized.endswith("/ollama"):
        return normalized[: -len("/ollama")]
    return normalized


def _request_headers(config: LocalLLMConfig) -> dict[str, str]:
    """Build common JSON headers without exposing secrets in logs or responses."""

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"
    return headers


def _post_json(
    url: str, payload: dict[str, Any], config: LocalLLMConfig
) -> dict[str, Any]:
    """HTTP POST helper with JSON payload and JSON response."""

    try:
        response = requests.post(
            url,
            json=payload,
            headers=_request_headers(config),
            timeout=config.timeout_seconds,
            verify=config.verify_ssl,
        )
        response.raise_for_status()
        return response.json()
    except requests.Timeout as exc:
        raise LocalLLMError(
            f"LLM service timed out after {config.timeout_seconds} seconds."
        ) from exc
    except requests.HTTPError as exc:
        response = exc.response
        code = response.status_code if response is not None else "unknown"
        body = response.text[:600] if response is not None else str(exc)
        raise LocalLLMError(f"LLM service returned HTTP {code}: {body}") from exc
    except requests.RequestException as exc:
        raise LocalLLMError(f"Failed to connect local LLM service: {exc}") from exc
    except (requests.JSONDecodeError, ValueError) as exc:
        raise LocalLLMError(f"LLM service returned invalid JSON: {exc}") from exc


def _post_stream_json(
    url: str,
    payload: dict[str, Any],
    config: LocalLLMConfig,
    accept: str = "application/json",
) -> Iterator[str]:
    """HTTP POST helper yielding raw UTF-8 lines from a streaming response."""

    headers = _request_headers(config)
    headers["Accept"] = accept
    try:
        with requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=config.timeout_seconds,
            verify=config.verify_ssl,
            stream=True,
        ) as response:
            response.raise_for_status()
            # Open WebUI commonly serves SSE as `text/event-stream` without a
            # charset. Requests then assumes ISO-8859-1, which corrupts UTF-8
            # German text (for example, `ä` becomes `Ã¤`). SSE data is UTF-8.
            response.encoding = "utf-8"
            for raw_line in response.iter_lines(decode_unicode=True):
                line = (
                    raw_line.decode("utf-8", errors="replace")
                    if isinstance(raw_line, bytes)
                    else raw_line
                ).strip()
                if line:
                    yield line
    except requests.Timeout as exc:
        raise LocalLLMError(
            f"LLM service timed out after {config.timeout_seconds} seconds."
        ) from exc
    except requests.HTTPError as exc:
        response = exc.response
        code = response.status_code if response is not None else "unknown"
        body = response.text[:600] if response is not None else str(exc)
        raise LocalLLMError(f"LLM service returned HTTP {code}: {body}") from exc
    except requests.RequestException as exc:
        raise LocalLLMError(f"Failed to connect local LLM service: {exc}") from exc


def _get_json(url: str, config: LocalLLMConfig) -> dict[str, Any]:
    """HTTP GET helper returning JSON payload."""

    try:
        response = requests.get(
            url,
            headers=_request_headers(config),
            timeout=config.timeout_seconds,
            verify=config.verify_ssl,
        )
        response.raise_for_status()
        return response.json()
    except requests.Timeout as exc:
        raise LocalLLMError(
            f"LLM service timed out after {config.timeout_seconds} seconds."
        ) from exc
    except requests.HTTPError as exc:
        response = exc.response
        code = response.status_code if response is not None else "unknown"
        body = response.text[:600] if response is not None else str(exc)
        raise LocalLLMError(f"LLM service returned HTTP {code}: {body}") from exc
    except requests.RequestException as exc:
        raise LocalLLMError(f"Failed to connect local LLM service: {exc}") from exc
    except (requests.JSONDecodeError, ValueError) as exc:
        raise LocalLLMError(f"LLM service returned invalid JSON: {exc}") from exc


def check_status(config: LocalLLMConfig = DEFAULT_CONFIG) -> dict[str, Any]:
    """Return current local-model status and installation info.

    Output is intentionally UI-friendly so frontend can show:
    - service reachability,
    - configured model,
    - installed model list.
    """

    provider = config.provider.strip().lower()
    if provider == "ollama":
        payload = _get_json(f"{config.base_url}/api/tags", config)
        models = payload.get("models", [])
        model_names = [
            str(item.get("name", "")).strip() for item in models if item.get("name")
        ]
    elif provider == "openwebui":
        base_url = _normalize_openwebui_base_url(config.base_url)
        payload = _get_json(f"{base_url}/api/models", config)
        models = payload.get("data", [])
        model_names = [
            str(item.get("id", "")).strip() for item in models if item.get("id")
        ]
    else:
        raise LocalLLMError(f"Unsupported LOCAL_LLM_PROVIDER: {config.provider}")

    installed = {name for name in model_names if name}
    generation_models: list[str] = []
    embedding_models: list[str] = []

    for item, name in zip(models, model_names, strict=False):
        if not name:
            continue
        details = item.get("details", {}) or {}
        family = str(details.get("family", "")).strip().lower()
        lower_name = name.lower()
        is_embedding = family in EMBEDDING_FAMILIES or any(
            marker in lower_name for marker in EMBEDDING_NAME_MARKERS
        )
        if is_embedding:
            embedding_models.append(name)
        else:
            generation_models.append(name)

    return {
        "reachable": True,
        "provider": provider,
        "base_url": config.base_url,
        "configured_model": config.model,
        "installed": config.model in installed,
        "available_models": sorted(installed),
        "generation_models": sorted(set(generation_models)),
        "embedding_models": sorted(set(embedding_models)),
        "verify_ssl": config.verify_ssl,
    }


def generate_text(
    prompt: str,
    system_prompt: str = "",
    temperature: float = 0.2,
    max_tokens: int = 512,
    model: str | None = None,
    reasoning_effort: str | None = None,
    seed: int | None = None,
    config: LocalLLMConfig = DEFAULT_CONFIG,
    timeout_seconds: int | None = None,
) -> dict[str, Any]:
    """Run one synchronous generation call via the configured LLM provider.

    Returns normalized fields used by frontend and benchmark pipelines:
    - `response`: text output
    - `latency_ms`: wall-clock latency measured client-side
    - `done`: provider completion flag
    """

    runtime_config = _resolve_config(
        config=config,
        model_override=model,
        timeout_override=timeout_seconds,
    )
    provider = runtime_config.provider.strip().lower()
    started = time.perf_counter()

    if provider == "ollama":
        payload = {
            "model": runtime_config.model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if seed is not None:
            payload["options"]["seed"] = seed
        result = _post_json(
            f"{runtime_config.base_url}/api/generate", payload, runtime_config
        )
        raw_response = result.get("response", "")
        response_text = raw_response if isinstance(raw_response, str) else ""
        response_model = result.get("model", runtime_config.model)
        done = bool(result.get("done", False))
        finish_reason = str(result.get("done_reason", "") or "")
        usage = {
            key: result[key]
            for key in (
                "prompt_eval_count",
                "eval_count",
                "total_duration",
                "load_duration",
                "prompt_eval_duration",
                "eval_duration",
            )
            if key in result
        }
        reasoning_char_count = 0
    elif provider == "openwebui":
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": runtime_config.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if reasoning_effort is not None:
            payload["reasoning_effort"] = reasoning_effort
        if seed is not None:
            payload["seed"] = seed
        base_url = _normalize_openwebui_base_url(runtime_config.base_url)
        result = _post_json(f"{base_url}/api/chat/completions", payload, runtime_config)
        choices = result.get("choices", [])
        choice = choices[0] if choices else {}
        message = choice.get("message", {}) or {}
        raw_response = message.get("content", "")
        response_text = raw_response if isinstance(raw_response, str) else ""
        response_model = result.get("model", runtime_config.model)
        done = bool(choices) or bool(result.get("done", False))
        finish_reason = str(choice.get("finish_reason", "") or "")
        raw_usage = result.get("usage", {})
        usage = raw_usage if isinstance(raw_usage, dict) else {}
        raw_reasoning = (
            message.get("reasoning_content") or message.get("reasoning") or ""
        )
        reasoning_char_count = (
            len(raw_reasoning) if isinstance(raw_reasoning, str) else 0
        )
    else:
        raise LocalLLMError(
            f"Unsupported LOCAL_LLM_PROVIDER: {runtime_config.provider}"
        )

    elapsed_ms = int((time.perf_counter() - started) * 1000)

    return {
        "provider": provider,
        "model": response_model,
        "response": response_text,
        "done": done,
        "latency_ms": elapsed_ms,
        "finish_reason": finish_reason,
        "usage": usage,
        "reasoning_content_present": reasoning_char_count > 0,
        "reasoning_content_char_count": reasoning_char_count,
    }


def stream_generate_text(
    prompt: str,
    system_prompt: str = "",
    temperature: float = 0.2,
    max_tokens: int = 512,
    model: str | None = None,
    config: LocalLLMConfig = DEFAULT_CONFIG,
) -> Iterator[dict[str, Any]]:
    """Stream generation tokens from the configured provider.

    The yielded event dictionaries are provider-neutral:
    - `{"type": "token", "content": "..."}`
    - `{"type": "done", "model": "...", "latency_ms": 123, ...}`
    """

    runtime_config = _resolve_config(config=config, model_override=model)
    provider = runtime_config.provider.strip().lower()
    started = time.perf_counter()
    response_model = runtime_config.model
    done = False
    finish_reason = ""

    if provider == "ollama":
        payload = {
            "model": runtime_config.model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        for line in _post_stream_json(
            f"{runtime_config.base_url}/api/generate", payload, runtime_config
        ):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            response_model = event.get("model", response_model)
            token = event.get("response", "")
            if token:
                yield {"type": "token", "content": token}
            done = bool(event.get("done", False))
            if done:
                finish_reason = str(event.get("done_reason", "") or "stop")
                break
    elif provider == "openwebui":
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": runtime_config.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        base_url = _normalize_openwebui_base_url(runtime_config.base_url)

        for line in _post_stream_json(
            f"{base_url}/api/chat/completions",
            payload,
            runtime_config,
            accept="text/event-stream",
        ):
            if line.startswith("data:"):
                line = line.removeprefix("data:").strip()
            if line == "[DONE]":
                done = True
                break

            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            response_model = event.get("model", response_model)
            choices = event.get("choices", [])
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta", {}) or {}
            message = choice.get("message", {}) or {}
            token = delta.get("content") or message.get("content") or ""
            if token:
                yield {"type": "token", "content": token}
            choice_finish_reason = str(choice.get("finish_reason", "") or "")
            if choice_finish_reason:
                finish_reason = choice_finish_reason
                done = True
    else:
        raise LocalLLMError(
            f"Unsupported LOCAL_LLM_PROVIDER: {runtime_config.provider}"
        )

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    yield {
        "type": "done",
        "provider": provider,
        "model": response_model,
        "done": done,
        "finish_reason": finish_reason,
        "latency_ms": elapsed_ms,
    }
