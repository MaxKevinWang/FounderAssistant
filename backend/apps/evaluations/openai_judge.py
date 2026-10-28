"""Blind, auditable Responses API Judge for completed development runs."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import statistics
import time
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

from django.conf import settings
from pydantic import BaseModel, ConfigDict, Field, create_model


class OpenAIJudgeError(RuntimeError):
    """Raised when Judge inputs, configuration, or outputs are invalid."""


class KeypointEvaluationValue(BaseModel):
    """Structured decision value stored under a required keypoint-ID field."""

    model_config = ConfigDict(extra="forbid")

    label: Literal["covered", "contradicted", "neither"]
    rationale: str = Field(description="Concise evidence-based explanation")
    answer_evidence: str = Field(
        description=(
            "Required short exact candidate-answer excerpt for covered or "
            "contradicted; optional exact partial-match excerpt for neither"
        )
    )


class KeypointEvaluationExcerptListValue(BaseModel):
    """Structured decision value with independently auditable excerpts."""

    model_config = ConfigDict(extra="forbid")

    label: Literal["covered", "contradicted", "neither"]
    rationale: str = Field(description="Concise evidence-based explanation")
    answer_evidence: list[str] = Field(
        description=(
            "Exact candidate-answer excerpts; one or more for covered or "
            "contradicted, zero or more partial-match excerpts for neither"
        )
    )


class KeypointEvaluationSegmentValue(BaseModel):
    """Structured decision value referencing immutable candidate segments."""

    model_config = ConfigDict(extra="forbid")

    label: Literal["covered", "contradicted", "neither"]
    rationale: str = Field(description="Concise evidence-based explanation")
    answer_evidence_segment_ids: list[str] = Field(
        description=(
            "Candidate segment IDs supporting the decision; one or more for "
            "covered or contradicted, zero or more for neither"
        )
    )


class UnsupportedClaim(BaseModel):
    """A material factual claim not supported by supplied ground truth."""

    model_config = ConfigDict(extra="forbid")

    claim: str
    reason: str
    severity: Literal["low", "medium", "high"]


JudgeFunction = Callable[[dict[str, Any]], dict[str, Any]]
ProgressCallback = Callable[[dict[str, Any]], None]

ALLOWED_MODES = {"baseline", "rag"}
ALLOWED_JUDGE_PROVIDERS = {"deepseek", "openai"}
ALLOWED_REASONING_EFFORTS = {"none", "low", "medium", "high", "xhigh"}
ALLOWED_UNSUPPORTED_CLAIMS_EVALUATIONS = {"model_evaluated", "not_evaluated"}
ALLOWED_ANSWER_EVIDENCE_FORMATS = {
    "exact_excerpt_list",
    "legacy_string",
    "segment_id_list",
}
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$")


def project_root() -> Path:
    """Return the repository root."""

    return Path(settings.BASE_DIR).parent


def default_prompt_path() -> Path:
    """Return the versioned keypoint-Judge prompt."""

    return (
        project_root()
        / "docs"
        / "evaluation"
        / "judges"
        / "prompts"
        / "keypoint_judge_prompt_v0.5.json"
    )


def default_output_root(source_run_path: Path) -> Path:
    """Keep API Judge runs next to the generation run they evaluate."""

    return source_run_path.resolve().parent / "judge" / "openai_api"


def sha256_file(path: Path) -> str:
    """Hash an artifact without reading it all into memory."""

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


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise OpenAIJudgeError(f"Required JSON artifact not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise OpenAIJudgeError(f"Invalid JSON artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise OpenAIJudgeError(f"Expected a JSON object in {path}")
    return payload


def _display_path(path: Path, root: Path | None = None) -> str:
    root = (root or project_root()).resolve()
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return str(path.resolve())


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_judge_prompt(path: Path | None = None) -> dict[str, Any]:
    """Load and verify the versioned Judge prompt artifact."""

    path = (path or default_prompt_path()).resolve()
    prompt = _load_json_object(path)
    required = {
        "prompt_id",
        "version",
        "status",
        "purpose",
        "system_prompt",
        "user_prompt_template",
    }
    missing = sorted(required - prompt.keys())
    if missing:
        raise OpenAIJudgeError(f"Judge prompt is missing fields: {missing}")
    template = str(prompt["user_prompt_template"])
    if template.count("{evaluation_item_json}") != 1:
        raise OpenAIJudgeError(
            "Judge user prompt must contain exactly one "
            "{evaluation_item_json} placeholder"
        )
    input_policy = str(prompt.get("judge_input_policy", "legacy_full_reference"))
    unsupported_evaluation = str(
        prompt.get("unsupported_claims_evaluation", "model_evaluated")
    )
    evidence_format = str(prompt.get("answer_evidence_format", "legacy_string"))
    if unsupported_evaluation not in ALLOWED_UNSUPPORTED_CLAIMS_EVALUATIONS:
        raise OpenAIJudgeError(
            "unsupported_claims_evaluation must be one of "
            f"{sorted(ALLOWED_UNSUPPORTED_CLAIMS_EVALUATIONS)}"
        )
    if evidence_format not in ALLOWED_ANSWER_EVIDENCE_FORMATS:
        raise OpenAIJudgeError(
            "answer_evidence_format must be one of "
            f"{sorted(ALLOWED_ANSWER_EVIDENCE_FORMATS)}"
        )
    if input_policy in {
        "keypoint_only_reference_isolated",
        "keypoint_segmented_reference_isolated",
    }:
        expected_fields = [
            "qra_id",
            "language",
            "question",
            "answerable",
            "keypoints",
            "candidate_answer",
        ]
        if prompt.get("judge_input_fields") != expected_fields:
            raise OpenAIJudgeError(
                "Reference-isolated prompts must declare the exact "
                f"judge_input_fields sequence: {expected_fields}"
            )
        if unsupported_evaluation != "not_evaluated":
            raise OpenAIJudgeError(
                "Reference-isolated prompts cannot evaluate unsupported claims"
            )
    if (
        input_policy == "keypoint_segmented_reference_isolated"
        and evidence_format != "segment_id_list"
    ):
        raise OpenAIJudgeError(
            "Segmented reference-isolated prompts require segment_id_list evidence"
        )
    return {
        **prompt,
        "_artifact": {
            "path": _display_path(path),
            "sha256": sha256_file(path),
        },
    }


def load_generation_run(path: Path) -> dict[str, Any]:
    """Load a completed, judge-ready generation artifact."""

    path = path.resolve()
    payload = _load_json_object(path)
    if payload.get("status") not in {"completed", "completed_with_errors"}:
        raise OpenAIJudgeError(
            "Source generation run must be completed or completed_with_errors"
        )
    dataset = payload.get("dataset")
    results = payload.get("results")
    if not isinstance(dataset, dict) or not isinstance(dataset.get("items"), list):
        raise OpenAIJudgeError("Source run has no valid dataset.items list")
    if not isinstance(results, list):
        raise OpenAIJudgeError("Source run has no valid results list")

    qra_ids: list[str] = []
    for item in dataset["items"]:
        if not isinstance(item, dict):
            raise OpenAIJudgeError("Source run contains a non-object QRA item")
        required = {
            "id",
            "domain",
            "question_type",
            "language",
            "question",
            "answerable",
            "reference_answer",
            "keypoints",
            "references",
        }
        missing = sorted(required - item.keys())
        if missing:
            raise OpenAIJudgeError(f"QRA item is missing Judge fields: {missing}")
        if not 3 <= len(item["keypoints"]) <= 5:
            raise OpenAIJudgeError(
                f"{item['id']} must contain between 3 and 5 keypoints"
            )
        qra_ids.append(str(item["id"]))
    if len(qra_ids) != len(set(qra_ids)):
        raise OpenAIJudgeError("Source run contains duplicate QRA IDs")

    return {
        **payload,
        "_artifact": {
            "path": _display_path(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        },
    }


def _normalize_filter(values: Iterable[str] | None) -> set[str]:
    return {str(value).strip() for value in values or [] if str(value).strip()}


def prepare_blinded_candidates(
    source_run: dict[str, Any],
    *,
    tested_models: Iterable[str] | None = None,
    qra_ids: Iterable[str] | None = None,
    modes: Iterable[str] = ("baseline", "rag"),
    seed: int = 20260728,
) -> dict[str, Any]:
    """Filter successful answers, shuffle them, and remove condition labels."""

    model_filter = _normalize_filter(tested_models)
    qra_filter = _normalize_filter(qra_ids)
    mode_filter = {value.strip().lower() for value in modes if value.strip()}
    if not mode_filter or mode_filter - ALLOWED_MODES:
        raise OpenAIJudgeError(f"Modes must be drawn from {sorted(ALLOWED_MODES)}")

    results = source_run["results"]
    available_models = {str(item.get("model", "")) for item in results}
    available_qras = {str(item.get("qra_id", "")) for item in results}
    missing_models = sorted(model_filter - available_models)
    missing_qras = sorted(qra_filter - available_qras)
    if missing_models:
        raise OpenAIJudgeError(
            f"Requested tested models are absent from source run: {missing_models}"
        )
    if missing_qras:
        raise OpenAIJudgeError(
            f"Requested QRA IDs are absent from source run: {missing_qras}"
        )

    qra_by_id = {str(item["id"]): item for item in source_run["dataset"]["items"]}
    selected: list[dict[str, Any]] = []
    skipped_errors = 0
    seen_conditions: set[tuple[str, str, int, str]] = set()
    for result in results:
        model = str(result.get("model", ""))
        qra_id = str(result.get("qra_id", ""))
        mode = str(result.get("mode", "")).lower()
        repeat = int(result.get("repeat", 1))
        if repeat < 1:
            raise OpenAIJudgeError(
                f"Source repeat must be at least 1: {model}/{qra_id}/{repeat}/{mode}"
            )
        if model_filter and model not in model_filter:
            continue
        if qra_filter and qra_id not in qra_filter:
            continue
        if mode not in mode_filter:
            continue
        condition = (model, qra_id, repeat, mode)
        if condition in seen_conditions:
            raise OpenAIJudgeError(
                "Duplicate source condition found: "
                f"{model}/{qra_id}/repeat-{repeat}/{mode}"
            )
        seen_conditions.add(condition)
        generation = result.get("generation", {})
        if generation.get("status") != "success":
            skipped_errors += 1
            continue
        qra = qra_by_id.get(qra_id)
        if qra is None:
            raise OpenAIJudgeError(f"Result references unknown QRA ID: {qra_id}")
        answer = str(generation.get("response", "")).strip()
        if not answer:
            raise OpenAIJudgeError(
                f"Successful generation has an empty answer: {condition}"
            )
        selected.append(
            {
                "source_sequence": int(result.get("sequence", 0)),
                "qra_id": qra_id,
                "domain": str(result.get("domain", qra["domain"])),
                "question_type": str(result.get("question_type", qra["question_type"])),
                "tested_model": model,
                "tested_model_type": str(result.get("model_type", "")),
                "mode": mode,
                "repeat": repeat,
                "judge_input": {
                    "qra_id": qra_id,
                    "language": qra["language"],
                    "question": qra["question"],
                    "answerable": qra["answerable"],
                    "reference_answer": qra["reference_answer"],
                    "keypoints": [
                        {
                            "id": str(keypoint["id"]),
                            "text": str(keypoint["text"]),
                        }
                        for keypoint in qra["keypoints"]
                    ],
                    "reference_evidence": [
                        {
                            "reference_id": str(reference["reference_id"]),
                            "quote": str(reference["quote"]),
                        }
                        for reference in qra["references"]
                    ],
                    "candidate_answer": answer,
                },
            }
        )
    if not selected:
        raise OpenAIJudgeError("No successful source answers matched the selection")

    random.Random(seed).shuffle(selected)
    for position, candidate in enumerate(selected, start=1):
        candidate["candidate_id"] = f"C{position:04d}"

    fingerprint_rows = [
        {
            "candidate_id": item["candidate_id"],
            "source_sequence": item["source_sequence"],
            "repeat": item["repeat"],
            "answer_sha256": _canonical_sha256(item["judge_input"]["candidate_answer"]),
        }
        for item in selected
    ]
    return {
        "candidates": selected,
        "seed": seed,
        "candidate_count": len(selected),
        "skipped_generation_errors": skipped_errors,
        "selection_sha256": _canonical_sha256(fingerprint_rows),
        "filters": {
            "tested_models": sorted(model_filter),
            "qra_ids": sorted(qra_filter),
            "modes": sorted(mode_filter),
        },
    }


def _value(source: Any, name: str, default: Any = None) -> Any:
    if source is None:
        return default
    if isinstance(source, dict):
        return source.get(name, default)
    return getattr(source, name, default)


def _extract_usage(response: Any) -> dict[str, int]:
    usage = _value(response, "usage")
    input_details = _value(usage, "input_tokens_details")
    output_details = _value(usage, "output_tokens_details")
    return {
        "input_tokens": int(_value(usage, "input_tokens", 0) or 0),
        "cached_input_tokens": int(_value(input_details, "cached_tokens", 0) or 0),
        "output_tokens": int(_value(usage, "output_tokens", 0) or 0),
        "reasoning_tokens": int(_value(output_details, "reasoning_tokens", 0) or 0),
        "total_tokens": int(_value(usage, "total_tokens", 0) or 0),
    }


def _segment_candidate_answer(candidate_answer: str) -> list[dict[str, str]]:
    """Expose immutable non-empty answer lines through stable local IDs."""

    texts = [line.strip() for line in candidate_answer.splitlines() if line.strip()]
    if not texts:
        raise OpenAIJudgeError("Candidate answer produced no non-empty segments")
    return [
        {"id": f"A{position:04d}", "text": value}
        for position, value in enumerate(texts, start=1)
    ]


def build_judge_decision_model(
    keypoint_ids: Iterable[str],
    *,
    answer_evidence_format: str = "legacy_string",
    candidate_segment_ids: Iterable[str] | None = None,
) -> type[BaseModel]:
    """Build a schema whose required fields are the current keypoint IDs."""

    ids = [str(value) for value in keypoint_ids]
    if not ids or len(ids) != len(set(ids)):
        raise OpenAIJudgeError("Keypoint IDs must be non-empty and unique")
    invalid = [
        value for value in ids if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", value)
    ]
    if invalid:
        raise OpenAIJudgeError(
            f"Keypoint IDs cannot become strict schema fields: {invalid}"
        )
    if answer_evidence_format not in ALLOWED_ANSWER_EVIDENCE_FORMATS:
        raise OpenAIJudgeError(
            "answer_evidence_format must be one of "
            f"{sorted(ALLOWED_ANSWER_EVIDENCE_FORMATS)}"
        )
    if answer_evidence_format == "exact_excerpt_list":
        evaluation_model = KeypointEvaluationExcerptListValue
    elif answer_evidence_format == "segment_id_list":
        segment_ids = [str(value) for value in candidate_segment_ids or []]
        if not segment_ids or len(segment_ids) != len(set(segment_ids)):
            raise OpenAIJudgeError("Candidate segment IDs must be non-empty and unique")
        allowed_segment_id = Literal.__getitem__(tuple(segment_ids))
        evaluation_model = create_model(
            "CandidateSegmentKeypointEvaluation",
            __config__=ConfigDict(extra="forbid"),
            label=(Literal["covered", "contradicted", "neither"], ...),
            rationale=(
                str,
                Field(description="Concise evidence-based explanation"),
            ),
            answer_evidence_segment_ids=(
                list[allowed_segment_id],
                Field(
                    description=(
                        "Only supplied candidate segment IDs; one or more for "
                        "covered or contradicted, zero or more for neither"
                    )
                ),
            ),
        )
    else:
        evaluation_model = KeypointEvaluationValue
    keypoint_map = create_model(
        "RequiredKeypointEvaluations",
        __config__=ConfigDict(extra="forbid"),
        **{
            keypoint_id: (
                evaluation_model,
                Field(description=f"Evaluation of supplied {keypoint_id}"),
            )
            for keypoint_id in ids
        },
    )
    return create_model(
        "DynamicJudgeDecision",
        __config__=ConfigDict(extra="forbid"),
        keypoint_evaluations=(
            keypoint_map,
            Field(description="One required property per supplied keypoint ID"),
        ),
        unsupported_claims=(list[UnsupportedClaim], ...),
        overall_assessment=(str, ...),
    )


class OpenAIKeypointJudge:
    """Configured callable around OpenAI-compatible Responses Structured Outputs."""

    def __init__(
        self,
        *,
        prompt: dict[str, Any],
        model: str,
        reasoning_effort: str = "low",
        max_output_tokens: int = 2500,
        timeout_seconds: int = 120,
        max_retries: int = 2,
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any | None = None,
    ) -> None:
        if reasoning_effort not in ALLOWED_REASONING_EFFORTS:
            raise OpenAIJudgeError(
                f"reasoning_effort must be one of {sorted(ALLOWED_REASONING_EFFORTS)}"
            )
        if max_output_tokens < 256:
            raise OpenAIJudgeError("max_output_tokens must be at least 256")
        if timeout_seconds < 1:
            raise OpenAIJudgeError("timeout_seconds must be positive")
        if max_retries < 0:
            raise OpenAIJudgeError("max_retries cannot be negative")
        if client is None:
            key = api_key or os.getenv("OPENAI_API_KEY", "")
            if not key:
                raise OpenAIJudgeError(
                    "OPENAI_API_KEY is missing from the process environment"
                )
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise OpenAIJudgeError(
                    "OpenAI SDK is not installed; install backend requirements"
                ) from exc
            client_kwargs: dict[str, Any] = {
                "api_key": key,
                "timeout": timeout_seconds,
                "max_retries": max_retries,
            }
            if base_url:
                client_kwargs["base_url"] = base_url.rstrip("/")
            client = OpenAI(
                **client_kwargs,
            )
        self.client = client
        self.prompt = prompt
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens

    def __call__(self, judge_input: dict[str, Any]) -> dict[str, Any]:
        visible_fields = self.prompt.get("judge_input_fields")
        if visible_fields is None:
            model_input = judge_input
        else:
            missing = [field for field in visible_fields if field not in judge_input]
            if missing:
                raise OpenAIJudgeError(
                    f"Judge input is missing model-visible fields: {missing}"
                )
            model_input = {field: judge_input[field] for field in visible_fields}
        evidence_format = str(
            self.prompt.get("answer_evidence_format", "legacy_string")
        )
        candidate_segments: list[dict[str, str]] = []
        if evidence_format == "segment_id_list":
            candidate_segments = _segment_candidate_answer(
                str(judge_input["candidate_answer"])
            )
            model_input = {
                key: value
                for key, value in model_input.items()
                if key != "candidate_answer"
            }
            model_input["candidate_answer_segments"] = candidate_segments
        item_json = json.dumps(model_input, ensure_ascii=False, indent=2)
        user_prompt = str(self.prompt["user_prompt_template"]).format(
            evaluation_item_json=item_json
        )
        decision_model = build_judge_decision_model(
            (item["id"] for item in judge_input["keypoints"]),
            answer_evidence_format=evidence_format,
            candidate_segment_ids=(segment["id"] for segment in candidate_segments),
        )
        response = self.client.responses.parse(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": str(self.prompt["system_prompt"]),
                },
                {"role": "user", "content": user_prompt},
            ],
            text_format=decision_model,
            reasoning={"effort": self.reasoning_effort},
            max_output_tokens=self.max_output_tokens,
            store=False,
        )
        parsed = response.output_parsed
        if parsed is None:
            refusal = ""
            for output in _value(response, "output", []) or []:
                for content in _value(output, "content", []) or []:
                    refusal = str(_value(content, "refusal", "") or refusal)
            detail = f": {refusal}" if refusal else ""
            raise OpenAIJudgeError(
                f"Judge returned no parsed Structured Output{detail}"
            )
        decision = parsed.model_dump(mode="json")
        if evidence_format == "segment_id_list":
            segment_by_id = {
                segment["id"]: segment["text"] for segment in candidate_segments
            }
            for keypoint_id, evaluation in decision["keypoint_evaluations"].items():
                segment_ids = evaluation["answer_evidence_segment_ids"]
                invalid_ids = [
                    segment_id
                    for segment_id in segment_ids
                    if segment_id not in segment_by_id
                ]
                if invalid_ids:
                    raise OpenAIJudgeError(
                        f"{keypoint_id} returned unknown candidate segment IDs: "
                        f"{invalid_ids}"
                    )
                evaluation["answer_evidence"] = [
                    segment_by_id[segment_id] for segment_id in segment_ids
                ]
        return {
            "response_id": str(_value(response, "id", "")),
            "response_model": str(_value(response, "model", self.model)),
            "raw_output_text": str(_value(response, "output_text", "")),
            "decision": decision,
            "usage": _extract_usage(response),
            "unsupported_claims_evaluation": str(
                self.prompt.get(
                    "unsupported_claims_evaluation",
                    "model_evaluated",
                )
            ),
        }


def _validate_and_score_decision(
    decision: dict[str, Any],
    judge_input: dict[str, Any],
    *,
    unsupported_claims_evaluation: str = "model_evaluated",
) -> dict[str, Any]:
    if unsupported_claims_evaluation not in ALLOWED_UNSUPPORTED_CLAIMS_EVALUATIONS:
        raise OpenAIJudgeError(
            "Invalid unsupported-claims evaluation status: "
            f"{unsupported_claims_evaluation}"
        )
    expected = [str(item["id"]) for item in judge_input["keypoints"]]
    evaluations = decision.get("keypoint_evaluations")
    by_id: dict[str, dict[str, Any]] = {}
    if isinstance(evaluations, dict):
        by_id = {
            str(keypoint_id): dict(evaluation)
            for keypoint_id, evaluation in evaluations.items()
        }
    elif isinstance(evaluations, list):
        for evaluation in evaluations:
            keypoint_id = str(evaluation.get("keypoint_id", ""))
            if keypoint_id in by_id:
                raise OpenAIJudgeError(
                    f"Judge returned duplicate keypoint ID: {keypoint_id}"
                )
            by_id[keypoint_id] = evaluation
    else:
        raise OpenAIJudgeError("Judge decision has no keypoint_evaluations object")
    if set(by_id) != set(expected):
        raise OpenAIJudgeError(
            "Judge keypoint IDs differ from expected IDs: "
            f"expected={expected}, returned={sorted(by_id)}"
        )

    ordered: list[dict[str, Any]] = []
    keypoint_text = {
        str(item["id"]): str(item["text"]) for item in judge_input["keypoints"]
    }
    candidate_answer = str(judge_input["candidate_answer"])
    for keypoint_id in expected:
        evaluation = dict(by_id[keypoint_id])
        label = str(evaluation.get("label", ""))
        if label not in {"covered", "contradicted", "neither"}:
            raise OpenAIJudgeError(
                f"Judge returned invalid label for {keypoint_id}: {label}"
            )
        raw_answer_evidence = evaluation.get("answer_evidence", "")
        evidence_is_list = isinstance(raw_answer_evidence, list)
        if evidence_is_list:
            evidence_excerpts = [str(value) for value in raw_answer_evidence]
            answer_evidence: str | list[str] = evidence_excerpts
        else:
            evidence_text = str(raw_answer_evidence)
            evidence_excerpts = [evidence_text] if evidence_text else []
            answer_evidence = evidence_text
        if label in {"covered", "contradicted"} and not evidence_excerpts:
            raise OpenAIJudgeError(
                f"{keypoint_id} label={label} requires a non-empty exact "
                "candidate-answer excerpt"
            )
        if any(
            not excerpt or excerpt not in candidate_answer
            for excerpt in evidence_excerpts
        ):
            if evidence_is_list:
                raise OpenAIJudgeError(
                    f"{keypoint_id} answer_evidence must contain only exact "
                    "candidate-answer excerpts"
                )
            raise OpenAIJudgeError(
                f"{keypoint_id} answer_evidence must be an exact "
                "candidate-answer excerpt"
            )
        ordered.append(
            {
                "keypoint_id": keypoint_id,
                "keypoint_text": keypoint_text[keypoint_id],
                "label": label,
                "rationale": str(evaluation.get("rationale", "")),
                "answer_evidence": answer_evidence,
                **(
                    {
                        "answer_evidence_segment_ids": list(
                            evaluation["answer_evidence_segment_ids"]
                        )
                    }
                    if "answer_evidence_segment_ids" in evaluation
                    else {}
                ),
            }
        )

    counts = {
        label: sum(item["label"] == label for item in ordered)
        for label in ("covered", "contradicted", "neither")
    }
    total = len(ordered)
    unsupported = decision.get("unsupported_claims", [])
    if not isinstance(unsupported, list):
        raise OpenAIJudgeError("Judge unsupported_claims must be a list")
    if unsupported_claims_evaluation == "not_evaluated":
        unsupported = []
    return {
        "keypoint_evaluations": ordered,
        "unsupported_claims": unsupported,
        "overall_assessment": str(decision.get("overall_assessment", "")),
        "unsupported_claims_evaluation": unsupported_claims_evaluation,
        "validation_warnings": [],
        "metrics": {
            "keypoint_count": total,
            "covered": counts["covered"],
            "contradicted": counts["contradicted"],
            "neither": counts["neither"],
            "completeness": round(counts["covered"] / total, 4),
            "hallucination": round(counts["contradicted"] / total, 4),
            "irrelevancy": round(counts["neither"] / total, 4),
            "unsupported_claim_count": (
                len(unsupported)
                if unsupported_claims_evaluation == "model_evaluated"
                else None
            ),
        },
    }


def _safe_judge_call(
    *,
    candidate: dict[str, Any],
    judge_fn: JudgeFunction,
) -> dict[str, Any]:
    started_at = datetime.now(UTC).isoformat()
    started_clock = time.perf_counter()
    payload: dict[str, Any] = {}
    try:
        payload = judge_fn(candidate["judge_input"])
        scored = _validate_and_score_decision(
            payload["decision"],
            candidate["judge_input"],
            unsupported_claims_evaluation=str(
                payload.get("unsupported_claims_evaluation", "model_evaluated")
            ),
        )
        return {
            "status": "success",
            "started_at": started_at,
            "completed_at": datetime.now(UTC).isoformat(),
            "latency_ms": round((time.perf_counter() - started_clock) * 1000),
            "response_id": str(payload.get("response_id", "")),
            "response_model": str(payload.get("response_model", "")),
            "usage": payload.get("usage", {}),
            "raw_output_text": str(payload.get("raw_output_text", "")),
            **scored,
            "error_type": "",
            "error": "",
        }
    except Exception as exc:  # noqa: BLE001 - retain progress in paid batches
        return {
            "status": "error",
            "started_at": started_at,
            "completed_at": datetime.now(UTC).isoformat(),
            "latency_ms": round((time.perf_counter() - started_clock) * 1000),
            "response_id": str(payload.get("response_id", "")),
            "response_model": str(payload.get("response_model", "")),
            "usage": payload.get("usage", {}),
            "raw_output_text": str(payload.get("raw_output_text", "")),
            "keypoint_evaluations": [],
            "unsupported_claims": [],
            "unsupported_claims_evaluation": str(
                payload.get("unsupported_claims_evaluation", "model_evaluated")
            ),
            "overall_assessment": "",
            "validation_warnings": [],
            "metrics": None,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def _mean(values: list[float]) -> float:
    return round(statistics.mean(values), 4) if values else 0.0


def _mean_or_none(values: list[float | None]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    return round(statistics.mean(numeric), 4) if numeric else None


def aggregate_scores(
    results: list[dict[str, Any]],
    *,
    group_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Aggregate answer-level metrics without micro-averaging keypoints."""

    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for result in results:
        key = tuple(str(result[field]) for field in group_fields)
        groups.setdefault(key, []).append(result)

    summaries: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        successful = [item for item in group if item["judge"]["status"] == "success"]
        metrics = [item["judge"]["metrics"] for item in successful]
        usage = [item["judge"].get("usage", {}) for item in successful]
        unsupported_counts = [item["unsupported_claim_count"] for item in metrics]
        summary = {field: value for field, value in zip(group_fields, key, strict=True)}
        summary.update(
            {
                "scheduled_answers": len(group),
                "successful_answers": len(successful),
                "failed_answers": len(group) - len(successful),
                "mean_completeness": _mean(
                    [float(item["completeness"]) for item in metrics]
                ),
                "mean_hallucination": _mean(
                    [float(item["hallucination"]) for item in metrics]
                ),
                "mean_irrelevancy": _mean(
                    [float(item["irrelevancy"]) for item in metrics]
                ),
                "unsupported_claims": (
                    sum(int(value) for value in unsupported_counts)
                    if unsupported_counts
                    and all(value is not None for value in unsupported_counts)
                    else None
                ),
                "mean_latency_seconds": _mean(
                    [float(item["judge"]["latency_ms"]) / 1000 for item in successful]
                ),
                "input_tokens": sum(int(item.get("input_tokens", 0)) for item in usage),
                "cached_input_tokens": sum(
                    int(item.get("cached_input_tokens", 0)) for item in usage
                ),
                "output_tokens": sum(
                    int(item.get("output_tokens", 0)) for item in usage
                ),
                "reasoning_tokens": sum(
                    int(item.get("reasoning_tokens", 0)) for item in usage
                ),
                "total_tokens": sum(int(item.get("total_tokens", 0)) for item in usage),
            }
        )
        summaries.append(summary)
    return summaries


def build_paired_comparison(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Calculate within-model RAG minus Baseline differences on matched QRA."""

    conditions: dict[tuple[str, str, int], dict[str, dict[str, Any]]] = {}
    for result in results:
        if result["judge"]["status"] != "success":
            continue
        conditions.setdefault(
            (
                result["tested_model"],
                result["qra_id"],
                int(result.get("repeat", 1)),
            ),
            {},
        )[result["mode"]] = result

    pairs: list[dict[str, Any]] = []
    for (model, qra_id, repeat), modes in sorted(conditions.items()):
        if set(modes) != {"baseline", "rag"}:
            continue
        baseline = modes["baseline"]["judge"]["metrics"]
        rag = modes["rag"]["judge"]["metrics"]
        unsupported_delta = (
            round(
                float(rag["unsupported_claim_count"])
                - float(baseline["unsupported_claim_count"]),
                4,
            )
            if rag["unsupported_claim_count"] is not None
            and baseline["unsupported_claim_count"] is not None
            else None
        )
        pairs.append(
            {
                "tested_model": model,
                "qra_id": qra_id,
                "repeat": repeat,
                "domain": modes["rag"]["domain"],
                "question_type": modes["rag"]["question_type"],
                "baseline": {
                    key: baseline[key]
                    for key in (
                        "completeness",
                        "hallucination",
                        "irrelevancy",
                        "unsupported_claim_count",
                    )
                },
                "rag": {
                    key: rag[key]
                    for key in (
                        "completeness",
                        "hallucination",
                        "irrelevancy",
                        "unsupported_claim_count",
                    )
                },
                "rag_minus_baseline": {
                    key: round(float(rag[key]) - float(baseline[key]), 4)
                    for key in (
                        "completeness",
                        "hallucination",
                        "irrelevancy",
                    )
                }
                | {"unsupported_claim_count": unsupported_delta},
            }
        )

    model_summaries: list[dict[str, Any]] = []
    models = sorted({item["tested_model"] for item in pairs})
    for model in models:
        model_pairs = [item for item in pairs if item["tested_model"] == model]
        qra_groups: dict[str, list[dict[str, Any]]] = {}
        for item in model_pairs:
            qra_groups.setdefault(str(item["qra_id"]), []).append(item)
        qra_level_deltas = []
        for qra_id, qra_pairs in sorted(qra_groups.items()):
            qra_level_deltas.append(
                {
                    "qra_id": qra_id,
                    "paired_repeat_count": len(qra_pairs),
                    "completeness": _mean(
                        [
                            item["rag_minus_baseline"]["completeness"]
                            for item in qra_pairs
                        ]
                    ),
                    "hallucination": _mean(
                        [
                            item["rag_minus_baseline"]["hallucination"]
                            for item in qra_pairs
                        ]
                    ),
                    "irrelevancy": _mean(
                        [
                            item["rag_minus_baseline"]["irrelevancy"]
                            for item in qra_pairs
                        ]
                    ),
                    "unsupported_claim_count": _mean_or_none(
                        [
                            item["rag_minus_baseline"]["unsupported_claim_count"]
                            for item in qra_pairs
                        ]
                    ),
                }
            )
        model_summaries.append(
            {
                "tested_model": model,
                "paired_qra_count": len(qra_groups),
                "paired_observation_count": len(model_pairs),
                "mean_rag_minus_baseline_completeness": _mean(
                    [item["completeness"] for item in qra_level_deltas]
                ),
                "mean_rag_minus_baseline_hallucination": _mean(
                    [item["hallucination"] for item in qra_level_deltas]
                ),
                "mean_rag_minus_baseline_irrelevancy": _mean(
                    [item["irrelevancy"] for item in qra_level_deltas]
                ),
                "mean_rag_minus_baseline_unsupported_claims": _mean_or_none(
                    [
                        item["unsupported_claim_count"] for item in qra_level_deltas
                    ]
                ),
                "qra_level_deltas": qra_level_deltas,
            }
        )
    return {"summary": model_summaries, "pairs": pairs}


def _usage_totals(results: list[dict[str, Any]]) -> dict[str, int]:
    fields = (
        "input_tokens",
        "cached_input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "total_tokens",
    )
    return {
        field: sum(
            int(item["judge"].get("usage", {}).get(field, 0)) for item in results
        )
        for field in fields
    }


def build_markdown_report(report: dict[str, Any]) -> str:
    """Render a concise human-review companion to the score JSON."""

    lines = [
        "# Responses API Judge development report",
        "",
        f"- Judge run: `{report['judge_run_id']}`",
        f"- Status: `{report['status']}`",
        f"- Evaluation status: `{report['evaluation_status']}`",
        f"- Source run: `{report['source_run']['run_id']}`",
        f"- Judge provider: `{report['judge']['provider']}`",
        f"- Requested Judge model: `{report['judge']['requested_model']}`",
        f"- Reasoning effort: `{report['judge']['reasoning_effort']}`",
        f"- Judge input policy: `{report['judge']['input_policy']}`",
        "- Unsupported-claim evaluation: "
        f"`{report['judge']['unsupported_claims_evaluation']}`",
        f"- Candidate answers: `{report['selection']['candidate_count']}`",
        f"- Preserved prior failed attempts: `{len(report.get('prior_attempts', []))}`",
        f"- Blind seed: `{report['selection']['seed']}`",
        "",
        "This is an automated development-set Judge run. It is not a formal "
        "thesis result until validated against independent human labels.",
        "",
        "## Overall comparison",
        "",
        "| Tested model | Mode | Success | Completeness | Hallucination | "
        "Irrelevancy | Unsupported claims | Input tokens | Output tokens |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["summary"]:
        unsupported_claims = (
            str(item["unsupported_claims"])
            if item["unsupported_claims"] is not None
            else "N/A"
        )
        lines.append(
            f"| `{item['tested_model']}` | {item['mode']} | "
            f"{item['successful_answers']}/{item['scheduled_answers']} | "
            f"{item['mean_completeness']:.4f} | "
            f"{item['mean_hallucination']:.4f} | "
            f"{item['mean_irrelevancy']:.4f} | "
            f"{unsupported_claims} | {item['input_tokens']} | "
            f"{item['output_tokens']} |"
        )

    lines.extend(
        [
            "",
            "## Paired RAG minus Baseline",
            "",
            "| Tested model | Paired QRA | Completeness delta | "
            "Hallucination delta | Irrelevancy delta | "
            "Unsupported-claim delta |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for item in report["paired_comparison"]["summary"]:
        unsupported_delta = item["mean_rag_minus_baseline_unsupported_claims"]
        unsupported_delta_text = (
            f"{unsupported_delta:+.4f}" if unsupported_delta is not None else "N/A"
        )
        lines.append(
            f"| `{item['tested_model']}` | {item['paired_qra_count']} | "
            f"{item['mean_rag_minus_baseline_completeness']:+.4f} | "
            f"{item['mean_rag_minus_baseline_hallucination']:+.4f} | "
            f"{item['mean_rag_minus_baseline_irrelevancy']:+.4f} | "
            f"{unsupported_delta_text} |"
        )

    lines.extend(["", "## Candidate decisions", ""])
    for result in sorted(
        report["results"],
        key=lambda item: (
            item["tested_model"],
            item["qra_id"],
            int(item.get("repeat", 1)),
            item["mode"],
        ),
    ):
        judge = result["judge"]
        lines.extend(
            [
                f"### {result['qra_id']} / {result['tested_model']} / "
                f"{result['mode']} / repeat {result.get('repeat', 1)}",
                "",
                f"- Candidate ID: `{result['candidate_id']}`",
                f"- Status: `{judge['status']}`",
                f"- Judge latency: `{judge['latency_ms'] / 1000:.3f} s`",
                f"- Response model: `{judge['response_model']}`",
                "",
            ]
        )
        if judge["status"] != "success":
            lines.extend(
                [
                    f"**Error:** `{judge['error_type']}` {judge['error']}",
                    "",
                ]
            )
            continue
        metrics = judge["metrics"]
        unsupported_count = metrics["unsupported_claim_count"]
        unsupported_count_text = (
            str(unsupported_count) if unsupported_count is not None else "N/A"
        )
        lines.extend(
            [
                f"- Completeness: `{metrics['completeness']:.4f}`",
                f"- Hallucination: `{metrics['hallucination']:.4f}`",
                f"- Irrelevancy: `{metrics['irrelevancy']:.4f}`",
                f"- Unsupported claims: `{unsupported_count_text}`",
                "",
                "| Keypoint | Label | Rationale |",
                "|---|---|---|",
            ]
        )
        for evaluation in judge["keypoint_evaluations"]:
            rationale = evaluation["rationale"].replace("|", "\\|")
            lines.append(
                f"| `{evaluation['keypoint_id']}` | "
                f"{evaluation['label']} | {rationale} |"
            )
        lines.extend(["", judge["overall_assessment"], ""])
    return "\n".join(lines).rstrip() + "\n"


def _openai_sdk_version() -> str:
    try:
        return version("openai")
    except PackageNotFoundError:
        return "not-installed"


def run_openai_judge(
    *,
    source_run_path: Path,
    output_root: Path | None = None,
    tested_models: Iterable[str] | None = None,
    qra_ids: Iterable[str] | None = None,
    modes: Iterable[str] = ("baseline", "rag"),
    judge_model: str = "gpt-5.4-mini",
    judge_provider: str = "openai",
    api_key: str | None = None,
    api_base_url: str | None = None,
    reasoning_effort: str = "low",
    max_output_tokens: int = 2500,
    timeout_seconds: int = 120,
    max_retries: int = 2,
    seed: int = 20260728,
    run_id: str | None = None,
    prompt_path: Path | None = None,
    judge_fn: JudgeFunction | None = None,
    progress_callback: ProgressCallback | None = None,
    fail_on_error: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    """Judge selected source answers and export auditable comparison artifacts."""

    source_run_path = source_run_path.resolve()
    source_run = load_generation_run(source_run_path)
    prompt = load_judge_prompt(prompt_path)
    selection = prepare_blinded_candidates(
        source_run,
        tested_models=tested_models,
        qra_ids=qra_ids,
        modes=modes,
        seed=seed,
    )
    if reasoning_effort not in ALLOWED_REASONING_EFFORTS:
        raise OpenAIJudgeError(f"Invalid reasoning effort: {reasoning_effort}")
    judge_provider = judge_provider.strip().lower()
    if judge_provider not in ALLOWED_JUDGE_PROVIDERS:
        raise OpenAIJudgeError(
            f"Judge provider must be one of {sorted(ALLOWED_JUDGE_PROVIDERS)}"
        )
    api_base_url = api_base_url.rstrip("/") if api_base_url else None
    if resume and not run_id:
        raise OpenAIJudgeError("resume requires an explicit existing run_id")
    now = datetime.now(UTC)
    run_id = run_id or f"OPENAI-JUDGE-{now.strftime('%Y%m%dT%H%M%S%fZ')}"
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise OpenAIJudgeError(
            "run_id must contain only letters, digits, dots, underscores, and hyphens"
        )
    output_root = (output_root or default_output_root(source_run_path)).resolve()
    run_dir = output_root / run_id
    if run_dir.exists() and not resume:
        raise OpenAIJudgeError(f"Judge output directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=resume)

    if judge_fn is None:
        judge_fn = OpenAIKeypointJudge(
            prompt=prompt,
            model=judge_model,
            reasoning_effort=reasoning_effort,
            max_output_tokens=max_output_tokens,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            api_key=api_key,
            base_url=api_base_url,
        )

    blinded_path = run_dir / f"{run_id}_blinded_inputs.json"
    mapping_path = run_dir / f"{run_id}_candidate_mapping.json"
    checkpoint_path = run_dir / f"{run_id}.partial.json"
    scores_path = run_dir / f"{run_id}_scores.json"
    markdown_path = run_dir / f"{run_id}_report.md"
    manifest_path = run_dir / f"{run_id}_manifest.json"

    blinded = {
        "judge_run_id": run_id,
        "seed": seed,
        "candidate_count": selection["candidate_count"],
        "selection_sha256": selection["selection_sha256"],
        "candidates": [
            {
                "candidate_id": item["candidate_id"],
                "judge_input": item["judge_input"],
            }
            for item in selection["candidates"]
        ],
    }
    mapping = {
        "judge_run_id": run_id,
        "candidate_count": selection["candidate_count"],
        "selection_sha256": selection["selection_sha256"],
        "candidates": [
            {
                key: item[key]
                for key in (
                    "candidate_id",
                    "source_sequence",
                    "qra_id",
                    "domain",
                    "question_type",
                    "tested_model",
                    "tested_model_type",
                    "mode",
                    "repeat",
                )
            }
            for item in selection["candidates"]
        ],
    }
    new_report: dict[str, Any] = {
        "judge_run_id": run_id,
        "status": "running",
        "evaluation_status": "development_api_judge_unvalidated",
        "started_at": now.isoformat(),
        "completed_at": None,
        "source_run": {
            "run_id": source_run["run_id"],
            **source_run["_artifact"],
            "dataset_id": source_run["dataset"].get("dataset_id", ""),
            "dataset_manifest_sha256": source_run["dataset"].get(
                "manifest_sha256",
                "",
            ),
        },
        "judge": {
            "provider": judge_provider,
            "api": "responses",
            "base_url": api_base_url,
            "requested_model": judge_model,
            "response_models": [],
            "reasoning_effort": reasoning_effort,
            "max_output_tokens": max_output_tokens,
            "timeout_seconds": timeout_seconds,
            "max_retries": max_retries,
            "structured_outputs": True,
            "store": False,
            "input_policy": str(
                prompt.get("judge_input_policy", "legacy_full_reference")
            ),
            "model_visible_input_fields": list(prompt.get("judge_input_fields", [])),
            "answer_evidence_format": str(
                prompt.get("answer_evidence_format", "legacy_string")
            ),
            "unsupported_claims_evaluation": str(
                prompt.get(
                    "unsupported_claims_evaluation",
                    "model_evaluated",
                )
            ),
            "openai_sdk_version": _openai_sdk_version(),
            "prompt": prompt["_artifact"],
        },
        "selection": {
            key: selection[key]
            for key in (
                "seed",
                "candidate_count",
                "skipped_generation_errors",
                "selection_sha256",
                "filters",
            )
        },
        "summary": [],
        "domain_summary": [],
        "paired_comparison": {"summary": [], "pairs": []},
        "usage": {},
        "results": [],
        "prior_attempts": [],
        "resume": {
            "resumed": False,
            "resume_count": 0,
            "preexisting_successful_answers": 0,
        },
    }
    if resume:
        if not checkpoint_path.exists():
            raise OpenAIJudgeError(
                f"Resume checkpoint does not exist: {checkpoint_path}"
            )
        if scores_path.exists() or manifest_path.exists():
            raise OpenAIJudgeError(
                "Cannot resume a Judge run that already has final artifacts"
            )
        report = _load_json_object(checkpoint_path)
        checks = (
            (
                report.get("source_run", {}).get("sha256"),
                new_report["source_run"]["sha256"],
                "source run hash",
            ),
            (
                report.get("selection", {}).get("selection_sha256"),
                new_report["selection"]["selection_sha256"],
                "selection hash",
            ),
            (
                report.get("judge", {}).get("provider"),
                new_report["judge"]["provider"],
                "Judge provider",
            ),
            (
                report.get("judge", {}).get("base_url"),
                new_report["judge"]["base_url"],
                "Judge base URL",
            ),
            (
                report.get("judge", {}).get("requested_model"),
                new_report["judge"]["requested_model"],
                "Judge model",
            ),
            (
                report.get("judge", {}).get("reasoning_effort"),
                new_report["judge"]["reasoning_effort"],
                "reasoning effort",
            ),
            (
                report.get("judge", {}).get("prompt", {}).get("sha256"),
                new_report["judge"]["prompt"]["sha256"],
                "Judge prompt hash",
            ),
        )
        mismatches = [name for old, new, name in checks if old != new]
        if mismatches:
            raise OpenAIJudgeError(f"Resume configuration differs for: {mismatches}")
        if not blinded_path.exists() or not mapping_path.exists():
            raise OpenAIJudgeError(
                "Resume requires the original blinded inputs and candidate mapping"
            )
        saved_blinded = _load_json_object(blinded_path)
        saved_mapping = _load_json_object(mapping_path)
        if (
            saved_blinded.get("selection_sha256") != selection["selection_sha256"]
            or saved_mapping.get("selection_sha256") != selection["selection_sha256"]
        ):
            raise OpenAIJudgeError(
                "Resume input artifacts do not match the selected candidates"
            )
        previous_resume_count = int(report.get("resume", {}).get("resume_count", 0))
        previous_results = report.get("results", [])
        successful_results = [
            item
            for item in previous_results
            if item.get("judge", {}).get("status") == "success"
        ]
        prior_attempts = list(report.get("prior_attempts", []))
        prior_attempts.extend(
            item
            for item in previous_results
            if item.get("judge", {}).get("status") != "success"
        )
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
        _write_json_atomic(blinded_path, blinded)
        _write_json_atomic(mapping_path, mapping)
        report = new_report
        _write_json_atomic(checkpoint_path, report)

    total = selection["candidate_count"]
    completed_candidate_ids = {item["candidate_id"] for item in report["results"]}
    for position, candidate in enumerate(selection["candidates"], start=1):
        if candidate["candidate_id"] in completed_candidate_ids:
            continue
        if progress_callback:
            progress_callback(
                {
                    "type": "start",
                    "position": position,
                    "total": total,
                    **{
                        key: candidate[key]
                        for key in (
                            "candidate_id",
                            "qra_id",
                            "tested_model",
                            "mode",
                            "repeat",
                        )
                    },
                }
            )
        judge = _safe_judge_call(candidate=candidate, judge_fn=judge_fn)
        result = {
            key: candidate[key]
            for key in (
                "candidate_id",
                "source_sequence",
                "qra_id",
                "domain",
                "question_type",
                "tested_model",
                "tested_model_type",
                "mode",
                "repeat",
            )
        }
        result["judge"] = judge
        report["results"].append(result)
        report["summary"] = aggregate_scores(
            report["results"],
            group_fields=("tested_model", "tested_model_type", "mode"),
        )
        report["domain_summary"] = aggregate_scores(
            report["results"],
            group_fields=("tested_model", "mode", "domain"),
        )
        report["paired_comparison"] = build_paired_comparison(report["results"])
        report["usage"] = _usage_totals(
            [*report.get("prior_attempts", []), *report["results"]]
        )
        _write_json_atomic(checkpoint_path, report)
        if progress_callback:
            progress_callback(
                {
                    "type": "complete",
                    "position": position,
                    "total": total,
                    "candidate_id": candidate["candidate_id"],
                    "qra_id": candidate["qra_id"],
                    "tested_model": candidate["tested_model"],
                    "mode": candidate["mode"],
                    "repeat": candidate["repeat"],
                    "status": judge["status"],
                    "latency_ms": judge["latency_ms"],
                    "error": judge["error"],
                }
            )
        if fail_on_error and judge["status"] == "error":
            report["status"] = "failed"
            report["completed_at"] = datetime.now(UTC).isoformat()
            _write_json_atomic(checkpoint_path, report)
            raise OpenAIJudgeError(
                f"Judge failed for {candidate['candidate_id']}: {judge['error']}"
            )

    report["status"] = (
        "completed_with_errors"
        if any(item["judge"]["status"] == "error" for item in report["results"])
        else "completed"
    )
    report["completed_at"] = datetime.now(UTC).isoformat()
    report["usage"] = _usage_totals(
        [*report.get("prior_attempts", []), *report["results"]]
    )
    report["judge"]["response_models"] = sorted(
        {
            item["judge"]["response_model"]
            for item in report["results"]
            if item["judge"]["response_model"]
        }
    )
    _write_json_atomic(scores_path, report)
    markdown_path.write_text(build_markdown_report(report), encoding="utf-8")

    code_path = Path(__file__).resolve()
    artifacts = {}
    for name, path in (
        ("blinded_inputs", blinded_path),
        ("candidate_mapping", mapping_path),
        ("scores", scores_path),
        ("markdown_report", markdown_path),
    ):
        artifacts[name] = {
            "path": path.name,
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    manifest = {
        "judge_run_id": run_id,
        "status": report["status"],
        "evaluation_status": report["evaluation_status"],
        "source_run": report["source_run"],
        "judge": report["judge"],
        "selection": report["selection"],
        "implementation": {
            "path": _display_path(code_path),
            "sha256": sha256_file(code_path),
        },
        "artifacts": artifacts,
        "limitations": [
            "Automated development-set Judge results are not formal thesis results.",
            "Human agreement validation and adjudication have not yet been completed.",
            "The requested model alias may move; freeze a dated model snapshot before formal evaluation.",
            *(
                [
                    "Unsupported-claim extraction was not evaluated because the "
                    "keypoint Judge input is reference-isolated; the corresponding "
                    "counts are null/N/A, not zero."
                ]
                if report["judge"]["unsupported_claims_evaluation"] == "not_evaluated"
                else []
            ),
        ],
    }
    _write_json_atomic(manifest_path, manifest)
    checkpoint_path.unlink(missing_ok=True)
    return {
        "report": report,
        "run_dir": str(run_dir),
        "scores_path": str(scores_path),
        "markdown_path": str(markdown_path),
        "manifest_path": str(manifest_path),
    }
