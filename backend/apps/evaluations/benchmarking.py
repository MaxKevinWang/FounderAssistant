"""Benchmark execution utilities for baseline-vs-RAG comparison.

Important note:
This is a project-specific, corpus-grounded benchmark (not MMLU/HELM/BIG-Bench).
It evaluates:
1) TF correctness on curated legal/compliance statements.
2) Citation presence/traceability proxies.
3) Optional scenario output latency snapshot.

Outputs are exported as JSON + Markdown, and can optionally be persisted into
the EvaluationRun / EvaluationArtifact / ComparisonResult tables.
"""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from django.conf import settings
from django.contrib.auth import get_user_model
from django.core.management.base import CommandError

from apps.advisors.local_llm import LocalLLMError, generate_text
from apps.advisors.model_profiles import DEFAULT_ANSWER_MAX_TOKENS
from apps.advisors.rag_pipeline import RAGError, rag_answer
from apps.ventures.models import Venture

from .models import ComparisonResult, EvaluationArtifact, EvaluationRun, ScenarioCase

CITATION_PATTERN = re.compile(r"(\[[^\]]{1,40}\]|https?://|§\s*\d+|Art\.?\s*\d+)", re.IGNORECASE)
TF_LABEL_PATTERN = re.compile(r"\b(TRUE|FALSE|WAHR|FALSCH)\b", re.IGNORECASE)


@dataclass(frozen=True)
class ModeResult:
    mode: str
    response: str
    predicted_label: str
    is_correct: bool
    citation_present: bool
    latency_ms: int
    retrieval_latency_ms: int
    retrieval_mode: str
    citation_count: int
    error: str


def project_root() -> Path:
    """Return repository root directory from Django backend base path."""

    return Path(settings.BASE_DIR).parent


def default_test_set_path() -> Path:
    """Default machine-readable benchmark question set path."""

    return project_root() / "docs" / "evaluation" / "test_question_set.json"


def default_output_dir() -> Path:
    """Default output folder for generated evaluation artifacts."""

    return project_root() / "docs" / "evaluation" / "runs"


def load_test_set(path: Path) -> dict[str, Any]:
    """Load and validate benchmark test-set JSON file."""

    if not path.exists():
        raise CommandError(f"Test set file not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CommandError(f"Invalid JSON in test set file {path}: {exc}") from exc


def extract_tf_label(response_text: str) -> str:
    """Extract normalized TRUE/FALSE label from model response text.

    Robustness goals:
    1) Tolerate markdown wrappers (for example `**TRUE**`).
    2) Tolerate numbered formats (`1. TRUE`, `1) TRUE`, `Line 1: TRUE`).
    3) Map German outputs (`WAHR`/`FALSCH`) to (`TRUE`/`FALSE`).
    """

    if not response_text:
        return "UNKNOWN"

    prefix = response_text[:500]
    lines = [line.strip() for line in prefix.splitlines() if line.strip()][:8]

    def _normalize_label(raw: str) -> str:
        value = raw.upper()
        if value == "WAHR":
            return "TRUE"
        if value == "FALSCH":
            return "FALSE"
        return value

    # Pass 1: line-based parsing in the initial lines.
    for line in lines:
        cleaned = re.sub(r"^[#>*`\\-\\s]+", "", line).strip()
        cleaned = re.sub(
            r"^(?:line\\s*1\\s*[:\\-]?\\s*|answer\\s*[:\\-]?\\s*|\\d+\\s*[\\).:\\-]?\\s*)",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = cleaned.strip().strip("*_` ").strip()
        match = TF_LABEL_PATTERN.match(cleaned)
        if match:
            return _normalize_label(match.group(1))

    # Pass 2: fallback search in the short prefix.
    match = TF_LABEL_PATTERN.search(prefix)
    if not match:
        return "UNKNOWN"
    return _normalize_label(match.group(1))


def has_citation(response_text: str) -> bool:
    """Return True when response appears to include citation-like markers."""

    return bool(CITATION_PATTERN.search(response_text))


def tf_prompt(statement: str) -> str:
    """Build constrained prompt for True/False benchmark items."""

    return (
        "Evaluate this statement using German startup/legal context from available knowledge.\n"
        "Output format (strict):\n"
        "Line 1: TRUE or FALSE\n"
        "Line 2: One short justification\n"
        "Line 3: One citation (source marker or legal reference)\n\n"
        f"Statement: {statement}"
    )


def scenario_prompt(base_prompt: str) -> str:
    """Build structured prompt for scenario-style qualitative benchmark item."""

    return (
        f"{base_prompt}\n\n"
        "Output format:\n"
        "1) Recommendation\n"
        "2) Compliance checklist\n"
        "3) Validation plan\n"
        "4) Evidence list with citations"
    )


def run_baseline(
    prompt: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
    generation_model: str | None = None,
    expected_label: str | None = None,
) -> ModeResult:
    """Execute one baseline generation run and normalize result payload."""

    try:
        payload = generate_text(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            model=generation_model,
        )
        response = payload.get("response", "")
        predicted = extract_tf_label(response) if expected_label else "N/A"
        return ModeResult(
            mode="baseline",
            response=response,
            predicted_label=predicted,
            is_correct=bool(expected_label and predicted == expected_label),
            citation_present=has_citation(response),
            latency_ms=int(payload.get("latency_ms", 0)),
            retrieval_latency_ms=0,
            retrieval_mode="none",
            citation_count=0,
            error="",
        )
    except LocalLLMError as exc:
        return ModeResult(
            mode="baseline",
            response="",
            predicted_label="ERROR",
            is_correct=False,
            citation_present=False,
            latency_ms=0,
            retrieval_latency_ms=0,
            retrieval_mode="none",
            citation_count=0,
            error=str(exc),
        )


def run_rag(
    prompt: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
    top_k: int,
    generation_model: str | None = None,
    expected_label: str | None = None,
    domain: str = "",
) -> ModeResult:
    """Execute one retrieval-augmented generation run and normalize result payload."""

    try:
        payload = rag_answer(
            question=prompt,
            domain=domain,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_k=top_k,
            model=generation_model,
        )
        response = payload.get("response", "")
        predicted = extract_tf_label(response) if expected_label else "N/A"
        citations = payload.get("citations", [])
        return ModeResult(
            mode="rag",
            response=response,
            predicted_label=predicted,
            is_correct=bool(expected_label and predicted == expected_label),
            citation_present=has_citation(response) or len(citations) > 0,
            latency_ms=int(payload.get("latency_ms", 0)),
            retrieval_latency_ms=int(payload.get("retrieval_latency_ms", 0)),
            retrieval_mode=str(payload.get("retrieval_mode", "")),
            citation_count=len(citations),
            error="",
        )
    except (LocalLLMError, RAGError) as exc:
        return ModeResult(
            mode="rag",
            response="",
            predicted_label="ERROR",
            is_correct=False,
            citation_present=False,
            latency_ms=0,
            retrieval_latency_ms=0,
            retrieval_mode="error",
            citation_count=0,
            error=str(exc),
        )


def aggregate_tf_metrics(items: list[dict[str, Any]], mode_key: str) -> dict[str, Any]:
    """Aggregate TF correctness, citation proxy, latency, and error metrics."""

    mode_items_all = [item[mode_key] for item in items]
    mode_items = [item for item in mode_items_all if item["predicted_label"] != "SKIPPED"]
    attempted = len(mode_items)
    correct = sum(1 for item in mode_items if item["is_correct"])
    citations = sum(1 for item in mode_items if item["citation_present"])
    latency_values = [item["latency_ms"] for item in mode_items if item["latency_ms"] > 0]
    retrieval_values = [item["retrieval_latency_ms"] for item in mode_items if item["retrieval_latency_ms"] > 0]
    errors = sum(1 for item in mode_items if item["error"])
    return {
        "attempted": attempted,
        "correct": correct,
        "accuracy": round(correct / attempted, 4) if attempted else 0.0,
        "citation_present_count": citations,
        "citation_rate": round(citations / attempted, 4) if attempted else 0.0,
        "avg_latency_ms": round(statistics.mean(latency_values), 2) if latency_values else 0.0,
        "avg_retrieval_latency_ms": round(statistics.mean(retrieval_values), 2) if retrieval_values else 0.0,
        "error_count": errors,
    }


def mode_result_to_dict(result: ModeResult) -> dict[str, Any]:
    """Convert `ModeResult` dataclass to JSON-serializable dictionary."""

    return {
        "mode": result.mode,
        "response": result.response,
        "predicted_label": result.predicted_label,
        "is_correct": result.is_correct,
        "citation_present": result.citation_present,
        "latency_ms": result.latency_ms,
        "retrieval_latency_ms": result.retrieval_latency_ms,
        "retrieval_mode": result.retrieval_mode,
        "citation_count": result.citation_count,
        "error": result.error,
    }


def build_markdown_report(report: dict[str, Any]) -> str:
    """Render single-run report as Markdown for thesis/demo readability."""

    summary = report["summary"]
    lines = [
        "# Baseline vs RAG Evaluation Report",
        "",
        f"- Generated at: {report['generated_at']}",
        f"- Test set: `{report['config']['test_set_path']}`",
        f"- Local model: `{report['config']['generation_model']}`",
        f"- Top-k: `{report['config']['top_k']}`",
        "",
        "## TF Summary",
        "",
        "| Mode | Accuracy | Citation Rate | Avg Latency (ms) | Avg Retrieval (ms) | Errors |",
        "| --- | --- | --- | --- | --- | --- |",
        (
            f"| Baseline | {summary['baseline_tf']['accuracy']:.2%} | "
            f"{summary['baseline_tf']['citation_rate']:.2%} | "
            f"{summary['baseline_tf']['avg_latency_ms']} | "
            f"{summary['baseline_tf']['avg_retrieval_latency_ms']} | "
            f"{summary['baseline_tf']['error_count']} |"
        ),
        (
            f"| RAG | {summary['rag_tf']['accuracy']:.2%} | "
            f"{summary['rag_tf']['citation_rate']:.2%} | "
            f"{summary['rag_tf']['avg_latency_ms']} | "
            f"{summary['rag_tf']['avg_retrieval_latency_ms']} | "
            f"{summary['rag_tf']['error_count']} |"
        ),
        "",
        "## TF Item Results",
        "",
        "| ID | Gold | Baseline | RAG |",
        "| --- | --- | --- | --- |",
    ]

    for item in report["tf_results"]:
        base = item["baseline"]["predicted_label"]
        rag = item["rag"]["predicted_label"]
        lines.append(f"| {item['id']} | {item['gold_label']} | {base} | {rag} |")

    if report.get("scenario_result"):
        lines.extend(
            [
                "",
                "## Scenario Snapshot",
                "",
                f"- Scenario ID: {report['scenario_result']['id']}",
                f"- Baseline latency: {report['scenario_result']['baseline']['latency_ms']} ms",
                (
                    f"- RAG latency: {report['scenario_result']['rag']['latency_ms']} ms "
                    f"(retrieval: {report['scenario_result']['rag']['retrieval_latency_ms']} ms, "
                    f"mode: {report['scenario_result']['rag']['retrieval_mode']})"
                ),
            ]
        )

    if report.get("db_persistence"):
        lines.extend(
            [
                "",
                "## DB Persistence",
                "",
                f"- Persisted: {report['db_persistence']['persisted']}",
                f"- Owner: {report['db_persistence'].get('owner_username', '')}",
                f"- Venture: {report['db_persistence'].get('venture_name', '')}",
                f"- Baseline run id: {report['db_persistence'].get('baseline_run_id', '')}",
                f"- RAG run id: {report['db_persistence'].get('rag_run_id', '')}",
                f"- Comparison id: {report['db_persistence'].get('comparison_id', '')}",
            ]
        )

    return "\n".join(lines) + "\n"


def _mean_std(values: list[float]) -> dict[str, float]:
    """Return rounded mean/std summary for numeric run metrics."""

    if not values:
        return {"mean": 0.0, "std": 0.0}
    if len(values) == 1:
        return {"mean": round(values[0], 4), "std": 0.0}
    return {"mean": round(statistics.mean(values), 4), "std": round(statistics.stdev(values), 4)}


def build_batch_markdown_report(batch_report: dict[str, Any]) -> str:
    """Render repeated-run aggregate report as Markdown."""

    aggregate = batch_report["aggregate"]
    lines = [
        "# Baseline vs RAG Batch Report",
        "",
        f"- Generated at: {batch_report['generated_at']}",
        f"- Repeats: {batch_report['repeats']}",
        f"- Test set: `{batch_report['config']['test_set_path']}`",
        f"- Mode: `{batch_report['config']['mode']}`",
        "",
        "## Aggregate",
        "",
        "| Metric | Baseline Mean | Baseline Std | RAG Mean | RAG Std |",
        "| --- | --- | --- | --- | --- |",
        (
            f"| Accuracy | {aggregate['baseline_accuracy']['mean']:.4f} | {aggregate['baseline_accuracy']['std']:.4f} | "
            f"{aggregate['rag_accuracy']['mean']:.4f} | {aggregate['rag_accuracy']['std']:.4f} |"
        ),
        (
            f"| Citation Rate | {aggregate['baseline_citation_rate']['mean']:.4f} | {aggregate['baseline_citation_rate']['std']:.4f} | "
            f"{aggregate['rag_citation_rate']['mean']:.4f} | {aggregate['rag_citation_rate']['std']:.4f} |"
        ),
        "",
        "## Win Count",
        "",
        f"- RAG better: {aggregate['rag_better_count']}",
        f"- Baseline better: {aggregate['baseline_better_count']}",
        f"- Tie: {aggregate['tie_count']}",
        "",
        "## Run Files",
    ]

    for run in batch_report["runs"]:
        lines.append(
            f"- Run {run['run_index']}: baseline_acc={run['baseline_accuracy']:.4f}, "
            f"rag_acc={run['rag_accuracy']:.4f}, json=`{run['json_path']}`"
        )

    return "\n".join(lines) + "\n"


def _decide_winner(summary: dict[str, Any]) -> str:
    """Decide winner by TF accuracy only (project's primary metric)."""

    baseline_accuracy = summary["baseline_tf"]["accuracy"]
    rag_accuracy = summary["rag_tf"]["accuracy"]
    if rag_accuracy > baseline_accuracy:
        return "rag"
    if baseline_accuracy > rag_accuracy:
        return "baseline"
    return "tie"


def _ensure_owner_and_venture(owner_username: str, venture_name: str) -> tuple[Any, Venture]:
    """Ensure benchmark persistence target user and venture exist."""

    User = get_user_model()
    owner, _ = User.objects.get_or_create(
        username=owner_username,
        defaults={
            "email": f"{owner_username}@local.demo",
            "is_active": True,
        },
    )
    if not owner.has_usable_password():
        owner.set_unusable_password()
        owner.save(update_fields=["password"])

    venture, _ = Venture.objects.get_or_create(
        owner=owner,
        name=venture_name,
        defaults={
            "stage": "validation",
            "problem_statement": "Benchmark venture generated by run_baseline_rag_eval.",
            "value_proposition": "Benchmarking baseline vs RAG.",
            "target_segment": "Internal evaluation users",
            "legal_form": "UG",
        },
    )
    return owner, venture


def _persist_report_to_db(
    report: dict[str, Any],
    owner_username: str,
    venture_name: str,
) -> dict[str, Any]:
    """Persist summary and item-level artifacts into evaluation tables."""

    owner, venture = _ensure_owner_and_venture(owner_username=owner_username, venture_name=venture_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    summary = report["summary"]

    baseline_run = EvaluationRun.objects.create(
        run_name=f"benchmark_{timestamp}_baseline",
        owner=owner,
        venture=venture,
        decision_domain="legal_form",
        run_mode="baseline",
        llm_model=report["config"]["llm_model"],
        retrieval_recall=0.0,
        citation_precision=summary["baseline_tf"]["citation_rate"],
        relevance_score=summary["baseline_tf"]["accuracy"] * 5,
        traceability_score=summary["baseline_tf"]["citation_rate"] * 5,
        perceived_usefulness=summary["baseline_tf"]["accuracy"] * 5,
        response_latency_ms=int(summary["baseline_tf"]["avg_latency_ms"] or 0),
        notes=f"Auto-generated benchmark run from {report['generated_at']}.",
    )
    rag_run = EvaluationRun.objects.create(
        run_name=f"benchmark_{timestamp}_rag",
        owner=owner,
        venture=venture,
        decision_domain="legal_form",
        run_mode="rag",
        llm_model=report["config"]["llm_model"],
        retrieval_recall=summary["rag_tf"]["citation_rate"],
        citation_precision=summary["rag_tf"]["citation_rate"],
        relevance_score=summary["rag_tf"]["accuracy"] * 5,
        traceability_score=summary["rag_tf"]["citation_rate"] * 5,
        perceived_usefulness=summary["rag_tf"]["accuracy"] * 5,
        response_latency_ms=int(summary["rag_tf"]["avg_latency_ms"] or 0),
        notes=f"Auto-generated benchmark run from {report['generated_at']}.",
    )

    EvaluationArtifact.objects.create(run=baseline_run, artifact_type="benchmark_summary", payload=report["summary"])
    EvaluationArtifact.objects.create(run=rag_run, artifact_type="benchmark_summary", payload=report["summary"])

    for tf_item in report["tf_results"]:
        EvaluationArtifact.objects.create(
            run=baseline_run,
            artifact_type="tf_result",
            payload={
                "id": tf_item["id"],
                "statement": tf_item["statement"],
                "gold_label": tf_item["gold_label"],
                "result": tf_item["baseline"],
            },
        )
        EvaluationArtifact.objects.create(
            run=rag_run,
            artifact_type="tf_result",
            payload={
                "id": tf_item["id"],
                "statement": tf_item["statement"],
                "gold_label": tf_item["gold_label"],
                "result": tf_item["rag"],
            },
        )

    comparison = None
    scenario_payload = report.get("scenario_result")
    if scenario_payload:
        scenario_case, _ = ScenarioCase.objects.get_or_create(
            code=str(scenario_payload.get("id", "SC-AUTO")),
            defaults={
                "domain": "market_validation",
                "title": str(scenario_payload.get("title", "Benchmark Scenario")),
                "problem_statement": "Auto-created benchmark scenario placeholder.",
                "founder_context": {},
                "expected_outcomes": [],
            },
        )
        baseline_run.scenario = scenario_case
        rag_run.scenario = scenario_case
        baseline_run.save(update_fields=["scenario"])
        rag_run.save(update_fields=["scenario"])

        EvaluationArtifact.objects.create(
            run=baseline_run,
            artifact_type="scenario_result",
            payload=scenario_payload.get("baseline", {}),
        )
        EvaluationArtifact.objects.create(
            run=rag_run,
            artifact_type="scenario_result",
            payload=scenario_payload.get("rag", {}),
        )

        comparison = ComparisonResult.objects.create(
            owner=owner,
            venture=venture,
            scenario=scenario_case,
            baseline_run=baseline_run,
            rag_run=rag_run,
            winner=_decide_winner(summary),
            judge_notes="Auto-generated comparison from benchmark command.",
        )

    return {
        "persisted": True,
        "owner_username": owner.username,
        "venture_name": venture.name,
        "baseline_run_id": baseline_run.id,
        "rag_run_id": rag_run.id,
        "comparison_id": comparison.id if comparison else None,
    }


def run_baseline_rag_benchmark(
    *,
    test_set_path: Path,
    output_dir: Path,
    mode: str = "both",
    tf_limit: int | None = None,
    skip_scenario: bool = False,
    temperature: float = 0.1,
    top_k: int = 5,
    max_tokens_tf: int = 220,
    max_tokens_scenario: int = DEFAULT_ANSWER_MAX_TOKENS,
    generation_model: str | None = None,
    persist_db: bool = False,
    owner_username: str = "benchmark_runner",
    venture_name: str = "Benchmark Venture",
) -> dict[str, Any]:
    """Execute one benchmark run and export one report pair (json + md)."""

    output_dir.mkdir(parents=True, exist_ok=True)
    test_set = load_test_set(test_set_path)
    tf_items: list[dict[str, Any]] = test_set.get("true_false", [])
    if tf_limit is not None:
        tf_items = tf_items[: max(0, tf_limit)]
    scenario_item = test_set.get("scenario")

    if not tf_items:
        raise CommandError("No true/false items found in test set.")

    effective_generation_model = generation_model or getattr(settings, "LOCAL_LLM_MODEL", "qwen2.5:7b")

    system_prompt = (
        "You are a German startup decision-support assistant. "
        "Be precise, avoid speculation, and cite evidence."
    )

    tf_results: list[dict[str, Any]] = []
    for item in tf_items:
        statement = item["statement"]
        gold = str(item["gold_label"]).upper()
        prompt = tf_prompt(statement)

        baseline_result = ModeResult(
            mode="baseline",
            response="",
            predicted_label="SKIPPED",
            is_correct=False,
            citation_present=False,
            latency_ms=0,
            retrieval_latency_ms=0,
            retrieval_mode="none",
            citation_count=0,
            error="skipped",
        )
        rag_result = ModeResult(
            mode="rag",
            response="",
            predicted_label="SKIPPED",
            is_correct=False,
            citation_present=False,
            latency_ms=0,
            retrieval_latency_ms=0,
            retrieval_mode="none",
            citation_count=0,
            error="skipped",
        )

        if mode in {"both", "baseline"}:
            baseline_result = run_baseline(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens_tf,
                generation_model=effective_generation_model,
                expected_label=gold,
            )
        if mode in {"both", "rag"}:
            rag_result = run_rag(
                prompt=prompt,
                domain=str(item.get("domain", "")),
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens_tf,
                top_k=top_k,
                generation_model=effective_generation_model,
                expected_label=gold,
            )

        tf_results.append(
            {
                "id": item["id"],
                "domain": item.get("domain", ""),
                "statement": statement,
                "gold_label": gold,
                "source": item.get("source", ""),
                "baseline": mode_result_to_dict(baseline_result),
                "rag": mode_result_to_dict(rag_result),
            }
        )

    scenario_result: dict[str, Any] | None = None
    if not skip_scenario and scenario_item:
        prompt = scenario_prompt(str(scenario_item.get("prompt", "")))
        baseline_scenario = ModeResult(
            mode="baseline",
            response="",
            predicted_label="N/A",
            is_correct=False,
            citation_present=False,
            latency_ms=0,
            retrieval_latency_ms=0,
            retrieval_mode="none",
            citation_count=0,
            error="skipped",
        )
        rag_scenario = ModeResult(
            mode="rag",
            response="",
            predicted_label="N/A",
            is_correct=False,
            citation_present=False,
            latency_ms=0,
            retrieval_latency_ms=0,
            retrieval_mode="none",
            citation_count=0,
            error="skipped",
        )

        if mode in {"both", "baseline"}:
            baseline_scenario = run_baseline(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens_scenario,
                generation_model=effective_generation_model,
            )
        if mode in {"both", "rag"}:
            rag_scenario = run_rag(
                prompt=prompt,
                domain=str(scenario_item.get("domain", "")),
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens_scenario,
                top_k=top_k,
                generation_model=effective_generation_model,
            )

        scenario_result = {
            "id": scenario_item.get("id", ""),
            "title": scenario_item.get("title", ""),
            "baseline": mode_result_to_dict(baseline_scenario),
            "rag": mode_result_to_dict(rag_scenario),
        }

    baseline_tf_summary = aggregate_tf_metrics(tf_results, "baseline")
    rag_tf_summary = aggregate_tf_metrics(tf_results, "rag")

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "config": {
            "test_set_path": str(test_set_path),
            "output_dir": str(output_dir),
            "generation_model": effective_generation_model,
            "llm_model": effective_generation_model,
            "rag_embedding_model": getattr(settings, "LOCAL_RAG_EMBED_MODEL", "bge-m3"),
            "mode": mode,
            "temperature": temperature,
            "top_k": top_k,
            "max_tokens_tf": max_tokens_tf,
            "max_tokens_scenario": max_tokens_scenario,
            "tf_count": len(tf_results),
            "scenario_included": bool(scenario_result),
        },
        "summary": {
            "baseline_tf": baseline_tf_summary,
            "rag_tf": rag_tf_summary,
        },
        "tf_results": tf_results,
        "scenario_result": scenario_result,
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    json_path = output_dir / f"benchmark_{timestamp}.json"
    md_path = output_dir / f"benchmark_{timestamp}.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(build_markdown_report(report), encoding="utf-8")

    db_info = {"persisted": False}
    if persist_db:
        db_info = _persist_report_to_db(
            report=report,
            owner_username=owner_username,
            venture_name=venture_name,
        )
        report["db_persistence"] = db_info
        md_path.write_text(build_markdown_report(report), encoding="utf-8")
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "report": report,
        "json_path": str(json_path),
        "md_path": str(md_path),
        "db_persistence": db_info,
    }


def run_baseline_rag_benchmark_batch(
    *,
    repeats: int,
    test_set_path: Path,
    output_dir: Path,
    mode: str = "both",
    tf_limit: int | None = None,
    skip_scenario: bool = False,
    temperature: float = 0.1,
    top_k: int = 5,
    max_tokens_tf: int = 220,
    max_tokens_scenario: int = DEFAULT_ANSWER_MAX_TOKENS,
    generation_model: str | None = None,
    persist_db: bool = False,
    owner_username: str = "benchmark_runner",
    venture_name: str = "Benchmark Venture",
) -> dict[str, Any]:
    """Execute repeated benchmark runs and export an aggregate batch report."""

    if repeats < 1:
        raise CommandError("repeats must be >= 1")

    output_dir.mkdir(parents=True, exist_ok=True)
    run_outputs: list[dict[str, Any]] = []
    baseline_accuracies: list[float] = []
    rag_accuracies: list[float] = []
    baseline_citation_rates: list[float] = []
    rag_citation_rates: list[float] = []
    rag_better_count = 0
    baseline_better_count = 0
    tie_count = 0

    for run_index in range(1, repeats + 1):
        result = run_baseline_rag_benchmark(
            test_set_path=test_set_path,
            output_dir=output_dir,
            mode=mode,
            tf_limit=tf_limit,
            skip_scenario=skip_scenario,
            temperature=temperature,
            top_k=top_k,
            max_tokens_tf=max_tokens_tf,
            max_tokens_scenario=max_tokens_scenario,
            generation_model=generation_model,
            persist_db=persist_db,
            owner_username=owner_username,
            venture_name=venture_name,
        )
        summary = result["report"]["summary"]
        baseline_acc = float(summary["baseline_tf"]["accuracy"])
        rag_acc = float(summary["rag_tf"]["accuracy"])
        baseline_citation = float(summary["baseline_tf"]["citation_rate"])
        rag_citation = float(summary["rag_tf"]["citation_rate"])

        baseline_accuracies.append(baseline_acc)
        rag_accuracies.append(rag_acc)
        baseline_citation_rates.append(baseline_citation)
        rag_citation_rates.append(rag_citation)

        if rag_acc > baseline_acc:
            rag_better_count += 1
        elif baseline_acc > rag_acc:
            baseline_better_count += 1
        else:
            tie_count += 1

        run_outputs.append(
            {
                "run_index": run_index,
                "json_path": result["json_path"],
                "md_path": result["md_path"],
                "db_persistence": result.get("db_persistence", {"persisted": False}),
                "baseline_accuracy": baseline_acc,
                "rag_accuracy": rag_acc,
                "baseline_citation_rate": baseline_citation,
                "rag_citation_rate": rag_citation,
                "summary": summary,
            }
        )

    batch_report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "repeats": repeats,
        "config": {
            "test_set_path": str(test_set_path),
            "output_dir": str(output_dir),
            "mode": mode,
            "tf_limit": tf_limit,
            "skip_scenario": skip_scenario,
            "temperature": temperature,
            "top_k": top_k,
            "max_tokens_tf": max_tokens_tf,
            "max_tokens_scenario": max_tokens_scenario,
            "generation_model": generation_model or getattr(settings, "LOCAL_LLM_MODEL", "qwen2.5:7b"),
            "persist_db": persist_db,
        },
        "aggregate": {
            "baseline_accuracy": _mean_std(baseline_accuracies),
            "rag_accuracy": _mean_std(rag_accuracies),
            "baseline_citation_rate": _mean_std(baseline_citation_rates),
            "rag_citation_rate": _mean_std(rag_citation_rates),
            "rag_better_count": rag_better_count,
            "baseline_better_count": baseline_better_count,
            "tie_count": tie_count,
        },
        "runs": run_outputs,
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    batch_json_path = output_dir / f"benchmark_batch_{timestamp}.json"
    batch_md_path = output_dir / f"benchmark_batch_{timestamp}.md"
    batch_json_path.write_text(json.dumps(batch_report, ensure_ascii=False, indent=2), encoding="utf-8")
    batch_md_path.write_text(build_batch_markdown_report(batch_report), encoding="utf-8")

    return {
        "batch_report": batch_report,
        "batch_json_path": str(batch_json_path),
        "batch_md_path": str(batch_md_path),
    }
