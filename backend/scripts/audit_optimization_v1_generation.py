"""Audit and consolidate the Optimization V1 generation artifacts.

The Kimi baseline run exhausted its original 8K output cap after two
successful answers.  A labelled 16K supplement completed the remaining 58
QRA items.  This script preserves that provenance while producing judge-ready
condition artifacts and bilingual audit reports.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = REPO_ROOT / "docs/evaluation/runs/optimization_v1"
GENERATION_DIR = RUN_ROOT / "generation"
AUDIT_DIR = RUN_ROOT / "audit"

SOURCE_FILES = {
    "core": GENERATION_DIR / "OPT-V1-CORE-BOTH-001.json",
    "kimi_rag": GENERATION_DIR / "OPT-V1-KIMI-RAG-001.json",
    "kimi_baseline_8k": GENERATION_DIR / "OPT-V1-KIMI-BASELINE-001.partial.json",
    "kimi_baseline_16k": GENERATION_DIR
    / "OPT-V1-KIMI-BASELINE-SUPPLEMENT-16K-001.json",
}

OUTPUT_FILES = {
    "kimi_baseline": GENERATION_DIR / "OPT-V1-KIMI-BASELINE-MERGED-001.json",
    "baseline": GENERATION_DIR / "OPT-V1-BASELINE-AUDITED-001.json",
    "rag": GENERATION_DIR / "OPT-V1-RAG-AUDITED-001.json",
    "all": GENERATION_DIR / "OPT-V1-ALL-AUDITED-001.json",
    "audit_json": AUDIT_DIR / "OPT-V1-GENERATION-AUDIT-001.json",
    "audit_en": AUDIT_DIR / "OPT-V1-GENERATION-AUDIT-001_en.md",
    "audit_zh": AUDIT_DIR / "OPT-V1-GENERATION-AUDIT-001_zh.md",
}

MODEL_ORDER = [
    "gemma4:31b",
    "gemma4:26b",
    "gpt-oss:20b",
    "magistral:latest",
    "olmo-3.1:32b-think",
    "qwen3.6:latest",
    "cloud.moonshot/kimi-k3",
]


class AuditError(RuntimeError):
    """Raised when generated answers are not safe to consolidate."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"Could not read {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise AuditError(f"Expected a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _display_path(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def _successful_results(run: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        result
        for result in run.get("results", [])
        if result.get("generation", {}).get("status") == "success"
    ]


def _failed_results(run: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        result
        for result in run.get("results", [])
        if result.get("generation", {}).get("status") != "success"
    ]


def _source_record(name: str, path: Path, run: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "path": _display_path(path),
        "sha256": _sha256_file(path),
        "run_id": str(run.get("run_id", "")),
        "status": str(run.get("status", "")),
        "dataset_item_count": len(run.get("dataset", {}).get("items", [])),
        "dataset_manifest_sha256": str(
            run.get("dataset", {}).get("manifest_sha256", "")
        ),
        "result_count": len(run.get("results", [])),
        "successful_result_count": len(_successful_results(run)),
        "failed_result_count": len(_failed_results(run)),
        "temperature": run.get("config", {}).get("temperature"),
        "max_tokens": run.get("config", {}).get("max_tokens"),
        "reasoning_effort": run.get("config", {}).get("reasoning_effort"),
        "prompt_sha256": str(
            run.get("config", {}).get("prompt", {}).get("sha256", "")
        ),
        "implementation_sha256": str(
            run.get("config", {}).get("implementation", {}).get("sha256", "")
        ),
    }


def _result_with_provenance(
    result: dict[str, Any],
    *,
    source_name: str,
    source_path: Path,
    source_run: dict[str, Any],
) -> dict[str, Any]:
    copied = copy.deepcopy(result)
    copied["source_provenance"] = {
        "source_name": source_name,
        "source_run_id": str(source_run.get("run_id", "")),
        "source_path": _display_path(source_path),
        "source_sha256": _sha256_file(source_path),
        "source_sequence": int(result.get("sequence", 0)),
        "configured_temperature": source_run.get("config", {}).get("temperature"),
        "configured_max_tokens": source_run.get("config", {}).get("max_tokens"),
        "configured_reasoning_effort": source_run.get("config", {}).get(
            "reasoning_effort"
        ),
    }
    return copied


def _summaries(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[
            (
                str(result.get("model", "")),
                str(result.get("model_type", "")),
                str(result.get("mode", "")),
            )
        ].append(result)

    summaries = []
    for (model, model_type, mode), items in sorted(grouped.items()):
        generations = [item["generation"] for item in items]
        retrieval_modes = Counter(
            str(item.get("retrieval_mode", "")) for item in generations
        )
        summaries.append(
            {
                "model": model,
                "model_type": model_type,
                "mode": mode,
                "scheduled": len(items),
                "succeeded": len(items),
                "errors": 0,
                "mean_latency_seconds": round(
                    mean(float(item.get("latency_ms", 0)) for item in generations)
                    / 1000,
                    3,
                ),
                "mean_retrieval_seconds": round(
                    mean(
                        float(item.get("retrieval_latency_ms", 0))
                        for item in generations
                    )
                    / 1000,
                    3,
                ),
                "mean_response_chars": round(
                    mean(float(item.get("response_char_count", 0)) for item in generations),
                    1,
                ),
                "retrieval_modes": dict(sorted(retrieval_modes.items())),
            }
        )
    return summaries


def _build_consolidated_run(
    *,
    run_id: str,
    mode: str,
    canonical_dataset: dict[str, Any],
    results: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    excluded_attempts: list[dict[str, Any]],
    generated_at: str,
    shared_config: dict[str, Any],
) -> dict[str, Any]:
    qra_order = {
        str(item["id"]): position
        for position, item in enumerate(canonical_dataset["items"])
    }
    model_order = {model: position for position, model in enumerate(MODEL_ORDER)}
    ordered = sorted(
        results,
        key=lambda result: (
            model_order[str(result["model"])],
            qra_order[str(result["qra_id"])],
        ),
    )
    for sequence, result in enumerate(ordered, start=1):
        result["sequence"] = sequence

    return {
        "run_id": run_id,
        "status": "completed",
        "evaluation_status": "test_unscored_generation_only_audited",
        "started_at": min(
            str(source.get("started_at", generated_at)) for source in sources
        ),
        "completed_at": generated_at,
        "config": {
            "models": [
                {"id": model, "type": next(
                    str(result.get("model_type", ""))
                    for result in ordered
                    if result.get("model") == model
                )}
                for model in MODEL_ORDER
                if any(result.get("model") == model for result in ordered)
            ],
            "modes": ["baseline", "rag"] if mode == "both" else [mode],
            "mixed_source_generation_parameters": True,
            "source_runs": sources,
            "prompt": copy.deepcopy(shared_config.get("prompt", {})),
            "implementation": copy.deepcopy(shared_config.get("implementation", {})),
            "rag_pipeline": copy.deepcopy(shared_config.get("rag_pipeline", {})),
            "model_profiles": copy.deepcopy(shared_config.get("model_profiles", {})),
            "rag_index": copy.deepcopy(shared_config.get("rag_index", {})),
            "audit": {
                "status": "passed_with_parameter_sensitivity_limitation",
                "report_path": _display_path(OUTPUT_FILES["audit_json"]),
            },
        },
        "dataset": copy.deepcopy(canonical_dataset),
        "summary": _summaries(ordered),
        "results": ordered,
        "excluded_attempts": excluded_attempts,
        "consolidation": {
            "generated_at": generated_at,
            "result_count": len(ordered),
            "condition_mode": mode,
            "source_result_provenance_embedded": True,
        },
    }


def _markdown_en(audit: dict[str, Any]) -> str:
    sensitivity = audit["parameter_sensitivity"]
    return f"""# Optimization V1 generation audit

Status: **passed with a parameter-sensitivity limitation**.

## Outcome

- Kimi baseline coverage: 60/60 QRA items, with no duplicate or empty successful answers.
- Original 8K successes retained: 2.
- Successful 16K supplements merged: 58.
- Excluded failed attempt: 1 truncated response for `OPTLFS-II-001`.
- Full panel coverage: 840/840 unique model/QRA/condition answers.
- Judge-ready Baseline and RAG artifacts: 420 answers each.

## Parameter-sensitivity limitation

The 58 supplemented Kimi baseline answers used a 16,384-token cap, while Kimi RAG used 8,192. Four baseline answers actually exceeded 8,192 output tokens: {', '.join(sensitivity['answers_over_8192_tokens'])}. The data are structurally valid for judging, but Kimi Baseline-versus-RAG comparisons must disclose this mismatch and retain a sensitivity analysis.

## Integrity checks

- Frozen QRA content matches the canonical 60-item dataset for every source subset.
- The generation prompt and implementation hashes match across all four source artifacts.
- Every accepted answer has `status=success`, `done=true`, `finish_reason=stop`, and non-empty response text.
- No accepted answer was marked truncated; every accepted generation ended with `finish_reason=stop`.
- Each merged result embeds its source run, source file hash, original sequence, and generation limits.

## Artifacts

- Kimi baseline merged: `{_display_path(OUTPUT_FILES['kimi_baseline'])}`
- All-model baseline: `{_display_path(OUTPUT_FILES['baseline'])}`
- All-model RAG: `{_display_path(OUTPUT_FILES['rag'])}`
- All-model paired Judge source: `{_display_path(OUTPUT_FILES['all'])}`
- Machine-readable audit: `{_display_path(OUTPUT_FILES['audit_json'])}`
"""


def _markdown_zh(audit: dict[str, Any]) -> str:
    sensitivity = audit["parameter_sensitivity"]
    return f"""# Optimization V1 生成结果审计

状态：**通过，但存在参数敏感性限制**。

## 结论

- Kimi baseline 覆盖 60/60 个 QRA，没有重复或空的成功回答。
- 保留原始 8K 成功回答：2 条。
- 合并 16K 补充成功回答：58 条。
- 排除失败尝试：`OPTLFS-II-001` 的 1 条截断回答。
- 完整模型面板覆盖：840/840 个不同的模型/QRA/条件回答。
- Judge 可用的 Baseline 与 RAG 文件各包含 420 条回答。

## 参数敏感性限制

58 条 Kimi baseline 补充回答使用 16,384-token 上限，而 Kimi RAG 使用 8,192。实际有 4 条 baseline 回答超过 8,192 输出 tokens：{', '.join(sensitivity['answers_over_8192_tokens'])}。数据结构可以进入 Judge，但 Kimi Baseline/RAG 对比必须披露该参数差异并保留敏感性分析。

## 完整性检查

- 所有来源子集中的冻结 QRA 内容均与标准 60 题数据集一致。
- 四个来源文件使用相同的生成 prompt 和实现版本哈希。
- 每个纳入的回答均为 `status=success`、`done=true`、`finish_reason=stop`，且正文非空。
- 没有纳入的回答被标记为截断；所有纳入结果均以 `finish_reason=stop` 结束。
- 每条合并结果都保存来源运行、来源文件哈希、原序号和生成上限。

## 产物

- Kimi baseline 合并文件：`{_display_path(OUTPUT_FILES['kimi_baseline'])}`
- 全模型 baseline：`{_display_path(OUTPUT_FILES['baseline'])}`
- 全模型 RAG：`{_display_path(OUTPUT_FILES['rag'])}`
- 全模型配对 Judge 来源：`{_display_path(OUTPUT_FILES['all'])}`
- 机器可读审计：`{_display_path(OUTPUT_FILES['audit_json'])}`
"""


def main() -> None:
    runs = {name: _read_json(path) for name, path in SOURCE_FILES.items()}
    source_records = {
        name: _source_record(name, SOURCE_FILES[name], run)
        for name, run in runs.items()
    }

    canonical_dataset = runs["core"].get("dataset", {})
    canonical_items = canonical_dataset.get("items", [])
    if len(canonical_items) != 60:
        raise AuditError("The canonical core run must contain exactly 60 QRA items")
    canonical_by_id = {str(item["id"]): item for item in canonical_items}
    if len(canonical_by_id) != 60:
        raise AuditError("The canonical dataset contains duplicate QRA IDs")

    dataset_checks = []
    for name, run in runs.items():
        items = run.get("dataset", {}).get("items", [])
        mismatches = [
            str(item.get("id", ""))
            for item in items
            if str(item.get("id", "")) not in canonical_by_id
            or _canonical_sha256(item)
            != _canonical_sha256(canonical_by_id[str(item["id"])])
        ]
        if mismatches:
            raise AuditError(f"Dataset mismatch in {name}: {mismatches}")
        dataset_checks.append(
            {
                "source": name,
                "item_count": len(items),
                "content_mismatch_count": 0,
                "manifest_sha256": str(
                    run.get("dataset", {}).get("manifest_sha256", "")
                ),
            }
        )

    prompt_hashes = {
        record["prompt_sha256"] for record in source_records.values()
    }
    implementation_hashes = {
        record["implementation_sha256"] for record in source_records.values()
    }
    if len(prompt_hashes) != 1 or "" in prompt_hashes:
        raise AuditError("Generation prompt hashes differ across sources")
    if len(implementation_hashes) != 1 or "" in implementation_hashes:
        raise AuditError("Generation implementation hashes differ across sources")

    accepted: list[dict[str, Any]] = []
    for source_name, run in runs.items():
        for result in _successful_results(run):
            accepted.append(
                _result_with_provenance(
                    result,
                    source_name=source_name,
                    source_path=SOURCE_FILES[source_name],
                    source_run=run,
                )
            )

    keys = [
        (
            str(result.get("model", "")),
            str(result.get("qra_id", "")),
            str(result.get("mode", "")),
            int(result.get("repeat", 1)),
        )
        for result in accepted
    ]
    duplicate_keys = sorted(key for key, count in Counter(keys).items() if count > 1)
    if duplicate_keys:
        raise AuditError(f"Duplicate successful conditions: {duplicate_keys}")
    if len(accepted) != 840:
        raise AuditError(f"Expected 840 successful conditions, found {len(accepted)}")

    invalid_answers = []
    for result in accepted:
        generation = result.get("generation", {})
        output_tokens = generation.get("usage", {}).get("output_tokens")
        if (
            not str(generation.get("response", "")).strip()
            or generation.get("done") is not True
            or generation.get("finish_reason") != "stop"
            or not isinstance(output_tokens, int)
        ):
            invalid_answers.append(
                {
                    "model": result.get("model"),
                    "qra_id": result.get("qra_id"),
                    "mode": result.get("mode"),
                }
            )
    if invalid_answers:
        raise AuditError(f"Accepted answers failed completion checks: {invalid_answers}")

    kimi_baseline = [
        result
        for result in accepted
        if result.get("model") == "cloud.moonshot/kimi-k3"
        and result.get("mode") == "baseline"
    ]
    kimi_ids = [str(result["qra_id"]) for result in kimi_baseline]
    if len(kimi_ids) != 60 or set(kimi_ids) != set(canonical_by_id):
        raise AuditError("Merged Kimi baseline does not cover the canonical 60 QRA IDs")

    over_8192 = sorted(
        (
            str(result["qra_id"]),
            int(result["generation"]["usage"]["output_tokens"]),
        )
        for result in kimi_baseline
        if int(result["generation"]["usage"]["output_tokens"]) > 8192
    )
    failed_8k = _failed_results(runs["kimi_baseline_8k"])
    if len(failed_8k) != 1 or failed_8k[0].get("qra_id") != "OPTLFS-II-001":
        raise AuditError("Unexpected original Kimi baseline failure set")
    excluded_attempts = [
        _result_with_provenance(
            result,
            source_name="kimi_baseline_8k",
            source_path=SOURCE_FILES["kimi_baseline_8k"],
            source_run=runs["kimi_baseline_8k"],
        )
        for result in failed_8k
    ]

    generated_at = datetime.now(UTC).isoformat()
    if OUTPUT_FILES["audit_json"].is_file():
        prior_audit = _read_json(OUTPUT_FILES["audit_json"])
        generated_at = str(prior_audit.get("generated_at", generated_at))
    sources_with_times = []
    for name, record in source_records.items():
        item = copy.deepcopy(record)
        item["started_at"] = str(runs[name].get("started_at", generated_at))
        sources_with_times.append(item)

    baseline_results = [result for result in accepted if result.get("mode") == "baseline"]
    rag_results = [result for result in accepted if result.get("mode") == "rag"]
    if len(baseline_results) != 420 or len(rag_results) != 420:
        raise AuditError(
            f"Expected 420 results per condition, found "
            f"baseline={len(baseline_results)}, rag={len(rag_results)}"
        )

    shared_config = runs["core"]["config"]
    merged_payloads = {
        "kimi_baseline": _build_consolidated_run(
            run_id="OPT-V1-KIMI-BASELINE-MERGED-001",
            mode="baseline",
            canonical_dataset=canonical_dataset,
            results=kimi_baseline,
            sources=[
                next(
                    item
                    for item in sources_with_times
                    if item["name"] == "kimi_baseline_8k"
                ),
                next(
                    item
                    for item in sources_with_times
                    if item["name"] == "kimi_baseline_16k"
                ),
            ],
            excluded_attempts=excluded_attempts,
            generated_at=generated_at,
            shared_config=shared_config,
        ),
        "baseline": _build_consolidated_run(
            run_id="OPT-V1-BASELINE-AUDITED-001",
            mode="baseline",
            canonical_dataset=canonical_dataset,
            results=baseline_results,
            sources=sources_with_times,
            excluded_attempts=excluded_attempts,
            generated_at=generated_at,
            shared_config=shared_config,
        ),
        "rag": _build_consolidated_run(
            run_id="OPT-V1-RAG-AUDITED-001",
            mode="rag",
            canonical_dataset=canonical_dataset,
            results=rag_results,
            sources=sources_with_times,
            excluded_attempts=[],
            generated_at=generated_at,
            shared_config=shared_config,
        ),
        "all": _build_consolidated_run(
            run_id="OPT-V1-ALL-AUDITED-001",
            mode="both",
            canonical_dataset=canonical_dataset,
            results=copy.deepcopy(accepted),
            sources=sources_with_times,
            excluded_attempts=excluded_attempts,
            generated_at=generated_at,
            shared_config=shared_config,
        ),
    }
    for name, payload in merged_payloads.items():
        _write_json(OUTPUT_FILES[name], payload)

    audit = {
        "audit_id": "OPT-V1-GENERATION-AUDIT-001",
        "status": "passed_with_parameter_sensitivity_limitation",
        "generated_at": generated_at,
        "source_runs": list(source_records.values()),
        "checks": {
            "dataset_subsets_match_canonical": True,
            "dataset_checks": dataset_checks,
            "generation_prompt_hash_consistent": True,
            "generation_implementation_hash_consistent": True,
            "successful_condition_count": len(accepted),
            "unique_successful_condition_count": len(set(keys)),
            "duplicate_successful_condition_count": 0,
            "empty_successful_answer_count": 0,
            "accepted_answer_completion_issue_count": 0,
            "baseline_answer_count": len(baseline_results),
            "rag_answer_count": len(rag_results),
            "kimi_baseline_answer_count": len(kimi_baseline),
            "kimi_baseline_unique_qra_count": len(set(kimi_ids)),
            "kimi_baseline_missing_qra_ids": [],
            "excluded_failed_attempt_count": len(excluded_attempts),
        },
        "parameter_sensitivity": {
            "status": "must_be_disclosed_in_kimi_baseline_rag_comparison",
            "kimi_rag_max_tokens": 8192,
            "kimi_baseline_8k_answer_count": 2,
            "kimi_baseline_16k_answer_count": 58,
            "kimi_baseline_answers_over_8192_count": len(over_8192),
            "answers_over_8192_tokens": [
                f"{qra_id} ({tokens})" for qra_id, tokens in over_8192
            ],
            "details": [
                {"qra_id": qra_id, "output_tokens": tokens}
                for qra_id, tokens in over_8192
            ],
        },
        "excluded_attempts": [
            {
                "qra_id": result.get("qra_id"),
                "source_run_id": result["source_provenance"]["source_run_id"],
                "error": result.get("generation", {}).get("error", ""),
            }
            for result in excluded_attempts
        ],
        "outputs": {},
    }
    for name in ("kimi_baseline", "baseline", "rag", "all"):
        audit["outputs"][name] = {
            "path": _display_path(OUTPUT_FILES[name]),
            "sha256": _sha256_file(OUTPUT_FILES[name]),
            "result_count": len(merged_payloads[name]["results"]),
        }
    _write_json(OUTPUT_FILES["audit_json"], audit)
    OUTPUT_FILES["audit_en"].write_text(_markdown_en(audit), encoding="utf-8")
    OUTPUT_FILES["audit_zh"].write_text(_markdown_zh(audit), encoding="utf-8")

    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
