"""Build a three-arm comparison for the formal reranker ablation."""

from __future__ import annotations

import hashlib
import json
import random
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from apps.evaluations.formal_codex_comparison import (
    _bootstrap_mean_ci,
    _generation_max_tokens,
)


ARMS = ("baseline", "dense5", "rerank")
METRICS = ("completeness", "hallucination", "irrelevancy")
CONTRASTS = (
    ("rerank_minus_dense5", "rerank", "dense5"),
    ("rerank_minus_baseline", "rerank", "baseline"),
)


class FormalRerankComparisonError(RuntimeError):
    """Raised when the three Judge runs cannot be compared safely."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FormalRerankComparisonError(f"Cannot load JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise FormalRerankComparisonError(f"Expected JSON object in {path}")
    return payload


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _round(value: float) -> float:
    return round(float(value), 4)


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return _round(statistics.mean(materialized)) if materialized else 0.0


def _validate_and_index(
    scores: dict[str, Any],
    manifest: dict[str, Any],
    *,
    expected_mode: str,
    label: str,
) -> dict[tuple[str, str], dict[str, Any]]:
    if scores.get("status") != "completed" or manifest.get("status") != "completed":
        raise FormalRerankComparisonError(f"{label} Judge run is not completed")
    results = scores.get("results", [])
    if not isinstance(results, list):
        raise FormalRerankComparisonError(f"{label} results must be a list")
    if len(results) != int(scores.get("result_count", -1)):
        raise FormalRerankComparisonError(f"{label} score count is inconsistent")
    if len(results) != int(manifest.get("answer_count", -1)):
        raise FormalRerankComparisonError(f"{label} manifest count is inconsistent")
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for item in results:
        if item.get("mode") != expected_mode:
            raise FormalRerankComparisonError(
                f"{label} contains mode {item.get('mode')!r}, expected {expected_mode!r}"
            )
        key = (str(item.get("model", "")), str(item.get("qra_id", "")))
        if key in index:
            raise FormalRerankComparisonError(f"Duplicate {label} score for {key}")
        index[key] = item
    return index


def _paired_panel(
    indexes: dict[str, dict[tuple[str, str], dict[str, Any]]],
) -> list[dict[str, Any]]:
    keys = sorted(indexes["rerank"])
    if not keys:
        raise FormalRerankComparisonError("Rerank Judge run is empty")
    models = {key[0] for key in keys}
    qras = {key[1] for key in keys}
    if len(keys) != len(models) * len(qras):
        raise FormalRerankComparisonError("Rerank scores are not a complete model/QRA panel")
    missing = {
        arm: [key for key in keys if key not in indexes[arm]]
        for arm in ("baseline", "dense5")
    }
    if any(missing.values()):
        raise FormalRerankComparisonError(f"Control scores are missing panel keys: {missing}")

    panel = []
    for key in keys:
        source_items = {arm: indexes[arm][key] for arm in ARMS}
        structural = {
            (
                str(item.get("domain", "")),
                str(item.get("question_type", "")),
                int(item.get("keypoint_count", -1)),
            )
            for item in source_items.values()
        }
        if len(structural) != 1:
            raise FormalRerankComparisonError(
                f"QRA metadata or keypoint count differs across arms for {key}"
            )
        domain, question_type, keypoint_count = structural.pop()
        panel.append(
            {
                "model": key[0],
                "model_type": str(source_items["rerank"].get("model_type", "")),
                "qra_id": key[1],
                "domain": domain,
                "question_type": question_type,
                "keypoint_count": keypoint_count,
                "arms": {
                    arm: {
                        "candidate_id": str(source_items[arm].get("candidate_id", "")),
                        **{
                            metric: float(source_items[arm][metric])
                            for metric in METRICS
                        },
                        **{
                            label: int(source_items[arm][label])
                            for label in ("covered", "contradicted", "neither")
                        },
                    }
                    for arm in ARMS
                },
            }
        )
    return panel


def _arm_summary(items: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    keypoints = sum(int(item["keypoint_count"]) for item in items)
    counts = {
        label: sum(int(item["arms"][arm][label]) for item in items)
        for label in ("covered", "contradicted", "neither")
    }
    return {
        "answer_count": len(items),
        "macro": {
            metric: _mean(item["arms"][arm][metric] for item in items)
            for metric in METRICS
        },
        "pooled_keypoints": {
            "total": keypoints,
            **counts,
            "covered_rate": _round(counts["covered"] / keypoints),
            "contradicted_rate": _round(counts["contradicted"] / keypoints),
            "neither_rate": _round(counts["neither"] / keypoints),
        },
    }


def _contrast_summary(
    items: list[dict[str, Any]],
    *,
    newer: str,
    older: str,
    bootstrap_iterations: int,
    rng: random.Random,
) -> dict[str, Any]:
    deltas = {
        metric: [
            float(item["arms"][newer][metric])
            - float(item["arms"][older][metric])
            for item in items
        ]
        for metric in METRICS
    }
    completeness = deltas["completeness"]
    epsilon = 1e-9
    return {
        "metric_deltas": {
            metric: {
                "mean": _mean(values),
                "bootstrap_95_ci": _bootstrap_mean_ci(
                    values,
                    iterations=bootstrap_iterations,
                    rng=rng,
                )
                if bootstrap_iterations
                else [],
            }
            for metric, values in deltas.items()
        },
        "completeness_win_tie_loss": {
            "win": sum(value > epsilon for value in completeness),
            "tie": sum(abs(value) <= epsilon for value in completeness),
            "loss": sum(value < -epsilon for value in completeness),
        },
    }


def _summary(
    items: list[dict[str, Any]],
    *,
    bootstrap_iterations: int,
    rng: random.Random,
) -> dict[str, Any]:
    return {
        "answer_count": len(items),
        "arms": {arm: _arm_summary(items, arm) for arm in ARMS},
        "contrasts": {
            name: _contrast_summary(
                items,
                newer=newer,
                older=older,
                bootstrap_iterations=bootstrap_iterations,
                rng=rng,
            )
            for name, newer, older in CONTRASTS
        },
    }


def _grouped_summaries(
    panel: list[dict[str, Any]],
    field: str,
    *,
    bootstrap_iterations: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    return [
        {
            field: value,
            **_summary(
                [item for item in panel if item[field] == value],
                bootstrap_iterations=bootstrap_iterations,
                rng=rng,
            ),
        }
        for value in sorted({str(item[field]) for item in panel})
    ]


def _generation_health(
    manifest: dict[str, Any], manifest_path: Path
) -> list[dict[str, Any]]:
    rows = []
    for source in manifest.get("source_runs", []):
        path_value = str(source.get("path", ""))
        path = None
        for parent in manifest_path.resolve().parents:
            candidate = parent / path_value
            if candidate.is_file():
                path = candidate
                break
        if path is None:
            raise FormalRerankComparisonError(
                f"Cannot resolve generation source {path_value}"
            )
        payload = _load_json(path)
        results = payload.get("results", [])
        successful = [
            item
            for item in results
            if item.get("generation", {}).get("status") == "success"
        ]
        prior = payload.get("prior_attempts", [])
        retried_qras = sorted({str(item.get("qra_id", "")) for item in prior})
        latencies = [
            float(item["generation"].get("latency_ms", 0)) / 1000
            for item in successful
        ]
        retrieval_modes = {
            str(item["generation"].get("retrieval_mode", ""))
            for item in successful
        }
        rows.append(
            {
                "run_id": str(payload.get("run_id", "")),
                "model": str(successful[0].get("model", "")) if successful else "",
                "final_success_count": len(successful),
                "final_error_count": len(results) - len(successful),
                "prior_attempt_count": len(prior),
                "retried_qra_count": len(retried_qras),
                "retried_qra_ids": retried_qras,
                "prior_timeout_count": sum(
                    "timed out" in str(item.get("generation", {}).get("error", ""))
                    for item in prior
                ),
                "prior_token_limit_count": sum(
                    "token limit" in str(item.get("generation", {}).get("error", ""))
                    for item in prior
                ),
                "mean_generation_seconds": _mean(latencies),
                "max_generation_seconds": _round(max(latencies)) if latencies else 0.0,
                "retrieval_modes": sorted(retrieval_modes),
                "keyword_fallback_count": sum(
                    "keyword" in str(item["generation"].get("retrieval_mode", ""))
                    for item in successful
                ),
                "temperature": float(payload.get("config", {}).get("temperature", 0)),
                "max_tokens": int(payload.get("config", {}).get("max_tokens", 0)),
                "resume_count": int(payload.get("resume", {}).get("resume_count", 0)),
                "source_path": path.as_posix(),
                "source_sha256": _file_sha256(path),
            }
        )
    return sorted(rows, key=lambda item: item["model"])


def _pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def _pp(value: float) -> str:
    return f"{100 * value:+.2f} pp"


def _markdown(report: dict[str, Any]) -> str:
    overall = report["overall"]
    lines = [
        "# 正式 12 题：七模型 rerank 三臂配对比较",
        "",
        f"- 报告 ID：`{report['report_id']}`",
        "- 条件：Baseline；domain-filtered Dense@5；"
        "domain-filtered Dense@20 → BGE rerank@5",
        f"- 配对面板：`{report['panel']['answer_count']}` 个回答（"
        f"{report['panel']['model_count']} 模型 × {report['panel']['qra_count']} QRA）",
        f"- Judge：`{report['judge']['requested_model']}` / "
        f"`{report['judge']['reasoning_effort']}`，盲评，"
        f"rubric `{report['judge']['prompt_version']}`",
        f"- Bootstrap：`{report['bootstrap']['iterations']}` 次，"
        f"seed `{report['bootstrap']['seed']}`",
        "",
        "## 主要结论",
        "",
    ]
    dense_delta = overall["contrasts"]["rerank_minus_dense5"]
    base_delta = overall["contrasts"]["rerank_minus_baseline"]
    lines.extend(
        [
            "- Rerank completeness 为 "
            f"{_pct(overall['arms']['rerank']['macro']['completeness'])}，"
            "相对 Dense@5 提升 "
            f"{_pp(dense_delta['metric_deltas']['completeness']['mean'])}；"
            "95% CI 为 "
            f"[{_pp(dense_delta['metric_deltas']['completeness']['bootstrap_95_ci'][0])}, "
            f"{_pp(dense_delta['metric_deltas']['completeness']['bootstrap_95_ci'][1])}]。",
            "- 相对 Baseline，rerank completeness 提升 "
            f"{_pp(base_delta['metric_deltas']['completeness']['mean'])}；"
            "95% CI 为 "
            f"[{_pp(base_delta['metric_deltas']['completeness']['bootstrap_95_ci'][0])}, "
            f"{_pp(base_delta['metric_deltas']['completeness']['bootstrap_95_ci'][1])}]。",
            "- 与 Dense@5 相比，rerank 的 contradiction/hallucination 比例变化为 "
            f"{_pp(dense_delta['metric_deltas']['hallucination']['mean'])}；"
            "因此本轮表现是覆盖率提高，但安全性指标有所回退。",
            "- 这是预选 12 题的探索性消融，不应替代新的 held-out 正式测试。",
            "",
            "## 总体结果",
            "",
            "| 条件 | Completeness | Hallucination | Irrelevancy | Covered / Contradicted / Neither |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for arm, label in (
        ("baseline", "Baseline"),
        ("dense5", "Dense@5"),
        ("rerank", "Dense@20 → rerank@5"),
    ):
        arm_summary = overall["arms"][arm]
        pooled = arm_summary["pooled_keypoints"]
        lines.append(
            f"| {label} | {_pct(arm_summary['macro']['completeness'])} | "
            f"{_pct(arm_summary['macro']['hallucination'])} | "
            f"{_pct(arm_summary['macro']['irrelevancy'])} | "
            f"{pooled['covered']} / {pooled['contradicted']} / {pooled['neither']} |"
        )
    lines.extend(
        [
            "",
            "| 配对差值 | ΔCompleteness [95% CI] | ΔHallucination | 胜 / 平 / 负 |",
            "|---|---:|---:|---:|",
        ]
    )
    for key, label in (
        ("rerank_minus_dense5", "Rerank − Dense@5"),
        ("rerank_minus_baseline", "Rerank − Baseline"),
    ):
        contrast = overall["contrasts"][key]
        completeness = contrast["metric_deltas"]["completeness"]
        wtl = contrast["completeness_win_tie_loss"]
        lines.append(
            f"| {label} | {_pp(completeness['mean'])} "
            f"[{_pp(completeness['bootstrap_95_ci'][0])}, "
            f"{_pp(completeness['bootstrap_95_ci'][1])}] | "
            f"{_pp(contrast['metric_deltas']['hallucination']['mean'])} | "
            f"{wtl['win']} / {wtl['tie']} / {wtl['loss']} |"
        )
    lines.extend(
        [
            "",
            "## 各模型",
            "",
            "| 模型 | Baseline C | Dense@5 C | Rerank C | Rerank−Dense@5 | "
            "Rerank−Baseline | Rerank/Dense@5 胜平负 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["by_model"]:
        dense = row["contrasts"]["rerank_minus_dense5"]
        base = row["contrasts"]["rerank_minus_baseline"]
        wtl = dense["completeness_win_tie_loss"]
        lines.append(
            f"| `{row['model']}` | "
            f"{_pct(row['arms']['baseline']['macro']['completeness'])} | "
            f"{_pct(row['arms']['dense5']['macro']['completeness'])} | "
            f"{_pct(row['arms']['rerank']['macro']['completeness'])} | "
            f"{_pp(dense['metric_deltas']['completeness']['mean'])} | "
            f"{_pp(base['metric_deltas']['completeness']['mean'])} | "
            f"{wtl['win']}/{wtl['tie']}/{wtl['loss']} |"
        )
    lines.extend(
        [
            "",
            "## 按领域",
            "",
            "| 领域 | n | Baseline C | Dense@5 C | Rerank C | Rerank−Dense@5 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["by_domain"]:
        delta = row["contrasts"]["rerank_minus_dense5"]["metric_deltas"]
        lines.append(
            f"| {row['domain']} | {row['answer_count']} | "
            f"{_pct(row['arms']['baseline']['macro']['completeness'])} | "
            f"{_pct(row['arms']['dense5']['macro']['completeness'])} | "
            f"{_pct(row['arms']['rerank']['macro']['completeness'])} | "
            f"{_pp(delta['completeness']['mean'])} |"
        )
    lines.extend(
        [
            "",
            "## 生成运行健康",
            "",
            "| 模型 | 最终成功 | 重试题 | 历史失败尝试 | 历史超时 | 平均/最大生成秒 | 回退 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["generation_health"]:
        lines.append(
            f"| `{row['model']}` | {row['final_success_count']}/12 | "
            f"{row['retried_qra_count']} | {row['prior_attempt_count']} | "
            f"{row['prior_timeout_count']} | {row['mean_generation_seconds']:.1f}/"
            f"{row['max_generation_seconds']:.1f} | {row['keyword_fallback_count']} |"
        )
    mismatches = report["generation_parameter_check"]["baseline_mismatches"]
    lines.extend(
        [
            "",
            "## 可比性与限制",
            "",
            "- Dense@5 与 rerank 的 84 个配对全部使用相同 max_tokens；"
            "Kimi 为 8192，其余模型为 900。",
            f"- Baseline 有 `{len(mismatches)}` 个 max_tokens 不一致配对："
            + ", ".join(
                f"`{item['model']}/{item['qra_id']}` "
                f"({item['control_max_tokens']} vs {item['rerank_max_tokens']})"
                for item in mismatches
            )
            + "。应优先把 Rerank−Dense@5 视为本次 reranker 的直接效应。",
            "- 排除上述 1 个 Baseline 参数不一致项后，Rerank−Baseline "
            "completeness 为 "
            f"{_pp(report['generation_parameter_check']['matched_sensitivity']['rerank_minus_baseline']['metric_deltas']['completeness']['mean'])}。",
            "- Judge 只按冻结 keypoints 判定 covered/contradicted/neither；"
            "hallucination 不等价于开放式事实核查。",
            "- OLMo 的 `FTLFS-F-001` 最终一次生成使用 600 秒连接上限并在 554 秒完成；"
            "其余标准尝试使用 360 秒。该例外只影响可用率，不改变检索或生成采样参数。",
            "- Kimi 的 900-token 预跑被判定截断并停止；正式 rerank run 使用与旧 Kimi "
            "Dense@5 相同的 8192 上限。未完成的预跑不进入 Judge。",
            "",
            "完整逐题三臂配对、输入哈希与 bootstrap 数据见同目录 JSON。",
            "",
        ]
    )
    return "\n".join(lines)


def build_formal_rerank_comparison(
    *,
    baseline_scores_path: Path,
    baseline_manifest_path: Path,
    dense5_scores_path: Path,
    dense5_manifest_path: Path,
    rerank_scores_path: Path,
    rerank_manifest_path: Path,
    output_dir: Path,
    report_id: str,
    bootstrap_iterations: int = 10_000,
    seed: int = 20260811,
) -> dict[str, Path]:
    """Validate, pair, aggregate, and export a three-arm reranker report."""

    if bootstrap_iterations < 1:
        raise FormalRerankComparisonError("bootstrap_iterations must be positive")
    paths = {
        "baseline_scores": baseline_scores_path.resolve(),
        "baseline_manifest": baseline_manifest_path.resolve(),
        "dense5_scores": dense5_scores_path.resolve(),
        "dense5_manifest": dense5_manifest_path.resolve(),
        "rerank_scores": rerank_scores_path.resolve(),
        "rerank_manifest": rerank_manifest_path.resolve(),
    }
    payloads = {name: _load_json(path) for name, path in paths.items()}
    manifests = {
        arm: payloads[f"{arm}_manifest"] for arm in ("baseline", "dense5", "rerank")
    }
    judges = {json.dumps(item.get("judge", {}), sort_keys=True) for item in manifests.values()}
    if len(judges) != 1:
        raise FormalRerankComparisonError("Judge configurations differ across arms")
    indexes = {
        "baseline": _validate_and_index(
            payloads["baseline_scores"],
            manifests["baseline"],
            expected_mode="baseline",
            label="Baseline",
        ),
        "dense5": _validate_and_index(
            payloads["dense5_scores"],
            manifests["dense5"],
            expected_mode="rag",
            label="Dense@5",
        ),
        "rerank": _validate_and_index(
            payloads["rerank_scores"],
            manifests["rerank"],
            expected_mode="rag",
            label="Rerank",
        ),
    }
    panel = _paired_panel(indexes)
    panel_keys = {(item["model"], item["qra_id"]) for item in panel}
    token_provenance = {
        arm: _generation_max_tokens(manifests[arm], paths[f"{arm}_manifest"])
        for arm in ARMS
    }
    if any(not panel_keys.issubset(values) for values in token_provenance.values()):
        raise FormalRerankComparisonError("Generation provenance misses panel keys")

    def mismatches(control: str) -> list[dict[str, Any]]:
        return [
            {
                "model": model,
                "qra_id": qra_id,
                "control_max_tokens": token_provenance[control][(model, qra_id)][
                    "max_tokens"
                ],
                "rerank_max_tokens": token_provenance["rerank"][(model, qra_id)][
                    "max_tokens"
                ],
            }
            for model, qra_id in sorted(panel_keys)
            if token_provenance[control][(model, qra_id)]["max_tokens"]
            != token_provenance["rerank"][(model, qra_id)]["max_tokens"]
        ]

    rng = random.Random(seed)
    generated_at = datetime.now(UTC).isoformat()
    report = {
        "report_id": report_id,
        "status": "completed",
        "evaluation_status": "formal_multi_judge_sensitivity_pending_validation",
        "generated_at": generated_at,
        "judge": manifests["rerank"]["judge"],
        "bootstrap": {
            "iterations": bootstrap_iterations,
            "seed": seed,
            "unit": "paired model/QRA answer",
        },
        "panel": {
            "answer_count": len(panel),
            "model_count": len({item["model"] for item in panel}),
            "qra_count": len({item["qra_id"] for item in panel}),
            "models": sorted({item["model"] for item in panel}),
            "qra_ids": sorted({item["qra_id"] for item in panel}),
        },
        "conditions": {
            "baseline": "No retrieved context",
            "dense5": "QRA domain filter -> Dense@5",
            "rerank": "QRA domain filter -> Dense@20 -> BGE rerank@5",
        },
        "overall": _summary(
            panel,
            bootstrap_iterations=bootstrap_iterations,
            rng=rng,
        ),
        "by_model": _grouped_summaries(
            panel,
            "model",
            bootstrap_iterations=bootstrap_iterations,
            rng=rng,
        ),
        "by_domain": _grouped_summaries(
            panel,
            "domain",
            bootstrap_iterations=0,
            rng=rng,
        ),
        "generation_parameter_check": {
            "dense5_mismatches": mismatches("dense5"),
            "baseline_mismatches": mismatches("baseline"),
        },
        "generation_health": _generation_health(
            manifests["rerank"], paths["rerank_manifest"]
        ),
        "paired_results": panel,
        "inputs": {
            name: {"path": path.as_posix(), "sha256": _file_sha256(path)}
            for name, path in paths.items()
        },
    }
    for contrast_name, newer, older, control in (
        ("rerank_minus_dense5", "rerank", "dense5", "dense5"),
        ("rerank_minus_baseline", "rerank", "baseline", "baseline"),
    ):
        excluded = {
            (item["model"], item["qra_id"])
            for item in report["generation_parameter_check"][f"{control}_mismatches"]
        }
        retained = [
            item
            for item in panel
            if (item["model"], item["qra_id"]) not in excluded
        ]
        report["generation_parameter_check"].setdefault(
            "matched_sensitivity", {}
        )[contrast_name] = {
            "excluded_pair_count": len(excluded),
            "retained_pair_count": len(retained),
            **_contrast_summary(
                retained,
                newer=newer,
                older=older,
                bootstrap_iterations=bootstrap_iterations,
                rng=rng,
            ),
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{report_id}.json"
    markdown_path = output_dir / f"{report_id}.md"
    _write_json(json_path, report)
    temporary = markdown_path.with_suffix(markdown_path.suffix + ".tmp")
    temporary.write_text(_markdown(report), encoding="utf-8")
    temporary.replace(markdown_path)
    return {"json": json_path, "markdown": markdown_path}
