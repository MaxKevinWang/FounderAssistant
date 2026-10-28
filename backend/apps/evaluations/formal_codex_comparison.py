"""Build a reproducible paired Baseline/RAG report from formal Codex scores."""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable


METRICS = ("completeness", "hallucination", "irrelevancy")


class FormalCodexComparisonError(RuntimeError):
    """Raised when two formal Judge runs cannot be paired safely."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FormalCodexComparisonError(f"Cannot load JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise FormalCodexComparisonError(f"Expected JSON object in {path}")
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


def _snapshot(values: Iterable[float]) -> dict[str, float]:
    materialized = [float(value) for value in values]
    if not materialized:
        return {"mean": 0.0, "median": 0.0, "sample_sd": 0.0}
    return {
        "mean": _mean(materialized),
        "median": _round(statistics.median(materialized)),
        "sample_sd": _round(statistics.stdev(materialized))
        if len(materialized) > 1
        else 0.0,
    }


def _percentile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        return 0.0
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def _bootstrap_mean_ci(
    values: list[float],
    *,
    iterations: int,
    rng: random.Random,
) -> list[float]:
    if not values or iterations < 1:
        return []
    size = len(values)
    means = sorted(
        sum(values[rng.randrange(size)] for _ in range(size)) / size
        for _ in range(iterations)
    )
    return [_round(_percentile(means, 0.025)), _round(_percentile(means, 0.975))]


def _condition_summary(items: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    total_keypoints = sum(int(item[mode]["keypoint_count"]) for item in items)
    return {
        "answer_count": len(items),
        **{
            metric: _snapshot(float(item[mode][metric]) for item in items)
            for metric in METRICS
        },
        "micro_keypoint_rates": {
            "completeness": _round(
                sum(int(item[mode]["covered"]) for item in items) / total_keypoints
            ),
            "hallucination": _round(
                sum(int(item[mode]["contradicted"]) for item in items)
                / total_keypoints
            ),
            "irrelevancy": _round(
                sum(int(item[mode]["neither"]) for item in items) / total_keypoints
            ),
            "keypoint_count": total_keypoints,
        },
    }


def _pair_summary(
    items: list[dict[str, Any]],
    *,
    bootstrap_iterations: int,
    rng: random.Random,
) -> dict[str, Any]:
    deltas: dict[str, dict[str, Any]] = {}
    for metric in METRICS:
        values = [float(item["rag_minus_baseline"][metric]) for item in items]
        deltas[metric] = _snapshot(values)
        if bootstrap_iterations:
            deltas[metric]["bootstrap_95_ci"] = _bootstrap_mean_ci(
                values,
                iterations=bootstrap_iterations,
                rng=rng,
            )

    completeness_deltas = [
        float(item["rag_minus_baseline"]["completeness"]) for item in items
    ]
    outcomes = {
        "rag_win": sum(value > 0 for value in completeness_deltas),
        "tie": sum(value == 0 for value in completeness_deltas),
        "rag_loss": sum(value < 0 for value in completeness_deltas),
    }
    joint = {"strict_benefit": 0, "strict_harm": 0, "mixed": 0, "exact_tie": 0}
    for item in items:
        completeness = float(item["rag_minus_baseline"]["completeness"])
        hallucination = float(item["rag_minus_baseline"]["hallucination"])
        if completeness == 0 and hallucination == 0:
            joint["exact_tie"] += 1
        elif (completeness > 0 and hallucination <= 0) or (
            completeness >= 0 and hallucination < 0
        ):
            joint["strict_benefit"] += 1
        elif (completeness < 0 and hallucination >= 0) or (
            completeness <= 0 and hallucination > 0
        ):
            joint["strict_harm"] += 1
        else:
            joint["mixed"] += 1

    return {
        "paired_answer_count": len(items),
        "baseline": _condition_summary(items, "baseline"),
        "rag": _condition_summary(items, "rag"),
        "rag_minus_baseline": deltas,
        "completeness_win_tie_loss": outcomes,
        "joint_completeness_hallucination_outcomes": joint,
    }


def _validate_run(
    scores: dict[str, Any],
    manifest: dict[str, Any],
    *,
    expected_mode: str,
    label: str,
) -> list[dict[str, Any]]:
    if scores.get("status") != "completed" or manifest.get("status") != "completed":
        raise FormalCodexComparisonError(f"{label} Judge run is not completed")
    results = scores.get("results", [])
    if len(results) != int(scores.get("result_count", -1)):
        raise FormalCodexComparisonError(f"{label} score result count is inconsistent")
    if len(results) != int(manifest.get("answer_count", -1)):
        raise FormalCodexComparisonError(f"{label} manifest answer count is inconsistent")
    modes = {str(item.get("mode", "")) for item in results}
    if modes != {expected_mode}:
        raise FormalCodexComparisonError(
            f"{label} expected only {expected_mode} scores, found {modes}"
        )
    return results


def _resolve_source_path(manifest_path: Path, value: str) -> Path | None:
    reference = Path(value)
    if reference.is_absolute() and reference.is_file():
        return reference.resolve()
    for parent in manifest_path.resolve().parents:
        candidate = parent / reference
        if candidate.is_file():
            return candidate.resolve()
    return None


def _generation_max_tokens(
    manifest: dict[str, Any],
    manifest_path: Path,
) -> dict[tuple[str, str], dict[str, Any]]:
    sources = manifest.get("source_runs", [])
    if not sources:
        return {}
    provenance: dict[tuple[str, str], dict[str, Any]] = {}
    for source in sources:
        path = _resolve_source_path(manifest_path, str(source.get("path", "")))
        if path is None:
            raise FormalCodexComparisonError(
                f"Cannot resolve generation source {source.get('path')} from {manifest_path}"
            )
        payload = _load_json(path)
        default_max_tokens = int(
            payload.get("config", {}).get("max_tokens", 0) or 0
        )
        for result in payload.get("results", []):
            if result.get("generation", {}).get("status") != "success":
                continue
            key = (str(result.get("model", "")), str(result.get("qra_id", "")))
            if key in provenance:
                raise FormalCodexComparisonError(
                    f"Duplicate successful generation provenance for {key}"
                )
            source_provenance = result.get("source_provenance", {})
            if not isinstance(source_provenance, dict):
                source_provenance = {}
            max_tokens = int(
                source_provenance.get(
                    "configured_max_tokens",
                    default_max_tokens,
                )
                or 0
            )
            provenance[key] = {
                "max_tokens": max_tokens,
                "source_run_id": str(
                    source_provenance.get(
                        "source_run_id",
                        payload.get("run_id", ""),
                    )
                ),
                "source_path": str(
                    source_provenance.get("source_path", path.as_posix())
                ),
            }
    return provenance


def _paired_results(
    baseline_results: list[dict[str, Any]],
    rag_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    def index(items: list[dict[str, Any]], label: str) -> dict[tuple[str, str], Any]:
        indexed: dict[tuple[str, str], Any] = {}
        for item in items:
            key = (str(item["model"]), str(item["qra_id"]))
            if key in indexed:
                raise FormalCodexComparisonError(f"Duplicate {label} score: {key}")
            indexed[key] = item
        return indexed

    baseline = index(baseline_results, "Baseline")
    rag = index(rag_results, "RAG")
    if set(baseline) != set(rag):
        missing_baseline = sorted(set(rag) - set(baseline))[:10]
        missing_rag = sorted(set(baseline) - set(rag))[:10]
        raise FormalCodexComparisonError(
            "Baseline/RAG score keys differ; "
            f"missing Baseline={missing_baseline}, missing RAG={missing_rag}"
        )

    pairs = []
    for model, qra_id in sorted(baseline):
        left, right = baseline[(model, qra_id)], rag[(model, qra_id)]
        for field in ("domain", "question_type", "model_type", "keypoint_count"):
            if left[field] != right[field]:
                raise FormalCodexComparisonError(
                    f"Paired metadata differs for {model}/{qra_id}/{field}"
                )
        compact = {}
        for mode, item in (("baseline", left), ("rag", right)):
            compact[mode] = {
                field: item[field]
                for field in (
                    "keypoint_count",
                    "covered",
                    "contradicted",
                    "neither",
                    *METRICS,
                )
            }
        pairs.append(
            {
                "model": model,
                "model_type": str(right["model_type"]),
                "qra_id": qra_id,
                "domain": str(right["domain"]),
                "question_type": str(right["question_type"]),
                **compact,
                "rag_minus_baseline": {
                    metric: _round(float(right[metric]) - float(left[metric]))
                    for metric in METRICS
                },
            }
        )
    return pairs


def _grouped_summary(
    pairs: list[dict[str, Any]],
    fields: tuple[str, ...],
    *,
    bootstrap_iterations: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for item in pairs:
        grouped[tuple(str(item[field]) for field in fields)].append(item)
    rows = []
    for key, items in sorted(grouped.items()):
        rows.append(
            {
                **dict(zip(fields, key, strict=True)),
                **_pair_summary(
                    items,
                    bootstrap_iterations=bootstrap_iterations,
                    rng=rng,
                ),
            }
        )
    return rows


def _macro_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        mode: {
            metric: _mean(row[mode][metric]["mean"] for row in rows)
            for metric in METRICS
        }
        for mode in ("baseline", "rag")
    } | {
        "rag_minus_baseline": {
            metric: _mean(
                row["rag_minus_baseline"][metric]["mean"] for row in rows
            )
            for metric in METRICS
        },
        "stratum_count": len(rows),
    }


def _fmt_percent(value: float, *, signed: bool = False) -> str:
    format_spec = "+.1%" if signed else ".1%"
    return format(float(value), format_spec)


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# 正式实验：Codex 5.6 Baseline 与 RAG 配对对比",
        "",
        f"- 报告 ID：`{report['report_id']}`",
        f"- Judge：`{report['judge']['requested_model']}` / "
        f"`{report['judge']['reasoning_effort']}`",
        f"- 配对回答：`{report['paired_answer_count']}`（"
        f"{report['model_count']} 个模型 × {report['qra_count']} 个 QRA）",
        f"- Bootstrap：`{report['bootstrap']['iterations']}` 次，"
        f"seed `{report['bootstrap']['seed']}`",
        "- 评价状态：`formal_multi_judge_sensitivity_pending_validation`",
        "",
        "## 主要结果",
        "",
        "| 模型 | 类型 | Baseline C | RAG C | ΔC [95% CI] | Baseline H | RAG H | ΔH | C 胜/平/负 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["model_summary"]:
        delta = row["rag_minus_baseline"]["completeness"]
        ci = delta["bootstrap_95_ci"]
        hallucination = row["rag_minus_baseline"]["hallucination"]
        outcomes = row["completeness_win_tie_loss"]
        lines.append(
            f"| `{row['model']}` | {row['model_type']} | "
            f"{_fmt_percent(row['baseline']['completeness']['mean'])} | "
            f"{_fmt_percent(row['rag']['completeness']['mean'])} | "
            f"{_fmt_percent(delta['mean'], signed=True)} "
            f"[{_fmt_percent(ci[0], signed=True)}, {_fmt_percent(ci[1], signed=True)}] | "
            f"{_fmt_percent(row['baseline']['hallucination']['mean'])} | "
            f"{_fmt_percent(row['rag']['hallucination']['mean'])} | "
            f"{_fmt_percent(hallucination['mean'], signed=True)} | "
            f"{outcomes['rag_win']}/{outcomes['tie']}/{outcomes['rag_loss']} |"
        )

    ranking = report["rankings"]
    best_lift = max(
        report["model_summary"],
        key=lambda item: item["rag_minus_baseline"]["completeness"]["mean"],
    )
    harmed = [
        item
        for item in report["model_summary"]
        if item["rag_minus_baseline"]["completeness"]["mean"] < 0
    ]
    lines.extend(
        [
            "",
            "## 自动摘要",
            "",
            f"- Baseline 完整性第一：`{ranking['baseline_completeness'][0]['model']}` "
            f"（{_fmt_percent(ranking['baseline_completeness'][0]['value'])}）。",
            f"- RAG 完整性第一：`{ranking['rag_completeness'][0]['model']}` "
            f"（{_fmt_percent(ranking['rag_completeness'][0]['value'])}）。",
            f"- 平均完整性增益最大：`{best_lift['model']}` "
            f"（{_fmt_percent(best_lift['rag_minus_baseline']['completeness']['mean'], signed=True)}）。",
            "- 平均完整性下降的模型："
            + (
                "、".join(
                    f"`{item['model']}`（{_fmt_percent(item['rag_minus_baseline']['completeness']['mean'], signed=True)}）"
                    for item in harmed
                )
                if harmed
                else "无"
            )
            + "。",
            "",
            "ΔC 越高越好；ΔH 越低越好。C 胜/平/负只按单题 completeness 比较，"
            "不把 hallucination 混入一个未经验证的复合分数。",
            "",
            "## Macro average 检查",
            "",
            "| 模型 | item-level ΔC | domain-macro ΔC | question-type-macro ΔC |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in report["model_summary"]:
        lines.append(
            f"| `{row['model']}` | "
            f"{_fmt_percent(row['rag_minus_baseline']['completeness']['mean'], signed=True)} | "
            f"{_fmt_percent(row['domain_macro']['rag_minus_baseline']['completeness'], signed=True)} | "
            f"{_fmt_percent(row['question_type_macro']['rag_minus_baseline']['completeness'], signed=True)} |"
        )

    lines.extend(
        [
            "",
            "## 按领域的完整性",
            "",
            "| 模型 | 领域 | n | Baseline | RAG | ΔC | ΔH |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["domain_summary"]:
        lines.append(
            f"| `{row['model']}` | {row['domain']} | {row['paired_answer_count']} | "
            f"{_fmt_percent(row['baseline']['completeness']['mean'])} | "
            f"{_fmt_percent(row['rag']['completeness']['mean'])} | "
            f"{_fmt_percent(row['rag_minus_baseline']['completeness']['mean'], signed=True)} | "
            f"{_fmt_percent(row['rag_minus_baseline']['hallucination']['mean'], signed=True)} |"
        )

    lines.extend(
        [
            "",
            "## 按题型汇总（跨模型描述性统计）",
            "",
            "| 题型 | n | Baseline C | RAG C | ΔC | ΔH |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["question_type_pooled_summary"]:
        lines.append(
            f"| {row['question_type']} | {row['paired_answer_count']} | "
            f"{_fmt_percent(row['baseline']['completeness']['mean'])} | "
            f"{_fmt_percent(row['rag']['completeness']['mean'])} | "
            f"{_fmt_percent(row['rag_minus_baseline']['completeness']['mean'], signed=True)} | "
            f"{_fmt_percent(row['rag_minus_baseline']['hallucination']['mean'], signed=True)} |"
        )

    lines.extend(
        [
            "",
            "## 模型类型（探索性）",
            "",
            "| 类型 | 模型数 | 模型级平均 ΔC | 模型级平均 ΔH |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in report["model_type_summary"]:
        lines.append(
            f"| {row['model_type']} | {row['model_count']} | "
            f"{_fmt_percent(row['mean_model_level_completeness_delta'], signed=True)} | "
            f"{_fmt_percent(row['mean_model_level_hallucination_delta'], signed=True)} |"
        )

    sensitivity = report["generation_parameter_sensitivity"]
    lines.extend(
        [
            "",
            "## 生成参数敏感性",
            "",
            f"检测到 `{sensitivity['max_tokens_mismatch_count']}` 个配对的 Baseline/RAG "
            "`max_tokens` 不一致。主结果保留所有成功回答；下表仅作为参数一致性敏感性分析。",
            "",
            "| 模型 | 排除数 | 剩余 n | 主分析 ΔC | 排除不一致项后 ΔC [95% CI] |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in sensitivity["per_model_excluding_mismatches"]:
        primary = next(
            item for item in report["model_summary"] if item["model"] == row["model"]
        )
        delta = row["sensitivity_summary"]["rag_minus_baseline"]["completeness"]
        ci = delta["bootstrap_95_ci"]
        lines.append(
            f"| `{row['model']}` | {row['excluded_pair_count']} | "
            f"{row['sensitivity_summary']['paired_answer_count']} | "
            f"{_fmt_percent(primary['rag_minus_baseline']['completeness']['mean'], signed=True)} | "
            f"{_fmt_percent(delta['mean'], signed=True)} "
            f"[{_fmt_percent(ci[0], signed=True)}, {_fmt_percent(ci[1], signed=True)}] |"
        )

    lines.extend(
        [
            "",
            "## 最大单题变化",
            "",
            "完整性提升最大：",
            "",
        ]
    )
    for item in report["extreme_pairs"]["largest_completeness_gains"]:
        lines.append(
            f"- `{item['model']}` / `{item['qra_id']}`: "
            f"{_fmt_percent(item['rag_minus_baseline']['completeness'], signed=True)} "
            f"({item['domain']}, {item['question_type']})"
        )
    lines.extend(["", "完整性下降最大：", ""])
    for item in report["extreme_pairs"]["largest_completeness_losses"]:
        lines.append(
            f"- `{item['model']}` / `{item['qra_id']}`: "
            f"{_fmt_percent(item['rag_minus_baseline']['completeness'], signed=True)} "
            f"({item['domain']}, {item['question_type']})"
        )

    lines.extend(
        [
            "",
            "## 解释限制",
            "",
            "- 每个模型/条件/QRA 只有一次生成；置信区间反映 60 个题目的配对差值，"
            "不反映同一提示重复生成的方差。",
            "- Baseline 与 RAG 均采用同一 Codex 5.6 rubric 并隐藏模型与条件；"
            "但它们是两个独立批次，仍需 GPT-5.4、DeepSeek V4 或人工子集做敏感性检查。",
            "- `hallucination` 这里只表示与 keypoint 明确冲突的比例；未做开放式 unsupported-claim 审核。",
            "- `irrelevancy` 是未覆盖且未矛盾的 keypoint 比例；三类标签穷尽，所以它不是独立于前两项的新信号。",
            f"- 检测到 {sensitivity['max_tokens_mismatch_count']} 个 Baseline/RAG "
            "max-token 不一致配对；主分析保留全部成功回答，另报告排除后的敏感性结果。",
            "- `expert` 是项目清单中的 MoE/专家模型分类标签；类型层只有少量模型，"
            "模型类型结果仅作探索性描述，不能作为架构因果结论。",
            "",
            "完整逐题配对结果、所有分层统计及输入哈希见同目录 JSON。",
            "",
        ]
    )
    return "\n".join(lines)


def build_formal_codex_comparison(
    *,
    baseline_scores_path: Path,
    baseline_manifest_path: Path,
    rag_scores_path: Path,
    rag_manifest_path: Path,
    output_dir: Path,
    report_id: str,
    bootstrap_iterations: int = 10_000,
    seed: int = 20260810,
) -> dict[str, Path]:
    """Validate, pair, aggregate, and export two completed formal Judge runs."""

    if bootstrap_iterations < 1:
        raise FormalCodexComparisonError("bootstrap_iterations must be positive")
    baseline_scores = _load_json(baseline_scores_path)
    baseline_manifest = _load_json(baseline_manifest_path)
    rag_scores = _load_json(rag_scores_path)
    rag_manifest = _load_json(rag_manifest_path)
    if baseline_manifest.get("dataset_manifest_sha256") != rag_manifest.get(
        "dataset_manifest_sha256"
    ):
        raise FormalCodexComparisonError("Baseline and RAG datasets differ")
    if baseline_manifest.get("judge") != rag_manifest.get("judge"):
        raise FormalCodexComparisonError("Baseline and RAG Judge configurations differ")

    baseline_results = _validate_run(
        baseline_scores,
        baseline_manifest,
        expected_mode="baseline",
        label="Baseline",
    )
    rag_results = _validate_run(
        rag_scores,
        rag_manifest,
        expected_mode="rag",
        label="RAG",
    )
    pairs = _paired_results(baseline_results, rag_results)
    models = sorted({item["model"] for item in pairs})
    qras = sorted({item["qra_id"] for item in pairs})
    expected = len(models) * len(qras)
    if len(pairs) != expected:
        raise FormalCodexComparisonError(
            f"Expected a complete {len(models)} x {len(qras)} panel, found {len(pairs)} pairs"
        )

    rng = random.Random(seed)
    model_summary = _grouped_summary(
        pairs,
        ("model", "model_type"),
        bootstrap_iterations=bootstrap_iterations,
        rng=rng,
    )
    domain_summary = _grouped_summary(
        pairs,
        ("model", "domain"),
        bootstrap_iterations=0,
        rng=rng,
    )
    question_type_summary = _grouped_summary(
        pairs,
        ("model", "question_type"),
        bootstrap_iterations=0,
        rng=rng,
    )
    question_type_pooled = _grouped_summary(
        pairs,
        ("question_type",),
        bootstrap_iterations=0,
        rng=rng,
    )
    domain_pooled = _grouped_summary(
        pairs,
        ("domain",),
        bootstrap_iterations=0,
        rng=rng,
    )
    overall_summary = _pair_summary(
        pairs,
        bootstrap_iterations=0,
        rng=rng,
    )
    for item in model_summary:
        model = str(item["model"])
        item["domain_macro"] = _macro_summary(
            [row for row in domain_summary if row["model"] == model]
        )
        item["question_type_macro"] = _macro_summary(
            [row for row in question_type_summary if row["model"] == model]
        )

    type_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in model_summary:
        type_groups[str(item["model_type"])].append(item)
    model_type_summary = [
        {
            "model_type": model_type,
            "model_count": len(items),
            "models": [str(item["model"]) for item in items],
            "mean_model_level_completeness_delta": _mean(
                item["rag_minus_baseline"]["completeness"]["mean"] for item in items
            ),
            "mean_model_level_hallucination_delta": _mean(
                item["rag_minus_baseline"]["hallucination"]["mean"] for item in items
            ),
            "mean_model_level_irrelevancy_delta": _mean(
                item["rag_minus_baseline"]["irrelevancy"]["mean"] for item in items
            ),
        }
        for model_type, items in sorted(type_groups.items())
    ]

    baseline_tokens = _generation_max_tokens(
        baseline_manifest,
        baseline_manifest_path,
    )
    rag_tokens = _generation_max_tokens(rag_manifest, rag_manifest_path)
    max_token_mismatches = []
    if baseline_tokens or rag_tokens:
        pair_keys = {(item["model"], item["qra_id"]) for item in pairs}
        if set(baseline_tokens) != pair_keys or set(rag_tokens) != pair_keys:
            raise FormalCodexComparisonError(
                "Generation provenance does not cover every paired Judge score"
            )
        for key in sorted(pair_keys):
            if baseline_tokens[key]["max_tokens"] != rag_tokens[key]["max_tokens"]:
                max_token_mismatches.append(
                    {
                        "model": key[0],
                        "qra_id": key[1],
                        "baseline": baseline_tokens[key],
                        "rag": rag_tokens[key],
                    }
                )
    mismatches_by_model: dict[str, set[str]] = defaultdict(set)
    for item in max_token_mismatches:
        mismatches_by_model[str(item["model"])].add(str(item["qra_id"]))
    sensitivity_rows = []
    for model, excluded_qras in sorted(mismatches_by_model.items()):
        retained = [
            item
            for item in pairs
            if item["model"] == model and item["qra_id"] not in excluded_qras
        ]
        sensitivity_rows.append(
            {
                "model": model,
                "excluded_pair_count": len(excluded_qras),
                "excluded_qra_ids": sorted(excluded_qras),
                "sensitivity_summary": _pair_summary(
                    retained,
                    bootstrap_iterations=bootstrap_iterations,
                    rng=rng,
                ),
            }
        )

    rankings = {
        f"{mode}_{metric}": [
            {"rank": index, "model": item["model"], "value": item[mode][metric]["mean"]}
            for index, item in enumerate(
                sorted(
                    model_summary,
                    key=lambda row: (
                        -float(row[mode][metric]["mean"])
                        if metric == "completeness"
                        else float(row[mode][metric]["mean"]),
                        str(row["model"]),
                    ),
                ),
                start=1,
            )
        ]
        for mode in ("baseline", "rag")
        for metric in ("completeness", "hallucination")
    }
    largest_gains = sorted(
        pairs,
        key=lambda item: (-item["rag_minus_baseline"]["completeness"], item["model"], item["qra_id"]),
    )[:10]
    largest_losses = sorted(
        pairs,
        key=lambda item: (item["rag_minus_baseline"]["completeness"], item["model"], item["qra_id"]),
    )[:10]

    report = {
        "report_id": report_id,
        "status": "completed",
        "evaluation_status": "formal_multi_judge_sensitivity_pending_validation",
        "created_at": datetime.now(UTC).isoformat(),
        "judge": baseline_manifest["judge"],
        "inputs": {
            "baseline": {
                "judge_run_id": baseline_scores["judge_run_id"],
                "scores_path": baseline_scores_path.resolve().as_posix(),
                "scores_sha256": _file_sha256(baseline_scores_path),
                "manifest_path": baseline_manifest_path.resolve().as_posix(),
                "manifest_sha256": _file_sha256(baseline_manifest_path),
            },
            "rag": {
                "judge_run_id": rag_scores["judge_run_id"],
                "scores_path": rag_scores_path.resolve().as_posix(),
                "scores_sha256": _file_sha256(rag_scores_path),
                "manifest_path": rag_manifest_path.resolve().as_posix(),
                "manifest_sha256": _file_sha256(rag_manifest_path),
            },
            "dataset_manifest_sha256": baseline_manifest[
                "dataset_manifest_sha256"
            ],
        },
        "bootstrap": {
            "unit": "paired QRA item within model",
            "iterations": bootstrap_iterations,
            "seed": seed,
            "interval": "percentile 95%",
        },
        "metric_definitions": {
            "completeness": "covered keypoints / all keypoints; higher is better",
            "hallucination": "contradicted keypoints / all keypoints; lower is better",
            "irrelevancy": "neither keypoints / all keypoints; lower is better",
            "delta": "RAG minus Baseline",
        },
        "model_count": len(models),
        "qra_count": len(qras),
        "paired_answer_count": len(pairs),
        "pooled_descriptive_summary": overall_summary,
        "model_summary": model_summary,
        "domain_summary": domain_summary,
        "domain_pooled_summary": domain_pooled,
        "question_type_summary": question_type_summary,
        "question_type_pooled_summary": question_type_pooled,
        "model_type_summary": model_type_summary,
        "generation_parameter_sensitivity": {
            "primary_analysis_policy": "retain all successful generations",
            "mismatch_definition": "Baseline max_tokens differs from paired RAG max_tokens",
            "max_tokens_mismatch_count": len(max_token_mismatches),
            "max_tokens_mismatches": max_token_mismatches,
            "per_model_excluding_mismatches": sensitivity_rows,
        },
        "rankings": rankings,
        "extreme_pairs": {
            "largest_completeness_gains": largest_gains,
            "largest_completeness_losses": largest_losses,
        },
        "paired_results": pairs,
        "limitations": [
            "One generation per model, condition, and QRA.",
            "One automated Judge family; external Judge and human sensitivity checks remain pending.",
            "Unsupported claims outside the reference keypoints were not assessed.",
            f"{len(max_token_mismatches)} paired completions used different Baseline/RAG max-token caps.",
            "Model-type aggregation is exploratory and uses models, not QRA rows, as its units.",
        ],
    }
    json_path = output_dir / f"{report_id}.json"
    markdown_path = output_dir / f"{report_id}.md"
    _write_json(json_path, report)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    return {"json": json_path, "markdown": markdown_path}
