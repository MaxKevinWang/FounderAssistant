"""Compare the complete Codex, GPT, and DeepSeek Optimization V1 judgements."""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = REPO_ROOT / "docs/evaluation/runs/optimization_v1"
JUDGE_ROOT = RUN_ROOT / "judge"
OUTPUT_ROOT = JUDGE_ROOT / "comparison"
COMPARISON_ID = "OPT-V1-THREE-JUDGE-COMPARISON-001"

INPUTS = {
    "deepseek_v4_flash": [
        JUDGE_ROOT
        / "deepseek_api/OPT-V1-DEEPSEEKV4-FULL-001/OPT-V1-DEEPSEEKV4-FULL-001_scores.json"
    ],
    "gpt_5_4_mini": [
        JUDGE_ROOT
        / "openai_api/OPT-V1-GPT54MINI-JUDGE-001/OPT-V1-GPT54MINI-JUDGE-001_scores.json"
    ],
    "codex_5_6": [
        JUDGE_ROOT
        / "codex/OPT-V1-CODEX56-BASELINE-JUDGE-001/OPT-V1-CODEX56-BASELINE-JUDGE-001_scores.json",
        JUDGE_ROOT
        / "codex/OPT-V1-CODEX56-RAG-JUDGE-001/OPT-V1-CODEX56-RAG-JUDGE-001_scores.json",
    ],
}

LABELS = ("covered", "contradicted", "neither")
METRICS = ("completeness", "hallucination", "irrelevancy")


class ComparisonError(RuntimeError):
    """Raised when Judge runs are incomplete or not directly comparable."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ComparisonError(f"Could not read {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ComparisonError(f"Expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _display(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def _condition(record: dict[str, Any]) -> tuple[str, str, int, str]:
    return (
        str(record["model"]),
        str(record["qra_id"]),
        int(record.get("repeat", 1)),
        str(record["mode"]),
    )


def _normalize_api(payload: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for result in payload.get("results", []):
        judge = result.get("judge", {})
        if judge.get("status") != "success":
            raise ComparisonError("API Judge input contains a non-success result")
        records.append(
            {
                "model": str(result["tested_model"]),
                "qra_id": str(result["qra_id"]),
                "repeat": int(result.get("repeat", 1)),
                "mode": str(result["mode"]),
                "domain": str(result["domain"]),
                "question_type": str(result["question_type"]),
                "metrics": {
                    metric: float(judge["metrics"][metric]) for metric in METRICS
                },
                "labels": {
                    str(item["keypoint_id"]): str(item["label"])
                    for item in judge["keypoint_evaluations"]
                },
            }
        )
    return records


def _normalize_codex(payload: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for result in payload.get("results", []):
        records.append(
            {
                "model": str(result["model"]),
                "qra_id": str(result["qra_id"]),
                "repeat": int(result.get("repeat", 1)),
                "mode": str(result["mode"]),
                "domain": str(result["domain"]),
                "question_type": str(result["question_type"]),
                "metrics": {metric: float(result[metric]) for metric in METRICS},
                "labels": {
                    str(item["keypoint_id"]): str(item["label"])
                    for item in result["keypoint_labels"]
                },
            }
        )
    return records


def _mean(values: list[float]) -> float:
    return round(statistics.mean(values), 4) if values else 0.0


def _aggregate(
    records: list[dict[str, Any]],
    fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for record in records:
        key = tuple(str(record[field]) for field in fields)
        groups.setdefault(key, []).append(record)
    rows = []
    for key, group in sorted(groups.items()):
        row = {field: value for field, value in zip(fields, key, strict=True)}
        row["answer_count"] = len(group)
        row.update(
            {
                metric: _mean([item["metrics"][metric] for item in group])
                for metric in METRICS
            }
        )
        rows.append(row)
    return rows


def _mode_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    rows = {row["mode"]: row for row in _aggregate(records, ("mode",))}
    baseline = rows["baseline"]
    rag = rows["rag"]
    return {
        "baseline": {metric: baseline[metric] for metric in METRICS},
        "rag": {metric: rag[metric] for metric in METRICS},
        "rag_minus_baseline": {
            metric: round(rag[metric] - baseline[metric], 4) for metric in METRICS
        },
    }


def _agreement(
    first: dict[tuple[str, str, int, str], dict[str, Any]],
    second: dict[tuple[str, str, int, str], dict[str, Any]],
) -> dict[str, Any]:
    if set(first) != set(second):
        raise ComparisonError("Judge condition sets differ")
    decisions: list[tuple[str, str]] = []
    exact_answers = 0
    confusion = {left: {right: 0 for right in LABELS} for left in LABELS}
    for condition in sorted(first):
        first_labels = first[condition]["labels"]
        second_labels = second[condition]["labels"]
        if set(first_labels) != set(second_labels):
            raise ComparisonError(f"Keypoint sets differ for {condition}")
        exact = True
        for keypoint_id in sorted(first_labels):
            pair = (first_labels[keypoint_id], second_labels[keypoint_id])
            if pair[0] not in LABELS or pair[1] not in LABELS:
                raise ComparisonError(f"Unknown label in {condition}/{keypoint_id}")
            decisions.append(pair)
            confusion[pair[0]][pair[1]] += 1
            exact = exact and pair[0] == pair[1]
        exact_answers += int(exact)

    total = len(decisions)
    observed = sum(left == right for left, right in decisions) / total
    first_counts = Counter(left for left, _ in decisions)
    second_counts = Counter(right for _, right in decisions)
    expected = sum(
        (first_counts[label] / total) * (second_counts[label] / total)
        for label in LABELS
    )
    kappa = (observed - expected) / (1 - expected) if expected < 1 else 1.0
    return {
        "keypoint_decision_count": total,
        "raw_agreement": round(observed, 4),
        "cohens_kappa": round(kappa, 4),
        "exact_answer_agreement": round(exact_answers / len(first), 4),
        "exact_answer_count": exact_answers,
        "confusion_matrix_first_rows_second_columns": confusion,
    }


def _load_all() -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
]:
    records_by_judge: dict[str, list[dict[str, Any]]] = {}
    payloads_by_judge: dict[str, list[dict[str, Any]]] = {}
    for judge_name, paths in INPUTS.items():
        payloads = [_read_json(path) for path in paths]
        if any(payload.get("status") != "completed" for payload in payloads):
            raise ComparisonError(f"{judge_name} has an incomplete source run")
        if judge_name == "codex_5_6":
            records = [
                record for payload in payloads for record in _normalize_codex(payload)
            ]
        else:
            records = _normalize_api(payloads[0])
        condition_count = len({_condition(record) for record in records})
        if len(records) != 840 or condition_count != 840:
            raise ComparisonError(
                f"{judge_name} must contain 840 unique conditions; "
                f"found records={len(records)}, conditions={condition_count}"
            )
        records_by_judge[judge_name] = records
        payloads_by_judge[judge_name] = payloads
    return records_by_judge, payloads_by_judge


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _overall_table(report: dict[str, Any], *, language: str) -> list[str]:
    judge_names = {
        "deepseek_v4_flash": "DeepSeek V4 Flash",
        "gpt_5_4_mini": "GPT-5.4-mini",
        "codex_5_6": "Codex 5.6",
    }
    header = (
        "| Judge | Mode | C | Keypoint contradiction | Neither |"
        if language == "en"
        else "| Judge | 模式 | C | Keypoint 矛盾率 | Neither |"
    )
    lines = [header, "|---|---|---:|---:|---:|"]
    for judge, summary in report["aggregate_metrics"].items():
        for mode in ("baseline", "rag", "rag_minus_baseline"):
            row = summary[mode]
            lines.append(
                f"| {judge_names[judge]} | {mode} | "
                f"{_fmt(row['completeness'])} | "
                f"{_fmt(row['hallucination'])} | "
                f"{_fmt(row['irrelevancy'])} |"
            )
    return lines


def _agreement_table(report: dict[str, Any], *, language: str) -> list[str]:
    header = (
        "| Judge pair | Keypoint agreement | Cohen's κ | Exact answers |"
        if language == "en"
        else "| Judge 对 | Keypoint 一致率 | Cohen's κ | 整题完全一致 |"
    )
    lines = [header, "|---|---:|---:|---:|"]
    for pair, values in report["keypoint_agreement"].items():
        lines.append(
            f"| {pair.replace('_vs_', ' vs. ')} | "
            f"{_fmt(values['raw_agreement'])} | "
            f"{_fmt(values['cohens_kappa'])} | "
            f"{values['exact_answer_count']}/840 "
            f"({_fmt(values['exact_answer_agreement'])}) |"
        )
    return lines


def _domain_table(report: dict[str, Any], *, language: str) -> list[str]:
    header = (
        "| Judge | Domain | Mode | C | Contradiction | Neither |"
        if language == "en"
        else "| Judge | 领域 | 模式 | C | 矛盾率 | Neither |"
    )
    lines = [header, "|---|---|---|---:|---:|---:|"]
    for judge, rows in report["domain_metrics"].items():
        for row in rows:
            lines.append(
                f"| {judge} | {row['domain']} | {row['mode']} | "
                f"{_fmt(row['completeness'])} | "
                f"{_fmt(row['hallucination'])} | "
                f"{_fmt(row['irrelevancy'])} |"
            )
    return lines


def _render(report: dict[str, Any], *, language: str) -> str:
    if language == "en":
        title = "# Final answer evaluation: complete three-Judge comparison"
        scope = "## Scope and metric semantics"
        scope_text = (
            "This final answer evaluation uses a new frozen 60-item dataset, separate "
            "from the earlier retrieval configuration-comparison questions. Historical "
            "OPT and optimization_v1 names are retained. This comparison covers 60 QRA, "
            "six institute-hosted models and the external Kimi reference, "
            "Baseline and RAG, 840 candidate answers, and "
            f"{report['scope']['keypoint_decision_count']} keypoint decisions per Judge. "
            "C is covered reference keypoints / all; the historical H field is "
            "strictly the reference-keypoint contradiction rate, not general answer "
            "hallucination; Neither is omitted, partial, or unverifiable reference "
            "keypoints / all. Claims outside reference keypoints were not evaluated."
        )
        overall = "## Overall macro-average results"
        agreement = "## Inter-Judge agreement"
        domains = "## Results by domain"
        reliability = "## Protocol reliability"
        reliability_text = (
            f"DeepSeek preserved {report['protocol_reliability']['deepseek_v4_flash']['preserved_failed_attempts']} "
            "failed attempts before recovering every one of the 840 final answers. "
            f"GPT preserved {report['protocol_reliability']['gpt_5_4_mini']['preserved_failed_attempts']} "
            "failed attempts. Failed parse attempts do not retain provider usage, so "
            "recorded token totals can understate billing."
        )
        conclusion = "## Interpretation"
        conclusion_text = (
            "Agreement describes consistency, not ground-truth accuracy. Judges compare "
            "anonymised answers with keypoints without complete reference answers or "
            "source passages. Independent human review remains needed for label accuracy. "
            "The six core models use matched 900-token budgets; Kimi uses an external "
            "service with 8192 RAG tokens versus 8192 (2 answers) or 16384 (58 answers) "
            "for baseline, so the seven-model summary includes an unmatched-budget reference. "
            "Final-set retrieval coverage is a retrospective saved-trace diagnostic, "
            "not further configuration selection. Dataset-role corrections and limitations "
            "are documented in [the reviewed report]"
            "(../../../../reports/en/OPT-V1-THREE-JUDGE-MODEL-COMPARISON-001_en.md). "
            "Raw execution and scoring records retain their historical identifiers."
        )
    else:
        title = "# 最终答案评价：三评委完整对比"
        scope = "## 范围与指标语义"
        scope_text = (
            "本最终答案评价使用新的冻结 60 题题集，与旧检索配置比较题集分离。"
            "OPT 和 optimization_v1 历史命名保留。对比覆盖 60 条 QRA、六个机构部署模型"
            "及外部 Kimi 参考模型、Baseline/RAG 两种模式、"
            f"840 个候选回答；每位评委共作出 {report['scope']['keypoint_decision_count']} "
            "个 keypoint 判断。C=覆盖的参考 keypoint/全部 keypoint；历史 H 字段严格表示"
            "参考 keypoint 矛盾率，不是一般意义的回答幻觉率；Neither 表示遗漏、仅部分"
            "覆盖或无法验证的参考 keypoint/全部 keypoint。参考 keypoint 之外的主张未评估。"
        )
        overall = "## 总体宏平均结果"
        agreement = "## 评委间一致性"
        domains = "## 分领域结果"
        reliability = "## 协议可靠性"
        reliability_text = (
            f"DeepSeek 在得到 840 条最终成功评分前保留了 "
            f"{report['protocol_reliability']['deepseek_v4_flash']['preserved_failed_attempts']} "
            f"次失败尝试；GPT 保留了 "
            f"{report['protocol_reliability']['gpt_5_4_mini']['preserved_failed_attempts']} "
            "次。解析失败不会保留供应商 usage，因此记录 token 可能低于实际计费。"
        )
        conclusion = "## 解读"
        conclusion_text = (
            "一致性衡量评委之间的稳定程度，不等于真实正确率。Judge 使用 keypoints 和匿名"
            "答案，不接收完整参考答案或证据片段；标签准确性仍需独立人工复核。六个核心"
            "模型使用匹配的 900-token 预算；外部 Kimi 的 RAG 预算为 8192，baseline 为"
            "8192（2 条）或 16384（58 条），因此七模型汇总包含预算不匹配的参考结果。"
            "最终题集检索覆盖来自保存轨迹的事后诊断，不属于进一步配置选择。题集用途"
            "修订和限制见[审阅报告]"
            "(../../../../reports/zh/OPT-V1-THREE-JUDGE-MODEL-COMPARISON-001_zh.md)。"
            "原始运行及评分记录保留历史标识。"
        )
    return "\n".join(
        [
            title,
            "",
            f"- Comparison ID: `{report['comparison_id']}`",
            f"- Status: `{report['status']}`",
            "",
            scope,
            "",
            scope_text,
            "",
            overall,
            "",
            *_overall_table(report, language=language),
            "",
            agreement,
            "",
            *_agreement_table(report, language=language),
            "",
            domains,
            "",
            *_domain_table(report, language=language),
            "",
            reliability,
            "",
            reliability_text,
            "",
            conclusion,
            "",
            conclusion_text,
            "",
        ]
    )


def main() -> None:
    records_by_judge, payloads_by_judge = _load_all()
    indexed = {
        judge: {_condition(record): record for record in records}
        for judge, records in records_by_judge.items()
    }
    common_conditions = set.intersection(*(set(items) for items in indexed.values()))
    if len(common_conditions) != 840:
        raise ComparisonError("The three Judges do not share all 840 conditions")

    first_judge = next(iter(indexed.values()))
    keypoint_decisions = sum(
        len(first_judge[condition]["labels"]) for condition in common_conditions
    )
    agreement_pairs = (
        ("deepseek_v4_flash", "gpt_5_4_mini"),
        ("deepseek_v4_flash", "codex_5_6"),
        ("gpt_5_4_mini", "codex_5_6"),
    )
    report = {
        "comparison_id": COMPARISON_ID,
        "status": "completed",
        "created_at": datetime.now(UTC).isoformat(),
        "scope": {
            "qra_count": 60,
            "tested_model_count": 7,
            "modes": ["baseline", "rag"],
            "candidate_answer_count": 840,
            "keypoint_decision_count": keypoint_decisions,
        },
        "metric_semantics": {
            "completeness": "covered reference keypoints / all reference keypoints",
            "keypoint_contradiction_rate": "contradicted reference keypoints / all reference keypoints",
            "neither_rate": "reference keypoints classified as neither / all reference keypoints",
            "unsupported_claims_outside_keypoints": "not evaluated",
            "aggregation": "macro-average over candidate-answer metric values",
        },
        "aggregate_metrics": {
            judge: _mode_summary(records)
            for judge, records in records_by_judge.items()
        },
        "domain_metrics": {
            judge: _aggregate(records, ("domain", "mode"))
            for judge, records in records_by_judge.items()
        },
        "model_metrics": {
            judge: _aggregate(records, ("model", "mode"))
            for judge, records in records_by_judge.items()
        },
        "keypoint_agreement": {
            f"{first}_vs_{second}": _agreement(indexed[first], indexed[second])
            for first, second in agreement_pairs
        },
        "protocol_reliability": {
            "deepseek_v4_flash": {
                "successful_final_answers": len(records_by_judge["deepseek_v4_flash"]),
                "preserved_failed_attempts": len(
                    payloads_by_judge["deepseek_v4_flash"][0].get(
                        "prior_attempts", []
                    )
                ),
                "recorded_usage": payloads_by_judge["deepseek_v4_flash"][0].get(
                    "usage", {}
                ),
            },
            "gpt_5_4_mini": {
                "successful_final_answers": len(records_by_judge["gpt_5_4_mini"]),
                "preserved_failed_attempts": len(
                    payloads_by_judge["gpt_5_4_mini"][0].get("prior_attempts", [])
                ),
                "recorded_usage": payloads_by_judge["gpt_5_4_mini"][0].get(
                    "usage", {}
                ),
            },
            "usage_caveat": "Failed parse attempts did not retain API usage.",
        },
        "sources": {
            judge: [
                {"path": _display(path), "sha256": _sha256(path)}
                for path in INPUTS[judge]
            ]
            for judge in INPUTS
        },
    }

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_ROOT / f"{COMPARISON_ID}.json"
    en_path = OUTPUT_ROOT / f"{COMPARISON_ID}_en.md"
    zh_path = OUTPUT_ROOT / f"{COMPARISON_ID}_zh.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    en_path.write_text(_render(report, language="en"), encoding="utf-8")
    zh_path.write_text(_render(report, language="zh"), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {en_path}")
    print(f"Wrote {zh_path}")


if __name__ == "__main__":
    main()
