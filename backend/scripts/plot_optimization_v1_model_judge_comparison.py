"""Render the figures used by the Optimization V1 three-Judge report."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = (
    REPO_ROOT
    / "docs/evaluation/runs/optimization_v1/judge/comparison"
    / "OPT-V1-THREE-JUDGE-COMPARISON-001.json"
)
OUTPUT_ROOT = REPO_ROOT / "docs/evaluation/reports/assets"
FIGURE_PREFIX = "OPT-V1-THREE-JUDGE-MODEL-COMPARISON-001"

JUDGE_LABELS = {
    "deepseek_v4_flash": "DeepSeek V4 Flash",
    "gpt_5_4_mini": "GPT-5.4-mini",
    "codex_5_6": "Codex 5.6",
}
JUDGE_COLORS = {
    "deepseek_v4_flash": "#2F6BFF",
    "gpt_5_4_mini": "#00A878",
    "codex_5_6": "#F28E2B",
}
MODEL_LABELS = {
    "cloud.moonshot/kimi-k3": "Kimi-K3",
    "gemma4:31b": "Gemma 31B",
    "qwen3.6:latest": "Qwen 3.6",
    "gemma4:26b": "Gemma 26B",
    "gpt-oss:20b": "GPT-OSS 20B",
    "magistral:latest": "Magistral",
    "olmo-3.1:32b-think": "OLMo 32B",
}


def _load() -> dict[str, Any]:
    return json.loads(SOURCE_PATH.read_text(encoding="utf-8"))


def _model_index(report: dict[str, Any]) -> dict[str, dict[tuple[str, str], Any]]:
    return {
        judge: {(row["model"], row["mode"]): row for row in rows}
        for judge, rows in report["model_metrics"].items()
    }


def _model_order(index: dict[str, dict[tuple[str, str], Any]]) -> list[str]:
    judges = list(index)
    models = {model for judge in judges for model, _ in index[judge]}
    return sorted(
        models,
        key=lambda model: statistics.mean(
            index[judge][(model, "rag")]["completeness"] for judge in judges
        ),
        reverse=True,
    )


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#FAFBFD",
            "axes.edgecolor": "#C8CED8",
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": "#DCE1E8",
            "grid.alpha": 0.8,
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "legend.frameon": False,
        }
    )


def _save(fig: plt.Figure, name: str) -> None:
    output = OUTPUT_ROOT / f"{FIGURE_PREFIX}_{name}.png"
    fig.savefig(output, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {output}")


def _annotate_bars(axis: plt.Axes, *, signed: bool = False) -> None:
    for container in axis.containers:
        labels = []
        for bar in container:
            value = bar.get_height()
            labels.append(f"{value:+.1%}" if signed else f"{value:.1%}")
        axis.bar_label(container, labels=labels, padding=3, fontsize=8)


def plot_baseline_vs_rag(
    index: dict[str, dict[tuple[str, str], Any]],
    models: list[str],
) -> None:
    judges = list(index)
    baseline = [
        statistics.mean(
            index[judge][(model, "baseline")]["completeness"]
            for judge in judges
        )
        for model in models
    ]
    rag = [
        statistics.mean(
            index[judge][(model, "rag")]["completeness"] for judge in judges
        )
        for model in models
    ]
    positions = list(range(len(models)))
    width = 0.36
    fig, axis = plt.subplots(figsize=(12.5, 6.5))
    axis.bar(
        [position - width / 2 for position in positions],
        baseline,
        width,
        label="Baseline",
        color="#A7B0C0",
    )
    axis.bar(
        [position + width / 2 for position in positions],
        rag,
        width,
        label="RAG",
        color="#2F6BFF",
    )
    axis.set_title("Three-Judge mean completeness: Baseline vs. RAG")
    axis.set_ylabel("Completeness (C)")
    axis.set_xticks(positions, [MODEL_LABELS[model] for model in models])
    axis.set_ylim(0, 0.72)
    axis.yaxis.set_major_formatter(PercentFormatter(1))
    axis.legend(loc="upper right")
    _annotate_bars(axis)
    fig.tight_layout()
    _save(fig, "baseline_vs_rag_completeness")


def plot_rag_by_judge(
    report: dict[str, Any],
    index: dict[str, dict[tuple[str, str], Any]],
    models: list[str],
) -> None:
    judges = list(report["model_metrics"])
    positions = list(range(len(models)))
    width = 0.24
    fig, axis = plt.subplots(figsize=(13, 6.5))
    for judge_position, judge in enumerate(judges):
        offset = (judge_position - 1) * width
        values = [
            index[judge][(model, "rag")]["completeness"] for model in models
        ]
        axis.bar(
            [position + offset for position in positions],
            values,
            width,
            label=JUDGE_LABELS[judge],
            color=JUDGE_COLORS[judge],
        )
    axis.set_title("RAG completeness by tested model and Judge")
    axis.set_ylabel("Completeness (C)")
    axis.set_xticks(positions, [MODEL_LABELS[model] for model in models])
    axis.set_ylim(0, 0.70)
    axis.yaxis.set_major_formatter(PercentFormatter(1))
    axis.legend(loc="upper right", ncols=3)
    _annotate_bars(axis)
    fig.tight_layout()
    _save(fig, "rag_completeness_by_judge")


def plot_metric_deltas(
    index: dict[str, dict[tuple[str, str], Any]],
    models: list[str],
) -> None:
    judges = list(index)
    metrics = (
        ("completeness", "ΔC", "#2F6BFF"),
        ("hallucination", "Δ keypoint contradiction", "#E15759"),
        ("irrelevancy", "ΔNeither", "#7B61A8"),
    )
    positions = list(range(len(models)))
    width = 0.24
    fig, axis = plt.subplots(figsize=(13, 7))
    for metric_position, (metric, label, color) in enumerate(metrics):
        offset = (metric_position - 1) * width
        values = []
        for model in models:
            values.append(
                statistics.mean(
                    index[judge][(model, "rag")][metric]
                    - index[judge][(model, "baseline")][metric]
                    for judge in judges
                )
            )
        axis.bar(
            [position + offset for position in positions],
            values,
            width,
            label=label,
            color=color,
        )
    axis.axhline(0, color="#4B5563", linewidth=1)
    axis.set_title("Three-Judge mean metric change after RAG")
    axis.set_ylabel("RAG minus Baseline")
    axis.set_xticks(positions, [MODEL_LABELS[model] for model in models])
    axis.set_ylim(-0.30, 0.34)
    axis.yaxis.set_major_formatter(PercentFormatter(1))
    axis.legend(loc="upper right", ncols=3)
    _annotate_bars(axis, signed=True)
    fig.tight_layout()
    _save(fig, "metric_deltas_by_model")


def plot_judge_agreement(report: dict[str, Any]) -> None:
    pair_labels = {
        "deepseek_v4_flash_vs_gpt_5_4_mini": "DeepSeek–GPT",
        "deepseek_v4_flash_vs_codex_5_6": "DeepSeek–Codex",
        "gpt_5_4_mini_vs_codex_5_6": "GPT–Codex",
    }
    agreement = report["keypoint_agreement"]
    pairs = list(pair_labels)
    positions = list(range(len(pairs)))
    width = 0.24
    series = (
        ("raw_agreement", "Keypoint agreement", "#2F6BFF"),
        ("cohens_kappa", "Cohen's κ", "#00A878"),
        ("exact_answer_agreement", "Exact-answer agreement", "#F28E2B"),
    )
    fig, axis = plt.subplots(figsize=(10, 6.5))
    for series_position, (field, label, color) in enumerate(series):
        offset = (series_position - 1) * width
        axis.bar(
            [position + offset for position in positions],
            [agreement[pair][field] for pair in pairs],
            width,
            label=label,
            color=color,
        )
    axis.set_title("Inter-Judge agreement across 2,702 keypoint decisions")
    axis.set_ylabel("Value (0–1)")
    axis.set_xticks(positions, [pair_labels[pair] for pair in pairs])
    axis.set_ylim(0, 1)
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.11),
        ncols=3,
    )
    for container, (field, _, _) in zip(axis.containers, series, strict=True):
        labels = [
            f"{bar.get_height():.3f}"
            if field == "cohens_kappa"
            else f"{bar.get_height():.1%}"
            for bar in container
        ]
        axis.bar_label(container, labels=labels, padding=3, fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _save(fig, "judge_agreement")


def main() -> None:
    report = _load()
    index = _model_index(report)
    models = _model_order(index)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    _style()
    plot_baseline_vs_rag(index, models)
    plot_rag_by_judge(report, index, models)
    plot_metric_deltas(index, models)
    plot_judge_agreement(report)


if __name__ == "__main__":
    main()
