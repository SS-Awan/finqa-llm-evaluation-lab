"""Generate repository charts from the saved FinQA analysis report."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPORT_PATH = (
    PROJECT_ROOT / "data" / "results" / "finqa_test_96_analysis.json"
)
CHARTS_DIRECTORY = PROJECT_ROOT / "docs" / "assets"

STRATEGY_LABELS = {
    "direct_answer": "Direct Answer",
    "structured_reasoning": "Structured Reasoning",
    "program_of_thought": "Program-of-Thought",
    "few_shot_program_of_thought": "Few-shot PoT",
}

STRATEGY_ORDER = [
    "direct_answer",
    "structured_reasoning",
    "program_of_thought",
    "few_shot_program_of_thought",
]

STRATEGY_COLORS = [
    "#64748B",
    "#2563EB",
    "#7C3AED",
    "#059669",
]


def save_overall_accuracy_chart(report: dict[str, object]) -> None:
    """Save the overall strategy comparison chart."""
    strategy_metrics = report["by_strategy"]
    assert isinstance(strategy_metrics, dict)

    strategies = [
        strategy for strategy in STRATEGY_ORDER if strategy in strategy_metrics
    ]
    accuracies = [
        float(strategy_metrics[strategy]["accuracy_percent"])
        for strategy in strategies
    ]

    figure, axis = plt.subplots(figsize=(9, 5.5))
    bars = axis.bar(
        [STRATEGY_LABELS[strategy] for strategy in strategies],
        accuracies,
        color=STRATEGY_COLORS[: len(strategies)],
    )

    axis.set_title(
        "FinQA Accuracy by Prompting Strategy",
        fontweight="bold",
        pad=14,
    )
    axis.set_ylabel("Accuracy (%)")
    axis.set_ylim(0, 100)
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)

    for bar, accuracy in zip(bars, accuracies, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            accuracy + 2,
            f"{accuracy:.1f}%",
            ha="center",
            fontweight="bold",
        )

    figure.tight_layout()
    figure.savefig(
        CHARTS_DIRECTORY / "strategy_accuracy.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_grouped_accuracy_chart(
    report: dict[str, object],
    report_key: str,
    group_order: list[str],
    title: str,
    filename: str,
) -> None:
    """Save a grouped comparison chart for one manifest characteristic."""
    grouped_metrics = report[report_key]
    assert isinstance(grouped_metrics, dict)

    groups = [
        group_name for group_name in group_order if group_name in grouped_metrics
    ]
    positions = list(range(len(groups)))
    bar_width = 0.19

    figure, axis = plt.subplots(figsize=(10, 5.8))

    for strategy_index, strategy in enumerate(STRATEGY_ORDER):
        accuracies = []
        sample_sizes = []

        for group_name in groups:
            group_metrics = grouped_metrics[group_name]
            assert isinstance(group_metrics, dict)
            strategy_metrics = group_metrics[strategy]

            accuracies.append(float(strategy_metrics["accuracy_percent"]))
            sample_sizes.append(int(strategy_metrics["total_results"]))

        offsets = [
            position + (strategy_index - 1.5) * bar_width
            for position in positions
        ]

        axis.bar(
            offsets,
            accuracies,
            width=bar_width,
            label=STRATEGY_LABELS[strategy],
            color=STRATEGY_COLORS[strategy_index],
        )

    group_labels = [
        f"{group_name} (n={int(grouped_metrics[group_name]['direct_answer']['total_results'])})"
        for group_name in groups
    ]

    axis.set_title(title, fontweight="bold", pad=14)
    axis.set_ylabel("Accuracy (%)")
    axis.set_xticks(positions, group_labels)
    axis.set_ylim(0, 100)
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    axis.legend(loc="upper right")

    figure.tight_layout()
    figure.savefig(
        CHARTS_DIRECTORY / filename,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def main() -> None:
    """Create the three charts used in project documentation."""
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    CHARTS_DIRECTORY.mkdir(parents=True, exist_ok=True)

    save_overall_accuracy_chart(report)
    save_grouped_accuracy_chart(
        report=report,
        report_key="by_evidence_type",
        group_order=["table", "text", "both"],
        title="Accuracy by Evidence Type",
        filename="accuracy_by_evidence_type.png",
    )
    save_grouped_accuracy_chart(
        report=report,
        report_key="by_program_depth",
        group_order=["1", "2", "3+"],
        title="Accuracy by Calculation Depth",
        filename="accuracy_by_program_depth.png",
    )

    print(f"Charts saved to: {CHARTS_DIRECTORY}")
    print("- strategy_accuracy.png")
    print("- accuracy_by_evidence_type.png")
    print("- accuracy_by_program_depth.png")


if __name__ == "__main__":
    main()