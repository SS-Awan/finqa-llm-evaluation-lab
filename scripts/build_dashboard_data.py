"""Build the public static-dashboard data file from saved analysis reports."""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ANALYSIS_PATH = (
    PROJECT_ROOT / "data" / "results" / "finqa_test_96_analysis.json"
)
FAILURE_ANALYSIS_PATH = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "finqa_test_96_failure_analysis.json"
)
OUTPUT_PATH = PROJECT_ROOT / "docs" / "dashboard-data.json"

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


def make_group_rows(
    grouped_metrics: dict[str, object],
) -> list[dict[str, object]]:
    """Convert grouped analysis metrics into chart-friendly dashboard rows."""
    rows = []

    for group_name, strategy_metrics in grouped_metrics.items():
        sample_size = strategy_metrics["direct_answer"]["total_results"]
        accuracies = {
            STRATEGY_LABELS[strategy]: strategy_metrics[strategy][
                "accuracy_percent"
            ]
            for strategy in STRATEGY_ORDER
        }

        rows.append(
            {
                "group": group_name,
                "sample_size": sample_size,
                "accuracies": accuracies,
            }
        )

    return rows


def main() -> None:
    """Write a small public JSON file for the static dashboard."""
    analysis = json.loads(ANALYSIS_PATH.read_text(encoding="utf-8"))
    failure_analysis = json.loads(
        FAILURE_ANALYSIS_PATH.read_text(encoding="utf-8")
    )

    strategy_rows = [
        {
            "strategy": STRATEGY_LABELS[strategy],
            "accuracy_percent": analysis["by_strategy"][strategy][
                "accuracy_percent"
            ],
            "correct_answers": analysis["by_strategy"][strategy][
                "correct_answers"
            ],
            "total_results": analysis["by_strategy"][strategy][
                "total_results"
            ],
            "mean_latency_ms": analysis["by_strategy"][strategy][
                "mean_latency_ms"
            ],
        }
        for strategy in STRATEGY_ORDER
    ]

    dashboard_data = {
        "title": "FinQA LLM Evaluation Lab",
        "model": "gemini-3.5-flash-lite",
        "evaluation_set": "96 locked numeric FinQA test records",
        "overall": analysis["overall"],
        "strategy_results": strategy_rows,
        "evidence_type_results": make_group_rows(
            analysis["by_evidence_type"]
        ),
        "program_depth_results": make_group_rows(
            analysis["by_program_depth"]
        ),
        "failure_summary": failure_analysis["summary"],
        "plan_consistency": failure_analysis["plan_consistency"],
    }

    OUTPUT_PATH.write_text(
        json.dumps(dashboard_data, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Dashboard data saved: {OUTPUT_PATH}")
    print(f"Strategies included: {len(strategy_rows)}")


if __name__ == "__main__":
    main()