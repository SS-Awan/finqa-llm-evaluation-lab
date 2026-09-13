"""Generate a reproducible analysis report from locked FinQA results."""

from __future__ import annotations

import json
from pathlib import Path

from finqa_eval.analysis import build_analysis_report, build_manifest_metadata
from finqa_eval.results import load_results

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = (
    PROJECT_ROOT / "data" / "evaluation" / "finqa_test_96_manifest.json"
)
RESULTS_PATH = (
    PROJECT_ROOT / "data" / "results" / "finqa_test_96_results.jsonl"
)
REPORT_PATH = (
    PROJECT_ROOT / "data" / "results" / "finqa_test_96_analysis.json"
)


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    metadata = build_manifest_metadata(manifest["records"])
    results = load_results(RESULTS_PATH)

    report = build_analysis_report(results, metadata)

    REPORT_PATH.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    overall = report["overall"]
    by_strategy = report["by_strategy"]

    print(f"Analysed results: {overall['total_results']}")
    print(f"Overall accuracy: {overall['accuracy_percent']}%")
    print()
    print("Accuracy by strategy:")

    for strategy, metrics in by_strategy.items():
        print(
            f"- {strategy}: "
            f"{metrics['accuracy_percent']}% "
            f"({metrics['correct_answers']}/{metrics['total_results']})"
        )

    print()
    print(f"Analysis report: {REPORT_PATH}")


if __name__ == "__main__":
    main()