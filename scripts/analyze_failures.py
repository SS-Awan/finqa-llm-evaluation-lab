"""Analyse strategy disagreements and plan consistency in saved FinQA results."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from finqa_eval.dataset import load_finqa_records
from finqa_eval.results import EvaluationResult, load_results
from finqa_eval.scoring import answers_match, parse_numeric_answer

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = (
    PROJECT_ROOT / "data" / "evaluation" / "finqa_test_96_manifest.json"
)
TEST_DATASET_PATH = (
    PROJECT_ROOT / "data" / "raw" / "finqa-source" / "dataset" / "test.json"
)
RESULTS_PATH = (
    PROJECT_ROOT / "data" / "results" / "finqa_test_96_results.jsonl"
)
REPORT_PATH = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "finqa_test_96_failure_analysis.json"
)

STRATEGIES = (
    "direct_answer",
    "structured_reasoning",
    "program_of_thought",
    "few_shot_program_of_thought",
)

PLAN_STRATEGIES = (
    "program_of_thought",
    "few_shot_program_of_thought",
)


def answers_agree(first: str | None, second: str | None) -> bool:
    """Compare two answer strings using the project's numeric scorer."""
    if first is None or second is None:
        return False

    first_numeric = parse_numeric_answer(first)
    second_numeric = parse_numeric_answer(second)

    if first_numeric is None or second_numeric is None:
        return False

    return answers_match(first_numeric, second_numeric)


def main() -> None:
    """Create a saved report of strategy disagreements and plan consistency."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    metadata: dict[str, dict[str, str]] = {
        str(record["record_id"]): {
            "evidence_type": str(record["evidence_type"]),
            "program_depth": str(record["program_depth"]),
            "first_operation": str(record["first_operation"]),
        }
        for record in manifest["records"]
    }

    test_records = load_finqa_records(TEST_DATASET_PATH)
    expected_answers = {
        record.record_id: record.answer for record in test_records
    }

    results = load_results(RESULTS_PATH)
    results_by_record: dict[str, dict[str, EvaluationResult]] = {}

    for result in results:
        results_by_record.setdefault(result.record_id, {})[result.strategy] = (
            result
        )

    all_correct: list[str] = []
    all_incorrect: list[str] = []
    direct_to_structured: list[str] = []
    direct_to_few_shot: list[str] = []
    structured_only_correct: list[str] = []

    all_incorrect_evidence: Counter[str] = Counter()
    all_incorrect_depth: Counter[str] = Counter()
    all_incorrect_operation: Counter[str] = Counter()
    examples: list[dict[str, object]] = []

    plan_metrics: dict[str, Counter[str]] = {
        strategy: Counter() for strategy in PLAN_STRATEGIES
    }

    for record_id, strategy_results in sorted(results_by_record.items()):
        missing_strategies = set(STRATEGIES) - set(strategy_results)

        if missing_strategies:
            raise ValueError(
                f"Missing saved strategies for {record_id}: "
                f"{sorted(missing_strategies)}"
            )

        if record_id not in metadata or record_id not in expected_answers:
            raise ValueError(f"Missing source data for {record_id}.")

        correct_strategies = [
            strategy
            for strategy in STRATEGIES
            if strategy_results[strategy].answer_correct is True
        ]

        if len(correct_strategies) == len(STRATEGIES):
            all_correct.append(record_id)

        if not correct_strategies:
            all_incorrect.append(record_id)
            record_metadata = metadata[record_id]
            all_incorrect_evidence[record_metadata["evidence_type"]] += 1
            all_incorrect_depth[record_metadata["program_depth"]] += 1
            all_incorrect_operation[record_metadata["first_operation"]] += 1

        if (
            strategy_results["direct_answer"].answer_correct is False
            and strategy_results["structured_reasoning"].answer_correct is True
        ):
            direct_to_structured.append(record_id)

        if (
            strategy_results["direct_answer"].answer_correct is False
            and strategy_results["few_shot_program_of_thought"].answer_correct
            is True
        ):
            direct_to_few_shot.append(record_id)

        if correct_strategies == ["structured_reasoning"]:
            structured_only_correct.append(record_id)

        for strategy in PLAN_STRATEGIES:
            result = strategy_results[strategy]

            if result.plan_valid is not True or result.plan_answer is None:
                continue

            metrics = plan_metrics[strategy]
            metrics["executable_plans"] += 1

            plan_is_correct = answers_agree(
                result.plan_answer,
                expected_answers[record_id],
            )
            final_is_correct = result.answer_correct is True
            plan_matches_final = answers_agree(
                result.plan_answer,
                result.final_answer,
            )

            if plan_is_correct:
                metrics["plan_correct"] += 1

            if final_is_correct:
                metrics["final_answer_correct"] += 1

            if not plan_matches_final:
                metrics["plan_final_disagreement"] += 1

            if plan_is_correct and not final_is_correct:
                metrics["plan_correct_final_answer_wrong"] += 1

            if not plan_is_correct and final_is_correct:
                metrics["plan_wrong_final_answer_correct"] += 1

        if (
            not correct_strategies
            or correct_strategies == ["structured_reasoning"]
        ):
            record_metadata = metadata[record_id]
            examples.append(
                {
                    "record_id": record_id,
                    "evidence_type": record_metadata["evidence_type"],
                    "program_depth": record_metadata["program_depth"],
                    "first_operation": record_metadata["first_operation"],
                    "correct_strategies": correct_strategies,
                }
            )

    report = {
        "summary": {
            "records_analysed": len(results_by_record),
            "all_strategies_correct": len(all_correct),
            "all_strategies_incorrect": len(all_incorrect),
            "direct_answer_wrong_structured_reasoning_correct": (
                len(direct_to_structured)
            ),
            "direct_answer_wrong_few_shot_pot_correct": (
                len(direct_to_few_shot)
            ),
            "structured_reasoning_only_correct": len(
                structured_only_correct
            ),
        },
        "plan_consistency": {
            strategy: dict(metrics)
            for strategy, metrics in plan_metrics.items()
        },
        "all_incorrect_by_evidence_type": dict(
            sorted(all_incorrect_evidence.items())
        ),
        "all_incorrect_by_program_depth": dict(
            sorted(all_incorrect_depth.items())
        ),
        "all_incorrect_by_first_operation": dict(
            sorted(all_incorrect_operation.items())
        ),
        "all_strategies_correct_record_ids": all_correct,
        "all_strategies_incorrect_record_ids": all_incorrect,
        "direct_to_structured_record_ids": direct_to_structured,
        "direct_to_few_shot_pot_record_ids": direct_to_few_shot,
        "structured_reasoning_only_correct_record_ids": (
            structured_only_correct
        ),
        "selected_examples": examples[:20],
    }

    REPORT_PATH.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    summary = report["summary"]

    print(f"Records analysed: {summary['records_analysed']}")
    print(f"All strategies correct: {summary['all_strategies_correct']}")
    print(f"All strategies incorrect: {summary['all_strategies_incorrect']}")
    print(
        "Direct wrong, Structured Reasoning correct: "
        f"{summary['direct_answer_wrong_structured_reasoning_correct']}"
    )
    print(
        "Direct wrong, Few-shot PoT correct: "
        f"{summary['direct_answer_wrong_few_shot_pot_correct']}"
    )
    print(
        "Structured Reasoning only correct: "
        f"{summary['structured_reasoning_only_correct']}"
    )
    print()
    print("Plan consistency:")

    for strategy in PLAN_STRATEGIES:
        metrics = plan_metrics[strategy]
        print(
            f"- {strategy}: "
            f"plan correct={metrics['plan_correct']}/"
            f"{metrics['executable_plans']}, "
            f"final correct={metrics['final_answer_correct']}/"
            f"{metrics['executable_plans']}, "
            f"disagreements={metrics['plan_final_disagreement']}"
        )

    print()
    print(f"Failure analysis report: {REPORT_PATH}")


if __name__ == "__main__":
    main()