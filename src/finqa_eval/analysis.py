"""Metrics and grouped analysis for saved FinQA evaluation results."""

from __future__ import annotations

from statistics import mean, median
from typing import Callable

from finqa_eval.results import EvaluationResult

ManifestMetadata = dict[str, dict[str, str]]
GroupKey = Callable[[EvaluationResult], str]


def build_manifest_metadata(
    manifest_records: list[dict[str, object]],
) -> ManifestMetadata:
    """Index locked manifest metadata by FinQA record ID."""
    metadata: ManifestMetadata = {}

    for record in manifest_records:
        record_id = str(record["record_id"])

        if record_id in metadata:
            raise ValueError(f"Duplicate record ID in manifest: {record_id}")

        metadata[record_id] = {
            "evidence_type": str(record["evidence_type"]),
            "program_depth": str(record["program_depth"]),
            "first_operation": str(record["first_operation"]),
        }

    return metadata


def calculate_metrics(results: list[EvaluationResult]) -> dict[str, int | float]:
    """Calculate reliability, accuracy, plan, and latency metrics."""
    total = len(results)
    correct = sum(result.answer_correct is True for result in results)
    valid_outputs = sum(result.output_valid for result in results)
    errors = sum(result.error is not None for result in results)

    plan_results = [
        result for result in results if result.plan_valid is not None
    ]
    executable_plans = sum(
        result.plan_valid is True for result in plan_results
    )

    latencies = [result.latency_ms for result in results]

    return {
        "total_results": total,
        "correct_answers": correct,
        "accuracy_percent": _percentage(correct, total),
        "valid_outputs": valid_outputs,
        "valid_output_percent": _percentage(valid_outputs, total),
        "errors": errors,
        "plan_results": len(plan_results),
        "executable_plans": executable_plans,
        "executable_plan_percent": _percentage(
            executable_plans,
            len(plan_results),
        ),
        "mean_latency_ms": round(mean(latencies), 1) if latencies else 0.0,
        "median_latency_ms": round(median(latencies), 1) if latencies else 0.0,
    }


def metrics_by_strategy(
    results: list[EvaluationResult],
) -> dict[str, dict[str, int | float]]:
    """Calculate metrics separately for every prompting strategy."""
    strategies = sorted({result.strategy for result in results})

    return {
        strategy: calculate_metrics(
            [result for result in results if result.strategy == strategy]
        )
        for strategy in strategies
    }


def metrics_by_manifest_field(
    results: list[EvaluationResult],
    metadata: ManifestMetadata,
    field_name: str,
) -> dict[str, dict[str, dict[str, int | float]]]:
    """Compare strategies within evidence, depth, or operation groups."""
    allowed_fields = {
        "evidence_type",
        "program_depth",
        "first_operation",
    }

    if field_name not in allowed_fields:
        raise ValueError(f"Unsupported manifest field: {field_name}")

    grouped_results: dict[str, list[EvaluationResult]] = {}

    for result in results:
        try:
            group_name = metadata[result.record_id][field_name]
        except KeyError as error:
            raise ValueError(
                f"Missing {field_name} metadata for {result.record_id}."
            ) from error

        grouped_results.setdefault(group_name, []).append(result)

    return {
        group_name: metrics_by_strategy(group_results)
        for group_name, group_results in sorted(grouped_results.items())
    }


def build_analysis_report(
    results: list[EvaluationResult],
    metadata: ManifestMetadata,
) -> dict[str, object]:
    """Build the complete reproducible evaluation report."""
    return {
        "overall": calculate_metrics(results),
        "by_strategy": metrics_by_strategy(results),
        "by_evidence_type": metrics_by_manifest_field(
            results,
            metadata,
            "evidence_type",
        ),
        "by_program_depth": metrics_by_manifest_field(
            results,
            metadata,
            "program_depth",
        ),
        "by_first_operation": metrics_by_manifest_field(
            results,
            metadata,
            "first_operation",
        ),
    }


def _percentage(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0

    return round((numerator / denominator) * 100, 1)