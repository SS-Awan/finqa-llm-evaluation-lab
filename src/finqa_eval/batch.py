"""Batch evaluation with checkpoint-based resume and request pacing."""

from __future__ import annotations

from time import sleep
from typing import Any, Sequence

from finqa_eval.dataset import FinQARecord
from finqa_eval.results import (
    EvaluationResult,
    append_result,
    completed_pairs,
    load_results,
)
from finqa_eval.schemas import StrategyName


def run_evaluation_batch(
    records: Sequence[FinQARecord],
    strategies: Sequence[StrategyName],
    runner: Any,
    output_path: str,
    delay_seconds: float = 0.0,
) -> list[EvaluationResult]:
    """Evaluate unfinished pairs, saving each result immediately."""
    if delay_seconds < 0:
        raise ValueError("delay_seconds cannot be negative.")

    existing_results = load_results(output_path)
    already_completed = completed_pairs(existing_results)

    pending_pairs = [
        (record, strategy)
        for record in records
        for strategy in strategies
        if (record.record_id, strategy) not in already_completed
    ]

    new_results: list[EvaluationResult] = []

    for index, (record, strategy) in enumerate(pending_pairs):
        result = runner.evaluate_record(record, strategy)
        append_result(output_path, result)
        new_results.append(result)

        if delay_seconds and index < len(pending_pairs) - 1:
            sleep(delay_seconds)

    return new_results