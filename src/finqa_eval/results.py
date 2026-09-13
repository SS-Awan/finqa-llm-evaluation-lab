"""Persistent JSONL result storage with resume support."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from finqa_eval.schemas import StrategyName


@dataclass(frozen=True)
class EvaluationResult:
    """One strategy result for one FinQA record."""

    record_id: str
    strategy: StrategyName
    model: str
    prompt_version: str
    output_valid: bool
    final_answer: str | None
    answer_correct: bool | None
    plan_valid: bool | None
    plan_answer: str | None
    latency_ms: int
    evidence: list[str] | None = None
    reasoning_summary: str | None = None
    calculation_plan: dict[str, object] | None = None
    error: str | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )

    def to_dict(self) -> dict[str, object]:
        """Convert the result into a JSON-serializable dictionary."""
        return asdict(self)


def append_result(path: str | Path, result: EvaluationResult) -> None:
    """Append one result immediately so interrupted runs can resume safely."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(result.to_dict()) + "\n")


def load_results(path: str | Path) -> list[EvaluationResult]:
    """Load prior JSONL results, returning an empty list if no file exists."""
    input_path = Path(path)

    if not input_path.exists():
        return []

    results: list[EvaluationResult] = []

    with input_path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                raw_result = json.loads(line)
                results.append(EvaluationResult(**raw_result))
            except (json.JSONDecodeError, TypeError) as error:
                raise ValueError(
                    f"Invalid result on line {line_number} of {input_path}."
                ) from error

    return results


def completed_pairs(results: list[EvaluationResult]) -> set[tuple[str, str]]:
    """Return record-and-strategy pairs already stored in a result file."""
    return {(result.record_id, result.strategy) for result in results}