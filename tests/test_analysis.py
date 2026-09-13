from finqa_eval.analysis import (
    build_analysis_report,
    build_manifest_metadata,
    calculate_metrics,
)
from finqa_eval.results import EvaluationResult


def make_result(
    record_id: str,
    strategy: str,
    correct: bool,
    latency_ms: int,
    plan_valid: bool | None = None,
) -> EvaluationResult:
    return EvaluationResult(
        record_id=record_id,
        strategy=strategy,  # type: ignore[arg-type]
        model="test-model",
        prompt_version="test",
        output_valid=True,
        final_answer="10",
        answer_correct=correct,
        plan_valid=plan_valid,
        plan_answer="10" if plan_valid is not None else None,
        latency_ms=latency_ms,
    )


def test_calculate_metrics() -> None:
    results = [
        make_result("record-1", "direct_answer", True, 100),
        make_result("record-2", "direct_answer", False, 300),
    ]

    metrics = calculate_metrics(results)

    assert metrics["total_results"] == 2
    assert metrics["correct_answers"] == 1
    assert metrics["accuracy_percent"] == 50.0
    assert metrics["mean_latency_ms"] == 200.0


def test_build_analysis_report_groups_by_manifest_metadata() -> None:
    metadata = build_manifest_metadata(
        [
            {
                "record_id": "record-1",
                "evidence_type": "text",
                "program_depth": "1",
                "first_operation": "add",
            },
            {
                "record_id": "record-2",
                "evidence_type": "table",
                "program_depth": "2",
                "first_operation": "divide",
            },
        ]
    )
    results = [
        make_result("record-1", "direct_answer", True, 100),
        make_result("record-2", "direct_answer", False, 200),
        make_result("record-1", "program_of_thought", True, 150, True),
        make_result("record-2", "program_of_thought", True, 250, True),
    ]

    report = build_analysis_report(results, metadata)

    assert report["overall"]["total_results"] == 4  # type: ignore[index]
    assert report["by_strategy"]["direct_answer"]["accuracy_percent"] == 50.0  # type: ignore[index]
    assert report["by_evidence_type"]["text"]["direct_answer"]["correct_answers"] == 1  # type: ignore[index]
    assert report["by_program_depth"]["2"]["program_of_thought"]["correct_answers"] == 1  # type: ignore[index]