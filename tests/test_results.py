from finqa_eval.results import (
    EvaluationResult,
    append_result,
    completed_pairs,
    load_results,
)


def make_result(
    record_id: str = "record-1",
    strategy: str = "direct_answer",
) -> EvaluationResult:
    return EvaluationResult(
        record_id=record_id,
        strategy=strategy,  # type: ignore[arg-type]
        model="gemini-3.5-flash-lite",
        prompt_version="v1",
        output_valid=True,
        final_answer="94",
        answer_correct=True,
        plan_valid=None,
        plan_answer=None,
        latency_ms=123,
    )


def test_load_results_returns_empty_list_for_missing_file(tmp_path) -> None:
    results = load_results(tmp_path / "missing.jsonl")

    assert results == []


def test_append_result_round_trips_through_jsonl(tmp_path) -> None:
    path = tmp_path / "results.jsonl"
    result = make_result()

    append_result(path, result)
    loaded_results = load_results(path)

    assert len(loaded_results) == 1
    assert loaded_results[0].record_id == "record-1"
    assert loaded_results[0].final_answer == "94"
    assert loaded_results[0].latency_ms == 123


def test_completed_pairs_identifies_saved_record_strategy_pairs() -> None:
    results = [
        make_result("record-1", "direct_answer"),
        make_result("record-1", "structured_reasoning"),
    ]

    assert completed_pairs(results) == {
        ("record-1", "direct_answer"),
        ("record-1", "structured_reasoning"),
    }