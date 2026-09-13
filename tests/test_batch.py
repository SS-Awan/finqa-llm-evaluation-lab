from finqa_eval.batch import run_evaluation_batch
from finqa_eval.dataset import FinQARecord
from finqa_eval.results import EvaluationResult, append_result


def make_record(record_id: str) -> FinQARecord:
    return FinQARecord(
        record_id=record_id,
        filename="example.html",
        question="What is the answer?",
        answer="94",
        executable_answer="94",
        program="subtract(5829, 5735)",
        gold_evidence={},
        table=[["Label", "Value"], ["Answer", "94"]],
        pre_text=[],
        post_text=[],
    )


class FakeRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def evaluate_record(self, record: FinQARecord, strategy: str) -> EvaluationResult:
        self.calls.append((record.record_id, strategy))

        return EvaluationResult(
            record_id=record.record_id,
            strategy=strategy,  # type: ignore[arg-type]
            model="test-model",
            prompt_version="v1",
            output_valid=True,
            final_answer="94",
            answer_correct=True,
            plan_valid=None,
            plan_answer=None,
            latency_ms=1,
        )


def test_batch_runs_each_record_strategy_pair_once(tmp_path) -> None:
    runner = FakeRunner()
    output_path = tmp_path / "results.jsonl"

    results = run_evaluation_batch(
        records=[make_record("record-1"), make_record("record-2")],
        strategies=["direct_answer", "structured_reasoning"],
        runner=runner,
        output_path=str(output_path),
    )

    assert len(results) == 4
    assert len(runner.calls) == 4
    assert output_path.exists()


def test_batch_skips_pairs_already_saved(tmp_path) -> None:
    output_path = tmp_path / "results.jsonl"
    append_result(
        output_path,
        EvaluationResult(
            record_id="record-1",
            strategy="direct_answer",
            model="test-model",
            prompt_version="v1",
            output_valid=True,
            final_answer="94",
            answer_correct=True,
            plan_valid=None,
            plan_answer=None,
            latency_ms=1,
        ),
    )
    runner = FakeRunner()

    results = run_evaluation_batch(
        records=[make_record("record-1")],
        strategies=["direct_answer", "structured_reasoning"],
        runner=runner,
        output_path=str(output_path),
    )

    assert len(results) == 1
    assert runner.calls == [("record-1", "structured_reasoning")]


def test_batch_reports_progress_for_each_new_result(tmp_path) -> None:
    runner = FakeRunner()
    progress_updates: list[tuple[int, int, str]] = []

    run_evaluation_batch(
        records=[make_record("record-1")],
        strategies=["direct_answer", "structured_reasoning"],
        runner=runner,
        output_path=str(tmp_path / "results.jsonl"),
        on_result=lambda result, completed, total: progress_updates.append(
            (completed, total, result.strategy)
        ),
    )

    assert progress_updates == [
        (1, 2, "direct_answer"),
        (2, 2, "structured_reasoning"),
    ]


def test_batch_rejects_negative_delay(tmp_path) -> None:
    runner = FakeRunner()

    try:
        run_evaluation_batch(
            records=[],
            strategies=[],
            runner=runner,
            output_path=str(tmp_path / "results.jsonl"),
            delay_seconds=-1,
        )
    except ValueError as error:
        assert str(error) == "delay_seconds cannot be negative."
    else:
        raise AssertionError("Expected a ValueError for a negative delay.")