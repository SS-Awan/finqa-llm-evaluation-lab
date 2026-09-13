from typing import Any

from finqa_eval.dataset import FinQARecord
from finqa_eval.gemini_client import GenerationError
from finqa_eval.runner import EvaluationRunner
from finqa_eval.schemas import DirectAnswerResponse, ProgramOfThoughtResponse


def make_record() -> FinQARecord:
    return FinQARecord(
        record_id="example-1",
        filename="example.html",
        question="What is the change in revenue?",
        answer="94",
        executable_answer="94",
        program="subtract(5829, 5735)",
        gold_evidence={"table_1": "2014 revenue", "table_2": "2015 revenue"},
        table=[
            ["Year", "Revenue"],
            ["2014", "5735"],
            ["2015", "5829"],
        ],
        pre_text=["The company reported annual revenue."],
        post_text=["Amounts are in millions."],
    )


class FakeClient:
    def __init__(self, response: Any | None = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[tuple[str, object]] = []

    def generate(self, prompt: str, response_model: object) -> Any:
        self.calls.append((prompt, response_model))

        if self.error is not None:
            raise self.error

        return self.response


def test_runner_scores_correct_direct_answer() -> None:
    client = FakeClient(response=DirectAnswerResponse(final_answer="94"))
    runner = EvaluationRunner(client=client, model="test-model")

    result = runner.evaluate_record(make_record(), "direct_answer")

    assert result.output_valid is True
    assert result.answer_correct is True
    assert result.plan_valid is None
    assert "What is the change in revenue?" in client.calls[0][0]


def test_runner_executes_program_of_thought_plan() -> None:
    response = ProgramOfThoughtResponse.model_validate(
        {
            "evidence": ["2014 revenue: 5735", "2015 revenue: 5829"],
            "calculation_plan": {
                "steps": [
                    {
                        "operation": "subtract",
                        "operands": [{"value": 5829}, {"value": 5735}],
                    }
                ]
            },
            "final_answer": "94",
        }
    )
    runner = EvaluationRunner(client=FakeClient(response=response), model="test-model")

    result = runner.evaluate_record(make_record(), "program_of_thought")

    assert result.answer_correct is True
    assert result.plan_valid is True
    assert result.plan_answer == "94"


def test_runner_records_generation_failure() -> None:
    runner = EvaluationRunner(
        client=FakeClient(error=GenerationError("Temporary API problem")),
        model="test-model",
    )

    result = runner.evaluate_record(make_record(), "direct_answer")

    assert result.output_valid is False
    assert result.answer_correct is None
    assert result.error == "Temporary API problem"