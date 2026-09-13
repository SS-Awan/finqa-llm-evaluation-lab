import pytest
from pydantic import ValidationError

from finqa_eval.schemas import (
    DirectAnswerResponse,
    ProgramOfThoughtResponse,
    StructuredReasoningResponse,
)


def test_direct_answer_requires_a_final_answer() -> None:
    response = DirectAnswerResponse(final_answer="94")

    assert response.final_answer == "94"


def test_structured_reasoning_requires_evidence() -> None:
    with pytest.raises(ValidationError, match="at least 1 item"):
        StructuredReasoningResponse(
            evidence=[],
            reasoning_summary="Subtract the earlier revenue from the later revenue.",
            final_answer="94",
        )


def test_program_of_thought_accepts_a_safe_plan() -> None:
    response = ProgramOfThoughtResponse.model_validate(
        {
            "evidence": ["2014 net revenue: 5735", "2015 net revenue: 5829"],
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

    assert response.calculation_plan.steps[0].operation == "subtract"


def test_direct_answer_rejects_unexpected_fields() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        DirectAnswerResponse.model_validate(
            {"final_answer": "94", "reasoning": "This should not be here."}
        )