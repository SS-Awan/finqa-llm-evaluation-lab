from decimal import Decimal

import pytest
from pydantic import ValidationError

from finqa_eval.plans import (
    CalculationPlan,
    CalculationStep,
    Operand,
    PlanValidationError,
    execute_plan,
)


def test_execute_plan_uses_previous_step_results() -> None:
    plan = CalculationPlan(
        steps=[
            CalculationStep(
                operation="subtract",
                operands=[Operand(value=5829), Operand(value=5735)],
            ),
            CalculationStep(
                operation="divide",
                operands=[Operand(reference=0), Operand(value=94)],
            ),
        ]
    )

    assert execute_plan(plan) == Decimal("1")


def test_execute_plan_supports_average() -> None:
    plan = CalculationPlan(
        steps=[
            CalculationStep(
                operation="average",
                operands=[Operand(value=10), Operand(value=20), Operand(value=30)],
            )
        ]
    )

    assert execute_plan(plan) == Decimal("20")


def test_execute_plan_rejects_future_step_references() -> None:
    plan = CalculationPlan(
        steps=[
            CalculationStep(
                operation="add",
                operands=[Operand(reference=0), Operand(value=1)],
            )
        ]
    )

    with pytest.raises(PlanValidationError, match="cannot reference"):
        execute_plan(plan)


def test_execute_plan_rejects_division_by_zero() -> None:
    plan = CalculationPlan(
        steps=[
            CalculationStep(
                operation="divide",
                operands=[Operand(value=10), Operand(value=0)],
            )
        ]
    )

    with pytest.raises(PlanValidationError, match="Division by zero"):
        execute_plan(plan)


def test_operand_rejects_value_and_reference_together() -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        Operand(value=10, reference=0)