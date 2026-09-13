"""Validated, restricted calculation plans for Program-of-Thought evaluation."""

from __future__ import annotations

from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator


Operation = Literal[
    "add",
    "subtract",
    "multiply",
    "divide",
    "power",
    "max",
    "min",
    "sum",
    "average",
]

NumericValue = Decimal | int | float


class PlanValidationError(ValueError):
    """Raised when a calculation plan cannot be safely executed."""


class Operand(BaseModel):
    """Either a literal number or a reference to a previous step result."""

    model_config = ConfigDict(extra="forbid")

    value: NumericValue | None = None
    reference: int | None = None

    @model_validator(mode="after")
    def validate_exactly_one_source(self) -> "Operand":
        if (self.value is None) == (self.reference is None):
            raise ValueError("An operand needs exactly one of: value or reference.")

        if self.reference is not None and self.reference < 0:
            raise ValueError("An operand reference cannot be negative.")

        return self


class CalculationStep(BaseModel):
    """One restricted mathematical operation."""

    model_config = ConfigDict(extra="forbid")

    operation: Operation
    operands: list[Operand]

    @model_validator(mode="after")
    def validate_operand_count(self) -> "CalculationStep":
        count = len(self.operands)

        if self.operation in {"subtract", "divide", "power"} and count != 2:
            raise ValueError(f"{self.operation} requires exactly two operands.")

        if self.operation in {"add", "multiply"} and count < 2:
            raise ValueError(f"{self.operation} requires at least two operands.")

        if self.operation in {"max", "min", "sum", "average"} and count < 1:
            raise ValueError(f"{self.operation} requires at least one operand.")

        return self


class CalculationPlan(BaseModel):
    """A sequence of calculation steps whose last result is the final answer."""

    model_config = ConfigDict(extra="forbid")

    steps: list[CalculationStep]

    @model_validator(mode="after")
    def validate_non_empty_steps(self) -> "CalculationPlan":
        if not self.steps:
            raise ValueError("A calculation plan must contain at least one step.")
        return self


def execute_plan(plan: CalculationPlan) -> Decimal:
    """Safely execute a validated calculation plan and return its final result."""
    results: list[Decimal] = []

    for step_index, step in enumerate(plan.steps):
        values = [
            _resolve_operand(operand, results, step_index)
            for operand in step.operands
        ]
        results.append(_execute_step(step.operation, values))

    return results[-1]


def _resolve_operand(
    operand: Operand,
    results: list[Decimal],
    current_step_index: int,
) -> Decimal:
    if operand.value is not None:
        return Decimal(str(operand.value))

    assert operand.reference is not None

    if operand.reference >= current_step_index:
        raise PlanValidationError(
            f"Step {current_step_index} cannot reference step {operand.reference}."
        )

    return results[operand.reference]


def _execute_step(operation: Operation, values: list[Decimal]) -> Decimal:
    if operation == "add":
        return sum(values, Decimal("0"))

    if operation == "subtract":
        return values[0] - values[1]

    if operation == "multiply":
        result = Decimal("1")
        for value in values:
            result *= value
        return result

    if operation == "divide":
        if values[1] == 0:
            raise PlanValidationError("Division by zero is not allowed.")
        return values[0] / values[1]

    if operation == "power":
        try:
            result = float(values[0]) ** float(values[1])
            return Decimal(str(result))
        except (OverflowError, ValueError):
            raise PlanValidationError("Invalid power operation.") from None

    if operation == "max":
        return max(values)

    if operation == "min":
        return min(values)

    if operation == "sum":
        return sum(values, Decimal("0"))

    if operation == "average":
        return sum(values, Decimal("0")) / Decimal(len(values))

    raise PlanValidationError(f"Unsupported operation: {operation}")