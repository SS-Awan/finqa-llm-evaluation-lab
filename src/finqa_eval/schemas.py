"""Structured response contracts for the evaluation strategies."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from finqa_eval.plans import CalculationPlan


class DirectAnswerResponse(BaseModel):
    """Response format for the Direct Answer strategy."""

    model_config = ConfigDict(extra="forbid")

    final_answer: str = Field(min_length=1)


class StructuredReasoningResponse(BaseModel):
    """Response format for concise evidence-and-reasoning responses."""

    model_config = ConfigDict(extra="forbid")

    evidence: list[str] = Field(min_length=1)
    reasoning_summary: str = Field(min_length=1)
    final_answer: str = Field(min_length=1)


class ProgramOfThoughtResponse(BaseModel):
    """Response format for safe executable calculation-plan strategies."""

    model_config = ConfigDict(extra="forbid")

    evidence: list[str] = Field(min_length=1)
    calculation_plan: CalculationPlan
    final_answer: str = Field(min_length=1)


StrategyName = Literal[
    "direct_answer",
    "structured_reasoning",
    "program_of_thought",
    "few_shot_program_of_thought",
]