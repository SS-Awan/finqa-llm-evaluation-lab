"""Controlled evaluation of one FinQA record under one prompt strategy."""

from __future__ import annotations

from decimal import Decimal
from time import perf_counter
from typing import Any

from pydantic import BaseModel

from finqa_eval.dataset import FinQARecord
from finqa_eval.few_shot import load_fixed_few_shot_examples
from finqa_eval.gemini_client import GenerationError
from finqa_eval.plans import PlanValidationError, execute_plan
from finqa_eval.prompts import FewShotExample, build_prompt
from finqa_eval.results import EvaluationResult
from finqa_eval.schemas import (
    DirectAnswerResponse,
    ProgramOfThoughtResponse,
    StrategyName,
    StructuredReasoningResponse,
)
from finqa_eval.scoring import answers_match, parse_numeric_answer


PROMPT_VERSION = "v1"


class EvaluationRunner:
    """Evaluate records while keeping strategy logic and result fields consistent."""

    def __init__(
        self,
        client: Any,
        model: str,
        few_shot_examples: list[FewShotExample] | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.few_shot_examples = few_shot_examples or []

    @classmethod
    def with_fixed_few_shot_examples(
        cls,
        client: Any,
        model: str,
        development_dataset_path: str,
    ) -> "EvaluationRunner":
        """Create a runner with the four registered development examples."""
        examples = load_fixed_few_shot_examples(development_dataset_path)
        return cls(client=client, model=model, few_shot_examples=examples)

    def evaluate_record(
        self,
        record: FinQARecord,
        strategy: StrategyName,
    ) -> EvaluationResult:
        """Run one strategy on one record and return a scoreable result."""
        prompt = build_prompt(
            record,
            strategy,
            few_shot_examples=self.few_shot_examples,
        )
        response_model = _response_model_for_strategy(strategy)
        started_at = perf_counter()

        try:
            response = self.client.generate(prompt, response_model)
        except GenerationError as error:
            return _failed_result(
                record=record,
                strategy=strategy,
                model=self.model,
                latency_ms=_elapsed_ms(started_at),
                error=str(error),
            )

        final_answer = response.final_answer
        plan_valid: bool | None = None
        plan_answer: str | None = None

        if isinstance(response, ProgramOfThoughtResponse):
            try:
                executed_answer = execute_plan(response.calculation_plan)
                plan_valid = True
                plan_answer = _format_decimal(executed_answer)
            except PlanValidationError as error:
                plan_valid = False
                return EvaluationResult(
                    record_id=record.record_id,
                    strategy=strategy,
                    model=self.model,
                    prompt_version=PROMPT_VERSION,
                    output_valid=True,
                    final_answer=final_answer,
                    answer_correct=_answer_is_correct(record.answer, final_answer),
                    plan_valid=plan_valid,
                    plan_answer=None,
                    latency_ms=_elapsed_ms(started_at),
                    error=str(error),
                )

        return EvaluationResult(
            record_id=record.record_id,
            strategy=strategy,
            model=self.model,
            prompt_version=PROMPT_VERSION,
            output_valid=True,
            final_answer=final_answer,
            answer_correct=_answer_is_correct(record.answer, final_answer),
            plan_valid=plan_valid,
            plan_answer=plan_answer,
            latency_ms=_elapsed_ms(started_at),
        )


def _response_model_for_strategy(strategy: StrategyName) -> type[BaseModel]:
    if strategy == "direct_answer":
        return DirectAnswerResponse

    if strategy == "structured_reasoning":
        return StructuredReasoningResponse

    if strategy in {"program_of_thought", "few_shot_program_of_thought"}:
        return ProgramOfThoughtResponse

    raise ValueError(f"Unsupported strategy: {strategy}")


def _answer_is_correct(expected_answer: str, predicted_answer: str) -> bool:
    """Score numeric answers while safely rejecting unsupported formats."""
    expected_numeric_answer = parse_numeric_answer(expected_answer)
    predicted_numeric_answer = parse_numeric_answer(predicted_answer)

    if expected_numeric_answer is None or predicted_numeric_answer is None:
        return False

    return answers_match(expected_numeric_answer, predicted_numeric_answer)


def _failed_result(
    record: FinQARecord,
    strategy: StrategyName,
    model: str,
    latency_ms: int,
    error: str,
) -> EvaluationResult:
    return EvaluationResult(
        record_id=record.record_id,
        strategy=strategy,
        model=model,
        prompt_version=PROMPT_VERSION,
        output_valid=False,
        final_answer=None,
        answer_correct=None,
        plan_valid=None,
        plan_answer=None,
        latency_ms=latency_ms,
        error=error,
    )


def _elapsed_ms(started_at: float) -> int:
    return round((perf_counter() - started_at) * 1000)


def _format_decimal(value: Decimal) -> str:
    text = format(value, "f")

    if "." in text:
        return text.rstrip("0").rstrip(".")

    return text