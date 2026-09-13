"""Prompt builders for the four FinQA evaluation strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from finqa_eval.dataset import FinQARecord
from finqa_eval.schemas import StrategyName


@dataclass(frozen=True)
class FewShotExample:
    """A fixed worked example used only by the few-shot strategy."""

    context: str
    question: str
    response_json: str


def build_prompt(
    record: FinQARecord,
    strategy: StrategyName,
    few_shot_examples: Sequence[FewShotExample] = (),
) -> str:
    """Build a deterministic prompt for one record and one strategy."""
    context = format_record_context(record)

    sections = [
        "You answer financial numerical-reasoning questions using only the supplied document.",
        "Do not use outside knowledge. Do not invent values.",
        _strategy_instructions(strategy),
    ]

    if strategy == "few_shot_program_of_thought":
        if not few_shot_examples:
            raise ValueError("Few-shot Program-of-Thought requires fixed examples.")
        sections.append(_format_few_shot_examples(few_shot_examples))

    sections.extend(
        [
            f"DOCUMENT:\n{context}",
            f"QUESTION:\n{record.question}",
        ]
    )

    return "\n\n".join(sections)


def format_record_context(record: FinQARecord) -> str:
    """Format a FinQA record's table and text into a readable prompt context."""
    table_lines = [" | ".join(row) for row in record.table]

    sections = [
        "TABLE:",
        "\n".join(table_lines),
        "PRE-TEXT:",
        "\n".join(record.pre_text),
        "POST-TEXT:",
        "\n".join(record.post_text),
    ]

    return "\n".join(sections)


def _strategy_instructions(strategy: StrategyName) -> str:
    if strategy == "direct_answer":
        return (
            'Return JSON only: {"final_answer": "numeric answer"}. '
            "Do not include reasoning."
        )

    if strategy == "structured_reasoning":
        return (
            "Return JSON only with: evidence (a non-empty list of quoted values or "
            "facts), reasoning_summary (a concise calculation description), and "
            "final_answer (a numeric answer)."
        )

    if strategy in {"program_of_thought", "few_shot_program_of_thought"}:
        return (
            "Return JSON only with: evidence (a non-empty list of quoted values or "
            "facts), calculation_plan, and final_answer. The calculation_plan must "
            "contain steps. Each step uses only add, subtract, multiply, divide, "
            "power, max, min, sum, or average. Each operand must be either "
            '{"value": number} or {"reference": earlier_step_index}. '
            "Never write Python code or use any operation outside this list."
        )

    raise ValueError(f"Unsupported strategy: {strategy}")


def _format_few_shot_examples(examples: Sequence[FewShotExample]) -> str:
    formatted_examples = []

    for index, example in enumerate(examples, start=1):
        formatted_examples.append(
            "\n".join(
                [
                    f"WORKED EXAMPLE {index}:",
                    f"DOCUMENT:\n{example.context}",
                    f"QUESTION:\n{example.question}",
                    f"JSON RESPONSE:\n{example.response_json}",
                ]
            )
        )

    return "\n\n".join(formatted_examples)