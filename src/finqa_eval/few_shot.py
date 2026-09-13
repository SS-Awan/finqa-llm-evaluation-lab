"""Fixed development-set examples for Few-shot Program-of-Thought."""

from __future__ import annotations

import json
from dataclasses import dataclass

from finqa_eval.dataset import FinQARecord, load_finqa_records
from finqa_eval.prompts import FewShotExample, format_record_context


@dataclass(frozen=True)
class FewShotSpecification:
    """A pre-registered development record and its target response format."""

    record_id: str
    response: dict[str, object]


FIXED_FEW_SHOT_SPECS: tuple[FewShotSpecification, ...] = (
    FewShotSpecification(
        record_id="PNC/2013/page_62.pdf-2",
        response={
            "evidence": [
                "Residential mortgages balance for 2013: 1356",
                "Residential mortgages balance for 2012: 2220",
            ],
            "calculation_plan": {
                "steps": [
                    {
                        "operation": "add",
                        "operands": [{"value": 1356}, {"value": 2220}],
                    }
                ]
            },
            "final_answer": "3576",
        },
    ),
    FewShotSpecification(
        record_id="PM/2017/page_38.pdf-1",
        response={
            "evidence": [
                "Operating income in 2017: 11503",
                "Operating income in 2016: 10815",
            ],
            "calculation_plan": {
                "steps": [
                    {
                        "operation": "subtract",
                        "operands": [{"value": 11503}, {"value": 10815}],
                    }
                ]
            },
            "final_answer": "688",
        },
    ),
    FewShotSpecification(
        record_id="ETR/2002/page_86.pdf-3",
        response={
            "evidence": [
                "Annual other sinking fund requirements: 30.2 million",
                "Annual long-term debt maturities: 475288 thousand",
            ],
            "calculation_plan": {
                "steps": [
                    {
                        "operation": "multiply",
                        "operands": [{"value": 30.2}, {"value": 1000}],
                    },
                    {
                        "operation": "divide",
                        "operands": [{"reference": 0}, {"value": 475288}],
                    },
                    {
                        "operation": "multiply",
                        "operands": [{"reference": 1}, {"value": 100}],
                    },
                ]
            },
            "final_answer": "6.35%",
        },
    ),
    FewShotSpecification(
        record_id="V/2008/page_17.pdf-1",
        response={
            "evidence": [
                "American Express payment volume: 637",
                "American Express transactions: 5",
            ],
            "calculation_plan": {
                "steps": [
                    {
                        "operation": "divide",
                        "operands": [{"value": 637}, {"value": 5}],
                    }
                ]
            },
            "final_answer": "127.40",
        },
    ),
)


def load_fixed_few_shot_examples(dataset_path: str) -> list[FewShotExample]:
    """Load the four pre-registered examples from the FinQA development split."""
    records = load_finqa_records(dataset_path)
    records_by_id = {record.record_id: record for record in records}

    examples: list[FewShotExample] = []

    for specification in FIXED_FEW_SHOT_SPECS:
        record = records_by_id.get(specification.record_id)

        if record is None:
            raise ValueError(
                f"Fixed few-shot record was not found: {specification.record_id}"
            )

        examples.append(
            FewShotExample(
                context=format_record_context(record),
                question=record.question,
                response_json=json.dumps(specification.response, indent=2),
            )
        )

    return examples