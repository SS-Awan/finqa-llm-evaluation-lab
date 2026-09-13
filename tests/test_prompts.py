import pytest

from finqa_eval.dataset import FinQARecord
from finqa_eval.prompts import FewShotExample, build_prompt, format_record_context


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


def test_format_record_context_includes_table_and_text() -> None:
    context = format_record_context(make_record())

    assert "2014 | 5735" in context
    assert "The company reported annual revenue." in context
    assert "Amounts are in millions." in context


def test_direct_answer_prompt_requests_no_reasoning() -> None:
    prompt = build_prompt(make_record(), "direct_answer")

    assert 'Return JSON only: {"final_answer": "numeric answer"}.' in prompt
    assert "Do not include reasoning." in prompt
    assert "What is the change in revenue?" in prompt


def test_few_shot_prompt_requires_examples() -> None:
    with pytest.raises(ValueError, match="requires fixed examples"):
        build_prompt(make_record(), "few_shot_program_of_thought")


def test_few_shot_prompt_includes_fixed_example() -> None:
    example = FewShotExample(
        context="TABLE:\nYear | Revenue\n2020 | 100",
        question="What is the revenue?",
        response_json='{"final_answer": "100"}',
    )

    prompt = build_prompt(
        make_record(),
        "few_shot_program_of_thought",
        few_shot_examples=[example],
    )

    assert "WORKED EXAMPLE 1:" in prompt
    assert "What is the revenue?" in prompt
    assert '"final_answer": "100"' in prompt