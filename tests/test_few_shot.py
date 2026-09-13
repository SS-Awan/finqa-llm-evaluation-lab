import pytest

import finqa_eval.few_shot as few_shot
from finqa_eval.dataset import FinQARecord


def make_record(record_id: str) -> FinQARecord:
    return FinQARecord(
        record_id=record_id,
        filename="example.html",
        question=f"Question for {record_id}",
        answer="100",
        executable_answer="100",
        program="add(40, 60)",
        gold_evidence={"table_1": "Example evidence"},
        table=[["Label", "Value"], ["Example", "100"]],
        pre_text=["Example pre-text."],
        post_text=["Example post-text."],
    )


def test_fixed_few_shot_specs_are_four_distinct_records() -> None:
    record_ids = [specification.record_id for specification in few_shot.FIXED_FEW_SHOT_SPECS]

    assert len(record_ids) == 4
    assert len(set(record_ids)) == 4


def test_load_fixed_few_shot_examples_uses_registered_records(monkeypatch) -> None:
    records = [
        make_record(specification.record_id)
        for specification in few_shot.FIXED_FEW_SHOT_SPECS
    ]
    monkeypatch.setattr(few_shot, "load_finqa_records", lambda _path: records)

    examples = few_shot.load_fixed_few_shot_examples("unused-dev.json")

    assert len(examples) == 4
    assert examples[0].question == "Question for PNC/2013/page_62.pdf-2"
    assert '"final_answer": "3576"' in examples[0].response_json
    assert "TABLE:" in examples[0].context


def test_load_fixed_few_shot_examples_rejects_missing_record(monkeypatch) -> None:
    monkeypatch.setattr(few_shot, "load_finqa_records", lambda _path: [])

    with pytest.raises(ValueError, match="was not found"):
        few_shot.load_fixed_few_shot_examples("unused-dev.json")