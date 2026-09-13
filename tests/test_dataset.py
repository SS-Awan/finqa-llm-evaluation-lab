import json

import pytest

from finqa_eval.dataset import DatasetValidationError, load_finqa_records


def _valid_record() -> dict:
    return {
        "id": "example-1",
        "filename": "report.pdf",
        "table": [["", "amount"], ["2014 revenue", "$ 5735"]],
        "pre_text": ["A short paragraph before the table."],
        "post_text": ["A short paragraph after the table."],
        "qa": {
            "question": "What was revenue?",
            "answer": 5735,
            "exe_ans": 5735,
            "program": "add(5735, 0)",
            "gold_inds": {"table_0": "2014 revenue; $ 5735"},
        },
    }


def test_load_finqa_records_normalizes_numeric_answers(tmp_path) -> None:
    dataset_path = tmp_path / "sample.json"
    dataset_path.write_text(json.dumps([_valid_record()]), encoding="utf-8")

    records = load_finqa_records(dataset_path)

    assert len(records) == 1
    assert records[0].record_id == "example-1"
    assert records[0].answer == "5735"
    assert records[0].executable_answer == "5735"
    assert records[0].text_context == [
        "A short paragraph before the table.",
        "A short paragraph after the table.",
    ]


def test_load_finqa_records_rejects_missing_required_fields(tmp_path) -> None:
    invalid_record = _valid_record()
    del invalid_record["table"]

    dataset_path = tmp_path / "invalid.json"
    dataset_path.write_text(json.dumps([invalid_record]), encoding="utf-8")

    with pytest.raises(DatasetValidationError, match="missing 'table'"):
        load_finqa_records(dataset_path)