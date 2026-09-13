"""Loading and validating FinQA dataset records."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class DatasetValidationError(ValueError):
    """Raised when a FinQA dataset file has an unexpected structure."""


@dataclass(frozen=True)
class FinQARecord:
    """The FinQA fields required by this evaluation lab."""

    record_id: str
    filename: str
    question: str
    answer: str
    executable_answer: str
    program: str
    gold_evidence: dict[str, str]
    table: list[list[str]]
    pre_text: list[str]
    post_text: list[str]

    @property
    def text_context(self) -> list[str]:
        """Return all narrative evidence in its original order."""
        return [*self.pre_text, *self.post_text]


def load_finqa_records(path: str | Path) -> list[FinQARecord]:
    """Load and validate records from one official FinQA JSON split."""
    source_path = Path(path)

    with source_path.open(encoding="utf-8") as file:
        raw_records = json.load(file)

    if not isinstance(raw_records, list):
        raise DatasetValidationError("FinQA dataset root must be a JSON list.")

    return [_parse_record(raw_record, index) for index, raw_record in enumerate(raw_records)]


def _parse_record(raw_record: Any, index: int) -> FinQARecord:
    if not isinstance(raw_record, dict):
        raise DatasetValidationError(f"Record {index} must be a JSON object.")

    qa = raw_record.get("qa")
    if not isinstance(qa, dict):
        raise DatasetValidationError(f"Record {index} is missing a valid 'qa' object.")

    required_record_fields = ("id", "filename", "table", "pre_text", "post_text")
    required_qa_fields = ("question", "answer", "exe_ans", "program", "gold_inds")

    for field in required_record_fields:
        if field not in raw_record:
            raise DatasetValidationError(f"Record {index} is missing '{field}'.")

    for field in required_qa_fields:
        if field not in qa:
            raise DatasetValidationError(f"Record {index} qa is missing '{field}'.")

    return FinQARecord(
        record_id=_require_string(raw_record["id"], f"Record {index} id"),
        filename=_require_string(raw_record["filename"], f"Record {index} filename"),
        question=_require_string(qa["question"], f"Record {index} question"),
        answer=_require_scalar_as_string(qa["answer"], f"Record {index} answer"),
        executable_answer=_require_scalar_as_string(qa["exe_ans"], f"Record {index} exe_ans"),
        program=_require_string(qa["program"], f"Record {index} program"),
        gold_evidence=_require_string_dict(qa["gold_inds"], f"Record {index} gold_inds"),
        table=_require_table(raw_record["table"], f"Record {index} table"),
        pre_text=_require_string_list(raw_record["pre_text"], f"Record {index} pre_text"),
        post_text=_require_string_list(raw_record["post_text"], f"Record {index} post_text"),
    )


def _require_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise DatasetValidationError(f"{field_name} must be a string.")
    return value


def _require_scalar_as_string(value: Any, field_name: str) -> str:
    """Accept a FinQA answer stored as either text or a JSON number."""
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise DatasetValidationError(f"{field_name} must be text or a number.")
    return str(value)


def _require_string_list(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise DatasetValidationError(f"{field_name} must be a list of strings.")
    return value


def _require_string_dict(value: Any, field_name: str) -> dict[str, str]:
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(item, str) for key, item in value.items()
    ):
        raise DatasetValidationError(f"{field_name} must be a dictionary of strings.")
    return value


def _require_table(value: Any, field_name: str) -> list[list[str]]:
    if not isinstance(value, list) or not all(
        isinstance(row, list) and all(isinstance(cell, str) for cell in row) for row in value
    ):
        raise DatasetValidationError(f"{field_name} must be a table of string cells.")
    return value