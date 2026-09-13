import pytest

from finqa_eval.dataset import FinQARecord
from finqa_eval.sampling import build_evaluation_manifest, evidence_type, program_depth


def _record(record_id: str, evidence: str, program: str) -> FinQARecord:
    gold_evidence = {}

    if evidence in {"table", "both"}:
        gold_evidence["table_0"] = "revenue; 100"

    if evidence in {"text", "both"}:
        gold_evidence["text_0"] = "Revenue increased by 10."

    return FinQARecord(
        record_id=record_id,
        filename="report.pdf",
        question="What changed?",
        answer="10",
        executable_answer="10",
        program=program,
        gold_evidence=gold_evidence,
        table=[["metric", "value"], ["revenue", "100"]],
        pre_text=["Revenue information."],
        post_text=[],
    )


def test_manifest_is_deterministic_and_stratified() -> None:
    records = [
        _record("table-a", "table", "add(1, 2)"),
        _record("table-b", "table", "subtract(3, 1)"),
        _record("both-a", "both", "add(1, 2), add(#0, 3), add(#1, 4)"),
    ]
    quotas = {
        ("table", "1"): 1,
        ("both", "3+"): 1,
    }

    first_manifest = build_evaluation_manifest(
        records,
        source_revision="abc123",
        seed="test-seed",
        quotas=quotas,
    )
    second_manifest = build_evaluation_manifest(
        records,
        source_revision="abc123",
        seed="test-seed",
        quotas=quotas,
    )

    assert first_manifest == second_manifest
    assert len(first_manifest["records"]) == 2
    assert evidence_type(records[2]) == "both"
    assert program_depth(records[2]) == "3+"


def test_manifest_rejects_an_unavailable_quota() -> None:
    records = [_record("table-a", "table", "add(1, 2)")]

    with pytest.raises(ValueError, match="requested quota"):
        build_evaluation_manifest(
            records,
            source_revision="abc123",
            quotas={("table", "1"): 2},
        )