"""Deterministic selection of a stratified FinQA evaluation sample."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Mapping

from finqa_eval.dataset import FinQARecord
from finqa_eval.scoring import is_numeric_answer


Stratum = tuple[str, str]

DEFAULT_SELECTION_SEED = "finqa-evaluation-lab-v1"

DEFAULT_STRATA_QUOTAS: dict[Stratum, int] = {
    ("table", "1"): 16,
    ("table", "2"): 20,
    ("table", "3+"): 12,
    ("text", "1"): 8,
    ("text", "2"): 10,
    ("text", "3+"): 8,
    ("both", "1"): 6,
    ("both", "2"): 8,
    ("both", "3+"): 8,
}


def evidence_type(record: FinQARecord) -> str:
    """Classify whether annotated evidence comes from table, text, or both."""
    has_table = any(key.startswith("table_") for key in record.gold_evidence)
    has_text = any(key.startswith("text_") for key in record.gold_evidence)

    if has_table and has_text:
        return "both"
    if has_table:
        return "table"
    if has_text:
        return "text"

    raise ValueError(f"Record {record.record_id} has no recognized evidence type.")


def program_depth(record: FinQARecord) -> str:
    """Return program length as 1, 2, or 3+ steps."""
    operation_count = len(re.findall(r"([a-z_]+)\(", record.program))

    if operation_count >= 3:
        return "3+"
    return str(operation_count)


def first_operation(record: FinQARecord) -> str:
    """Return the first calculation operation in a FinQA gold program."""
    operations = re.findall(r"([a-z_]+)\(", record.program)

    if not operations:
        raise ValueError(f"Record {record.record_id} has no recognized operation.")

    return operations[0].removeprefix("table_")


def build_evaluation_manifest(
    records: list[FinQARecord],
    *,
    source_revision: str,
    seed: str = DEFAULT_SELECTION_SEED,
    quotas: Mapping[Stratum, int] = DEFAULT_STRATA_QUOTAS,
) -> dict:
    """Create a deterministic, stratified manifest containing FinQA record IDs."""
    grouped_records: dict[Stratum, list[FinQARecord]] = defaultdict(list)

    for record in records:
        if not is_numeric_answer(record.answer):
            continue

        stratum = (evidence_type(record), program_depth(record))
        if stratum in quotas:
            grouped_records[stratum].append(record)

    selected_records: list[dict[str, str]] = []

    for stratum, quota in quotas.items():
        candidates = grouped_records[stratum]

        if len(candidates) < quota:
            raise ValueError(
                f"Stratum {stratum} has {len(candidates)} eligible records, "
                f"but the requested quota is {quota}."
            )

        ranked_candidates = sorted(
            candidates,
            key=lambda record: _selection_rank(record.record_id, seed),
        )

        for record in ranked_candidates[:quota]:
            selected_records.append(
                {
                    "record_id": record.record_id,
                    "evidence_type": evidence_type(record),
                    "program_depth": program_depth(record),
                    "first_operation": first_operation(record),
                    "selection_rank": _selection_rank(record.record_id, seed),
                }
            )

    selected_ids = [record["record_id"] for record in selected_records]
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("Evaluation manifest contains duplicate record IDs.")

    return {
        "schema_version": 1,
        "dataset": "FinQA",
        "source_split": "test",
        "source_revision": source_revision,
        "selection_seed": seed,
        "selection_method": "stratified deterministic hash ranking",
        "eligibility_rule": "ground-truth answer parses as a number or percentage",
        "strata_quotas": {
            f"{evidence}|{depth}": quota
            for (evidence, depth), quota in quotas.items()
        },
        "records": sorted(selected_records, key=lambda record: record["record_id"]),
    }


def write_evaluation_manifest(
    records: list[FinQARecord],
    destination: str | Path,
    *,
    source_revision: str,
) -> dict:
    """Create and save a reproducible evaluation manifest."""
    manifest = build_evaluation_manifest(
        records,
        source_revision=source_revision,
    )

    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    return manifest


def _selection_rank(record_id: str, seed: str) -> str:
    value = f"{seed}:{record_id}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()