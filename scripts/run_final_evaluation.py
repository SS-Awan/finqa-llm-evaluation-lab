"""Run the locked 96-record FinQA evaluation with four prompt strategies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from finqa_eval.batch import run_evaluation_batch
from finqa_eval.config import load_gemini_settings
from finqa_eval.dataset import FinQARecord, load_finqa_records
from finqa_eval.gemini_client import GeminiStructuredClient
from finqa_eval.results import load_results
from finqa_eval.runner import EvaluationRunner


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_DATASET_PATH = (
    PROJECT_ROOT / "data" / "raw" / "finqa-source" / "dataset" / "test.json"
)
MANIFEST_PATH = (
    PROJECT_ROOT / "data" / "evaluation" / "finqa_test_96_manifest.json"
)
OUTPUT_PATH = PROJECT_ROOT / "data" / "results" / "finqa_test_96_results.jsonl"

EXPECTED_RECORD_COUNT = 96
REQUEST_DELAY_SECONDS = 4.1

STRATEGIES = (
    "direct_answer",
    "structured_reasoning",
    "program_of_thought",
    "few_shot_program_of_thought",
)


def load_manifest_records() -> list[FinQARecord]:
    """Load the frozen manifest IDs and resolve them to source test records."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest_records = manifest["records"]
    record_ids = [item["record_id"] for item in manifest_records]

    if len(record_ids) != EXPECTED_RECORD_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_RECORD_COUNT} manifest records, found {len(record_ids)}."
        )

    if len(set(record_ids)) != len(record_ids):
        raise ValueError("The evaluation manifest contains duplicate record IDs.")

    test_records = load_finqa_records(TEST_DATASET_PATH)
    records_by_id = {record.record_id: record for record in test_records}

    selected_records = []

    for record_id in record_ids:
        record = records_by_id.get(record_id)

        if record is None:
            raise ValueError(f"Manifest record was not found in test data: {record_id}")

        selected_records.append(record)

    return selected_records


def print_progress(completed: int, total: int, record_id: str, strategy: str) -> None:
    """Print one compact progress line after a safely saved result."""
    print(f"[{completed}/{total}] {record_id} | {strategy}", flush=True)


def main(dry_run: bool) -> None:
    records = load_manifest_records()
    expected_calls = len(records) * len(STRATEGIES)
    existing_results = load_results(OUTPUT_PATH)

    print(f"Frozen test records: {len(records)}")
    print(f"Strategies per record: {len(STRATEGIES)}")
    print(f"Expected total results: {expected_calls}")
    print(f"Previously saved results: {len(existing_results)}")

    if dry_run:
        print("Dry run complete. No Gemini request was made.")
        return

    settings = load_gemini_settings()
    client = GeminiStructuredClient(settings)
    runner = EvaluationRunner.with_fixed_few_shot_examples(
        client=client,
        model=settings.model,
        development_dataset_path=str(
            PROJECT_ROOT / "data" / "raw" / "finqa-source" / "dataset" / "dev.json"
        ),
    )

    new_results = run_evaluation_batch(
        records=records,
        strategies=STRATEGIES,
        runner=runner,
        output_path=str(OUTPUT_PATH),
        delay_seconds=REQUEST_DELAY_SECONDS,
        on_result=lambda result, completed, total: print_progress(
            completed,
            total,
            result.record_id,
            result.strategy,
        ),
    )

    all_results = load_results(OUTPUT_PATH)

    print(f"New results saved: {len(new_results)}")
    print(f"Total saved results: {len(all_results)}")
    print(f"Result file: {OUTPUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate selection and request count without calling Gemini.",
    )
    arguments = parser.parse_args()

    main(dry_run=arguments.dry_run)