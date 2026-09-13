"""Run a small, paced Gemini pilot on development-set records."""

from __future__ import annotations

from pathlib import Path

from finqa_eval.batch import run_evaluation_batch
from finqa_eval.config import load_gemini_settings
from finqa_eval.dataset import load_finqa_records
from finqa_eval.gemini_client import GeminiStructuredClient
from finqa_eval.runner import EvaluationRunner


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT_DATASET_PATH = (
    PROJECT_ROOT / "data" / "raw" / "finqa-source" / "dataset" / "dev.json"
)
OUTPUT_PATH = PROJECT_ROOT / "data" / "results" / "pilot_dev_results.jsonl"

PILOT_RECORD_IDS = (
    "ETR/2011/page_341.pdf-3",
    "CME/2017/page_97.pdf-5",
    "ETR/2004/page_213.pdf-2",
)

STRATEGIES = (
    "direct_answer",
    "structured_reasoning",
    "program_of_thought",
    "few_shot_program_of_thought",
)

REQUEST_DELAY_SECONDS = 4.1


def main() -> None:
    settings = load_gemini_settings()
    all_records = load_finqa_records(DEVELOPMENT_DATASET_PATH)
    records_by_id = {record.record_id: record for record in all_records}

    pilot_records = []

    for record_id in PILOT_RECORD_IDS:
        record = records_by_id.get(record_id)

        if record is None:
            raise ValueError(f"Pilot record was not found: {record_id}")

        pilot_records.append(record)

    client = GeminiStructuredClient(settings)
    runner = EvaluationRunner.with_fixed_few_shot_examples(
        client=client,
        model=settings.model,
        development_dataset_path=str(DEVELOPMENT_DATASET_PATH),
    )

    new_results = run_evaluation_batch(
        records=pilot_records,
        strategies=STRATEGIES,
        runner=runner,
        output_path=str(OUTPUT_PATH),
        delay_seconds=REQUEST_DELAY_SECONDS,
    )

    print(f"New results saved: {len(new_results)}")
    print(f"Result file: {OUTPUT_PATH}")

    for result in new_results:
        print(
            f"{result.record_id} | {result.strategy} | "
            f"correct={result.answer_correct} | "
            f"plan_valid={result.plan_valid}"
        )


if __name__ == "__main__":
    main()