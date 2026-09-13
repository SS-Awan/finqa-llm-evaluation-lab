# System Design

## Objective

FinQA LLM Evaluation Lab measures whether prompting strategies improve an LLM's ability to answer financial numerical-reasoning questions grounded in tables and supporting text.

This is an evaluation system, not a chatbot. Every strategy is tested on the same fixed examples, using the same model and generation settings.

## Evaluation Task

The system uses FinQA, a public dataset of financial-report questions with:

- table and text evidence,
- known numerical answers,
- supporting facts,
- and annotated calculation programs.

The initial evaluation set will contain 96 deterministic, stratified examples from FinQA's held-out test split.

## Model and Free-Tier Constraint

Initial model: `gemini-3.5-flash-lite`

The Gemini Free tier provides 500 requests per day for this model. The initial experiment uses:

- 96 examples
- 4 prompt strategies
- 384 planned requests

This leaves 116 requests of daily headroom for controlled retries and validation.

## Prompt Strategies

1. Direct Answer
2. Structured Reasoning
3. Program-of-Thought
4. Few-shot Program-of-Thought

Self-check is intentionally excluded from the initial run. It may be tested later only if the initial results justify the extra API usage.

## Architecture

```text
Fixed FinQA sample + versioned prompt strategy
                ↓
         Gemini API client
                ↓
      Pydantic output validation
                ↓
Restricted calculation-plan validation and local execution
                ↓
    SQLite experiment-result storage
                ↓
 Metrics, paired comparison, and failure analysis
                ↓
 Precomputed data for a static public dashboard