# T05 · Run log

Tier: 1 · Owner: _ · Depends: T01

## What

`run.py` writes one folder per run, `runs/<run_id>/`, where `run_id` = date-time + condition, e.g. `2026-10-14T1530_A`:

- `meta.json`: date and time, condition, dataset, question set, model per role, backend, git commit, and a full copy of the config used.
- `questions.jsonl`: one line per question:
  - `id`, `question`, `gold_answer`, `gold_titles`
  - `rounds_used`, `final_answer`, `em`, `f1`
  - `total_prompt_tokens`, `total_completion_tokens`, `llm_calls` (shadow answers counted separately)
  - `rounds`: a list, one entry per Round, with `query`, `retrieved_titles`, `judge_raw` (the Judge's full output, including its reasoning), `stop` (yes/no), `p_yes`, `steer`, `shadow_answer`, `shadow_em`, and tokens per role
- `evaluate.py` has `load_run(run_id) -> DataFrame` for analysis.

**Resumable:** when a run restarts, question ids already in `questions.jsonl` are skipped. Kaggle and Colab sessions die, and we must not lose or duplicate work.

## Done when

- A fake run of 5 questions (dummy values) writes all files, and `load_run` reads them back.
- Killing the fake run halfway and restarting it finishes the missing questions with no duplicates.

## Description

Every decision the Judge makes is written down with its reasoning, so we can later ask "why did it stop too early here?" without rerunning anything. Logging the shadow answer at every Round is what makes the Answerability key and the Answer-oracle computable after the fact.
