# T09 · Scoring and per-run summary

Tier: 1 · Owner: _ · Depends: T05, T08

## What

`evaluate.py`, which reads run logs only (no model calls):

- **Answer EM and F1** using HotpotQA's official answer normalization (lowercase, strip punctuation and articles).
- **Judge precision and recall** of its "stop" verdicts, per Round, against:
  - the **Coverage key** (all Gold paragraphs in the Evidence), and
  - the **Answerability key** (the shadow answer at that Round is correct).
- **Cost:** mean tokens and LLM calls per question (shadow answers excluded).
- **Rounds used:** the distribution per setting.
- `python evaluate.py <run_id> ...` prints and saves a summary table to `results/<set>/summary.csv`.

## Done when

- The summary table is produced for the 10-question runs from T08.
- EM/F1 matches the official HotpotQA eval script on 20 hand-picked answers.
- The table has one row per setting and the columns EM, F1, judge P/R (both keys), tokens, calls, mean Rounds.

## Description

This turns raw logs into the paper's main numbers. Because it only reads logs, anyone can rerun or extend scoring without GPU time.
