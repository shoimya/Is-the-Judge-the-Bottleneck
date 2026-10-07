# T11 · Main runs on the Test set

Tier: 1 · Owner: _ · Depends: T10, decisions in [CONTEXT.md](../CONTEXT.md)

## What

From the `v1-frozen` commit, on Kaggle (vLLM), run on all 1,000 Test set questions:

1. Closed-book
2. Single-turn
3. Always-loop
4. Conditions A, B, C, D

- Keep a run sheet in the Runs section of `NOTES.md`: run_id, setting, date, GPU hours used, who ran it, and notes.
- `runs/` is on Drive or Kaggle output, so logs survive session limits. Resume interrupted runs (T05) instead of restarting them.
- Before starting, recompute research 03's budget from T04's measured throughput. The estimate is ~5 GPU-hours for all seven settings. If the measurement is more than 5× slower, reopen research 03.

## Done when

- Every setting has a complete log with 1,000 questions, all from the same frozen commit.
- `python evaluate.py` produces `results/test/summary.csv` for all seven settings.

## Description

This is the week the GPU works hardest. Order matters: Closed-book and Single-turn are cheapest and give early numbers; the four conditions come next. Nothing about prompts or settings changes during this ticket. If something is broken, stop, fix, re-freeze and rerun the affected settings, and log that in the run sheet.
