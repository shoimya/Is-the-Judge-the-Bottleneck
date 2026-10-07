# T19 · Hosted 70B judge (optional, ≤ $40)

Tier: 3 · Owner: _ · Depends: T18

## What

- **Go/no-go:** only if Tier 1 shows a clear A-vs-D gap, and T17/T18 didn't already close it.
- Add an API backend to `pipeline/llm.py`, used only for `judge_model`. It must return log probabilities, or `p_yes` is marked unavailable for this judge.
- Estimate cost on 20 questions first. Then pick the largest seeded subset that stays under **$40**, and run Conditions A and B on it (the settings where the LLM judge's Stop decision is used; in B the Oracle steers).
- Rerun the Tier 1 judge's numbers on the same subset for comparison.

## Done when

A fourth row in `results/test/judge_comparison.csv`, with the subset size and the total dollars spent recorded in the Runs section of `NOTES.md`.

## Description

This is a stretch goal. A 70B judge tells us whether the gap is a "small model" problem or a "judging is hard" problem. The paper is complete without it.
