# Notes

Working notes for the team. The paper's setup section is drawn from Pilot.

## Pilot

What was tried on the Pilot set, what was chosen, and why (T10).

## Runs

One row per run (T11, T15, T17–T19).

| run_id | setting | set | date | GPU hours | who | notes |
|---|---|---|---|---|---|---|

## Throughput

Measured on Kaggle 2× T4 (T04): prompt tok/s, generated tok/s, wall time for 20 questions.

| date | model | GPUs used | questions × Rounds | wall time | prompt tokens (tok/s) | generated tokens (tok/s) |
|---|---|---|---|---|---|---|
| Oct 6 | Qwen3-4B-Instruct-2507, fp16, vLLM 0.30.0 | 1 of 2 T4 | 20 × 3 (Judge + Answerer + Rewriter every Round, 5 BM25 paragraphs added per Round) | 110 s | 97,366 (887/s) | 12,918 (118/s) |

- 5.5 s per question for a full 3-Round loop → Tier 1 main runs (1,000 questions × 7 settings) ≈ **10.7 GPU-hours**. The benchmark printed 7.6 because it multiplies by 5 settings, not 7.
- This is an upper bound: every question ran all 3 Rounds (real runs stop early), Closed-book and Single-turn are cheaper, batches were only 20 prompts (1,000 fill the GPU better), and only one T4 was used (one model copy per T4 would roughly halve the wall time, T10).
- Engine start: ~3.5 min the first time in a session (download + compile), ~1 min after that.
