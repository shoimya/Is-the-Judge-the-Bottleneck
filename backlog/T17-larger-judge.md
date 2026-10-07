# T17 · Larger judge

Tier: 3 · Owner: _ · Depends: T12, decisions in [CONTEXT.md](../CONTEXT.md)

## What

- Set `judge_model` to the larger model from research 02: **Qwen3-14B-AWQ** (same family as the Tier 1 model if Qwen3 was picked in T10; fits on one T4). The Rewriter and Answerer stay on the Tier 1 model.
- Rerun only the settings where the LLM judge's output is used: **A, B, C and Always-loop**, on HotpotQA's Test set (or a seeded subset if the GPU budget is tight; then rerun the Tier 1 model on the same subset for a fair comparison).
- Rerun T12's main table and gap decomposition for the new Judge.

## Done when

`results/test/judge_comparison.csv` shows EM, B − A, C − A and judge P/R for the Tier 1 judge and the larger judge side by side, with CIs.

## Description

RQ3: does a bigger judge close the gap to the oracle? Only the Judge changes, so any difference is the Judge's.

## Notes
- From the Oct 4 code review: `pipeline/llm.py` loads one vLLM engine per model at `gpu_memory_utilization: 0.90`, so a second model (the larger judge) won't fit on the same T4. Plan memory or one GPU per model before running.
