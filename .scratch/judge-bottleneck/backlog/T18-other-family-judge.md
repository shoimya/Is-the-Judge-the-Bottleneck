# T18 · Judge from a different model family

Tier: 3 · Owner: _ · Depends: T17

## What

Same as T17, with `models.judge` set to a similar-size model from a **different family**: **Llama-3.1-8B-Instruct** (needs an HF token on Kaggle) or **Granite-4.2-8B**, both fp16 across the two T4s (research 02). Add it as a third row in `results/test/judge_comparison.csv`.

## Done when

The judge comparison table has three rows (Tier 1, larger, other family), all on the same questions.

## Description

When the Judge and the Answerer come from the same family, they may share blind spots: the Judge might think evidence is enough exactly when its sibling Answerer can answer. A different-family Judge tests whether that closeness helps or hurts.
