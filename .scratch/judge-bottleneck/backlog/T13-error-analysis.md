# T13 · Error analysis of the Judge

Tier: 1 · Owner: _ · Depends: T12

## What

- Sample 60 Condition A questions (seeded): 30 where A is wrong but D is right, and 30 where the LLM judge's stop verdict disagrees with the Coverage key.
- For each, read the logged Judge reasoning and label it in `results/test/error_labels.csv` using a small fixed scheme, drafted first on 10 questions:
  - **Early stop:** said "enough" with a Gold paragraph missing.
  - **Late stop:** said "not enough" with everything present.
  - **Vague steer:** MISSING doesn't name a useful entity.
  - **Wrong steer:** MISSING points at the wrong entity.
  - **Good steer, bad retrieval:** the right entity was named but BM25 didn't find it.
  - **Answerer error:** the Judge was right but the answer was wrong.
- Two teammates label the first 20 independently; report their agreement (Cohen's kappa).
- Pick 2–3 short example traces for the paper.

## Done when

The labels CSV, category counts, the agreement number and the chosen examples are saved in `results/test/`.

## Description

The numbers say *how much* the Judge loses. This says *why*, in the Judge's own logged words. It's a cheap, high-value paper section, and it's only possible because T05 logs every reasoning.
