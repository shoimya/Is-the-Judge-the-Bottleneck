# T07 · Oracle judge

Tier: 1 · Owner: _ · Depends: T02

## What

In `pipeline/judge.py`, next to the LLM judge. Pure Python with no model calls:

- `oracle_stop(evidence_titles, gold_titles) -> bool`: True when every Gold paragraph title is in the Evidence.
- `oracle_steer(evidence_titles, gold_titles) -> str`: the title of the first missing Gold paragraph (in the dataset's order), phrased as `"Missing information about: <title>"`. If nothing is missing, returns `"nothing"`.
- Tests in `test_pipeline.py` covering: nothing retrieved, one of two found, both found, a title with different capitalization or whitespace, and a duplicate title in the Evidence.

## Done when

`pytest test_pipeline.py` passes.

## Description

The Oracle judge is the "perfect judge" we compare against. It cheats by looking at the answer key (the Gold paragraph titles). For steering, it only names **which** paragraph is missing, never its text, so the Rewriter still has to write a query and BM25 still has to find it. That keeps the steering comparison fair.
