# T02 · Pilot set and Test set

Tier: 1 · Owner: _ · Depends: T01

## What

- `pipeline/dataset.py` downloads HotpotQA dev (fullwiki setting) into `data/hotpotqa/raw/` from Hugging Face (`hotpotqa/hotpot_qa`); the official CMU server was down. Run with `python -m pipeline.dataset`.
- Samples a 100-question **Pilot set** and a disjoint 1,000-question **Test set** with a seed from `config.yaml`.
- Writes `data/hotpotqa/pilot.jsonl` and `data/hotpotqa/test.jsonl`, one question per line: `id`, `question`, `answer`, `type` (bridge/comparison), `level`, `gold_titles` (the titles of the Gold paragraphs, from `supporting_facts`).
- Commits only the list of sampled question ids to `results/question_ids/` (small, and it pins the sample for everyone).

## Done when

- Running the script twice gives identical files.
- The Pilot set and Test set share no ids, and every question has 2 `gold_titles`.
- The id lists are committed.

## Description

The Pilot set is our playground: we tune prompts and settings on it as often as we like. The Test set is the exam: we don't look at it until everything is frozen (T10), so the reported numbers aren't tuned to it. Keeping the Gold paragraph titles is what later lets the Oracle judge and the Coverage key work by simple title matching.

## Notes

- Verified Oct 3: 7,405 dev questions in the source; 100 Pilot + 1,000 Test, no overlap, every question has exactly 2 `gold_titles`; a second run gives byte-identical files. Test set mix: 789 bridge, 211 comparison.
- The Hugging Face cache is pointed at `data/cache/` (set in `pipeline/__init__.py`), so nothing is written to the home folder.
