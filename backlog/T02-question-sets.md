# T02 · Pilot set and Test set

Tier: 1 · Owner: _ · Depends: T01

## What

- `question_sets.py` at the top of the repo downloads HotpotQA dev (fullwiki setting) into `data/hotpotqa/raw/` from Hugging Face (`hotpotqa/hotpot_qa`); the official CMU server was down. Run with `python question_sets.py`; `setup_project.py` runs it when the sets are missing.
- Samples a 100-question **Pilot set** and a disjoint 1,000-question **Test set** with `SAMPLE_SEED = 0`, set at the top of the file.
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
- Oct 4 code review: added `load_question_set(set_name)` so every module reads the Pilot/Test sets the same way (file closed properly); used by `retriever check` and `llm benchmark`.
- Oct 9: rebuilt from scratch under the owner's coding rules, after the restart removed the old code. Decided with the owner: one file, `question_sets.py`; no reader function until a ticket needs one; tests for `raw_row_to_question`, `split_pilot_and_test`, `write_question_ids`, plus one real-download test that skips without internet. Added `huggingface_hub==1.33.0` and `pyarrow==25.0.1`. Setup gained a step that builds the sets only if missing. Logging reuses `note`, `warn` and `write_run_log` from `setup_project.py`; `write_setup_log` was renamed to `write_run_log` and takes the log's name. Each run logs to `runs/question_sets/<date>/` and warns if a committed id list would change.
- Oct 9: verified on the Mac. 7,405 questions read; Pilot 81 bridge / 19 comparison, Test 789 / 211; both id lists byte-identical to the ones committed before the restart (commit c73d377), so the Oct 4 search recall numbers still apply. A second run left all four files unchanged; setup skips the step when the files exist and rebuilds an identical file when one is missing. 6 tests pass. Not yet run on Kaggle. Oct 9: marked Done by the owner; the owner commits `results/question_ids/`.
- Oct 9 cleanup review: the two file writers share `write_lines_to_file`; the id list comes from one function, `question_ids_of`, used by the code and the tests; `PROJECT_ROOT` is imported from `setup_project.py` instead of being defined twice; dense lines split; both test files now run on their own with `python test/<file>.py`. Outputs byte-identical after the change; 6 tests pass.
