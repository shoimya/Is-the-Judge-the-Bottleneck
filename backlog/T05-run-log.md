# T05 · Run log

Tier: 1 · Owner: _ · Depends: T01

## What

Every run that uses the pipeline is logged (a whole question set, or one question followed with `run.py`); tests, `setup/` scripts and `run.py --check` write nothing.

**Folders:** one per scenario inside `runs/`, created the first time that scenario runs: `closed-book`, `single-turn`, `always-loop`, `coin-flip`, `A_llm-stops_llm-steers`, `B_llm-stops_oracle-steers`, `C_oracle-stops_llm-steers`, `D_oracle-stops_oracle-steers`. Inside, one folder per run named by its start time, e.g. `runs/single-turn/2026-10-14T153012/`, holding two files:

- **`meta.json`:** run id, scenario, dataset, question set and its question ids, start and finish time, backend, model per role, prompt versions, git commit (and whether there were uncommitted changes), and a full copy of the config.
- **`questions.jsonl`:** one line per question, written as soon as that question finishes:
  - `question_id`, `question` (English), `gold_answer`, `gold_titles`
  - `started_at`, `finished_at`
  - `final_answer`, `rounds_used`, `total_prompt_tokens`, `total_completion_tokens`, `llm_calls`
  - `rounds`: one entry per Round, in the order things happen:
    - `retriever`: the `query` sent and everything it brought back (`title`, full `text`, `score` per paragraph)
    - `judge`: what the Judge said (full output with reasoning, Stop decision, `p_yes`, Steer signal, parse error, tokens), filled in from T06
    - `answerer`: which paragraphs it saw (titles), its prompt version, its answer, tokens
    - `rewriter`: the query it sent back to the retriever, tokens, filled in from T06
  - Closed-book has one entry with no retriever (Round 0); Single-turn has one Round with retriever and answerer only.
  - The answerer's full prompt isn't stored: it is rebuilt exactly from the paragraphs it saw and its prompt version.
- **Resumable:** rerunning the same scenario on the same questions with the same config continues the newest unfinished run, skipping questions already in `questions.jsonl` (a half-written last line from a crash is dropped first). Kaggle and Colab sessions die, and we must not lose or duplicate work.

Scores (`em`, `f1`, `shadow_em`) are added by T09. Reading a run back is `read_question_records(run_folder)`; turning it into a table for analysis comes in T09, once pandas is pinned on Kaggle.

## Done when

- A fake run of 5 questions writes `meta.json` and `questions.jsonl`, and reading it back gives the 5 records.
- A run stopped halfway and restarted finishes the missing questions with no duplicates.

## Description

Every decision the Judge makes is written down with its reasoning, so we can later ask "why did it stop too early here?" without rerunning anything. Logging the shadow answer at every Round is what makes the Answerability key and the Answer-oracle computable after the fact. Logging what the retriever returned, in full, makes each log readable on its own.

## Notes
- Oct 6: decided with the owner: one folder per scenario in `runs/` (A–D named by who stops and who steers); full paragraph text logged; the answerer logged as the paragraphs it saw plus its prompt version; `meta.json` + `questions.jsonl` kept from the original plan.
