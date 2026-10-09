# Is the Judge the Bottleneck?

What the project studies, how, and every decision that shapes the code or the paper. Ticket status lives only in [backlog/README.md](backlog/README.md); how to run the code is in [README.md](README.md); how code is written and how work is done are in [Coding rules](#coding-rules). Last updated Oct 9, 2026.

Some QA systems answer hard questions by searching several times. After each search an LLM **Judge** decides whether the **Evidence** is enough to answer (the **Stop decision**) and, if not, what is missing (the **Steer signal**, which drives the next search). We measure how much accuracy and cost the LLM judge loses compared with an **Oracle judge** that knows which paragraphs are needed, and which of the two decisions causes more of the loss.

- **Course:** Masters NLP class project. Paper due **Dec 1, 2026**; started Sep 22, 2026.
- **Team:** 4 people; tickets run in order and are picked up by whoever is free. The repo owner writes the paper.
- **Repo:** https://github.com/shoimya/Is-the-Judge-the-Bottleneck (public; work branch `SC.V2`).

## Research questions

- **RQ1.** In a self-correcting retrieval loop, how much accuracy and cost does an LLM sufficiency judge lose compared with an oracle judge that reads the gold evidence?
- **RQ2.** Which of the judge's two jobs causes more of that loss: stopping ("enough, answer now") or steering ("here's what's missing")?
- **RQ3 (supporting).** Does a larger judge, or one from a different model family, close the gap?

**Gap in the literature.** Self-RAG (Asai et al., 2023) and CRAG (Yan et al., 2024) build judges but check them outside the loop (against GPT-4 labels, or on PopQA only). Sufficient Context (Joren et al., ICLR 2025) shows "sufficient" isn't "relevant", but studies it statically. IRCoT (Trivedi et al., 2023) and Adaptive-RAG (Jeong et al., 2024) loop on multi-hop benchmarks but report only end accuracy. None swaps an oracle into the loop, and none separates stopping errors from steering errors. We don't build or train a judge: we replace one decision at a time with a perfect one and measure what changes, logging every judge reasoning so errors can be read, not just counted.

## Scope

| Tier | Contents | Target |
|---|---|---|
| **1 (must-have)** | RQ1 + RQ2 on HotpotQA: Conditions A–D and all baselines, one model. A complete paper on its own. | ~Nov 12 |
| **2** | The same frozen pipeline on MuSiQue. | ~Nov 21 |
| **3** | RQ3: a larger judge, a different-family judge, optionally a hosted 70B judge (≤ $40). | ~Nov 26 |

If time runs short, drop Tier 3 first, then Tier 2. Nov 26 – Dec 1 is buffer and paper polish.
**Out of scope:** training or fine-tuning a judge; dense or hybrid retrievers (BM25 stays fixed); more than 3 Rounds; datasets beyond HotpotQA and MuSiQue.

## Glossary

Use these terms in code, tickets and the paper.

| Term | Meaning | Avoid |
|---|---|---|
| **Round** | One retrieve-then-judge cycle. A question gets at most 3. | iteration, hop (a hop is a reasoning step in the question) |
| **Evidence** | All paragraphs retrieved for a question so far. | context (alone), documents |
| **Gold paragraph** | A paragraph the dataset marks as needed to answer. | supporting fact, ground truth |
| **Judge** | Reads the question and the Evidence; gives a Stop decision and a Steer signal. | critic, evaluator, verifier |
| **Stop decision** | Yes/no: is the Evidence enough to answer now? | sufficiency label, halt |
| **Steer signal** | Free text saying what is missing. Only the Rewriter reads it. | feedback, critique |
| **LLM judge** | A Judge made by prompting a language model, with no access to Gold paragraphs. | deployed judge, model judge |
| **Oracle judge** | A Judge that sees the Gold paragraphs: stops once every Gold paragraph is in the Evidence; steers with the title of the first missing one. | gold judge, perfect judge |
| **Rewriter** | Turns the question, the queries already tried and the Steer signal into the next search query. | query generator, reformulator |
| **Answerer** | Gives the short answer from the question and the Evidence. | generator, reader |
| **Shadow answer** | The Answerer's answer after a Round where the loop didn't stop. Logged, not counted toward accuracy. | |
| **Condition A / B / C / D** | The 2×2 that crosses who stops with who steers (see Experiment). | |
| **Closed-book, Single-turn, Always-loop, Coin-flip** | Baselines: fixed rules instead of a Judge (see Experiment). | reference point |
| **Answer-oracle** | Upper bound: correct if the Answerer was right after any Round. | |
| **Coverage key** | Marks a Stop decision correct when every Gold paragraph is in the Evidence. | |
| **Answerability key** | Marks a Stop decision correct when the Answerer's answer on the current Evidence is correct. | |
| **Pilot set** | 100 questions for tuning prompts and settings; never reported. | dev set |
| **Test set** | The questions behind every reported result; run only after the freeze. | |

## Data

**HotpotQA (Tier 1).** Dev split, fullwiki setting, 7,405 questions, from Hugging Face (`hotpotqa/hotpot_qa`, because the CMU server was down). We keep both bridge and comparison questions and ignore its `context` field: we search with our own BM25.
- Pilot set 100 and a disjoint Test set 1,000, sampled with `sample_seed: 0`. Only the id lists are committed (`results/question_ids/`); every machine rebuilds identical sets.
- Each question: `id`, `question`, `answer`, `type`, `level`, `gold_titles` (each title once, in the dataset's order).

**Wikipedia (for HotpotQA).** HotpotQA's own processed 2017 intro paragraphs (~5.23M paragraphs, 1.55 GB download). Its titles match the gold labels exactly, so Gold coverage is an exact title match. We index `title + " " + text` with BM25 (English stopwords, stemming); the title is the paragraph's id. Built once on Kaggle (~12 min), saved as the private Kaggle dataset `hotpotqa-bm25-index` (~2.8 GB), loaded with memory mapping (~1–3 GB RAM).
- Search quality on the Pilot set (all Gold paragraphs found / at least one), top 2 / 5 / 10 / 20: 11 / 24 / 34 / 44 % and 77 / 83 / 84 / 90 %. One Gold paragraph is usually found and the second often isn't, so the loop has room to help.

**MuSiQue (Tier 2).** MuSiQue-Ans (no unanswerable questions), from the authors' GitHub. Test labels aren't public, so both sets come from dev (2,417 questions: 1,252 two-hop, 760 three-hop, 405 four-hop).
- Pilot set 100 and Test set 500, stratified by hop count, seeded.
- One pooled corpus of every paragraph in all six files (~139k, deduplicated), as IRCoT and Adaptive-RAG do. Per-question paragraphs were rejected: 3 Rounds would retrieve most of them, leaving the Judge little to decide.
- Titles repeat (34% of Gold paragraphs share a title), so each paragraph gets an id (hash of title + text) and the Oracle checks coverage by id; its Steer signal still names the title.
- 3 Rounds is tight for 4 hops (an oracle-loop test found all Gold paragraphs for 76% of 2-hop, 47% of 3-hop, 17% of 4-hop questions). Keep 3 Rounds and report every MuSiQue result by hop count.

## The loop

```
evidence = search(question, k)
for round in 1..3:
    judge = LLM judge                                       # called in every looping scenario, so it is always logged
    shadow_answer = Answerer(question, evidence)            # logged, not counted
    stop  = stop_source(judge, oracle)                      # LLM judge, Oracle judge, or "never"
    if stop or round == 3: final answer = shadow_answer; break
    steer = steer_source(judge, oracle)                     # LLM judge or Oracle judge
    query = Rewriter(question, past queries, steer)
    evidence += search(query, k, exclude=evidence titles)   # trimmed to the token budget
```

- **Evidence accumulates:** each Round adds k new paragraphs (titles already held are skipped). The Judge and Answerer always see all Evidence so far.
- **Fixed parts:** the retriever, Rewriter and Answerer are identical in every scenario. Only who stops and who steers changes.
- **The LLM judge is called in every looping scenario**, even where its output isn't used (e.g. D), so its verdict can always be scored. Closed-book and Single-turn have no Judge: Single-turn's Round-1 Evidence is the same as every looping scenario's, so the judge's verdict on it is already logged there.
- **A query with no searchable words** (empty, or only words like "the") returns nothing, so that Round adds no Evidence.
- **Shadow answers** are recorded after every Round; only the answer at the stopping Round counts. They feed the Answerability key, the Answer-oracle and offline stop-threshold sweeps.
- **Code:** `pipeline/loop.py` has one function per scenario; the looping ones share `run_rounds` and differ only in who stops and who steers.

## Roles

**LLM judge output**, reasoning first so there are thoughts worth logging:

```
REASONING: <a few sentences>
ENOUGH: yes|no
MISSING: <what information is still needed, or "nothing">
```

- **Confidence:** `p_yes` = P("yes") / (P("yes") + P("no")) at the ENOUGH position, summing "yes"/"Yes"/" yes" and "no"/"No"/" no" from that token's top 5 alternatives. If neither appears, `p_yes` is marked unavailable.
- **Malformed output** counts as "no" and is flagged `parse_error: true`.
- Qwen3's thinking mode is off; the reasoning goes in REASONING.

**Oracle judge** (pure Python, no model): stops once every Gold paragraph is in the Evidence (HotpotQA by title, MuSiQue by id). Steers with `"Missing information about: <first missing Gold title>"`, or `"nothing"`. It never gives the paragraph text: the Rewriter still has to write a query and BM25 still has to find it, which keeps the steering comparison fair.

**Rewriter** sees the question, the queries already tried and the Steer signal; writes one search query. **Answerer** sees the question and the Evidence; writes a short answer only (the format exact match needs). Each role's prompt sits at the top of its file, version-numbered (`v0`, `v1` after the pilot), and the version goes in the run log.

**Plug-and-play:** every role calls `llm.generate(role, prompt, config)`. All three roles use `model`; Tier 3 sets only `judge_model`.

## Experiment

**The 2×2 (from the proposal):**

| Condition | Who decides to stop | Who says what is missing |
|---|---|---|
| **A** | LLM judge | LLM judge (the deployed system) |
| **B** | LLM judge | Oracle judge |
| **C** | Oracle judge | LLM judge |
| **D** | Oracle judge | Oracle judge (best case) |

- **B − A** = what the LLM judge's **steering** costs (only the steering was made perfect).
- **C − A** = what the LLM judge's **stopping** costs (only the stopping was made perfect).
- The larger of the two is the bigger bottleneck (**RQ2**). **D − A** is the total gap (**RQ1**). If A is nearly as good as D, the judge is not the bottleneck; that is still a valid answer.

**Baselines** (fixed rules, not judges) and the upper bound:

| Scenario | Stop | Steer | Note |
|---|---|---|---|
| **Closed-book** | no search | – | Answers from memory; flags memorized answers. |
| **Single-turn** | after one search | – | No Judge. The floor. |
| **Always-loop** | never early (all 3 Rounds) | LLM judge | |
| **Coin-flip** *(optional)* | a random Round | LLM judge | Computed from Always-loop's shadow answers; no model calls. |
| **Answer-oracle** | the best Round, in hindsight | LLM judge | Upper bound, computed from Always-loop's shadow answers. Comparing it with C shows where "has all Gold paragraphs" and "can actually answer" come apart. |

**Measures**
- **Answers:** exact match (EM) and F1 with HotpotQA's official normalization.
- **Judge:** precision and recall of the LLM judge's Stop decisions, per Round, against the Coverage key and the Answerability key; early-stop and late-stop rates.
- **Calibration:** reliability diagram and ECE of `p_yes` against each key.
- **Cost:** tokens (prompt + generated) and model calls per question, shadow answers excluded. Rounds used per scenario.
- **Accuracy vs cost:** EM against mean tokens per scenario, plus an offline sweep of a stop threshold on `p_yes`.

**Statistics**
- Paired bootstrap 95% CIs (10,000 resamples) for every headline number and for B − A, C − A, D − A; McNemar tests on per-question correctness for those gaps.
- Memorization control: every headline result on all questions and on the not-memorized subset (questions Closed-book got wrong).
- Greedy decoding (temperature 0), so seeds only matter for sampling question sets (`sample_seed`) and the bootstrap (`seeds`).

**Error analysis** (the only manual labeling): 60 Condition A questions, seeded: 30 where A is wrong but D is right, 30 where the Stop decision disagrees with the Coverage key. Label each from its logged reasoning as early stop, late stop, vague steer, wrong steer, good steer but bad retrieval, or Answerer error (scheme drafted on 10 first). Two teammates label the first 20 independently; report Cohen's kappa. Pick 2–3 example traces for the paper.

**Pilot and freeze** (Pilot set only, T10):
1. Choose the Tier 1 model, Qwen3-8B-AWQ or Qwen3-4B-Instruct-2507: run A and Single-turn with each; higher parse rate and EM wins, ties go to the faster.
2. Choose k from 2, 5, 10: the smallest k where D's Gold coverage after 3 Rounds is clearly above Single-turn's.
3. Choose the token budget so 3 Rounds of Evidence fit without trimming for most questions.
4. Fix prompts (v0 → v1) only for format problems. Never tune to make the judge look better or worse.
5. Freeze: final values in `config.yaml`, commit, tag `v1-frozen`. If something breaks after that: stop, fix, re-freeze, rerun what's affected, log it.
6. Write what was tried and chosen in the Pilot section of `NOTES.md` (it becomes the paper's setup section).

MuSiQue keeps the frozen Tier 1 prompts; a 20-question check only confirms parsing.

## Models and compute

| Use | Model | On Kaggle's 2× T4 |
|---|---|---|
| Tier 1 candidates | Qwen3-8B-AWQ (4-bit, ~6 GB), Qwen3-4B-Instruct-2507 (fp16) | one model copy per T4 |
| Tier 1 fallback | Llama-3.1-8B-Instruct | fp16 across both T4s, or 4-bit |
| Tier 3 larger judge | Qwen3-14B-AWQ (~10 GB) | one T4 |
| Tier 3 other family | Llama-3.1-8B-Instruct (needs an HF token) or Granite-4.2-8B | fp16 across both T4s |
| Tier 3 optional | Hosted 70B judge via API | ≤ $40 |

Ruled out on the T4: Gemma 4 (no vLLM support on T4), Qwen3.5 (fp16 overflow risk). Development model until T10: Qwen3-4B-Instruct-2507.

- **Every reported number comes from vLLM on Kaggle.** Ollama on the Mac is for writing and testing code only (its compressed weights can answer differently). Statistics are computed on the Mac from Kaggle run logs. Colab is a backup; never mix Colab and Kaggle runs within one comparison. Exception: the hosted 70B judge (T19), stated in the paper.
- **GPU budget:** Kaggle gives ~30 GPU-hours per week per account, four accounts. Measured Oct 6 (T04, one T4, Qwen3-4B fp16): 887 prompt tok/s, 118 generated tok/s, 5.5 s per question for a full 3-Round loop, so all Tier 1 main runs (1,000 questions × 7 scenarios) take **≤ 10.7 GPU-hours** (upper bound). Whole project ≈ 30–100 of ~300 available. T11 re-measures with the final model; if it's more than 2× slower, revisit.
- **Money:** $0 for Tiers 1–2. Up to $40 only for the hosted 70B judge, decided after Tier 1 (only if a clear A-vs-D gap remains after T17/T18): cost 20 questions first, then run A and B on the largest seeded subset the budget allows, and rerun the Tier 1 judge on that subset. The API must return log probabilities.
- **Tier 3 reruns** only the scenarios that use the LLM judge's output: A, B, C and Always-loop.

## Environment

- **Python 3.12 or newer.** The Mac runs 3.12 in `.venv`; Kaggle runs 3.13 (not changeable). Both give byte-identical question sets and search results. The paper reports Kaggle's version.
- **Pins:** exact versions only, using what pip installs on Kaggle. vLLM (0.30.0) is in `requirements-gpu.txt` because it doesn't install on the Mac. `huggingface_hub` 1.33.0, NumPy 2.4.6 and numba 0.65.0 match what vLLM needs.
- **Kaggle:** Internet on (phone-verified account), GPU T4 x2, dataset `hotpotqa-bm25-index` attached. Fixes built into the code and notebook:
  - TorchAudio is uninstalled (built for a different CUDA; crashes vLLM).
  - `JAX_PLATFORMS=cpu` (JAX, picked up by bm25s, would take 75% of GPU memory).
  - vLLM starts its worker with `spawn` (forking after the GPU is touched fails).
  - The end-of-answer token's log probability is dropped on vLLM, so both backends return the same format.
  - Model downloads go to `/tmp`, not `/kaggle/working` (~20 GB limit, saved with every version).
  - The vLLM engine is shut down after each command, so notebook cells end by themselves.
- **Mac setup:** `project_set_up.py` sets up a machine in one run: `.venv`, the pinned libraries, the question sets and the Ollama model. It checks each piece first and only installs or downloads what is missing; what it can't do (installing Python or Ollama, downloading the index from Kaggle) it prints as steps.
- **Ollama on the Mac:** `pipeline/server_launcher.py` opens and closes the Ollama server, pointed at `data/ollama/`. The server keeps ~3 GB of memory while open, so close it when done; the launcher only closes a server it opened.

## Coding rules

The owner's, Oct 9. They apply to every file, including tests and the notebook.

**Readability**
- Use descriptive names written out in full: `user_count`, not `uc`.
- Function names say what the function does: `load_config`, not `load`.
- Every function has a short plain-language docstring saying what it's for.
- Important lines get a short comment explaining the goal, in words a beginner can follow.
- Break up dense lines. A line that does several things becomes several lines, or calls a small, well-named helper.

**Lean**
- Build the smallest solution that does today's job, with no layers, wrappers or features "for later".
- No placeholders, dead code or unused options.
- Each job has one function, and every caller uses it. Don't copy it.
- Explain any new file, folder or dependency before adding it.

**Structure**
- Each function has one goal.
- Shared code lives in one place, and each feature's own logic lives in its own file.
- Settings a feature may change go as named variables at the top of its file, not scattered through the code. The one exception: experiment choices that can change a reported number go in `config.yaml`, so T10 freezes a single file.

**Runnable and followable**
- Every file is run with the editor's Run button: open it, press Run (or Debug), and it shows a short demo of what it does. No terminal commands and no command-line arguments are needed to run any file.
- The demo lives at the end of the file, in its `if __name__ == "__main__":` block. Its defaults run something useful with no setup.
- Anything the demo lets you choose (which scenario, which question, start or stop) is a named variable at the top of the file that you edit before pressing Run, not a command-line option.
- The README tells people which file to open and press Run, never a terminal command to type.
- A script cleans up only what it started itself.

**Testing**
- Build test-first: a failing test, then just enough code to pass it, one small piece at a time.
- Agree on which public functions get tested before writing tests.
- Test against the real thing, such as the real service or model, when that's what matters. Skip with a clear message if it isn't available. Don't use fakes unless they're agreed.
- Tests never write to the real output folders.

**Logging**
- Every run writes a readable log to a predictable place, `runs/<script>/<date>/`, so reruns never overwrite each other.
- A log records exactly what went in, what came out, and the totals that matter, such as calls, cost and time.
- Warn loudly in the log when something may have gone silently wrong.

**Working rules**
- Propose first; the owner decides.
- One task at a time, confirmed before starting.
- The owner does all git work and commits and pushes for the team: no commits, pushes, `git rm` or other git changes unless asked, and nothing is pushed on anyone's behalf. When asked to ignore something, only edit `.gitignore`.
- Ask before installing system software or deleting anything.
- Keep everything the project downloads or installs inside the project folder.
- Every ticket ends by updating the setup and the README. Anything a machine needs to run the project (a library, a folder, a download, the index, a model) becomes a step in `project_set_up.py`, one function each, which checks whether it already exists and only installs or downloads what is missing. `README.md` is then updated so someone who just cloned the repo can get started: what setup now does, which file to open and run for the new piece, and the repository layout.
- The project lives on an external SSD and the Mac is short on space. Store datasets, indexes, model files and caches under `data/`, never in the home folder: the Hugging Face cache in `data/cache/`, Ollama models in `data/ollama/` (the server launcher points Ollama there). (Exception, agreed: on Kaggle, model downloads go to `/tmp`.)
- Keep project knowledge in this file and status in the backlog; don't create memory or notes files elsewhere. Explain choices in plain language.

## Run log

Every run that uses the pipeline is logged (`run.py` on one question or a whole set, `loop.py`'s demo); tests and setup scripts write nothing to it. Code: `pipeline/run_log.py`.

- **Folders:** `runs/<scenario>/<run id>/`, one scenario folder created the first time it runs: `closed-book`, `single-turn`, `always-loop`, `coin-flip`, `A_llm-stops_llm-steers`, `B_llm-stops_oracle-steers`, `C_oracle-stops_llm-steers`, `D_oracle-stops_oracle-steers`. The run id is the start time (e.g. `2026-10-14T153012`). `runs/` points to persistent storage on Kaggle (`/kaggle/working/runs`) and Colab (Drive).
- **`meta.json`:** run id, scenario, dataset, question set and its ids, start and finish time, backend, model per role, prompt versions, git commit (and whether there were uncommitted changes), full config.
- **`questions.jsonl`**, one line per question, saved as soon as it finishes: `question_id`, `question`, `gold_answer`, `gold_titles`, `started_at`, `finished_at`, `final_answer`, `rounds_used`, token totals, `llm_calls`, and `rounds`. Each Round records, in order: **retriever** (the query and every paragraph returned, with full text and score), **judge** (full output with reasoning, Stop decision, `p_yes`, Steer signal, parse error, tokens; from T06), **answerer** (titles of the paragraphs it saw, prompt version, answer, tokens; its full prompt is rebuilt from these), **rewriter** (the query it sent back, tokens; from T06). Closed-book is one Round 0 with no retriever.
- **Resumable:** running the same scenario on the same questions with the same config continues the newest unfinished run, skipping finished questions and dropping a half-written last line.
- Scores (`em`, `f1`, `shadow_em`) are added in T09.

## Outputs for the paper

- `results/<set>/summary.csv`: one row per scenario (EM, F1, judge P/R on both keys, tokens, calls, mean Rounds).
- `results/test/`: main table, gap decomposition, memorization table, judge quality, calibration, error labels, as CSV and LaTeX. `results/figures/`: reliability diagrams and accuracy-vs-cost curves (PDF).
- Tier 2: `results/test_musique/` and cross-dataset, by-hop tables. Tier 3: `results/test/judge_comparison.csv`.
- `NOTES.md`: Pilot choices, the Runs sheet (run_id, scenario, set, date, GPU hours, who, notes), Throughput.
- Main-run order (T11): Closed-book and Single-turn first, then Always-loop, then A–D.

## Risks

| Risk | Mitigation |
|---|---|
| 3 Rounds too few for 4-hop MuSiQue | Report MuSiQue by hop count; treat it as a finding. |
| Tuning to the Test set | Tune only on the Pilot set; freeze before any Test run. |
| Memorized answers inflate retrieval's value | Closed-book run; report the not-memorized subset. |
| Mixing inference setups | Every reported number from vLLM on Kaggle. |
| Kaggle sessions end mid-run | Resumable logs on persistent storage. |
| Judge output doesn't parse | Fixed format, ≥ 95% parse-rate check, malformed = "no" and flagged. |
| Time runs out | Drop Tier 3, then Tier 2. |

## Open questions

- Whether to spend the $40 on a hosted 70B judge (after Tier 1).
- Whether to report Coin-flip (costs no GPU time).
- Set in the pilot (T10): the Tier 1 model, k, and the token budget.
