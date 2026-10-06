# Project Specification: Is the Judge the Bottleneck?

Last updated: Sep 25, 2026. This is the single reference for what the project studies, how, and every decision made so far. Terms in **bold capitals** (Round, Evidence, Judge, …) are defined in [CONTEXT.md](CONTEXT.md); build tickets are in [backlog/](backlog/README.md).

---

## 1. Summary

Some QA systems answer hard questions by searching several times. After each search, an LLM **Judge** decides two things: whether the Evidence so far is enough to answer (the **Stop decision**), and if not, what is missing (the **Steer signal**, which drives the next search). This project measures how much accuracy and cost the LLM judge loses compared with an **Oracle judge** that knows which paragraphs are needed, and which of the two decisions causes more of the loss.

- **Course:** Masters NLP class project, about 2.5 months of work.
- **Deadline:** paper due **Dec 1, 2026**. Project started Sep 22–23, 2026.
- **Team:** 4 people. Tickets run one after another; owners are assigned as people pick them up. The repo owner writes the paper.
- **Repo:** https://github.com/shoimya/Is-the-Judge-the-Bottleneck (public, cloned over HTTPS).

## 2. Research questions

- **RQ1.** In a self-correcting retrieval loop, how much accuracy and cost does an LLM sufficiency judge lose compared with an oracle judge that reads the gold evidence?
- **RQ2.** Which of the judge's two jobs causes more of that loss: stopping ("enough, answer now") or steering ("here's what's missing", which drives the next search)?
- **RQ3 (supporting).** Does a larger judge, or one from a different model family, close the gap?

## 3. Literature and gap

| Work | What it does | What it leaves open |
|---|---|---|
| **Self-RAG** (Asai et al., 2023) | One LM learns reflection tokens that decide when to retrieve and critique relevance and support. | Its critic is checked against GPT-4 labels, not measured inside the loop. |
| **CRAG** (Yan et al., 2024) | A T5 evaluator grades retrieved documents and triggers refinement or web search. | The evaluator's accuracy is reported on PopQA only. |
| **Sufficient Context** (Joren et al., ICLR 2025) | Shows "sufficient" is not the same as "relevant"; models often answer correctly with insufficient context and hallucinate with sufficient context. | Studies sufficiency statically, not inside a loop. |
| **IRCoT** (Trivedi et al., 2023), **Adaptive-RAG** (Jeong et al., 2024) | Iterative retrieval on multi-hop benchmarks, including MuSiQue with a pooled corpus. | Report end-task accuracy, not the quality of the decision to keep searching. |

**Gap:** none of these swaps an oracle inside the loop, and none separates stopping errors from steering errors.

**How we differ:** we don't build or train a judge. We put a perfect (oracle) judge into the loop for one decision at a time and measure what changes, and we log every judge reasoning so errors can be read, not just counted.

## 4. Scope and tiers

| Tier | Contents | Status |
|---|---|---|
| **Tier 1 (must-have)** | RQ1 + RQ2 on HotpotQA: all four conditions, all reference points, one model. A complete paper on its own. | Planned, target ~Nov 12 |
| **Tier 2** | The same frozen pipeline on MuSiQue. | Planned, target ~Nov 21 |
| **Tier 3** | RQ3: a larger judge, a different-family judge, and optionally a hosted 70B judge (≤ $40). | Planned, target ~Nov 26 |

If time runs short, **drop Tier 3 first, then Tier 2.** Nov 26 – Dec 1 is buffer and paper polish.

**Out of scope** (ruled out, will not return in this project):
- Training or fine-tuning any judge (Self-RAG / CRAG style). This study swaps judges; it doesn't build them.
- Dense or hybrid retrievers. The retriever stays fixed as BM25.
- More than 3 Rounds.
- Datasets beyond HotpotQA and MuSiQue.

## 5. Task and data

**Task:** open-domain multi-hop question answering with up to **3 Rounds** of retrieval per question.

### 5.1 HotpotQA (primary, Tier 1)

- **Source:** HotpotQA dev split, **fullwiki** setting: 7,405 questions. Downloaded from the HotpotQA Hugging Face repo (`hotpotqa/hotpot_qa`, `fullwiki/validation-00000-of-00001.parquet`) because the official CMU download server was down (Oct 3, 2026). Same questions, answers and supporting facts; we ignore its `context` field (HotpotQA's own retrieval) and search with our own BM25.
- **Why:** 2-hop questions with gold supporting paragraphs marked (`supporting_facts`), so the oracle judge and coverage checks are possible.
- **Question types kept:** both bridge and comparison questions.
- **Question sets:** a **Pilot set** of 100 questions and a disjoint **Test set** of 1,000 questions, both sampled from dev with `hotpotqa.sample_seed` (0) from `config.yaml`.
  - The Pilot set is for tuning prompts and settings as often as needed and is **never reported**.
  - The Test set is run **only after the config is frozen** (tag `v1-frozen`).
- **Files:** `data/hotpotqa/pilot.jsonl` and `data/hotpotqa/test.jsonl`, one question per line with `id`, `question`, `answer`, `type`, `level`, `gold_titles`.
- **Pinning the sample:** only the lists of sampled question ids are committed, to `results/question_ids/`.

### 5.2 Wikipedia corpus (for HotpotQA)

- **Source:** HotpotQA's own processed 2017 Wikipedia (intro paragraphs), `https://nlp.stanford.edu/projects/hotpotqa/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2` (1.55 GB, bz2 JSON lines, ~5.23M paragraphs).
- **Why this dump and not a newer one:** its paragraph titles match the gold labels exactly, so Gold coverage is a simple **exact title match**. It is also the easier option; there was no trade-off.
- **What is indexed:** `title + " " + text` (sentences joined), not `text_with_links`. `title` is the document id. Wikipedia titles are unique here.

### 5.3 MuSiQue (secondary, Tier 2)

- **Version:** **MuSiQue-Ans**. It has no unanswerable questions; those exist only in MuSiQue-Full.
- **Source:** the authors' GitHub repository (StonyBrookNLP/musique, Google Drive zip).
- **Split:** test labels aren't public (leaderboard only), so both sets are sampled from **dev**: 2,417 questions (1,252 two-hop, 760 three-hop, 405 four-hop). Each question has 20 paragraphs, with one `is_supporting` paragraph per hop.
- **Question sets:** 100-question Pilot set and a disjoint **500-question Test set**, stratified by hop count, seeded.
- **Search setting (decided):** **one pooled corpus** of every paragraph in all six MuSiQue files (~139,416 paragraphs, ~64 MB), deduplicated, following IRCoT and Adaptive-RAG. Using only each question's own 20 paragraphs was rejected: 3 Rounds would retrieve most of them and the Judge would have little to decide.
- **Titles repeat:** 34% of dev Gold paragraphs share a title with another paragraph, and 116 questions have two Gold paragraphs with the same title. So each MuSiQue paragraph gets an id = hash of title + text, stored as `gold_ids`, and the oracle checks coverage **by paragraph id**. The Steer signal still names the title.
- **Known limit:** 3 Rounds is tight for 4-hop questions. A rough oracle-loop BM25 test found all Gold paragraphs within 3 Rounds for 76% of 2-hop, 47% of 3-hop and 17% of 4-hop questions. Decision: **keep 3 Rounds and report every MuSiQue result by hop count.**

### 5.4 Annotation

- No new labels are needed for the main results: gold answers and gold paragraphs come with the datasets, and scoring is automatic.
- The only manual labeling is the error analysis (§10.4): 60 logged judge reasonings labeled by error type, with two teammates labeling the first 20 independently to measure agreement.

## 6. System design

### 6.1 Pipeline pieces (roles)

| Piece | Job |
|---|---|
| **Dataset** | Provides the question, gold answer and Gold paragraph titles/ids. |
| **Retriever** | BM25 search: `search(query, k, exclude_titles)`. Fixed for the whole study. |
| **Judge** | Reads the question and all Evidence; emits a Stop decision and a Steer signal. |
| **Rewriter** | Turns the question, past queries and the Steer signal into the next search query. Only the Rewriter reads the Steer signal. |
| **Answerer** | Produces a short final answer from the question and the Evidence. |

### 6.2 The loop

```
evidence = search(question, k)
for round in 1..3:
    judge = LLM judge (always run, so its verdict is logged in every setting)
    shadow_answer = Answerer(question, evidence)            # logged, not counted
    stop  = stop_source(judge, oracle)                      # LLM, oracle, never, or always
    if stop or round == 3: final answer = shadow_answer; break
    steer = steer_source(judge, oracle)                     # LLM or oracle
    query = Rewriter(question, past queries, steer)
    evidence += search(query, k, exclude=evidence titles)   # trimmed to token budget
```

Decisions built into the loop:
- **A Round** is one retrieve-then-judge cycle; at most 3 per question. At Round 3 the Answerer answers regardless.
- **Evidence accumulates:** each Round adds k new paragraphs (paragraphs already held are skipped). The Judge and Answerer always see all Evidence so far, trimmed to the token budget if needed.
- **Fixed parts:** the retriever, Rewriter and Answerer are identical across all conditions. Only who makes the Stop decision and who produces the Steer signal changes.
- **The LLM judge is called in every setting**, even where its output isn't used (e.g. Condition D), so its verdict can always be scored against the answer keys.
- **Shadow answers:** the Answerer answers after **every Round in every setting**. Only the answer at the stopping Round counts toward accuracy and cost. The extra answers feed the Answerability key, the Answer-oracle, and offline stop-threshold sweeps.

### 6.3 LLM judge output format

```
REASONING: <a few sentences>
ENOUGH: yes|no
MISSING: <what information is still needed, or "nothing">
```

- The reasoning comes first so there are "thoughts" worth logging.
- **Confidence:** `p_yes` = P("yes") / (P("yes") + P("no")) from the token probabilities at the ENOUGH position. Every model call returns, per generated token, the chosen token and its top 5 alternatives with log probabilities (same format on Ollama and vLLM); "yes"/"Yes"/" yes" and "no"/"No"/" no" are read from that position. If neither appears in the top 5, `p_yes` is flagged unavailable for that call.
- **Malformed output** counts as "no" and is flagged `parse_error: true` in the log.
- Qwen3's built-in thinking mode is switched off (`enable_thinking=False`); the reasoning goes in the REASONING field instead.

### 6.4 Rewriter and Answerer

- **Rewriter:** sees the question, the queries already tried, and the Steer signal. Outputs one search query.
- **Answerer:** sees the question and the Evidence. Outputs a short answer only (a name, date, yes/no…), which is the format exact match needs.
- Each role's prompt is a string at the top of its own file, version-numbered (`v0`, then `v1` after the pilot); the version is written into the run log.

### 6.5 Oracle judge

- **Stop decision:** "yes" once **every** Gold paragraph is in the Evidence (HotpotQA: by title; MuSiQue: by paragraph id).
- **Steer signal:** the **title** of the first missing Gold paragraph (in the dataset's order), phrased `"Missing information about: <title>"`. If nothing is missing it returns `"nothing"`.
- **Never the paragraph text.** The Rewriter still has to write a query and BM25 still has to find the paragraph, which keeps the steering comparison fair. Alternatives rejected: giving the full paragraph text (leaks the answer) and gold sub-questions (HotpotQA doesn't have them).
- Pure Python, no model calls. Title matching tolerates capitalization/whitespace differences and duplicate titles in the Evidence.

### 6.6 Plug-and-play roles

- `config.yaml` has **one model setting per role**: `models.judge`, `models.rewriter`, `models.answerer`. Each holds the model's name for **both backends** (`ollama` and `vllm`), and one `backend` line picks which is used, so moving between the Mac and Kaggle never touches the roles.
- `pipeline/llm.py` is the **only** file that talks to a model: each role calls `llm.generate("<role>", prompt)`, which looks up that role's model in the config.
- **Tier 1:** all three roles use the **same model** (chosen in the pilot), so a difference between conditions can only come from the judge.
- **Tier 3:** only `models.judge` changes; the Rewriter and Answerer stay on the Tier 1 model. Swapping the judge is a one-line config change.

## 7. Experimental design

### 7.1 Main experiment: the 2×2

| | LLM steers | Oracle steers |
|---|---|---|
| **LLM stops** | **A**: deployed system | **C**: C − A = cost of bad steering |
| **Oracle stops** | **B**: B − A = cost of bad stopping | **D**: best case |

- **B − A** answers "how much do we lose because the LLM stops at the wrong time?"
- **C − A** answers "how much do we lose because the LLM searches for the wrong thing?"
- Whichever is larger is the bigger bottleneck (**RQ2**). **D − A** is the total gap (**RQ1**).
- If A is nearly as good as D, the judge is **not** the bottleneck. That is still a valid answer.

### 7.2 Baselines and reference points

The LLM judge and oracle judge are **not** baselines; they make the decisions in the main experiment. Baselines are "dumb judges" that follow a fixed rule.

| Run | Who decides when to stop | Who decides what to search next | Role |
|---|---|---|---|
| **A** | LLM judge | LLM judge | Main experiment |
| **B** | Oracle | LLM judge | Main experiment |
| **C** | LLM judge | Oracle | Main experiment |
| **D** | Oracle | Oracle | Main experiment |
| **Closed-book** | nobody (no retrieval) | nobody | Baseline: flags memorized answers |
| **Single-turn** | fixed rule: always stop after Round 1 (a judge that always says "enough") | nobody | Baseline: the floor |
| **Always-loop** | fixed rule: never stop before Round 3 (a judge that always says "keep going") | LLM judge | Baseline |
| **Coin-flip** *(proposed, optional)* | random | LLM judge | Baseline; computed offline from Always-loop logs, no GPU |
| **Answer-oracle** | best Round, chosen with hindsight | LLM judge | Upper bound |

**Answer-oracle** (upper bound, not a run): a question counts as correct if the Answerer is correct after **any** Round. It is the best score any stopping rule could reach with our retriever and rewriter. It needs no extra run: it is computed from Always-loop's per-Round shadow answers. Comparing it with B shows where "has all Gold paragraphs" and "can actually answer" come apart (the Joren et al. point).

**Comparisons:** A vs the baselines asks "is the LLM judge good at all?" A vs B, C, D asks "how much does it lose, and from which job?"

### 7.3 Measures

- **Answer quality:** exact match (EM) and F1, using HotpotQA's official answer normalization (lowercase, strip punctuation and articles).
- **Judge quality:** precision and recall of the LLM judge's stop verdicts, per Round, against **two answer keys**:
  - **Coverage key:** the stop is correct when every Gold paragraph is in the Evidence.
  - **Answerability key:** the stop is correct when the Answerer's shadow answer at that Round is correct.

  Reporting both shows where "sufficient" and "enough to answer" come apart. Also reported: early-stop and late-stop rates.
- **Calibration:** reliability diagram and ECE of `p_yes` against each key.
- **Cost:** total tokens (prompt + generated) and LLM calls per question. Shadow answers are excluded from cost.
- **Rounds used:** distribution per setting.
- **Accuracy vs cost curve:** EM against mean tokens for every setting, plus a curve from sweeping a stop threshold on `p_yes` offline.

### 7.4 Statistics

- **Paired bootstrap** 95% confidence intervals (10,000 resamples, seed from config) for every headline number and for B − A, C − A, D − A.
- **McNemar tests** on per-question correctness for those gaps.
- **Memorization control:** one closed-book Answerer run on the Test set. Every headline result is reported on **all questions** and on the **not-memorized subset** (questions closed-book got wrong).
- **Decoding:** greedy (temperature 0). Seeds therefore only matter for sampling question sets (`hotpotqa.sample_seed: 0`) and for the bootstrap and other repeated randomness (`seeds: [0, 1, 2]`).

### 7.5 Error analysis

- Sample 60 Condition A questions (seeded): 30 where A is wrong but D is right, and 30 where the LLM judge's stop verdict disagrees with the Coverage key.
- Label each from its logged judge reasoning with a fixed scheme, drafted first on 10 questions:
  - **Early stop:** said "enough" with a Gold paragraph missing.
  - **Late stop:** said "not enough" with everything present.
  - **Vague steer:** MISSING doesn't name a useful entity.
  - **Wrong steer:** MISSING points at the wrong entity.
  - **Good steer, bad retrieval:** the right entity was named, but BM25 didn't find it.
  - **Answerer error:** the judge was right, but the answer was wrong.
- Two teammates label the first 20 independently; report Cohen's kappa.
- Pick 2–3 short example traces for the paper.

### 7.6 Pilot and freeze

On the **Pilot set only**:
1. **Choose the Tier 1 model** between Qwen3-8B-AWQ and Qwen3-4B-Instruct-2507: run Condition A and Single-turn with each; pick the higher parse rate and EM; tie goes to the faster one.
2. **Choose k** (paragraphs per Round) from 2, 5, 10: the smallest k where Condition D's Gold coverage after 3 Rounds is clearly above Single-turn's.
3. **Choose the token budget** so 3 Rounds of Evidence fit in the model's context without trimming for most questions.
4. **Refine prompts** (v0 → v1) only for format problems (parse failures, answers too long for EM). Never tune to make the judge look better or worse.
5. **Freeze:** write final values into `config.yaml`, commit, tag `v1-frozen`. After this, nobody changes prompts or settings for Tier 1. If something is broken: stop, fix, re-freeze, rerun the affected settings, and log it.
6. Write up what was tried, chosen and why in the Pilot section of `NOTES.md` (becomes the paper's setup section).

For MuSiQue (Tier 2), prompts stay frozen from Tier 1; a 20-question pilot check only confirms parsing works.

## 8. Models, compute and environment

### 8.1 Models

| Use | Model | How it runs on Kaggle's 2× T4 |
|---|---|---|
| Tier 1 candidate 1 | **Qwen3-8B-AWQ** (official 4-bit, ~6.1 GB) | One copy per T4 (data parallel); thinking off |
| Tier 1 candidate 2 | **Qwen3-4B-Instruct-2507** | fp16, fits on one T4, no quantization |
| Tier 1 fallback | Llama-3.1-8B-Instruct | fp16 across both T4s or 4-bit AWQ |
| Tier 3 larger judge | **Qwen3-14B-AWQ** (~10 GB) | One T4 |
| Tier 3 other-family judge | **Llama-3.1-8B-Instruct** (needs an HF token) or **Granite-4.2-8B** | fp16 across both T4s |
| Tier 3 optional | Hosted 70B judge via paid API | ≤ $40, see §8.4 |

Ruled out on the T4: **Gemma 4** (doesn't run in vLLM on T4) and **Qwen3.5** (fp16 overflow risk).

### 8.2 Where things run

- **Every reported number comes from vLLM on Kaggle** (free 2× NVIDIA T4). Reason: Ollama on the Mac uses differently compressed weights and can answer slightly differently, which would confound comparisons between conditions.
- **Mac + Ollama:** writing and testing code only. Statistics and figures are computed on the Mac, but only from Kaggle run logs.
- **Exception:** the optional hosted 70B judge (T19) runs through an API; everything else in those runs stays on Kaggle, and the paper states this.
- **Colab (free T4):** backup only. Its software setup differs from Kaggle, so never mix Colab and Kaggle runs inside one comparison; if Kaggle became unavailable, rerun every setting of that comparison on Colab.

### 8.3 Compute budget

- **Nothing is trained**; we only run existing 4–8B models.
- **BM25:** CPU only. The index is built once on Kaggle in ~12 minutes (not on Colab, whose 12 GB RAM may be too little), saved (~1.1 GB index, ~2.9 GB with corpus) as a private Kaggle Dataset plus a Drive copy, and loaded with memory mapping everywhere (~1–3 GB RAM, ~21–48 ms per query).
- **GPU:** Kaggle gives ~30 GPU-hours per week per account; four members.
- **Estimate:** all Tier 1 main runs (1,000 questions × 7 settings) ≈ **5 GPU-hours** estimated; **measured Oct 6 (T04): ≤ 10.7 GPU-hours** on one T4 (upper bound, see `NOTES.md` Throughput). Whole project ≈ **30–100 GPU-hours** vs ~300 available before Dec 1. No cuts needed; the Test set stays at 1,000.
  - Sizing basis: worst case ~7.6k prompt tokens and ~450 generated tokens per question per loop setting; ~3k prompt tok/s and ~500 generated tok/s combined on 2× T4 (source-free estimate).
  - vLLM prefix caching should lower cost further, since the Judge and shadow answer read the same Evidence.
- **Throughput is not yet measured.** T04 measures it on 20 questions; T11 recomputes the budget before main runs. If it's more than 5× slower than estimated, reopen the budget question.

### 8.4 Money

- **$0** for Tiers 1 and 2.
- **Up to $40** of API credit, held **only** for Tier 3's hosted 70B judge. Go/no-go decided after Tier 1: only if there's a clear A-vs-D gap that T17/T18 didn't already close. Estimate cost on 20 questions first, then run Conditions A and C on the largest seeded subset the budget covers, and rerun the Tier 1 judge on the same subset for comparison. The API must return log probabilities, or `p_yes` is marked unavailable for that judge.

### 8.5 Software environment

- **Python 3.12 everywhere** (decided Sep 23). Kaggle runs 3.12.12, Colab runs 3.12; the Mac runs 3.12.14 (Homebrew) in `.venv`. Python 3.11 is also installed on the owner's Mac from earlier and kept until the project ends; nothing uses it.
- **Kaggle notebooks need Internet switched on** (requires a phone-verified account). `kaggle.ipynb` stops with that hint if it's off.
- **Version pins:** exact versions only (`name==x.y.z`). Final pins are generated on Kaggle (`pip freeze`) and reused everywhere, so a Mac-only version never gets pinned. vLLM goes in a separate `requirements-gpu.txt` because it doesn't install on the Mac.

## 9. Libraries: taken vs built

### 9.1 Existing libraries and tools

| Library / tool | Used for | Status |
|---|---|---|
| **vLLM** 0.30.0 (fallback 0.28.0) | Model inference on Kaggle T4s; token log probabilities for `p_yes`. Avoid FP8 KV cache and act-order GPTQ checkpoints on T4. Kept in `requirements-gpu.txt` (doesn't install on the Mac). | Pinned |
| **Ollama** (≥ v0.12.11 for logprobs) | Development on the Mac only | Chosen |
| **Hugging Face Hub** | Model weights | Chosen |
| **bm25s** 0.3.11 + PyStemmer 3.1.0, numba 0.65.0 | BM25 index build, search, save/load with mmap | Pinned |
| **huggingface_hub** 1.33.0 (vLLM's `transformers` needs < 2.0) | Downloading HotpotQA dev from Hugging Face | Pinned |
| **pyarrow** 25.0.1 | Reading the HotpotQA Parquet file | Pinned |
| **PyYAML** 6.0.3 | Reading `config.yaml` | Pinned |
| **pytest** 9.1.1 | Small tests | Pinned |
| **NumPy** 2.4.6 (the version vLLM installs on Kaggle) | Arrays for BM25 (installed with bm25s); bootstrap resampling | Pinned |
| **pandas** | Loading run logs | To pin |
| **SciPy** | Exact McNemar test (binomial test on discordant pairs) | To pin |
| **matplotlib** | Reliability diagrams, accuracy-vs-cost plots | To pin |
| **HotpotQA official eval script** | Answer normalization and EM/F1, adapted so scores match the standard | Adapted |
| **Kaggle / Colab / GitHub** | Free GPUs (Kaggle primary, Colab backup); shared code | In use |

**Not used:** Pyserini (needs Python ≥ 3.12 plus Java 21, a ~7 GB environment, and is slower on HotpotQA than bm25s).

### 9.2 What we build ourselves

| Component | Built or taken |
|---|---|
| Retrieval loop with swappable stop/steer sources (`loop.py`) | Built |
| Judge, Rewriter, Answerer prompts and output parsing | Built |
| Oracle judge | Built |
| Model access (`llm.py`) | Thin wrapper over vLLM / Ollama |
| Search (`retriever.py`) | Thin wrapper over bm25s |
| Question-set sampling, gold extraction, MuSiQue paragraph ids (`dataset.py`) | Built |
| Run logging and resume | Built |
| EM / F1 | Taken (official HotpotQA script) |
| Judge P/R on two keys, calibration (ECE), cost accounting, threshold sweep | Built |
| Paired bootstrap CIs | Built (~20 lines of NumPy) |
| McNemar test | Taken (SciPy) |
| Error-analysis labeling scheme | Built |

In short: we borrow the heavy machinery (inference, search, standard metrics) and build everything that is the study itself.

## 10. Code and repository

### 10.1 Layout (decided Sep 23: lean, pipeline-shaped)

The first, conventional scaffold (`src/`, `configs/`, `notebooks/`, `docs/`) was built and then removed as too complicated. The layout now follows the pipeline: questions → retriever → judge → rewriter → answerer.

```
Is-the-Judge-the-Bottleneck/
├── pipeline/
│   ├── __init__.py    # load_config(), so every piece reads config.yaml the same way
│   ├── dataset.py     # HotpotQA / MuSiQue → Pilot set & Test set questions
│   ├── retriever.py   # BM25 search
│   ├── judge.py       # LLM judge + Oracle judge (Stop decision, Steer signal)
│   ├── rewriter.py    # Steer signal → next query
│   ├── answerer.py    # Evidence → answer
│   ├── llm.py         # the one place that talks to the model
│   └── loop.py        # wires the pieces together; Conditions A–D, reference points
├── run.py             # run one setting on one question set → writes the run log
├── evaluate.py        # scores, statistics, tables, figures (--report)
├── test_pipeline.py   # small checks, run with `pytest`
├── kaggle.ipynb       # setup + run on Kaggle / Colab
├── config.yaml        # every setting
├── requirements.txt
├── NOTES.md           # Pilot choices, Runs sheet, Throughput numbers
├── README.md
├── CONTEXT.md         # glossary
├── LICENSE            # MIT, for the code only (datasets keep their own licenses)
├── specification.md   # this document
├── backlog/           # build tickets T01–T19 and their status
├── results/           # final tables + figures for the paper (committed)
├── data/              # datasets, index (git-ignored)
└── runs/              # run logs (git-ignored)
```

- `data/` and `runs/` stay off GitHub (several GB, constantly changing).
- `llm.py` exists even though it isn't a pipeline step: without it, model-loading code would be copied three times and swapping the judge would stop being a config change.
- `loop.py` is the pipeline itself: the pieces don't call each other; the loop decides the order and where the conditions differ.
- Prompts live at the top of each role's file (no separate prompts folder). Notes live in `NOTES.md` (no `docs/`). Analysis is `python evaluate.py --report` (no analysis notebook).

### 10.2 Configuration

- **Rule:** no settings hard-coded in code; everything is in `config.yaml`, read via `pipeline.load_config()`.
- Current blocks:
  - `seeds: [0, 1, 2]` (bootstrap)
  - `hotpotqa`: source file, `pilot_size: 100`, `test_size: 1000`, `sample_seed: 0` (T02)
  - `wiki`: dump URL, `dump_dir`, `index_dir`, `recall_ks: [2, 5, 10, 20]` (T03)
  - `backend`: `ollama` on the Mac, `vllm` on Kaggle (T04)
  - `models`: per role, a name for each backend (T04)
  - `generation`: `temperature: 0`, `max_tokens` per role, `top_logprobs: 5` (T04)
  - `vllm`: `max_model_len: 8192`, `gpu_memory_utilization: 0.90` (T04)
  - `retrieval`, `loop` (`rounds: 3`), `paths` for data/runs/results
- Development values until the pilot (T10) decides: `models.*` = Qwen3-4B-Instruct-2507 (`qwen3:4b-instruct` on Ollama). Still `null` until T10: `retrieval.k`, `loop.token_budget`.

### 10.3 Run log

Every run writes `runs/<run_id>/`, where `run_id` = date-time + setting (e.g. `2026-10-14T1530_A`):
- **`meta.json`:** date and time, condition, dataset, question set, model per role, backend, git commit, full copy of the config, prompt versions.
- **`questions.jsonl`**, one line per question:
  - `id`, `question`, `gold_answer`, `gold_titles`
  - `rounds_used`, `final_answer`, `em`, `f1`
  - `total_prompt_tokens`, `total_completion_tokens`, `llm_calls` (shadow answers counted separately)
  - `rounds`: per Round, `query`, `retrieved_titles`, `judge_raw` (the judge's full output including its reasoning), `stop`, `p_yes`, `steer`, `shadow_answer`, `shadow_em`, tokens per role, `parse_error`
- **Resumable:** on restart, question ids already in `questions.jsonl` are skipped (Kaggle/Colab sessions die; no lost or duplicated work).
- **Persistence:** on Kaggle, `runs/` points to `/kaggle/working/runs`; on Colab, to Google Drive.
- `evaluate.py` provides `load_run(run_id) → DataFrame`.

### 10.4 Outputs for the paper

- `results/<set>/summary.csv`: one row per setting (EM, F1, judge P/R on both keys, tokens, calls, mean Rounds).
- `results/test/`: main table, gap decomposition, memorization table, judge quality, calibration, error labels; as CSV and LaTeX.
- `results/figures/`: reliability diagrams, accuracy-vs-cost curves (PDF).
- `results/test/judge_comparison.csv` (Tier 3), `results/test_musique/` (Tier 2), cross-dataset and by-hop tables.
- `NOTES.md`: Pilot section, Runs sheet (run_id, setting, set, date, GPU hours, who, notes), Throughput section.

## 11. Plan and timeline

Tickets are worked in order; each lists its dependencies and a "Done when" check. A few can overlap: T03 ∥ T04, T05 ∥ T07, T13 ∥ T14. **Current status is tracked only in [backlog/README.md](backlog/README.md).**

| # | Ticket | Target |
|---|---|---|
| T01 | Repo and environment | Sep 23 – 29 |
| T02 | Pilot set and Test set | Sep 23 – 29 |
| T03 | Wikipedia corpus and BM25 search | Sep 30 – Oct 6 |
| T04 | Model backend (plug-and-play roles), throughput measurement | Sep 30 – Oct 6 |
| T05 | Run log | Oct 7 – 13 |
| T06 | Role prompts and output parsing | Oct 7 – 13 |
| T07 | Oracle judge | Oct 7 – 13 |
| T08 | The loop and all seven settings | Oct 14 – 20 |
| T09 | Scoring and per-run summary | Oct 14 – 20 |
| T10 | Pilot and freeze | Oct 21 – 27 |
| T11 | Main runs on the Test set | Oct 28 – Nov 6 |
| T12 | Analysis, statistics and figures | Nov 3 – 10 |
| T13 | Error analysis of the Judge | Nov 7 – 12 |
| T14 | MuSiQue data and search setting (Tier 2) | Nov 11 – 14 |
| T15 | MuSiQue pilot check and main runs | Nov 14 – 19 |
| T16 | MuSiQue analysis and cross-dataset table | Nov 19 – 21 |
| T17 | Larger judge (Tier 3) | Nov 19 – 23 |
| T18 | Judge from a different model family | Nov 22 – 25 |
| T19 | Hosted 70B judge (optional, ≤ $40) | Nov 24 – 26 |
| – | Buffer and paper polish | Nov 26 – Dec 1 |

- **Tier 1 finish line (~Nov 12):** a complete paper's worth of results.
- **Main-run order (T11):** Closed-book and Single-turn first (cheapest, early numbers), then Always-loop, then Conditions A–D.
- **Tier 3 reruns** only the settings where the LLM judge's output is used: A, B, C and Always-loop.

## 12. Research findings behind the decisions

| Question | Answer |
|---|---|
| Can a free machine hold a BM25 index over HotpotQA's Wikipedia? | Yes, with `bm25s`: built once on Kaggle (~12 min, ~3 GB saved), loaded with mmap everywhere (~1–3 GB RAM). |
| Which 3–9B models run on Kaggle's free GPU and give token probabilities? | Kaggle is 2× T4 only (P100 retired Sep 15, 2026); vLLM 0.30.0 works and returns logprobs. Tier 1: Qwen3-8B-AWQ or Qwen3-4B-Instruct-2507. Tier 3: Qwen3-14B-AWQ; Llama-3.1-8B or Granite-4.2-8B. |
| Do the runs fit in free GPU hours before Dec 1? | Yes: measured ≤ 10.7 GPU-hours for all Tier 1 main runs (T04, Oct 6). |
| How is MuSiQue set up and searched? | MuSiQue-Ans from dev (2,417 q), pooled corpus of 139k paragraphs, coverage by paragraph id (titles repeat), 3 Rounds kept with results by hop count. |


## 13. Risks and mitigations

| Risk | Mitigation |
|---|---|
| 3 Rounds is too few for 4-hop MuSiQue questions | Report MuSiQue by hop count; treat it as a finding. |
| Throughput on T4 is unmeasured | Measure in T04; recompute the budget before T11; ~3× headroom. |
| Building the BM25 index may not fit on Colab | Build on Kaggle only. |
| Mixing inference setups confounds comparisons | All reported numbers from vLLM on Kaggle; never mix Colab and Kaggle within a comparison. |
| Tuning to the Test set | Tune only on the Pilot set; freeze (`v1-frozen`) before any Test run. |
| Memorized answers inflate retrieval's apparent value | Closed-book run; report the not-memorized subset. |
| Kaggle sessions end mid-run | Resumable logs; `runs/` on persistent storage. |
| Judge output doesn't parse | Fixed format, ≥ 95% parse-rate check, malformed = "no" and flagged. |
| Time runs out | Drop Tier 3, then Tier 2. Tier 1 alone is a complete paper. |

## 14. Open questions

- **Whether to spend the $40** on a hosted 70B judge: decided after Tier 1 results.
- **Coin-flip baseline:** proposed, not yet confirmed. Costs no GPU time.
- Values set in the pilot (T10): the Tier 1 model, k, and the token budget.

## 15. Working agreements

- Lean code shaped like the pipeline; propose the smallest thing and explain any extra file before adding it.
- The repo owner commits and pushes; nothing is pushed on anyone's behalf.
- Ask before installing system software or deleting anything.
- One ticket at a time, confirmed before starting.
- Use the glossary terms from `CONTEXT.md` in code, tickets and the paper.
