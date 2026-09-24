# Is the Judge the Bottleneck?

## Destination

A locked study design plus an ordered build backlog ([backlog/](backlog/)) that a 4-person team works through one ticket after another, Tier 1 first, so the paper can be written by **Dec 1, 2026**. The map is done when every backlog ticket can be started without an open question.

## Notes

- Vocabulary: [CONTEXT.md](../../CONTEXT.md) (Round, Evidence, Gold paragraph, Judge, Stop decision, Steer signal, Rewriter, Answerer, Oracle judge, LLM judge, Conditions A–D, Single-turn, Always-loop, Answer-oracle, Coverage key, Answerability key, Pilot set, Test set).
- This is a **planning** map. Its tickets in [issues/](issues/) settle facts and decisions. Build work lives in [backlog/](backlog/) and is done outside the map.
- The paper is written by the map owner. Backlog tickets produce the tables, figures and logs the paper needs.
- Budget: M-series Mac, Colab free, Kaggle free (~30 GPU-h/week), up to $40 of API spend reserved for Tier 3.
- Tiers: **Tier 1** = RQ1 + RQ2 on HotpotQA (must-have, a complete paper on its own). **Tier 2** = MuSiQue. **Tier 3** = RQ3 (larger or different-family judge).

### Decisions locked while charting (Sep 23, 2026)

These predate the tickets, so they're recorded here rather than under Decisions so far.

- **Oracle judge Stop decision:** "yes" once every Gold paragraph is in the Evidence. At Round 3 the Answerer answers regardless.
- **Oracle judge Steer signal:** the title of one missing Gold paragraph, handed to the Rewriter. Never the paragraph text.
- **Roles are plug-and-play:** each role (Judge, Rewriter, Answerer) has its own model setting. In Tier 1 all three use one model, chosen in the pilot. Tier 3 changes only the Judge.
- **Lean repo layout:** code follows the pipeline, one file per piece in `pipeline/` (dataset, retriever, judge, rewriter, answerer, llm, loop), plus `run.py`, `evaluate.py`, `config.yaml`, `kaggle.ipynb` and `NOTES.md` at the top level. Full tree in [T01](backlog/T01-repo-and-environment.md).
- **Python 3.12 everywhere:** matches Kaggle (3.12.12) and Colab. Final version pins are generated on Kaggle.
- **One backend for reported numbers:** every number in the paper comes from vLLM on Kaggle. Ollama on the Mac is for development only.
- **Reference points:** Single-turn, Always-loop and Answer-oracle, as defined in CONTEXT.md.
- **Judge answer keys:** Coverage key and Answerability key. Precision and recall are reported against both.
- **Evidence accumulates:** each Round adds k new paragraphs (duplicates skipped), trimmed to the token budget.
- **Shadow answers:** the Answerer answers after every Round in every condition. Only the answer at the stopping Round counts toward accuracy and cost, but the extra answers feed the Answerability key, the Answer-oracle, and offline stop-threshold sweeps.
- **Corpus:** HotpotQA's own processed 2017 Wikipedia (intro paragraphs), searched with BM25. Gold coverage is an exact title match.
- **Question sets:** 100-question Pilot set and a disjoint 1,000-question Test set, both from HotpotQA dev (fullwiki), seeded. The Test set is run only after the config is frozen.
- **Decoding and cost:** greedy decoding. Cost = total tokens (prompt + generated) and LLM calls per question.
- **Memorization control:** one closed-book Answerer run on the Test set. Headline results are reported on all questions and on the not-memorized subset.
- **Judge output format:** `REASONING / ENOUGH: yes|no / MISSING`. Confidence = P("yes") from token probabilities at the verdict.
- **Run log:** every run writes JSONL to `runs/<run_id>/`: run metadata (date, condition, models, git commit, config), plus per question the question, gold data, Rounds used, final answer, scores and cost, and per Round the query, retrieved titles, the Judge's full raw output, Stop decision, P("yes"), Steer signal and tokens.
- **Statistics:** paired bootstrap confidence intervals and McNemar tests.
- **API money:** the $40 is held for Tier 3's hosted 70B judge, with go/no-go decided after Tier 1 results.

## Decisions so far

<!-- one line per resolved ticket in issues/: [title](issues/NN-slug.md): gist -->

- [Can a free machine hold a BM25 index over HotpotQA's Wikipedia?](issues/01-bm25-index-fits-free-tier.md): yes, with `bm25s`, built once on Kaggle (~12 min, ~3 GB saved) and loaded with mmap everywhere (~1–3 GB RAM).
- [Which 3–9B models run on Kaggle's free GPU and give token probabilities?](issues/02-models-on-free-gpu.md): Kaggle is 2× T4 only; vLLM 0.30.0 works on it and returns log probabilities. Tier 1 candidates are Qwen3-8B-AWQ and Qwen3-4B-Instruct-2507; Tier 3 candidates are Qwen3-14B-AWQ (larger) and Llama-3.1-8B or Granite-4.2-8B (other family).
- [Do the Tier 1 runs fit in the free GPU hours before Dec 1?](issues/03-gpu-hour-budget.md): yes, ~5 GPU-hours estimated for all Tier 1 main runs (~15 h if 3× slower), with no cuts needed; T04 measures real throughput to confirm.
- [How is MuSiQue set up, and how do others search it?](issues/04-musique-setup.md): MuSiQue-Ans from dev (2,417 questions), one pooled corpus of 139k paragraphs, Gold coverage by paragraph identity (titles repeat), 3 Rounds kept with results split by hop count.

## Not yet specified

- **Whether to spend the $40:** decided from Tier 1 results (is the A-vs-D gap big enough that a 70B judge is worth testing?).

## Out of scope

- Training or fine-tuning any Judge (Self-RAG / CRAG style): this study swaps judges, it doesn't build them.
- Dense or hybrid retrievers: the retriever stays fixed as BM25.
- More than 3 Rounds, and datasets beyond HotpotQA and MuSiQue.
