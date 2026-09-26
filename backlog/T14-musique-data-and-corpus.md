# T14 · MuSiQue data and search setting

Tier: 2 · Owner: _ · Depends: T12, research [04](../.scratch/judge-bottleneck/issues/04-musique-setup.md)

## What

- Use **MuSiQue-Ans** (no unanswerable questions). Test labels aren't public, so sample both sets from **dev** (2,417 questions): a 100-question Pilot set and a disjoint **500-question Test set**, stratified by hop count (2/3/4), seeded.
- `pipeline/dataset.py` writes them in **the same format as T02** plus `hops`, and a `gold_ids` field: MuSiQue titles repeat (34% of Gold paragraphs share a title), so each paragraph's id is a hash of title + text.
- Build **one pooled corpus** of every paragraph in all six MuSiQue files (~139k paragraphs, ~64 MB), deduplicated by that id, and index it with the same BM25 code from T03. It's small enough to build on the Mac.
- Update T07's oracle to check coverage by paragraph id when `gold_ids` is present. HotpotQA keeps using titles (Wikipedia titles are unique there). The Steer signal still names the title.
- Rerun the T03 sanity check on the MuSiQue Pilot set.

## Done when

- The MuSiQue question files and index exist, and `search` works on them.
- `results/pilot/musique_bm25_recall.csv` is saved.
- The oracle's coverage check passes a unit test with two Gold paragraphs that share a title.

## Description

MuSiQue asks 2–4 hop questions that are harder to shortcut than HotpotQA. It tests whether the Tier 1 finding holds when more steps are needed. The prompts stay frozen from Tier 1, so the only thing that changes is the data. Expect the 3-Round cap to bite on 4-hop questions (research 04 found an oracle loop reaches every Gold paragraph for only ~17% of them), so every MuSiQue result is reported by hop count.
