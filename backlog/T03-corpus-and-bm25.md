# T03 · Wikipedia corpus and BM25 search

Tier: 1 · Owner: _ · Depends: T02, research [01](../specification.md#12-research-findings-behind-the-decisions)

## What

- On **Kaggle** (not Colab: the build may not fit in 12 GB RAM), download HotpotQA's processed 2017 Wikipedia, `https://nlp.stanford.edu/projects/hotpotqa/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2` (1.55 GB). Index `title + " " + text` (sentences joined), not `text_with_links`. Keep `title` as the doc id so Gold paragraph titles match.
- Build with `bm25s` (English stopwords + PyStemmer), about 12 minutes. Add `bm25s[core]==0.3.11`, `PyStemmer==3.1.0`, `numba==0.67.0` to `requirements.txt`, checking the versions pip actually resolves.
- Save the index and corpus once (~3 GB) as a private Kaggle Dataset and a copy on Drive. On every machine, load with `bm25s.BM25.load(dir, mmap=True, load_corpus=True)` (~1–3 GB RAM).
- `pipeline/retriever.py`: `search(query, k, exclude_titles) -> list[{title, text, score}]`. `exclude_titles` lets a Round skip paragraphs already in the Evidence.
- Sanity script: for each Pilot set question, search with the raw question and report how often each Gold paragraph appears in the top k for k = 2, 5, 10, 20.

## Done when

- The index loads in under a couple of minutes on Kaggle and on the Mac.
- `search` returns results for any query and skips excluded titles.
- The sanity numbers are saved to `results/pilot/bm25_recall.csv`. Expect one Gold paragraph to be found far more often than the other: that's the second hop the loop exists to find.

## Description

BM25 is a classic keyword search: it ranks paragraphs by how many query words they share, weighted by how rare those words are. It stays fixed for the whole study, so building it well once matters. The sanity check tells us whether retrieval alone already finds both Gold paragraphs (then the loop has nothing to do) or usually misses one (then the loop has room to help).

## Notes

- Oct 3: `pipeline/retriever.py` written test-first (4 tests on tiny fake data). Commands: `python -m pipeline.retriever build` (Kaggle) and `python -m pipeline.retriever check` (Mac).
- Dump format checked on the first 5 MB of the real file: each line has `title` and `text` as a list of sentence strings; the archive is 1,553,565,403 bytes.
- The full build tokenizes into word ids, not word lists, to keep memory down for 5.2M paragraphs.
- Remaining: build on Kaggle (`kaggle.ipynb`, T03 cell), save the index as a private Kaggle Dataset, copy it to `data/wiki/bm25_index/` on the SSD, run `check`.
