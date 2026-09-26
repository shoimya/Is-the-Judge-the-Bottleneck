# Can a free machine hold a BM25 index over HotpotQA's Wikipedia?

Type: research
Status: resolved
Blocked by: none

## Question

HotpotQA's processed 2017 Wikipedia has about 5M intro paragraphs. Can we build and query a BM25 index over it within the memory of free Colab (~12 GB RAM), free Kaggle (~30 GB RAM) and a 16 GB Mac? Which library should we use (`bm25s`, Pyserini, or something else), how big are the index on disk and in RAM, how long does indexing take, and what is query latency? Where exactly is the dump downloaded from, and what is its format (title + paragraph fields)?

Feeds backlog: [T03](../../../backlog/T03-corpus-and-bm25.md).

## Answer

Yes, it fits. Use `bm25s` (0.3.11, with PyStemmer and numba). Build the index **once on Kaggle** (about 12 min for 5.23M paragraphs), save it (about 1.1 GB index, 2.9 GB with the corpus), and load it with `mmap=True` everywhere (roughly 1–3 GB RAM, 21–48 ms per query). The corpus is `enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2` from Stanford; index `title` + plain `text`. Skip Pyserini (needs Python 3.12+ and Java 21). Nobody has measured peak build RAM on Colab's 12 GB, so don't build there.
