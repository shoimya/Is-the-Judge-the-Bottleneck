"""BM25 over HotpotQA's 2017 Wikipedia abstracts. Built in T03.

    python -m pipeline.retriever build                 # download, extract, index
    python -m pipeline.retriever search "query" --k 10 # try a query
    python -m pipeline.retriever sanity                # Pilot recall -> results/pilot/bm25_recall.csv

Search is `search(query, k, exclude_titles) -> list[{title, text, score}]`;
`exclude_titles` lets a Round skip paragraphs already in the Evidence.

The corpus is indexed as `title + " " + text` (sentences joined, not
`text_with_links`) with English stopwords and PyStemmer. Titles are the doc
ids, so Gold paragraph titles match retrieved titles exactly. The tar from
Stanford holds one json-lines shard per wiki_*.bz2 (one article dict per
line), which is streamed shard by shard.

Memory: bm25s's own tokenize() keeps every token as a Python int (~36 bytes),
which for ~300M tokens (~11 GB) plus the ~5.5 GB corpus does not fit in 16 GB
RAM (the reason T03 says to build on Kaggle). `build_index` instead keeps
token ids as uint32 arrays (4 bytes each) and streams the shards, so the
peak is ~5 GB and the build fits on the Mac too. The saved corpus is written
as `{"id", "text", "title"}` dicts and loads memory-mapped.
"""

import argparse
import bz2
import csv
import json
import tarfile
import time
import urllib.request
from pathlib import Path

import bm25s
import numpy as np
import Stemmer
from bm25s.tokenization import Tokenized

from pipeline import REPO_ROOT, load_config

CORPUS_URL = (
    "https://nlp.stanford.edu/projects/hotpotqa/"
    "enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2"
)
CORPUS_TAR = "enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2"
SHARDS_DIRNAME = "enwiki-20171001-pages-meta-current-withlinks-abstracts"
INDEX_DIRNAME = "bm25_index"
WIKI_DIRNAME = "wiki"

_stemmer = Stemmer.Stemmer("english")
_retriever: bm25s.BM25 | None = None


def wiki_dir() -> Path:
    return REPO_ROOT / load_config()["paths"]["data_dir"] / WIKI_DIRNAME


def index_dir() -> Path:
    return wiki_dir() / INDEX_DIRNAME


# ---------------------------------------------------------------- corpus I/O


def download_corpus(corpus_dir: Path) -> Path:
    """Download the abstracts tar.bz2 (~1.55 GB), unless it is already there."""
    dest = corpus_dir / CORPUS_TAR
    if dest.exists() and dest.stat().st_size > 0:
        print(f"Using existing {dest}")
        return dest

    corpus_dir.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    print(f"Downloading {CORPUS_URL}")
    request = urllib.request.Request(CORPUS_URL, headers={"User-Agent": "cmsc723/1.0"})
    with urllib.request.urlopen(request, timeout=3600) as response, open(tmp, "wb") as out:
        size = 0
        next_report = 100 << 20
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            size += len(chunk)
            if size >= next_report:
                print(f"  {size / (1 << 20):.0f} MB")
                next_report += 100 << 20
    tmp.rename(dest)
    return dest


def extract_corpus(corpus_dir: Path) -> Path:
    """Extract the wiki_*.bz2 shards from the tar.bz2."""
    shards_dir = corpus_dir / SHARDS_DIRNAME
    if shards_dir.exists() and any(shards_dir.rglob("wiki_*.bz2")):
        print(f"Using existing {shards_dir}")
        return shards_dir

    print("Extracting shards (this takes a few minutes)...")
    t0 = time.time()
    with tarfile.open(corpus_dir / CORPUS_TAR, "r:bz2") as tar:
        tar.extractall(path=corpus_dir, filter="data")
    print(f"  done in {time.time() - t0:.0f}s")
    return shards_dir


def iter_abstracts(shards_dir: Path):
    """Yield one article dict at a time, streaming every shard in sorted order.

    Each wiki_*.bz2 shard is json-lines: one `{"id", "title", "text", ...}`
    object per line, so no file is ever fully in memory.
    """
    for shard in sorted(shards_dir.rglob("wiki_*.bz2")):
        with bz2.open(shard, "rt", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)


# ---------------------------------------------------------------- build


def _tokenize_batch(batch: list[str], vocab: dict[str, int], ids_per_doc: list[np.ndarray]) -> None:
    """Tokenize one batch; append each doc's token ids (uint32) to ids_per_doc."""
    token_lists = bm25s.tokenize(
        batch, stopwords="english", stemmer=_stemmer, return_ids=False, show_progress=False
    )
    for tokens in token_lists:
        doc_ids = np.empty(len(tokens), dtype=np.uint32)
        for j, token in enumerate(tokens):
            doc_ids[j] = vocab.setdefault(token, len(vocab))
        ids_per_doc.append(doc_ids)


def build_index(shards_dir: Path, out_dir: Path, batch_size: int = 100_000) -> dict:
    """Build and save the bm25s index. Two passes over the shards:

    1. tokenize (batched, uint32 ids) and index,
    2. stream the texts again as the saved corpus (written by BM25.save).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    vocab: dict[str, int] = {}
    ids_per_doc: list[np.ndarray] = []
    batch: list[str] = []
    n_docs = 0
    print("Pass 1: tokenizing...")
    for item in iter_abstracts(shards_dir):
        batch.append(f"{item['title']} {' '.join(item['text'])}")
        n_docs += 1
        if len(batch) >= batch_size:
            _tokenize_batch(batch, vocab, ids_per_doc)
            print(f"  {n_docs} docs, vocab {len(vocab)}")
            batch = []
    if batch:
        _tokenize_batch(batch, vocab, ids_per_doc)
    print(f"  {n_docs} docs, vocab {len(vocab)}, {time.time() - t0:.0f}s")

    print("Indexing...")
    retriever = bm25s.BM25()
    retriever.index(Tokenized(ids_per_doc, vocab))
    del ids_per_doc

    def corpus_stream():
        """Second pass: the saved corpus, one {'id', 'text', 'title'} per doc."""
        for i, item in enumerate(iter_abstracts(shards_dir)):
            yield {"id": i, "text": " ".join(item["text"]), "title": item["title"]}

    print("Saving...")
    retriever.save(out_dir, corpus=corpus_stream())
    return {"docs": n_docs, "vocab": len(vocab), "seconds": round(time.time() - t0)}


# ---------------------------------------------------------------- search


def load_retriever(index_directory: Path | None = None) -> bm25s.BM25:
    index_directory = Path(index_directory) if index_directory is not None else index_dir()
    return bm25s.BM25.load(index_directory, mmap=True, load_corpus=True, show_progress=False)


def get_retriever(index_directory: Path | None = None) -> bm25s.BM25:
    """The process-wide retriever, loaded once (first call is the slow one)."""
    global _retriever
    if _retriever is None:
        _retriever = load_retriever(index_directory)
    return _retriever


def search(query: str, k: int, exclude_titles: list[str] | set[str] = (), retriever=None) -> list[dict]:
    """Top-k paragraphs for `query`, skipping `exclude_titles`.

    Returns a list of {"title", "text", "score"} in rank order. More than k
    results are fetched so that excluded titles can be topped up.
    """
    if retriever is None:
        retriever = get_retriever()
    exclude_titles = set(exclude_titles)
    query_tokens = bm25s.tokenize(query, stopwords="english", stemmer=_stemmer, show_progress=False)
    results = retriever.retrieve(query_tokens, k=k + len(exclude_titles), show_progress=False)
    docs, scores = results.documents[0], results.scores[0]
    out = []
    for doc, score in zip(docs, scores):
        if doc["title"] in exclude_titles:
            continue
        out.append({"title": doc["title"], "text": doc["text"], "score": float(score)})
        if len(out) >= k:
            break
    return out


# ---------------------------------------------------------------- sanity


def sanity_recall(pilot_jsonl: Path, out_csv: Path, retriever=None) -> dict:
    """For each Pilot question, search the raw question and record whether each
    Gold paragraph appears in the top k, for k = 2, 5, 10, 20.

    Writes one row per question to out_csv and returns per-k recalls.
    """
    if retriever is None:
        retriever = get_retriever()
    questions = [json.loads(line) for line in open(pilot_jsonl, encoding="utf-8")]
    ks = (2, 5, 10, 20)
    header = ["id", "question", "gold_1", "gold_2"]
    header += [f"gold1@{k}" for k in ks] + [f"gold2@{k}" for k in ks]

    counts = {f"gold{g}@{k}": 0 for g in (1, 2) for k in ks}
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for q in questions:
            results = search(q["question"], k=max(ks), retriever=retriever)
            titles = [r["title"] for r in results]
            row = [q["id"], q["question"], q["gold_titles"][0], q["gold_titles"][1]]
            for g in (1, 2):
                for k in ks:
                    hit = 1 if q["gold_titles"][g - 1] in titles[:k] else 0
                    counts[f"gold{g}@{k}"] += hit
                    row.append(hit)
            writer.writerow(row)

    n = len(questions)
    return {f"gold{g}@{k}": counts[f"gold{g}@{k}"] / n for g in (1, 2) for k in ks}


# ---------------------------------------------------------------- CLI


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("build", help="download, extract, and build the index")

    p = sub.add_parser("search", help="print top-k paragraphs for one query")
    p.add_argument("query")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--exclude", nargs="*", default=[], help="titles to skip")

    sub.add_parser("sanity", help="Pilot recall -> results/pilot/bm25_recall.csv")

    args = parser.parse_args()

    if args.command == "build":
        t0 = time.time()
        corpus_dir = wiki_dir()
        download_corpus(corpus_dir)
        shards_dir = extract_corpus(corpus_dir)
        summary = build_index(shards_dir, index_dir())
        print(f"Built {summary['docs']} docs (vocab {summary['vocab']}) in {summary['seconds']}s "
              f"-> {index_dir()}")
        print(f"Total {time.time() - t0:.0f}s")

    elif args.command == "search":
        for hit in search(args.query, args.k, exclude_titles=args.exclude):
            print(f"{hit['score']:.4f}\t{hit['title']}")

    elif args.command == "sanity":
        cfg = load_config()
        pilot = REPO_ROOT / cfg["paths"]["data_dir"] / "hotpotqa" / "pilot.jsonl"
        out_csv = REPO_ROOT / cfg["paths"]["results_dir"] / "pilot" / "bm25_recall.csv"
        recalls = sanity_recall(pilot, out_csv)
        print(f"Wrote {out_csv}")
        for k in (2, 5, 10, 20):
            print(f"k={k:2d}  gold1 {recalls[f'gold1@{k}']:.3f}  gold2 {recalls[f'gold2@{k}']:.3f}")


if __name__ == "__main__":
    main()
