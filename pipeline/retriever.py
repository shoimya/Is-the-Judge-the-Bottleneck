"""BM25 search over HotpotQA's 2017 Wikipedia intro paragraphs. Built in T03.

    python -m pipeline.retriever build    # on Kaggle: download the dump, build and save the index
    python -m pipeline.retriever check    # on the Mac: Gold paragraph recall on the Pilot set
"""

import argparse
import bz2
import csv
import json
import tarfile
import urllib.request
from collections.abc import Collection, Iterable, Iterator
from pathlib import Path

import bm25s
import Stemmer

from pipeline import DATA_DIR, REPO_ROOT, load_config
from pipeline.dataset import load_question_set

STEMMER = Stemmer.Stemmer("english")


def read_wiki_paragraphs(dump_files: Iterable[Path]) -> Iterator[dict]:
    """HotpotQA Wikipedia dump files (bz2, one JSON article per line) -> {title, text} per paragraph."""
    for dump_file in dump_files:
        with bz2.open(dump_file, "rt") as lines:
            for line in lines:
                article = json.loads(line)
                yield {"title": article["title"], "text": "".join(article["text"])}


def tokenize(texts: list[str], as_ids: bool = False, show_progress: bool = False):
    """Lowercased, stemmed word tokens with English stopwords removed; the same rule for paragraphs and queries.

    as_ids=True returns word ids plus a vocabulary, which needs far less memory for millions of paragraphs.
    """
    return bm25s.tokenize(texts, stopwords="en", stemmer=STEMMER, return_ids=as_ids, show_progress=show_progress)


def build_index(paragraphs: Iterable[dict], index_dir: Path, show_progress: bool = False) -> None:
    """Index each paragraph's title + text with BM25 and save the index and the paragraphs to index_dir."""
    corpus = list(paragraphs)
    paragraph_tokens = tokenize([f"{paragraph['title']} {paragraph['text']}" for paragraph in corpus],
                                as_ids=True, show_progress=show_progress)
    index = bm25s.BM25()
    index.index(paragraph_tokens, show_progress=show_progress)
    index.save(index_dir, corpus=corpus)


class Retriever:
    """A loaded BM25 index that answers search queries."""

    def __init__(self, index: bm25s.BM25):
        self.index = index

    @classmethod
    def load(cls, index_dir: Path) -> "Retriever":
        # mmap keeps the index on disk and reads only what a query needs (~1-3 GB RAM for full Wikipedia).
        return cls(bm25s.BM25.load(index_dir, mmap=True, load_corpus=True))

    def search(self, query: str, k: int, exclude_titles: Collection[str] = frozenset()) -> list[dict]:
        """Top k paragraphs for the query as {title, text, score}, skipping titles already in the Evidence."""
        # Ask for extra results to make up for excluded ones, but never more than the index holds.
        results_to_fetch = min(k + len(exclude_titles), self.index.scores["num_docs"])
        paragraphs, scores = self.index.retrieve(tokenize([query]), k=results_to_fetch, show_progress=False)
        results = [
            {"title": paragraph["title"], "text": paragraph["text"], "score": float(score)}
            for paragraph, score in zip(paragraphs[0], scores[0])
            if paragraph["title"] not in exclude_titles
        ]
        return results[:k]


def gold_recall_at_k(questions: list[dict], retriever: Retriever, ks: list[int]) -> list[dict]:
    """Search with each raw question; per k, the share of questions with all / at least one Gold paragraph found.

    "both_found" means every Gold paragraph was found: both of HotpotQA's two, or all of a MuSiQue question's 2-4.
    """
    retrieved_titles = [
        [result["title"] for result in retriever.search(question["question"], k=max(ks))]
        for question in questions
    ]
    rows = []
    for k in ks:
        gold_found_counts = [
            sum(title in titles[:k] for title in question["gold_titles"])
            for question, titles in zip(questions, retrieved_titles)
        ]
        rows.append({
            "k": k,
            "both_found": sum(found == len(question["gold_titles"])
                              for found, question in zip(gold_found_counts, questions)) / len(questions),
            "at_least_one_found": sum(count >= 1 for count in gold_found_counts) / len(questions),
        })
    return rows


def download_and_unpack_dump(dump_url: str, dump_dir: Path) -> list[Path]:
    """Download the Wikipedia dump archive into dump_dir once, unpack it, and return its files in a fixed order."""
    dump_dir.mkdir(parents=True, exist_ok=True)
    archive_path = dump_dir / Path(dump_url).name
    if not archive_path.exists():
        print(f"Downloading {dump_url} (1.55 GB) ...")
        urllib.request.urlretrieve(dump_url, archive_path)
    unpacked_dir = dump_dir / "unpacked"
    if not unpacked_dir.exists():
        print("Unpacking ...")
        with tarfile.open(archive_path) as archive:
            archive.extractall(unpacked_dir, filter="data")
    return sorted(unpacked_dir.rglob("*.bz2"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["build", "check"])
    command = parser.parse_args().command

    config = load_config()
    wiki_config = config["wiki"]
    index_dir = DATA_DIR / wiki_config["index_dir"]

    if command == "build":
        dump_files = download_and_unpack_dump(wiki_config["dump_url"], DATA_DIR / wiki_config["dump_dir"])
        print(f"Reading {len(dump_files)} dump files and building the index ...")
        build_index(read_wiki_paragraphs(dump_files), index_dir, show_progress=True)
        print(f"Saved the BM25 index to {index_dir}")

    if command == "check":
        pilot_set = load_question_set("pilot")
        recall_rows = gold_recall_at_k(pilot_set, Retriever.load(index_dir), wiki_config["recall_ks"])
        recall_csv = REPO_ROOT / config["paths"]["results_dir"] / "pilot" / "bm25_recall.csv"
        recall_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(recall_csv, "w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=["k", "both_found", "at_least_one_found"])
            writer.writeheader()
            writer.writerows(recall_rows)
        for row in recall_rows:
            print(f"k={row['k']:>2}  both Gold paragraphs found: {row['both_found']:.0%}   "
                  f"at least one: {row['at_least_one_found']:.0%}")
        print(f"Saved to {recall_csv}")


if __name__ == "__main__":
    main()
