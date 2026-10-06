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

# Here we set up the stemmer, which cuts words down to their root ("directed", "directing" -> "direct")
# so a query matches a paragraph even when the word endings differ.
STEMMER = Stemmer.Stemmer("english")


def read_wiki_paragraphs(dump_files: Iterable[Path]) -> Iterator[dict]:
    """HotpotQA Wikipedia dump files (bz2, one JSON article per line) -> {title, text} per paragraph."""
    for dump_file in dump_files:
        # Here we open one compressed dump file and read it as text, one line at a time.
        with bz2.open(dump_file, "rt") as lines:
            for line in lines:
                # Here we turn the line into an article. Its text is a list of sentences, so we join
                # them back into one paragraph, and hand it out one at a time to save memory.
                article = json.loads(line)
                yield {"title": article["title"], "text": "".join(article["text"])}


def tokenize(texts: list[str], return_word_ids: bool = False, show_progress: bool = False):
    """Lowercased, stemmed word tokens with English stopwords removed; the same rule for paragraphs and queries.

    return_word_ids=True returns word ids plus a vocabulary, which needs far less memory for millions of paragraphs.
    """
    # Here we split each text into words, drop common words like "the" and "of", and stem the rest.
    return bm25s.tokenize(texts, stopwords="en", stemmer=STEMMER, return_ids=return_word_ids,
                          show_progress=show_progress)


def build_index(paragraphs: Iterable[dict], index_dir: Path, show_progress: bool = False) -> None:
    """Index each paragraph's title + text with BM25 and save the index and the paragraphs to index_dir."""
    # Here we load every paragraph into memory, because the index is saved together with them.
    corpus = list(paragraphs)
    # Here we tokenize each paragraph's title and text together, so a search can match words in either.
    paragraph_tokens = tokenize([f"{paragraph['title']} {paragraph['text']}" for paragraph in corpus],
                                return_word_ids=True, show_progress=show_progress)
    # Here we build the BM25 index (word -> which paragraphs contain it, and how strongly) and save it to disk.
    index = bm25s.BM25()
    index.index(paragraph_tokens, show_progress=show_progress)
    index.save(index_dir, corpus=corpus)


class Retriever:
    """A loaded BM25 index that answers search queries."""

    def __init__(self, index: bm25s.BM25):
        self.index = index

    @classmethod
    def load(cls, index_dir: Path) -> "Retriever":
        """Open a saved index from index_dir."""
        # Here we use mmap, which keeps the index on disk and reads only what a query needs (~1-3 GB RAM for full Wikipedia).
        return cls(bm25s.BM25.load(index_dir, mmap=True, load_corpus=True))

    def search(self, query: str, k: int, exclude_titles: Collection[str] = frozenset()) -> list[dict]:
        """Top k paragraphs for the query as {title, text, score}, skipping titles already in the Evidence."""
        # Here we ask for extra results to make up for the ones we will skip, but never more than the index holds.
        results_to_fetch = min(k + len(exclude_titles), self.index.scores["num_docs"])
        # Here we run the search. bm25s searches many queries at once, so we pass a list of one
        # and read back the first (only) row of paragraphs and scores.
        paragraphs, scores = self.index.retrieve(tokenize([query]), k=results_to_fetch, show_progress=False)
        # Here we drop paragraphs the loop already has in its Evidence, so every search brings something new.
        results = [
            {"title": paragraph["title"], "text": paragraph["text"], "score": float(score)}
            for paragraph, score in zip(paragraphs[0], scores[0])
            if paragraph["title"] not in exclude_titles
        ]
        # Here we keep only the best k that are left.
        return results[:k]


def gold_recall_at_k(questions: list[dict], retriever: Retriever, cutoffs: list[int]) -> list[dict]:
    """Search with each raw question; per cutoff k, the share of questions with all / at least one Gold paragraph found.

    "all_found" means every Gold paragraph was found: both of HotpotQA's two, or all of a MuSiQue question's 2-4.
    """
    # Here we search once per question with the largest cutoff and keep the titles in ranked order.
    # Smaller cutoffs are just the top of the same list, so we don't need to search again.
    retrieved_titles = [
        [result["title"] for result in retriever.search(question["question"], k=max(cutoffs))]
        for question in questions
    ]
    rows = []
    for k in cutoffs:
        # Here we count, for each question, how many of its Gold paragraphs are in the top k results.
        gold_found_counts = [
            sum(gold_title in top_titles[:k] for gold_title in question["gold_titles"])
            for question, top_titles in zip(questions, retrieved_titles)
        ]
        # Here we turn those counts into two shares of the question set:
        #   all_found           every Gold paragraph was in the top k
        #   at_least_one_found  at least one Gold paragraph was in the top k
        all_found_count = sum(gold_found_count == len(question["gold_titles"])
                              for gold_found_count, question in zip(gold_found_counts, questions))
        at_least_one_found_count = sum(gold_found_count >= 1 for gold_found_count in gold_found_counts)
        rows.append({
            "k": k,
            "all_found": all_found_count / len(questions),
            "at_least_one_found": at_least_one_found_count / len(questions),
        })
    return rows


def download_and_unpack_dump(dump_url: str, dump_dir: Path) -> list[Path]:
    """Download the Wikipedia dump archive into dump_dir once, unpack it, and return its files in a fixed order."""
    dump_dir.mkdir(parents=True, exist_ok=True)
    # Here we download the archive, unless an earlier run already did.
    archive_path = dump_dir / Path(dump_url).name
    if not archive_path.exists():
        print(f"Downloading {dump_url} (1.55 GB) ...")
        urllib.request.urlretrieve(dump_url, archive_path)
    # Here we unpack the archive into dump_dir/unpacked, unless an earlier run already did.
    unpacked_dir = dump_dir / "unpacked"
    if not unpacked_dir.exists():
        print("Unpacking ...")
        with tarfile.open(archive_path) as archive:
            archive.extractall(unpacked_dir, filter="data")
    # Here we list the compressed dump files in sorted order, so the index is built the same way every time.
    return sorted(unpacked_dir.rglob("*.bz2"))


def main() -> None:
    # Here we read which command was asked for: "build" or "check".
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["build", "check"])
    command = parser.parse_args().command

    config = load_config()
    wiki_config = config["wiki"]
    index_dir = DATA_DIR / wiki_config["index_dir"]

    # "build": here we download Wikipedia, build the BM25 index and save it (run once, on Kaggle).
    if command == "build":
        dump_files = download_and_unpack_dump(wiki_config["dump_url"], DATA_DIR / wiki_config["dump_dir"])
        print(f"Reading {len(dump_files)} dump files and building the index ...")
        build_index(read_wiki_paragraphs(dump_files), index_dir, show_progress=True)
        print(f"Saved the BM25 index to {index_dir}")

    # "check": here we measure how often search finds the Gold paragraphs for the Pilot questions.
    if command == "check":
        pilot_set = load_question_set("pilot")
        recall_rows = gold_recall_at_k(pilot_set, Retriever.load(index_dir), wiki_config["recall_ks"])

        # Here we save the results as a small table (results/pilot/bm25_recall.csv) for the paper.
        recall_csv = REPO_ROOT / config["paths"]["results_dir"] / "pilot" / "bm25_recall.csv"
        recall_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(recall_csv, "w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=["k", "all_found", "at_least_one_found"])
            writer.writeheader()
            writer.writerows(recall_rows)

        # Here we print the same results as percentages.
        for row in recall_rows:
            print(f"k={row['k']:>2}  all Gold paragraphs found: {row['all_found']:.0%}   "
                  f"at least one: {row['at_least_one_found']:.0%}")
        print(f"Saved to {recall_csv}")


if __name__ == "__main__":
    main()
