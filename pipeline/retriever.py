"""BM25 search over HotpotQA's 2017 Wikipedia intro paragraphs. (The index is built once by setup/build_index.py, T03.)

    Press Debug on this file to search with the first Pilot question and print the top paragraphs.
"""

import os
from collections.abc import Collection

# Here we keep JAX (preinstalled on Kaggle and picked up by bm25s) off the GPU. It must be set before bm25s is
# imported: otherwise JAX reserves 75% of the GPU's memory and the model (vLLM) no longer fits.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import bm25s     # noqa: E402  (imported after the JAX line above on purpose)
import Stemmer   # noqa: E402

from pipeline.config import INDEX_DIR   # noqa: E402

# Here we set up the stemmer, which cuts words down to their root ("directed", "directing" -> "direct")
# so a query matches a paragraph even when the word endings differ.
ENGLISH_STEMMER = Stemmer.Stemmer("english")


def tokenize(texts: list[str], return_word_ids: bool = False, show_progress: bool = False):
    """Turn texts into search words: lowercase, drop common words ("the", "of"), cut words to their root.

    Paragraphs (when the index is built) and queries (when searching) go through this same rule.
    return_word_ids=True gives numbers instead of words, which saves memory for millions of paragraphs.
    """
    return bm25s.tokenize(texts, stopwords="en", stemmer=ENGLISH_STEMMER,
                          return_ids=return_word_ids, show_progress=show_progress)


class Retriever:
    """A loaded BM25 index that finds the paragraphs best matching a search query."""

    def __init__(self, bm25_index: bm25s.BM25):
        """Keep the loaded index (use Retriever.load to open one from disk)."""
        self.bm25_index = bm25_index

    @classmethod
    def load(cls, index_dir=INDEX_DIR) -> "Retriever":
        """Open the saved index. It stays on disk and only the parts a search needs are read (~1-3 GB RAM)."""
        return cls(bm25s.BM25.load(index_dir, mmap=True, load_corpus=True))

    def paragraph_count(self) -> int:
        """How many paragraphs the index holds."""
        return self.bm25_index.scores["num_docs"]

    def search(self, query: str, k: int, exclude_titles: Collection[str] = frozenset()) -> list[dict]:
        """Return the k best paragraphs for the query as {title, text, score}, skipping titles already in the Evidence."""
        query_words = tokenize([query])
        # Here we stop if no search words are left (an empty query, or only words like "the"): BM25 would
        # otherwise return arbitrary paragraphs, all scored 0, and they would pollute the Evidence.
        if not query_words[0]:
            return []

        # Here we ask for extra results to make up for the ones we will skip, but never more than the index holds.
        results_to_fetch = min(k + len(exclude_titles), self.paragraph_count())
        # bm25s searches a list of queries at once; we pass one, so we read back row 0.
        paragraphs, scores = self.bm25_index.retrieve(query_words, k=results_to_fetch, show_progress=False)
        ranked_results = [{"title": paragraph["title"], "text": paragraph["text"], "score": float(score)}
                          for paragraph, score in zip(paragraphs[0], scores[0])]
        return drop_titles_already_held(ranked_results, exclude_titles)[:k]


def drop_titles_already_held(ranked_results: list[dict], exclude_titles: Collection[str]) -> list[dict]:
    """Remove results whose title is already in the Evidence, so every search brings something new."""
    return [result for result in ranked_results if result["title"] not in exclude_titles]


if __name__ == "__main__":
    from pipeline.dataset import load_question_set

    first_question = load_question_set("pilot")[0]
    retriever = Retriever.load()
    print("Question:   ", first_question["question"])
    print("Gold titles:", first_question["gold_titles"])
    for rank, result in enumerate(retriever.search(first_question["question"], k=5), start=1):
        print(f"  {rank}. [{result['score']:.1f}] {result['title']}: {result['text'][:80]}...")
