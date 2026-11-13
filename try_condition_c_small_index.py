"""TEST ONLY: run Condition C on your Mac against a small stand-in search index. Never use it for reported numbers.

    Set the setting below, then press Run. Don't commit this file.

The real index (data/wiki/bm25_index/, all ~5.2M Wikipedia paragraphs) isn't on this Mac yet. This builds a small
stand-in from HotpotQA's own "distractor" dev file on Hugging Face: every question there comes with 10 paragraphs
(its 2 Gold ones plus 8 look-alikes), and pooling them for all 7,405 dev questions gives ~66,000 paragraphs. Search
is much easier than on the real index (every Gold paragraph is guaranteed to be in it), so the numbers mean nothing.
What it does test: that Condition C stops when both Gold titles are in, steers with the LLM judge's missing-information signal, and logs it.

It keeps everything apart from the real thing:
    the stand-in index:  data/test_index/bm25_index/      (data/ is git-ignored)
    the run logs:        runs/condition_c_TEST_INDEX/     (not runs/condition_c/)
Delete data/test_index/ when you're done.
"""

import urllib.request
from datetime import datetime
from pathlib import Path

import bm25s
import pyarrow.parquet as pq
from bm25s.tokenization import Tokenized

from pipeline import REPO_ROOT, load_config
from pipeline.Methods import condition_c
from pipeline.retriever import Retriever, word_ids_for_batch

# --- Setting: edit it, then press Run ---
HOW_MANY_QUESTIONS = 3       # from the start of the Pilot set; the 8B model is slow on a Mac, so start small

DISTRACTOR_DEV_URL = ("https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/"
                      "distractor/validation-00000-of-00001.parquet")
TEST_FOLDER = REPO_ROOT / load_config()["paths"]["data_dir"] / "test_index"
TEST_INDEX_FOLDER = TEST_FOLDER / "bm25_index"


def download_distractor_dev() -> Path:
    """HotpotQA's distractor dev file (~45 MB), downloaded once into data/test_index/."""
    TEST_FOLDER.mkdir(parents=True, exist_ok=True)
    parquet_path = TEST_FOLDER / "hotpot_dev_distractor.parquet"
    if not parquet_path.exists():
        print(f"Downloading {DISTRACTOR_DEV_URL} ...")
        partial_path = parquet_path.with_suffix(".part")
        urllib.request.urlretrieve(DISTRACTOR_DEV_URL, partial_path)
        partial_path.rename(parquet_path)
    return parquet_path


def pooled_paragraphs(parquet_path: Path) -> list[dict]:
    """Every question's 10 paragraphs, pooled, each title kept once, in a fixed order."""
    paragraphs = {}
    for row in pq.read_table(parquet_path, columns=["context"]).to_pylist():
        for title, sentences in zip(row["context"]["title"], row["context"]["sentences"]):
            # Here we join the sentences the same way the real index does: each one after the first has its space.
            paragraphs.setdefault(title, "".join(sentences))
    return [{"title": title, "text": paragraphs[title]} for title in sorted(paragraphs)]


def build_test_index() -> None:
    """Build the stand-in index with the real index's word rules (same stopwords, same stemmer), once."""
    if (TEST_INDEX_FOLDER / "params.index.json").exists():
        print(f"Using the stand-in index already in {TEST_INDEX_FOLDER}")
        return
    paragraphs = pooled_paragraphs(download_distractor_dev())
    print(f"Building a stand-in index over {len(paragraphs):,} paragraphs ...")
    vocabulary, word_ids_per_paragraph = {}, []
    for start in range(0, len(paragraphs), 10_000):
        batch = [f"{p['title']} {p['text']}" for p in paragraphs[start:start + 10_000]]
        word_ids_for_batch(batch, vocabulary, word_ids_per_paragraph)
    index = bm25s.BM25()
    index.index(Tokenized(ids=word_ids_per_paragraph, vocab=vocabulary), show_progress=False)
    index.save(TEST_INDEX_FOLDER, corpus=paragraphs, show_progress=False)
    print(f"Saved the stand-in index to {TEST_INDEX_FOLDER}")


def point_condition_c_at_test_index() -> None:
    """Make condition_c search the stand-in index and log to runs/condition_c_TEST_INDEX/, without editing it."""
    test_retriever = Retriever.load(TEST_INDEX_FOLDER)
    condition_c.Retriever = type("StandInRetriever", (), {"load": staticmethod(lambda: test_retriever)})

    def test_log_file(start_time: datetime) -> Path:
        date_folder = (REPO_ROOT / load_config()["paths"]["runs_dir"] / "condition_c_TEST_INDEX"
                       / start_time.strftime("%Y-%m-%d"))
        date_folder.mkdir(parents=True, exist_ok=True)
        log_file = date_folder / f"{start_time:%H%M%S}.txt"
        log_file.touch(exist_ok=False)
        return log_file

    real_header = condition_c.log_header

    def test_header(log_file: Path, start_time: datetime, question_count: int) -> None:
        condition_c.write_to_log(log_file, "TEST ONLY: stand-in index from HotpotQA's distractor paragraphs. "
                                           "Not the real Wikipedia index; don't report these numbers.\n")
        real_header(log_file, start_time, question_count)

    condition_c.new_log_file = test_log_file
    condition_c.log_header = test_header


if __name__ == "__main__":
    build_test_index()
    point_condition_c_at_test_index()
    condition_c.HOW_MANY_QUESTIONS = HOW_MANY_QUESTIONS
    finished_log_file = condition_c.run_condition_c()
    print(f"\nLog: {finished_log_file}")