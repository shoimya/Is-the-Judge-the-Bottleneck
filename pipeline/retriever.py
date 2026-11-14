"""BM25 search over HotpotQA's 2017 Wikipedia intro paragraphs, plus the code that builds the search index.

    Open this file, set the settings below, and press Run (or Debug).

By default it searches with one Pilot question and prints the top results, marking the Gold paragraphs.
Three settings add slower jobs before that:
    BUILD_INDEX = True               downloads Wikipedia (1.55 GB) into data/wiki/ and builds the index into
                                     data/wiki/bm25_index/, if that folder is missing (delete it to build a fresh one)
    WRITE_RECALL_CSV = True          rewrites results/pilot/bm25_recall.csv: per Pilot question, whether each
                                     Gold paragraph made the top 2 / 5 / 10 / 20
    WRITE_RECALL_METRIC_CSV = True   rewrites results/pilot/bm25_recall_metrics.csv: per cutoff, the share of
                                     Pilot questions with all, and with at least one, Gold paragraph found

Other code uses it like this:
    retriever = Retriever.load()
    results = retriever.search("Who wrote ...?", k=5, exclude_titles={"A title already in the Evidence"})
"""

import bz2
import csv
import http.client
import json
import os
import shutil
import tarfile
import time
import urllib.error
import urllib.request
from collections.abc import Collection, Iterable, Iterator
from pathlib import Path

# Here we keep JAX (preinstalled on Kaggle and picked up by bm25s) off the GPU. It must be set before bm25s is
# imported: otherwise JAX reserves 75% of the GPU's memory and the model (vLLM) no longer fits.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import bm25s                                  # noqa: E402  (imported after the JAX line above on purpose)
import numpy as np                            # noqa: E402
import Stemmer                                # noqa: E402
from bm25s.tokenization import Tokenized      # noqa: E402

from pipeline import REPO_ROOT, llm2, load_config, pipeline_number, rewriter   # noqa: E402
from pipeline.dataset import load_questions   # noqa: E402

# --- Settings for the demo at the bottom: edit them, then press Run ---
QUESTION_NUMBER = 50        # which Pilot question to search with, counting from 1
RESULTS_TO_SHOW = 15        # k: how many paragraphs to fetch and print
BUILD_INDEX = False         # True = build the index first, if data/wiki/bm25_index/ is missing (slow, see above)
WRITE_RECALL_CSV = True    # True = rewrite results/pilot/bm25_recall.csv first
WRITE_RECALL_METRIC_CSV = True   # True = rewrite results/pilot/bm25_recall_metrics.csv first, and print it

# --- Fixed settings ---
# Stanford's server refuses connections from the Mac; Kaggle reached it when the index was built (Oct 4).
# So build the index on Kaggle (see CONTEXT.md, "Wikipedia").
WIKI_DUMP_URL =("https://nlp.stanford.edu/projects/hotpotqa/"
                 "enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2")
DOWNLOAD_ATTEMPTS = 20               # how many times to retry a dropped download before giving up
SECONDS_BETWEEN_ATTEMPTS = 10
ONE_MEGABYTE = 1024 * 1024
PARAGRAPHS_PER_BATCH = 100_000       # how many paragraphs are cut into words at once while building
RECALL_CUTOFFS = [2, 5, 10, 20]      # the top-k sizes the recall check looks at
QUESTION_COLUMNS = ["id", "question", "gold_1", "gold_2"]   # the recall CSV's first columns

# Here we set up the stemmer, which cuts words down to their root ("directed", "directing" -> "direct")
# so a query matches a paragraph even when the word endings differ.
ENGLISH_STEMMER = Stemmer.Stemmer("english")


# ---------------------------------------------------------------------------
# Where things live
# ---------------------------------------------------------------------------

def wiki_dir() -> Path:
    """Where the Wikipedia download and its unpacked files go: data/wiki/ (only needed to build the index)."""
    return REPO_ROOT / load_config()["paths"]["data_dir"] / "wiki"


def index_dir() -> Path:
    """Where the saved BM25 index lives: data/wiki/bm25_index/."""
    return REPO_ROOT / load_config()["paths"]["data_dir"] / "wiki" / "bm25_index"


# ---------------------------------------------------------------------------
# Download Wikipedia, resuming after interruptions
# ---------------------------------------------------------------------------

def download_remaining_bytes(url: str, partial_path: Path) -> int | None:
    """One download attempt: fetch the bytes partial_path doesn't have yet and add them to it.

    Returns the full file size the server promised (None if it didn't say).
    """
    bytes_already_downloaded = partial_path.stat().st_size if partial_path.exists() else 0
    # Here we ask the server for only the missing bytes ("Range: bytes=<first missing byte>-").
    range_request = urllib.request.Request(url, headers={"Range": f"bytes={bytes_already_downloaded}-"})
    try:
        with urllib.request.urlopen(range_request, timeout=60) as response:
            # Status 206 = the server sent only the rest, so we add it on. Status 200 = it sent the whole file,
            # so we start the .part file over.
            resuming = response.status == 206
            size_being_sent = response.headers.get("Content-Length")
            with open(partial_path, "ab" if resuming else "wb") as partial_file:
                shutil.copyfileobj(response, partial_file, length=ONE_MEGABYTE)
    except urllib.error.HTTPError as http_error:
        # Status 416 = "nothing left to send": the .part file already holds every byte.
        if http_error.code == 416:
            return bytes_already_downloaded
        raise
    if size_being_sent is None:
        return None
    return int(size_being_sent) + (bytes_already_downloaded if resuming else 0)


def download_with_resume(url: str, destination: Path) -> None:
    """Download url to destination, retrying and resuming until the file is complete.

    The bytes go into "<destination>.part" and the file only gets its real name once complete,
    so a file named destination is always a whole download.
    """
    partial_path = destination.with_name(destination.name + ".part")
    for attempt_number in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
            expected_size = download_remaining_bytes(url, partial_path)
        except (OSError, http.client.HTTPException) as download_error:
            # Here we wait and try again after a dropped connection; the next attempt resumes the .part file.
            print(f"Download interrupted ({download_error}); attempt {attempt_number} of {DOWNLOAD_ATTEMPTS}.")
            time.sleep(SECONDS_BETWEEN_ATTEMPTS)
            continue

        downloaded_size = partial_path.stat().st_size
        # Here we accept the file only if it has every byte the server promised.
        if expected_size is None or downloaded_size == expected_size:
            partial_path.rename(destination)
            return
        print(f"Download incomplete ({downloaded_size:,} of {expected_size:,} bytes); "
              f"attempt {attempt_number} of {DOWNLOAD_ATTEMPTS}.")
    # Here we say what went wrong: nothing downloaded at all means the server can't be reached from here.
    if not partial_path.exists():
        raise RuntimeError(f"Could not reach {url} in {DOWNLOAD_ATTEMPTS} attempts: the server may be down, "
                           f"or refusing this network. Try again later, or build the index on Kaggle.")
    raise RuntimeError(f"Could not finish downloading {url}. Run this file again to keep resuming {partial_path}.")


# ---------------------------------------------------------------------------
# Unpack and read Wikipedia
# ---------------------------------------------------------------------------

def unpack_archive(archive_path: Path, unpacked_dir: Path) -> None:
    """Unpack the archive into unpacked_dir, all or nothing.

    Files go into "<unpacked_dir>.part" first, which is renamed only when unpacking finishes,
    so a folder named unpacked_dir always holds every file.
    """
    unpacking_dir = unpacked_dir.with_name(unpacked_dir.name + ".part")
    shutil.rmtree(unpacking_dir, ignore_errors=True)   # here we throw away a half-finished earlier unpack
    with tarfile.open(archive_path) as archive:
        archive.extractall(unpacking_dir, filter="data")
    unpacking_dir.rename(unpacked_dir)


def download_and_unpack_wikipedia(dump_dir: Path) -> list[Path]:
    """Make sure Wikipedia is downloaded and unpacked in dump_dir, skipping finished steps.

    Returns the unpacked files, in order.
    """
    dump_dir.mkdir(parents=True, exist_ok=True)
    archive_path = dump_dir / Path(WIKI_DUMP_URL).name
    if not archive_path.exists():
        print(f"Downloading {WIKI_DUMP_URL} (1.55 GB) ...")
        download_with_resume(WIKI_DUMP_URL, archive_path)

    unpacked_dir = dump_dir / "unpacked"
    if not unpacked_dir.exists():
        print("Unpacking (this takes a few minutes) ...")
        unpack_archive(archive_path, unpacked_dir)
    # Here we sort the file list, so the index is built in the same order every time.
    return sorted(unpacked_dir.rglob("wiki_*.bz2"))


def read_wiki_paragraphs(dump_files: Iterable[Path]) -> Iterator[dict]:
    """Read the dump files (bz2, one JSON article per line) and hand out one {title, text} paragraph at a time."""
    for dump_file in dump_files:
        with bz2.open(dump_file, "rt", encoding="utf-8") as dump_lines:
            for line in dump_lines:
                article = json.loads(line)
                # Here we join the article's sentences back into one paragraph; each sentence after the first
                # already starts with its own space.
                yield {"title": article["title"], "text": "".join(article["text"])}


# ---------------------------------------------------------------------------
# Cut text into search words, and build the index
# ---------------------------------------------------------------------------

def tokenize(texts: list[str]) -> list[list[str]]:
    """Turn each text into its search words: lowercase, drop common words ("the", "of"), cut words to their root.

    Paragraphs (when the index is built) and queries (when searching) go through this same rule.
    """
    return bm25s.tokenize(texts, stopwords="en", stemmer=ENGLISH_STEMMER, return_ids=False, show_progress=False)


def word_ids_for_batch(batch_texts: list[str], vocabulary: dict, word_ids_per_paragraph: list) -> None:
    """Cut a batch of paragraphs into words, and add each paragraph's words as numbers to word_ids_per_paragraph.

    vocabulary gives every word its number and grows as new words appear. Each number is stored as a 4-byte
    uint32 instead of a ~36-byte Python number: that is what keeps the build at ~5 GB of RAM.
    """
    word_lists = tokenize(batch_texts)
    for words in word_lists:
        # Here a new word gets the next free number; a word seen before keeps the number it already has.
        word_ids = [vocabulary.setdefault(word, len(vocabulary)) for word in words]
        word_ids_per_paragraph.append(np.array(word_ids, dtype=np.uint32))


def build_index(dump_files: list[Path], index_folder: Path) -> int:
    """Build the BM25 index over every paragraph's title + text and save it to index_folder.

    Returns the number of paragraphs indexed.

    It reads Wikipedia twice: first to cut every paragraph into words and index them, then to save the paragraphs
    next to the index (search results show their text). It saves into "<index_folder>.part" and renames it only
    at the end, so a folder named index_folder always holds a whole index.
    """
    vocabulary = {}
    word_ids_per_paragraph = []
    batch_texts = []
    print("Pass 1 of 2: cutting every paragraph into words ...")
    for paragraph in read_wiki_paragraphs(dump_files):
        # Here we index title and text together, so a search can match words in either.
        batch_texts.append(f"{paragraph['title']} {paragraph['text']}")
        if len(batch_texts) == PARAGRAPHS_PER_BATCH:
            word_ids_for_batch(batch_texts, vocabulary, word_ids_per_paragraph)
            print(f"  {len(word_ids_per_paragraph):,} paragraphs, {len(vocabulary):,} different words")
            batch_texts = []
    if batch_texts:
        word_ids_for_batch(batch_texts, vocabulary, word_ids_per_paragraph)
    paragraph_total = len(word_ids_per_paragraph)

    print(f"Indexing {paragraph_total:,} paragraphs ...")
    bm25_index = bm25s.BM25()
    bm25_index.index(Tokenized(ids=word_ids_per_paragraph, vocab=vocabulary), show_progress=True)
    # Here we free the word numbers before saving: the index already holds what it needs from them.
    del word_ids_per_paragraph

    print("Pass 2 of 2: saving the index and the paragraphs ...")
    saving_folder = index_folder.with_name(index_folder.name + ".part")
    shutil.rmtree(saving_folder, ignore_errors=True)   # here we throw away a half-saved earlier build
    bm25_index.save(saving_folder, corpus=read_wiki_paragraphs(dump_files))
    saving_folder.rename(index_folder)
    return paragraph_total


def build_index_if_missing() -> None:
    """Download Wikipedia and build the index, unless data/wiki/bm25_index/ already exists (delete it to rebuild)."""
    index_folder = index_dir()
    if index_folder.exists():
        print(f"The index is already in {index_folder}; delete that folder to build a fresh one.")
        return
    start_time = time.time()
    dump_files = download_and_unpack_wikipedia(wiki_dir())
    print(f"Building the index from {len(dump_files):,} Wikipedia files (this takes a long time) ...")
    paragraph_total = build_index(dump_files, index_folder)
    minutes_taken = (time.time() - start_time) / 60
    print(f"Saved {paragraph_total:,} paragraphs to {index_folder} in {minutes_taken:.0f} minutes.\n")


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

class Retriever:
    """A loaded BM25 index that finds the paragraphs best matching a search query."""

    def __init__(self, bm25_index: bm25s.BM25):
        """Keep the loaded index (use Retriever.load to open one from disk)."""
        self.bm25_index = bm25_index

    @classmethod
    def load(cls, index_folder: Path | None = None) -> "Retriever":
        """Open the saved index (data/wiki/bm25_index/ unless told otherwise).

        It stays on disk, and only the parts a search needs are read (~1-3 GB RAM).
        """
        if index_folder is None:
            index_folder = index_dir()
        bm25_index = bm25s.BM25.load(index_folder, mmap=True, load_corpus=True, show_progress=False)
        return cls(bm25_index)

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


def steer_says_nothing_is_missing(steer: str | None) -> bool:
    """True when there is no steer, or the Judge wrote "nothing" (every Gold paragraph is already there)."""
    return steer is None or steer.strip().rstrip(".").casefold() in ("", "nothing")


def choose_query_for_round(question: str, previous_steer: str | None, past_queries: list[str]) -> dict:
    """Choose the search query for one Round, by config.yaml's loop.pipeline.

    Round 1 (no steer yet) searches with the question in both pipelines. After that:
        Pipeline 1: the Rewriter turns the question, the queries already tried and the steer into the query.
        Pipeline 2: the steer is the query, as written. A steer of "nothing" searches with the question again; the
                    search skips titles already held, so it still brings new paragraphs.

    Returns a dictionary with:
        query      the text sent to BM25
        rewrite    the Rewriter call's record (Pipeline 1, Rounds 2+), or None when the Rewriter was not called
    """
    if previous_steer is None:
        return {"query": question, "rewrite": None}

    if pipeline_number() == 1:
        rewrite_result = rewriter.rewrite_query(question, past_queries, previous_steer)
        # Here a Rewriter reply that never fitted its shape gives no query, so we search with the question again.
        return {"query": rewrite_result["query"] or question, "rewrite": rewrite_result}

    if steer_says_nothing_is_missing(previous_steer):
        return {"query": question, "rewrite": None}
    return {"query": previous_steer, "rewrite": None}


def drop_titles_already_held(ranked_results: list[dict], exclude_titles: Collection[str]) -> list[dict]:
    """Remove results whose title is already in the Evidence, so every search brings something new."""
    return [result for result in ranked_results if result["title"] not in exclude_titles]


# ---------------------------------------------------------------------------
# Recall check: how often a plain search of each Pilot question finds its Gold paragraphs
# ---------------------------------------------------------------------------

def ranked_titles_for_questions(retriever: Retriever, questions: list[dict]) -> list[list[str]]:
    """Search once with each question as written, and keep the titles of its top 20 results, best first.

    Both recall checks read from this, so the questions are searched only once.
    """
    ranked_titles_per_question = []
    for question in questions:
        results = retriever.search(question["question"], k=max(RECALL_CUTOFFS))
        ranked_titles_per_question.append([result["title"] for result in results])
    return ranked_titles_per_question


def recall_csv_path() -> Path:
    """Where the recall check is saved: results/pilot/bm25_recall.csv."""
    return REPO_ROOT / load_config()["paths"]["results_dir"] / "pilot" / "bm25_recall.csv"


def hit_column_names() -> list[str]:
    """The 0/1 columns: gold1@2, gold1@5, ... for the first Gold paragraph, then the same for the second."""
    column_names = []
    for gold_number in (1, 2):
        column_names += [f"gold{gold_number}@{cutoff}" for cutoff in RECALL_CUTOFFS]
    return column_names


def gold_hits(gold_titles: list[str], ranked_titles: list[str]) -> list[int]:
    """For each Gold title, then each cutoff: 1 if that title is among the top-cutoff results, else 0."""
    hits = []
    for gold_title in gold_titles:
        for cutoff in RECALL_CUTOFFS:
            hits.append(1 if gold_title in ranked_titles[:cutoff] else 0)
    return hits


def write_pilot_recall_csv(pilot_questions: list[dict], ranked_titles_per_question: list[list[str]]) -> dict:
    """Save one CSV row per Pilot question with its 0/1 Gold hits (from ranked_titles_for_questions).

    Returns the recall of each hit column: the share of Pilot questions where that Gold paragraph made that cutoff.
    """
    hit_columns = hit_column_names()
    hit_totals = dict.fromkeys(hit_columns, 0)
    csv_path = recall_csv_path()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(QUESTION_COLUMNS + hit_columns)
        for question, ranked_titles in zip(pilot_questions, ranked_titles_per_question):
            hits = gold_hits(question["gold_titles"], ranked_titles)
            csv_writer.writerow([question["id"], question["question"], *question["gold_titles"], *hits])
            # Here we add this question's hits to the running totals, column by column.
            for column_name, hit in zip(hit_columns, hits):
                hit_totals[column_name] += hit
    return {column_name: hit_total / len(pilot_questions) for column_name, hit_total in hit_totals.items()}


def print_recall(recalls: dict) -> None:
    """Show, for each cutoff, how often each Gold paragraph was found."""
    print(f"Wrote {recall_csv_path()}")
    for cutoff in RECALL_CUTOFFS:
        print(f"  top {cutoff:>2}:  Gold 1 found {recalls[f'gold1@{cutoff}']:.0%},"
              f"  Gold 2 found {recalls[f'gold2@{cutoff}']:.0%}")
    print()


def recall_metric_csv_path() -> Path:
    """Where the recall summary is saved: results/pilot/bm25_recall_metrics.csv."""
    return REPO_ROOT / load_config()["paths"]["results_dir"] / "pilot" / "bm25_recall_metrics.csv"


def count_gold_found(gold_titles: list[str], top_titles: list[str]) -> int:
    """How many of a question's Gold paragraphs are among the given top titles."""
    return sum(gold_title in top_titles for gold_title in gold_titles)


def write_pilot_recall_metric_csv(pilot_questions: list[dict],
                                  ranked_titles_per_question: list[list[str]]) -> list[dict]:
    """For each cutoff k, save the share of Pilot questions with ALL, and with AT LEAST ONE, Gold paragraph in the top k
    (from ranked_titles_for_questions).

    One row per k (k, all_found, at_least_one_found), shares rounded to 3 decimals. Returns the same rows.
    """
    metric_rows = []
    # Here a smaller cutoff is just the top of the same ranked list, so no extra searching is needed.
    for cutoff in RECALL_CUTOFFS:
        all_found_count = 0
        at_least_one_found_count = 0
        for question, ranked_titles in zip(pilot_questions, ranked_titles_per_question):
            gold_found_count = count_gold_found(question["gold_titles"], ranked_titles[:cutoff])
            if gold_found_count == len(question["gold_titles"]):
                all_found_count += 1
            if gold_found_count >= 1:
                at_least_one_found_count += 1
        metric_rows.append({"k": cutoff,
                            "all_found": f"{all_found_count / len(pilot_questions):.3f}",
                            "at_least_one_found": f"{at_least_one_found_count / len(pilot_questions):.3f}"})

    csv_path = recall_metric_csv_path()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=["k", "all_found", "at_least_one_found"])
        csv_writer.writeheader()
        csv_writer.writerows(metric_rows)
    return metric_rows


def print_recall_metrics(metric_rows: list[dict]) -> None:
    """Show the recall summary in the same layout as the CSV."""
    print(f"Wrote {recall_metric_csv_path()}")
    print("k,all_found,at_least_one_found")
    for row in metric_rows:
        print(f"{row['k']},{row['all_found']},{row['at_least_one_found']}")
    print()


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def print_search_results(question: dict, results: list[dict]) -> None:
    """Show a question, its Gold titles, then each result (rank, score, title, start of its text), marking Gold ones."""
    gold_titles = question["gold_titles"]
    print(f"Question:    {question['question']}")
    print(f"Gold titles: {', '.join(gold_titles)}")
    print(f"\nTop {len(results)} results:")
    for rank, result in enumerate(results, start=1):
        gold_mark = "   <-- Gold" if result["title"] in gold_titles else ""
        print(f"{rank:>3}. [{result['score']:5.1f}] {result['title']}{gold_mark}")
        print(f"       {result['text'][:100]}...")
    found_titles = {result["title"] for result in results} & set(gold_titles)
    print(f"\nGold found: {len(found_titles)} of {len(gold_titles)}")


if __name__ == "__main__":
    # Currently facing an error.
    # if BUILD_INDEX:
    #     build_index_if_missing()

    retriever = Retriever.load()
    # if WRITE_RECALL_CSV or WRITE_RECALL_METRIC_CSV:
    #     # Here we search every Pilot question once; both recall checks read from the same results.
    #     pilot_questions = load_questions("pilot")
    #     ranked_titles_per_question = ranked_titles_for_questions(retriever, pilot_questions)
    # if WRITE_RECALL_CSV:
    #     print_recall(write_pilot_recall_csv(pilot_questions, ranked_titles_per_question))
    # if WRITE_RECALL_METRIC_CSV:
    #     print_recall_metrics(write_pilot_recall_metric_csv(pilot_questions, ranked_titles_per_question))

    # # Here QUESTION_NUMBER counts from 1, so question 1 is the first line of pilot.jsonl.
    # question = load_questions("pilot")[QUESTION_NUMBER - 1]
    # results = retriever.search(question["question"], k=RESULTS_TO_SHOW)
    # print_search_results(question, results)

    # Here we search with a made-up question. It has no Gold paragraphs, so we list the Wikipedia pages we expect
    # it to need instead; the printout marks them the same way (use [] to mark nothing).
    made_up_question = {
        "question": "Which river flows through the capital city of Hungary?",
        "gold_titles": ["Budapest", "Danube"],
    }
    made_up_results = retriever.search(made_up_question["question"], k=5)
    print("\n" + "-" * 80 + "\n")
    print_search_results(made_up_question, made_up_results)

    # Here we show which query Round 2 would search with under the pipeline config.yaml asks for. Under Pipeline 2
    # this needs no model; under Pipeline 1 it calls the Rewriter, so we start the server first.
    made_up_steer = "the name of the river that flows through Budapest"
    made_up_past_queries = [made_up_question["question"]]
    server_started_here = llm2.start_ollama_server() if pipeline_number() == 1 else False
    try:
        round_2_choice = choose_query_for_round(made_up_question["question"], made_up_steer, made_up_past_queries)
        print(f"Pipeline {pipeline_number()}: steer \"{made_up_steer}\"  ->  "
              f"Round 2 query \"{round_2_choice['query']}\"")
    finally:
        # Here we stop the server only if this demo started it.
        if server_started_here:
            llm2.stop_ollama_server()
