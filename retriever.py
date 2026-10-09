"""Search HotpotQA's 2017 Wikipedia with BM25: give a query, get back the best-matching paragraphs. (T03)

    python retriever.py        (or press Run / Debug on this file)

Other scripts use two functions from here: load_index() opens the saved index, and
search(index, query, k, exclude_titles) returns the k best paragraphs as {title, text, score}.

Running this file is the demo. It searches with the first Pilot question and prints the top 5 paragraphs. Then
it measures search quality on the whole Pilot set: for the top 2, 5, 10 and 20 results, the share of questions
where all Gold paragraphs were found, and where at least one was. That table is saved to
results/pilot/bm25_recall.csv, and each run is logged in runs/retriever/<date>/.
"""

import csv
import io
import os
import time
from pathlib import Path

from question_sets import load_question_set
from setup_project import PROJECT_ROOT, note, warn, write_run_log

# Settings for this script.
INDEX_DIR = PROJECT_ROOT / "data" / "wiki" / "bm25_index"
RECALL_CUTOFFS = [2, 5, 10, 20]   # how many top results the search-quality check looks at
RECALL_FILE = PROJECT_ROOT / "results" / "pilot" / "bm25_recall.csv"
DEMO_RESULT_COUNT = 5             # how many paragraphs the demo search prints
LOGS_DIR = PROJECT_ROOT / "runs" / "retriever"

# bm25s uses JAX if it is installed, and Kaggle has it preinstalled. By default JAX grabs 75% of the GPU's memory,
# which leaves too little for the language model. This keeps JAX on the CPU. It must be set before bm25s is imported.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

# These two imports sit below the setting on purpose, so JAX sees it.
import bm25s
import Stemmer

# The stemmer cuts words down to their root, so "directed" and "directing" both become "direct".
# That way a query still matches a paragraph when the word endings differ.
ENGLISH_STEMMER = Stemmer.Stemmer("english")


# --- Searching ---

def split_into_search_words(texts: list[str], as_word_numbers: bool = False):
    """Turn each text into its search words: lowercase, without common words like "the", each cut to its root.

    The index (built in build_index.py) and every query must go through this same function, or they won't match.
    as_word_numbers=True gives numbers instead of words, which saves memory when indexing millions of paragraphs.
    """
    return bm25s.tokenize(texts, stopwords="en", stemmer=ENGLISH_STEMMER,
                          return_ids=as_word_numbers, show_progress=False)


def load_index(index_dir: Path = INDEX_DIR) -> bm25s.BM25:
    """Open the saved index. Most of it stays on disk and only the parts a search needs are read into memory."""
    return bm25s.BM25.load(index_dir, mmap=True, load_corpus=True)


def count_paragraphs(index: bm25s.BM25) -> int:
    """How many paragraphs the index holds."""
    return index.scores["num_docs"]


def search(index: bm25s.BM25, query: str, k: int, exclude_titles: set[str] = frozenset()) -> list[dict]:
    """Return the k paragraphs that best match the query, best first, each as {title, text, score}.

    Paragraphs whose title is in exclude_titles are skipped, so a later Round only brings new Evidence.
    """
    query_words = split_into_search_words([query])

    # If no search words are left (an empty query, or only words like "the"), we return nothing.
    # Otherwise bm25s would return random paragraphs, all scored 0, and they would pollute the Evidence.
    if len(query_words[0]) == 0:
        return []

    # Some results may be skipped, so we ask for extra to still end up with k. We never ask for more
    # paragraphs than the index holds, because bm25s fails if we do.
    results_to_fetch = min(k + len(exclude_titles), count_paragraphs(index))

    # bm25s can search many queries at once, so it takes a list and gives back one row per query.
    # We pass one query, so we read row 0.
    paragraphs, scores = index.retrieve(query_words, k=results_to_fetch, show_progress=False)
    best_paragraphs = paragraphs[0]
    best_scores = scores[0]

    search_results = []
    for paragraph, score in zip(best_paragraphs, best_scores):
        if paragraph["title"] in exclude_titles:
            continue   # already in the Evidence, so skip it
        search_results.append({
            "title": paragraph["title"],
            "text": paragraph["text"],
            "score": float(score),
        })
    return search_results[:k]


# --- Search quality on the Pilot set ---

def count_gold_titles_found(gold_titles: list[str], found_titles: list[str]) -> int:
    """How many of a question's Gold paragraphs are among the titles the search found."""
    found_count = 0
    for gold_title in gold_titles:
        if gold_title in found_titles:
            found_count += 1
    return found_count


def measure_gold_recall(index: bm25s.BM25, questions: list[dict], cutoffs: list[int]) -> list[dict]:
    """For each cutoff k: the share of questions with all, and with at least one, Gold paragraph in the top k."""
    # We search once per question with the largest cutoff. The top k for a smaller cutoff is the start of that list.
    largest_cutoff = max(cutoffs)
    found_titles_per_question = []
    for question in questions:
        search_results = search(index, question["question"], k=largest_cutoff)
        found_titles_per_question.append([result["title"] for result in search_results])

    recall_rows = []
    for k in cutoffs:
        questions_with_all_found = 0
        questions_with_at_least_one_found = 0

        for question, found_titles in zip(questions, found_titles_per_question):
            gold_found_count = count_gold_titles_found(question["gold_titles"], found_titles[:k])
            if gold_found_count == len(question["gold_titles"]):
                questions_with_all_found += 1
            if gold_found_count >= 1:
                questions_with_at_least_one_found += 1

        recall_rows.append({
            "k": k,
            "all_found": questions_with_all_found / len(questions),
            "at_least_one_found": questions_with_at_least_one_found / len(questions),
        })
    return recall_rows


def recall_rows_as_csv_text(recall_rows: list[dict]) -> str:
    """The recall table as the text of a CSV file: a header line, then one line per cutoff."""
    csv_text = io.StringIO()
    csv_writer = csv.DictWriter(csv_text, fieldnames=["k", "all_found", "at_least_one_found"])
    csv_writer.writeheader()
    csv_writer.writerows(recall_rows)
    return csv_text.getvalue()


def recall_file_would_change(recall_rows: list[dict], recall_file: Path) -> bool:
    """True if a recall file already exists and holds different numbers, meaning the search now behaves differently."""
    if not recall_file.exists():
        return False
    # newline="" reads the file's line endings exactly as they are, so the comparison is fair.
    with open(recall_file, newline="") as saved_file:
        saved_text = saved_file.read()
    return saved_text != recall_rows_as_csv_text(recall_rows)


def write_recall_file(recall_rows: list[dict], recall_file: Path) -> None:
    """Save the recall table as a CSV file, for the paper."""
    recall_file.parent.mkdir(parents=True, exist_ok=True)
    # newline="" writes the text exactly as the csv module made it.
    with open(recall_file, "w", newline="") as output_file:
        output_file.write(recall_rows_as_csv_text(recall_rows))


def describe_recall_row(recall_row: dict) -> str:
    """One readable line of the recall table, as percentages."""
    all_found = f"{recall_row['all_found']:.0%}"
    at_least_one_found = f"{recall_row['at_least_one_found']:.0%}"
    return f"top {recall_row['k']:>2}: all Gold paragraphs found {all_found:>4}, at least one {at_least_one_found:>4}"


# --- The demo ---

def show_demo_search(index: bm25s.BM25, question: dict, log_lines: list[str]) -> None:
    """Search with one question and log the top paragraphs, marking the ones that are Gold paragraphs."""
    note(log_lines, f"Demo search: {question['question']}")
    note(log_lines, f"  Gold titles: {question['gold_titles']}")

    search_results = search(index, question["question"], k=DEMO_RESULT_COUNT)
    for rank, result in enumerate(search_results, start=1):
        gold_marker = "GOLD" if result["title"] in question["gold_titles"] else "    "
        note(log_lines, f"  {rank}. {gold_marker} [{result['score']:.1f}] {result['title']}: {result['text'][:70]}...")


def check_search_quality(log_lines: list[str]) -> None:
    """Load the index, show one demo search, then measure and save search quality on the Pilot set."""
    started_at = time.time()
    if not (INDEX_DIR / "params.index.json").exists():
        raise SystemExit(f"No search index in {INDEX_DIR}. Run `python3 setup_project.py`: it explains how to get it.")

    index = load_index(INDEX_DIR)
    note(log_lines, f"Index: {count_paragraphs(index):,} paragraphs from {INDEX_DIR.relative_to(PROJECT_ROOT)}")

    pilot_set = load_question_set("pilot")
    show_demo_search(index, pilot_set[0], log_lines)

    note(log_lines, f"Search quality on the {len(pilot_set)} Pilot questions (searching with the question itself):")
    recall_rows = measure_gold_recall(index, pilot_set, RECALL_CUTOFFS)
    for recall_row in recall_rows:
        note(log_lines, "  " + describe_recall_row(recall_row))

    # Here we compare with the saved table before overwriting it, so a change in search behaviour can't go unnoticed.
    if recall_file_would_change(recall_rows, RECALL_FILE):
        warn(log_lines, f"{RECALL_FILE.name} changed: the search now finds different paragraphs than before.")
    write_recall_file(recall_rows, RECALL_FILE)
    note(log_lines, f"Saved to {RECALL_FILE.relative_to(PROJECT_ROOT)}")

    elapsed_seconds = time.time() - started_at
    note(log_lines, f"Done in {elapsed_seconds:.1f} s")


if __name__ == "__main__":
    retriever_log_lines = []
    try:
        check_search_quality(retriever_log_lines)
    except BaseException as search_error:
        # Here we record a failed run in the log too, then let the error show as usual.
        warn(retriever_log_lines, f"the search check failed: {search_error!r}")
        raise
    finally:
        print("Log saved in", write_run_log(retriever_log_lines, LOGS_DIR, "retriever"))
