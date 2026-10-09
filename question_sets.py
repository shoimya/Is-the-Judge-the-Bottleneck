"""Make the Pilot set and the Test set from HotpotQA's dev questions. (T02)

    python question_sets.py        (or press Run / Debug on this file)

It downloads HotpotQA dev (fullwiki setting), picks 100 Pilot questions and a separate 1,000 Test questions with a
fixed seed, and writes:
  - data/hotpotqa/pilot.jsonl and test.jsonl: the questions, one per line (git ignores data/)
  - results/question_ids/hotpotqa_pilot.txt and hotpotqa_test.txt: only their ids, committed so everyone uses the same sets
The fixed seed means every machine picks exactly the same questions. Each run is logged in runs/question_sets/<date>/.
Other scripts read a saved set with load_question_set("pilot") or load_question_set("test").
"""

import json
import os
import random
import time
from collections import Counter
from pathlib import Path

from setup_project import PROJECT_ROOT, note, warn, write_run_log

# Settings for this script.
PILOT_SIZE = 100
TEST_SIZE = 1000
SAMPLE_SEED = 0
GOLD_TITLES_PER_QUESTION = 2   # every HotpotQA question needs exactly 2 Gold paragraphs
# HotpotQA comes from Hugging Face because the official CMU server is down. We use the dev split, fullwiki setting.
HOTPOTQA_REPOSITORY = "hotpotqa/hotpot_qa"
HOTPOTQA_DEV_FILE = "fullwiki/validation-00000-of-00001.parquet"
QUESTIONS_DIR = PROJECT_ROOT / "data" / "hotpotqa"
DOWNLOAD_DIR = QUESTIONS_DIR / "raw"
QUESTION_IDS_DIR = PROJECT_ROOT / "results" / "question_ids"
HUGGING_FACE_CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "huggingface"
LOGS_DIR = PROJECT_ROOT / "runs" / "question_sets"

# Hugging Face keeps a cache in the home folder unless told otherwise. The Mac is short on space, so we point it
# into data/cache/. This must happen before huggingface_hub is imported, because it reads the setting on import.
os.environ.setdefault("HF_HOME", str(HUGGING_FACE_CACHE_DIR))

# These two imports sit below the setting on purpose, so Hugging Face sees it.
import pyarrow.parquet
from huggingface_hub import hf_hub_download


# --- Getting the questions ---

def download_hotpotqa(download_dir: Path) -> Path:
    """Download HotpotQA's dev questions into download_dir, or reuse the file if it's already there. Return its path."""
    downloaded_file_path = hf_hub_download(HOTPOTQA_REPOSITORY, HOTPOTQA_DEV_FILE,
                                           repo_type="dataset", local_dir=download_dir)
    return Path(downloaded_file_path)


def raw_row_to_question(raw_row: dict) -> dict:
    """Turn one raw HotpotQA row into our question record, keeping only the fields the project uses."""
    # HotpotQA repeats a Gold paragraph's title once for every supporting sentence in it.
    # dict.fromkeys keeps the first copy of each title and drops the repeats, without changing their order.
    gold_titles = list(dict.fromkeys(raw_row["supporting_facts"]["title"]))

    return {
        "id": raw_row["id"],
        "question": raw_row["question"],
        "answer": raw_row["answer"],
        "type": raw_row["type"],
        "level": raw_row["level"],
        "gold_titles": gold_titles,
    }


def read_hotpotqa_questions(hotpotqa_file_path: Path) -> list[dict]:
    """Read every row of the downloaded HotpotQA file and turn each one into our question record."""
    raw_rows = pyarrow.parquet.read_table(hotpotqa_file_path).to_pylist()
    return [raw_row_to_question(raw_row) for raw_row in raw_rows]


# --- Picking the two sets ---

def split_pilot_and_test(questions: list[dict], pilot_size: int, test_size: int,
                         seed: int) -> tuple[list[dict], list[dict]]:
    """Pick the Pilot set and the Test set at random. The same seed always picks the same questions."""
    # We draw both sets in one random sample, so no question can be picked twice.
    random_picker = random.Random(seed)
    sampled_questions = random_picker.sample(questions, pilot_size + test_size)

    # The first pilot_size questions become the Pilot set, and the rest become the Test set.
    pilot_set = sampled_questions[:pilot_size]
    test_set = sampled_questions[pilot_size:]
    return pilot_set, test_set


# --- Saving the sets ---

def question_ids_of(question_set: list[dict]) -> list[str]:
    """The ids of a question set, in the set's order."""
    return [question["id"] for question in question_set]


def write_lines_to_file(lines: list[str], file_path: Path) -> None:
    """Write each string on its own line, creating the file's folder first if it doesn't exist."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("".join(line + "\n" for line in lines))


def write_question_set(question_set: list[dict], question_file_path: Path) -> None:
    """Save a question set as a .jsonl file: one question per line, written as JSON."""
    question_lines = [json.dumps(question) for question in question_set]
    write_lines_to_file(question_lines, question_file_path)


def write_question_ids(question_set: list[dict], id_file_path: Path) -> None:
    """Save only the question ids, one per line. These small files go on GitHub and pin the sets for everyone."""
    write_lines_to_file(question_ids_of(question_set), id_file_path)


# --- Reading a saved set back ---

def load_question_set(set_name: str) -> list[dict]:
    """Read a saved question set ("pilot" or "test") back from its .jsonl file, one question per line."""
    question_file_path = QUESTIONS_DIR / f"{set_name}.jsonl"
    question_lines = question_file_path.read_text().splitlines()
    return [json.loads(question_line) for question_line in question_lines]


# --- Checks for the log ---

def id_list_would_change(question_set: list[dict], id_file_path: Path) -> bool:
    """True if an id file already exists and holds different ids, meaning the sets on GitHub would change."""
    if not id_file_path.exists():
        return False
    saved_ids = id_file_path.read_text().splitlines()
    return saved_ids != question_ids_of(question_set)


def describe_question_set(set_name: str, question_set: list[dict]) -> str:
    """One log line about a set: how many questions, and how many of each type (bridge or comparison)."""
    # Counter tallies how often each type appears, e.g. {"bridge": 81, "comparison": 19}.
    type_counts = Counter(question["type"] for question in question_set)

    type_descriptions = []
    for question_type, count in sorted(type_counts.items()):
        type_descriptions.append(f"{count} {question_type}")

    return f"{set_name} set: {len(question_set)} questions ({', '.join(type_descriptions)})"


def count_questions_without_two_gold_titles(question_set: list[dict]) -> int:
    """How many questions don't have exactly 2 Gold titles. It should be 0 for HotpotQA."""
    wrong_count = 0
    for question in question_set:
        if len(question["gold_titles"]) != GOLD_TITLES_PER_QUESTION:
            wrong_count += 1
    return wrong_count


# --- All steps, in order ---

def save_question_set(set_name: str, question_set: list[dict], log_lines: list[str]) -> None:
    """Write one set's question file and id file, log what was written, and warn if anything looks wrong."""
    question_file_path = QUESTIONS_DIR / f"{set_name}.jsonl"
    id_file_path = QUESTION_IDS_DIR / f"hotpotqa_{set_name}.txt"

    # Here we compare with the id list already saved before overwriting it, so a changed sample can't go unnoticed.
    if id_list_would_change(question_set, id_file_path):
        warn(log_lines, f"{id_file_path.name} changed: these are not the questions on GitHub. Check the seed and sizes.")

    write_question_set(question_set, question_file_path)
    write_question_ids(question_set, id_file_path)
    note(log_lines, describe_question_set(set_name, question_set))
    question_file_name = question_file_path.relative_to(PROJECT_ROOT)
    id_file_name = id_file_path.relative_to(PROJECT_ROOT)
    note(log_lines, f"  wrote {question_file_name} and {id_file_name}")

    questions_with_wrong_title_count = count_questions_without_two_gold_titles(question_set)
    if questions_with_wrong_title_count > 0:
        warn(log_lines, f"{questions_with_wrong_title_count} {set_name} questions don't have exactly "
                        f"{GOLD_TITLES_PER_QUESTION} Gold titles")


def build_question_sets(log_lines: list[str]) -> list[dict]:
    """Download HotpotQA dev, pick the Pilot set and the Test set, and save both. Return the Pilot set."""
    started_at = time.time()
    note(log_lines, f"Question sets: seed {SAMPLE_SEED}, Pilot size {PILOT_SIZE}, Test size {TEST_SIZE}")

    hotpotqa_file_path = download_hotpotqa(DOWNLOAD_DIR)
    questions = read_hotpotqa_questions(hotpotqa_file_path)
    note(log_lines, f"Read {len(questions)} questions from {hotpotqa_file_path.relative_to(PROJECT_ROOT)}")

    pilot_set, test_set = split_pilot_and_test(questions, PILOT_SIZE, TEST_SIZE, SAMPLE_SEED)
    save_question_set("pilot", pilot_set, log_lines)
    save_question_set("test", test_set, log_lines)

    elapsed_seconds = time.time() - started_at
    note(log_lines, f"Done in {elapsed_seconds:.1f} s")
    return pilot_set


if __name__ == "__main__":
    question_sets_log_lines = []
    try:
        pilot_set = build_question_sets(question_sets_log_lines)

        # A short demo: show the first Pilot question, so you can see what one question record looks like.
        print("\nThe first Pilot question:")
        for field_name, field_value in pilot_set[0].items():
            print(f"  {field_name}: {field_value}")
    except BaseException as build_error:
        # Here we record a failed run in the log too, then let the error show as usual.
        warn(question_sets_log_lines, f"building the question sets failed: {build_error!r}")
        raise
    finally:
        print("Log saved in", write_run_log(question_sets_log_lines, LOGS_DIR, "question_sets"))
