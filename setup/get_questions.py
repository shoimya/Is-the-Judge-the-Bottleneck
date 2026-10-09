"""Make the Pilot set and Test set from HotpotQA's dev questions (fullwiki setting). Run once per machine. (T02)

    Open this file and press Run (or Debug).

Writes data/hotpotqa/{pilot,test}.jsonl (the questions) and results/question_ids/hotpotqa_{pilot,test}.txt
(their ids, on GitHub). The fixed seed means every machine picks exactly the same questions.
"""

import json
import os
import random
from pathlib import Path

import pyarrow.parquet as pq

from pipeline.config import HUGGING_FACE_CACHE_DIR, QUESTION_IDS_DIR, QUESTIONS_DIR, load_config

# Where HotpotQA is downloaded from (Hugging Face, because the official CMU server is down).
HOTPOTQA_REPO = "hotpotqa/hotpot_qa"
HOTPOTQA_FILE = "fullwiki/validation-00000-of-00001.parquet"


def download_hotpotqa(download_dir: Path) -> Path:
    """Download HotpotQA dev (fullwiki) into download_dir, or reuse it if it's already there; return its path."""
    # Here we keep Hugging Face's own cache inside data/, not the home folder (the Mac is short on space).
    os.environ.setdefault("HF_HOME", str(HUGGING_FACE_CACHE_DIR))
    from huggingface_hub import hf_hub_download   # imported after HF_HOME is set, so the setting applies

    return Path(hf_hub_download(HOTPOTQA_REPO, HOTPOTQA_FILE, repo_type="dataset", local_dir=download_dir))


def raw_row_to_question(raw_row: dict) -> dict:
    """One raw HotpotQA row -> our question record, keeping only the fields the project uses."""
    # Here we list each Gold paragraph's title once, in order: HotpotQA repeats a title for every supporting
    # sentence, and dict.fromkeys drops the repeats while keeping the order.
    gold_titles = list(dict.fromkeys(raw_row["supporting_facts"]["title"]))
    return {
        "id": raw_row["id"],
        "question": raw_row["question"],
        "answer": raw_row["answer"],
        "type": raw_row["type"],
        "level": raw_row["level"],
        "gold_titles": gold_titles,
    }


def read_hotpotqa_questions(parquet_path: Path) -> list[dict]:
    """Read every row of the downloaded HotpotQA file as a question record."""
    raw_rows = pq.read_table(parquet_path).to_pylist()
    return [raw_row_to_question(raw_row) for raw_row in raw_rows]


def split_pilot_and_test(questions: list[dict], pilot_size: int, test_size: int,
                         seed: int) -> tuple[list[dict], list[dict]]:
    """Pick a Pilot set and a Test set at random (same seed -> same picks), with no question in both."""
    pilot_and_test_sample = random.Random(seed).sample(questions, pilot_size + test_size)
    # Here we cut the sample in two, so the sets can't overlap.
    return pilot_and_test_sample[:pilot_size], pilot_and_test_sample[pilot_size:]


def write_question_set(question_set: list[dict], question_file_path: Path) -> None:
    """Save a question set as .jsonl: one question per line, as JSON."""
    question_file_path.parent.mkdir(parents=True, exist_ok=True)
    question_lines = [json.dumps(question) + "\n" for question in question_set]
    question_file_path.write_text("".join(question_lines))


def write_question_ids(question_set: list[dict], id_file_path: Path) -> None:
    """Save just the question ids, one per line, so anyone can check which questions were used."""
    id_file_path.parent.mkdir(parents=True, exist_ok=True)
    id_lines = [question["id"] + "\n" for question in question_set]
    id_file_path.write_text("".join(id_lines))


def build_question_sets(parquet_path: Path, questions_dir: Path, question_ids_dir: Path,
                        pilot_size: int, test_size: int, seed: int) -> None:
    """HotpotQA file -> the Pilot and Test question files and their id lists."""
    questions = read_hotpotqa_questions(parquet_path)
    pilot_set, test_set = split_pilot_and_test(questions, pilot_size, test_size, seed)
    for set_name, question_set in [("pilot", pilot_set), ("test", test_set)]:
        write_question_set(question_set, questions_dir / f"{set_name}.jsonl")
        write_question_ids(question_set, question_ids_dir / f"hotpotqa_{set_name}.txt")


if __name__ == "__main__":
    config = load_config()
    parquet_path = download_hotpotqa(download_dir=QUESTIONS_DIR / "raw")
    build_question_sets(parquet_path, QUESTIONS_DIR, QUESTION_IDS_DIR,
                        config["pilot_size"], config["test_size"], config["sample_seed"])
    print(f"Wrote {config['pilot_size']} Pilot and {config['test_size']} Test questions to {QUESTIONS_DIR}")
    print(f"and their id lists to {QUESTION_IDS_DIR}")
