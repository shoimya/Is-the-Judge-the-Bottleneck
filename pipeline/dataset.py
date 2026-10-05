"""HotpotQA / MuSiQue -> Pilot set and Test set question files. Built in T02 (HotpotQA) and T14 (MuSiQue).

    python -m pipeline.dataset    # download HotpotQA dev (fullwiki) and write the question sets
"""

import json
import random
from pathlib import Path

import pyarrow.parquet as pq

from pipeline import DATA_DIR, REPO_ROOT, load_config


def to_question(raw_row: dict) -> dict:
    """One raw HotpotQA row -> our question record, with each Gold paragraph title once, in order."""
    gold_titles = list(dict.fromkeys(raw_row["supporting_facts"]["title"]))
    return {
        "id": raw_row["id"],
        "question": raw_row["question"],
        "answer": raw_row["answer"],
        "type": raw_row["type"],
        "level": raw_row["level"],
        "gold_titles": gold_titles,
    }


def split_pilot_and_test(questions: list[dict], pilot_size: int, test_size: int,
                         seed: int) -> tuple[list[dict], list[dict]]:
    """Seeded sample of a Pilot set and a disjoint Test set."""
    sampled = random.Random(seed).sample(questions, pilot_size + test_size)
    return sampled[:pilot_size], sampled[pilot_size:]


def build_question_sets(parquet_path: Path, questions_dir: Path, question_ids_dir: Path,
                        pilot_size: int, test_size: int, seed: int) -> None:
    """Raw HotpotQA Parquet -> questions_dir/{pilot,test}.jsonl and the sampled id lists in question_ids_dir."""
    questions = [to_question(row) for row in pq.read_table(parquet_path).to_pylist()]
    pilot_set, test_set = split_pilot_and_test(questions, pilot_size, test_size, seed)
    questions_dir.mkdir(parents=True, exist_ok=True)
    question_ids_dir.mkdir(parents=True, exist_ok=True)
    for set_name, question_set in [("pilot", pilot_set), ("test", test_set)]:
        (questions_dir / f"{set_name}.jsonl").write_text(
            "".join(json.dumps(question) + "\n" for question in question_set))
        (question_ids_dir / f"hotpotqa_{set_name}.txt").write_text(
            "".join(question["id"] + "\n" for question in question_set))


def load_question_set(set_name: str, questions_dir: Path = DATA_DIR / "hotpotqa") -> list[dict]:
    """Read a question set written by build_question_sets: set_name is "pilot" or "test"."""
    with open(questions_dir / f"{set_name}.jsonl") as question_file:
        return [json.loads(line) for line in question_file]


def download_hotpotqa(hotpotqa_config: dict, raw_dir: Path) -> Path:
    """Fetch HotpotQA dev (fullwiki) from Hugging Face into raw_dir, once."""
    # Imported here so the Hugging Face cache location set in pipeline/__init__.py applies first.
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(hotpotqa_config["hf_repo"], hotpotqa_config["hf_file"],
                                repo_type="dataset", local_dir=raw_dir))


def main() -> None:
    config = load_config()
    hotpotqa_config = config["hotpotqa"]
    questions_dir = DATA_DIR / "hotpotqa"
    question_ids_dir = REPO_ROOT / config["paths"]["results_dir"] / "question_ids"

    parquet_path = download_hotpotqa(hotpotqa_config, raw_dir=questions_dir / "raw")
    build_question_sets(parquet_path, questions_dir, question_ids_dir,
                        hotpotqa_config["pilot_size"], hotpotqa_config["test_size"],
                        hotpotqa_config["sample_seed"])
    print(f"Wrote {hotpotqa_config['pilot_size']} Pilot and {hotpotqa_config['test_size']} Test questions "
          f"to {questions_dir}, and their id lists to {question_ids_dir}")


if __name__ == "__main__":
    main()
