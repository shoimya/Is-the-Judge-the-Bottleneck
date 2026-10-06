"""HotpotQA / MuSiQue -> Pilot set and Test set question files. Built in T02 (HotpotQA) and T14 (MuSiQue).

    python -m pipeline.dataset    # download HotpotQA dev (fullwiki) and write the question sets
"""

import json
import random
from pathlib import Path

import pyarrow.parquet as pq

from pipeline import DATA_DIR, REPO_ROOT, load_config


def raw_row_to_question(raw_row: dict) -> dict:
    """One raw HotpotQA row -> our question record, with each Gold paragraph title once, in order."""
    # Here we collect the titles of the Gold paragraphs. HotpotQA lists a title once per supporting
    # sentence, so a title can repeat; dict.fromkeys drops the repeats but keeps the original order.
    gold_titles = list(dict.fromkeys(raw_row["supporting_facts"]["title"]))
    # Here we keep only the fields the project uses and drop the rest of the raw row.
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
    # Here we draw pilot_size + test_size different questions at random. The fixed seed means
    # everyone, on any machine, draws exactly the same questions.
    pilot_and_test_sample = random.Random(seed).sample(questions, pilot_size + test_size)
    # Here we cut the sample in two: the first part is the Pilot set, the rest is the Test set,
    # so no question can be in both.
    return pilot_and_test_sample[:pilot_size], pilot_and_test_sample[pilot_size:]


def build_question_sets(parquet_path: Path, questions_dir: Path, question_ids_dir: Path,
                        pilot_size: int, test_size: int, seed: int) -> None:
    """Raw HotpotQA Parquet -> questions_dir/{pilot,test}.jsonl and the sampled id lists in question_ids_dir."""
    # Here we read every row of the downloaded HotpotQA file and turn each one into our question record.
    questions = [raw_row_to_question(row) for row in pq.read_table(parquet_path).to_pylist()]
    pilot_set, test_set = split_pilot_and_test(questions, pilot_size, test_size, seed)

    # Here we make sure both output folders exist.
    questions_dir.mkdir(parents=True, exist_ok=True)
    question_ids_dir.mkdir(parents=True, exist_ok=True)

    # Here we write two files for each set:
    #   data/hotpotqa/<set>.jsonl     the full questions, one JSON record per line (stays on this machine)
    #   results/question_ids/...txt   just the question ids, one per line (goes on GitHub, so anyone can check them)
    for set_name, question_set in [("pilot", pilot_set), ("test", test_set)]:
        question_lines = [json.dumps(question) + "\n" for question in question_set]
        (questions_dir / f"{set_name}.jsonl").write_text("".join(question_lines))

        id_lines = [question["id"] + "\n" for question in question_set]
        (question_ids_dir / f"hotpotqa_{set_name}.txt").write_text("".join(id_lines))


def load_question_set(set_name: str, questions_dir: Path = DATA_DIR / "hotpotqa") -> list[dict]:
    """Read a question set written by build_question_sets: set_name is "pilot" or "test"."""
    # Here we read the .jsonl file line by line; each line is one question record.
    with open(questions_dir / f"{set_name}.jsonl") as question_file:
        return [json.loads(line) for line in question_file]


def download_hotpotqa(hotpotqa_config: dict, raw_dir: Path) -> Path:
    """Fetch HotpotQA dev (fullwiki) from Hugging Face into raw_dir, once."""
    # Imported here so the Hugging Face cache location set in pipeline/__init__.py applies first.
    from huggingface_hub import hf_hub_download

    # Here we download the file (or reuse it if it is already in raw_dir) and return where it is saved.
    return Path(hf_hub_download(hotpotqa_config["hf_repo"], hotpotqa_config["hf_file"],
                                repo_type="dataset", local_dir=raw_dir))


def main() -> None:
    # Here we read the settings and work out where the files go.
    config = load_config()
    hotpotqa_config = config["hotpotqa"]
    questions_dir = DATA_DIR / "hotpotqa"
    question_ids_dir = REPO_ROOT / config["paths"]["results_dir"] / "question_ids"

    # Here we download HotpotQA, then sample and save the Pilot set and Test set.
    parquet_path = download_hotpotqa(hotpotqa_config, raw_dir=questions_dir / "raw")
    build_question_sets(parquet_path, questions_dir, question_ids_dir,
                        hotpotqa_config["pilot_size"], hotpotqa_config["test_size"],
                        hotpotqa_config["sample_seed"])
    print(f"Wrote {hotpotqa_config['pilot_size']} Pilot and {hotpotqa_config['test_size']} Test questions "
          f"to {questions_dir}, and their id lists to {question_ids_dir}")


if __name__ == "__main__":
    main()
