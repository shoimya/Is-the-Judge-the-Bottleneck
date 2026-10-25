"""HotpotQA questions: download them, keep the usable ones, and split them into a Pilot set and a Test set.

    Press Run to see the first Pilot question and its Gold paragraph titles.

build() (called by project_set_up.py) downloads HotpotQA's dev questions (fullwiki setting), samples a Pilot set
and a disjoint Test set with the seed from config.yaml (`seeds[0]`), and writes:

    data/hotpotqa/pilot.jsonl, test.jsonl                   (git-ignored)
    results/question_ids/pilot_ids.txt, test_ids.txt        (committed: they pin the sample for everyone)

One question per line: `id`, `question`, `answer`, `type` (bridge/comparison), `level`, and `gold_titles`
(the two Gold paragraph titles, from `supporting_facts`).

The download is HotpotQA's dev file as republished on the Hugging Face Hub (the original CMU server is
unreliable). Same data, 28 MB.
"""

import json
import random
import shutil
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

from pipeline import REPO_ROOT, load_config

# The fullwiki dev split of hotpotqa/hotpot_qa, the same questions as HotpotQA's hotpot_dev_fullwiki_v1.json.
DEV_URL = (
    "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/"
    "fullwiki/validation-00000-of-00001.parquet"
)
DEV_PARQUET = "hotpot_dev_fullwiki_v1.parquet"
ONE_MEGABYTE = 1024 * 1024

PILOT_SIZE = 100
TEST_SIZE = 1000


def questions_dir() -> Path:
    """Where the question files live: data/hotpotqa/."""
    return REPO_ROOT / load_config()["paths"]["data_dir"] / "hotpotqa"


def download_dev(questions_folder: Path) -> Path:
    """Download HotpotQA's dev questions (~28 MB) into questions_folder, unless they are already there."""
    parquet_path = questions_folder / DEV_PARQUET
    if parquet_path.exists() and parquet_path.stat().st_size > 0:
        print(f"Using the HotpotQA dev file already in {parquet_path}")
        return parquet_path

    questions_folder.mkdir(parents=True, exist_ok=True)
    # Here the bytes go into a ".part" file that is renamed only when the download ends,
    # so a half-finished download never looks like a finished one.
    partial_path = parquet_path.with_suffix(".part")
    print(f"Downloading {DEV_URL}")
    # Here we say who is asking: some servers refuse requests that don't name their program.
    request = urllib.request.Request(DEV_URL, headers={"User-Agent": "cmsc723/1.0"})
    with urllib.request.urlopen(request, timeout=600) as response, open(partial_path, "wb") as partial_file:
        shutil.copyfileobj(response, partial_file, length=ONE_MEGABYTE)
    partial_path.rename(parquet_path)
    return parquet_path


def read_dev_questions(parquet_path: Path) -> list[dict]:
    """Every question in the downloaded dev file, one dict per question."""
    return pq.read_table(parquet_path).to_pylist()


def distinct_gold_titles(dev_question: dict) -> list[str] | None:
    """The question's two Gold paragraph titles, in the dataset's order, or None if the question isn't usable.

    HotpotQA marks supporting facts sentence by sentence, so one Gold paragraph can appear several times.
    Here we keep each title once. A usable question then has exactly two titles, and the Oracle judge's
    title matching never sees duplicates.
    """
    titles = []
    for title in dev_question["supporting_facts"]["title"]:
        if title not in titles:
            titles.append(title)
    if len(titles) != 2:
        return None
    return titles


def question_record(dev_question: dict, gold_titles: list[str]) -> dict:
    """The fields the project keeps for one question: one line of pilot.jsonl or test.jsonl."""
    return {
        "id": dev_question["id"],
        "question": dev_question["question"],
        "answer": dev_question["answer"],
        "type": dev_question["type"],
        "level": dev_question["level"],
        "gold_titles": gold_titles,
    }


def sample_ids(
    question_ids: list[str],
    seed: int,
    pilot_size: int = PILOT_SIZE,
    test_size: int = TEST_SIZE,
) -> tuple[list[str], list[str]]:
    """Pick pilot_size + test_size different question ids at random with the seed: Pilot ids first, then Test ids.

    Here we sort the ids before picking, so the same seed gives the same ids on any machine,
    whatever order the downloaded questions arrive in.
    """
    random_generator = random.Random(seed)
    picked_ids = sorted(random_generator.sample(sorted(question_ids), pilot_size + test_size))
    return picked_ids[:pilot_size], picked_ids[pilot_size:]


def write_jsonl(jsonl_path: Path, records: list[dict]) -> None:
    """Save records to a .jsonl file, one JSON object per line."""
    with open(jsonl_path, "w", encoding="utf-8") as jsonl_file:
        for record in records:
            # Here ensure_ascii=False keeps names such as "Bogotá" readable instead of escaping them.
            jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_id_lists(results_dir: Path, pilot_ids: list[str], test_ids: list[str]) -> None:
    """Save the Pilot and Test question ids. These lists are committed: they pin the sample for everyone."""
    question_ids_dir = results_dir / "question_ids"
    question_ids_dir.mkdir(parents=True, exist_ok=True)
    (question_ids_dir / "pilot_ids.txt").write_text("\n".join(pilot_ids) + "\n")
    (question_ids_dir / "test_ids.txt").write_text("\n".join(test_ids) + "\n")


def build(
    items: list[dict] | None = None,
    *,
    data_dir: Path | None = None,
    results_dir: Path | None = None,
    seed: int | None = None,
    pilot_size: int = PILOT_SIZE,
    test_size: int = TEST_SIZE,
) -> dict:
    """Download the dev questions (or use `items`), sample them, and write the question files and id lists.

    Returns a small summary. Overwrites the output files, so running it twice gives identical files.
    """
    config = load_config()
    seed = config["seeds"][0] if seed is None else seed
    data_dir = Path(data_dir) if data_dir is not None else questions_dir()
    results_dir = Path(results_dir) if results_dir is not None else REPO_ROOT / config["paths"]["results_dir"]

    if items is None:
        items = read_dev_questions(download_dev(data_dir))

    usable: dict[str, dict] = {}
    dropped = 0
    for item in items:
        titles = distinct_gold_titles(item)
        if titles is None:
            dropped += 1
            continue
        usable[item["id"]] = question_record(item, titles)
    if len(usable) != len(items) - dropped:
        raise ValueError("duplicate question ids in the source data")

    pilot_ids, test_ids = sample_ids(list(usable), seed, pilot_size, test_size)

    data_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(data_dir / "pilot.jsonl", [usable[question_id] for question_id in pilot_ids])
    write_jsonl(data_dir / "test.jsonl", [usable[question_id] for question_id in test_ids])
    write_id_lists(results_dir, pilot_ids, test_ids)

    return {
        "seed": seed,
        "source_total": len(items),
        "dropped": dropped,
        "usable": len(usable),
        "pilot": len(pilot_ids),
        "test": len(test_ids),
    }


def load_questions(split_name: str) -> list[dict]:
    """Every question in one split ("pilot" or "test"), in the order of its file."""
    questions_file = questions_dir() / f"{split_name}.jsonl"
    with open(questions_file, encoding="utf-8") as opened_file:
        return [json.loads(line) for line in opened_file if line.strip()]


def print_question_and_gold_titles(question: dict) -> None:
    """Show a question and the titles of the Gold paragraphs the dataset marks as needed to answer it."""
    print(f"Question: {question['question']}")
    print("Gold paragraphs:")
    for gold_title in question["gold_titles"]:
        print(f"  - {gold_title}")


if __name__ == "__main__":
    # Here the demo shows the first Pilot question; project_set_up.py builds the sets themselves.
    pilot_questions = load_questions("pilot")
    print_question_and_gold_titles(pilot_questions[0])
