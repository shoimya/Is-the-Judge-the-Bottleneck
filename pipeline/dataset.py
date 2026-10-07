"""HotpotQA / MuSiQue -> Pilot set and Test set question files. Built in T02 (HotpotQA) and T14 (MuSiQue).

    python -m pipeline.dataset

downloads HotpotQA dev (fullwiki setting), samples a Pilot set and a disjoint
Test set with the seed from config.yaml (`seeds[0]`), and writes

    data/hotpotqa/pilot.jsonl  data/hotpotqa/test.jsonl     (git-ignored)
    results/question_ids/pilot_ids.txt, test_ids.txt        (committed, pins the sample)

One question per line: `id`, `question`, `answer`, `type` (bridge/comparison),
`level`, `gold_titles` (the two Gold paragraph titles, from `supporting_facts`).

The download is HotpotQA's dev JSON republished on the Hugging Face Hub as
parquet (the canonical curtis.ml.cmu.edu host is flaky). Same data, 28 MB.
"""

import json
import random
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

from pipeline import REPO_ROOT, load_config

# The fullwiki validation split of hotpotqa/hotpot_qa == hotpot_dev_fullwiki_v1.json
DEV_URL = (
    "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/"
    "fullwiki/validation-00000-of-00001.parquet"
)
DEV_PARQUET = "hotpot_dev_fullwiki_v1.parquet"

PILOT_SIZE = 100
TEST_SIZE = 1000


def download_dev(data_dir: Path) -> Path:
    """Download the dev parquet into `data_dir`, unless it is already there."""
    dest = data_dir / DEV_PARQUET
    if dest.exists() and dest.stat().st_size > 0:
        print(f"Using existing {dest}")
        return dest

    data_dir.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    print(f"Downloading {DEV_URL}")
    request = urllib.request.Request(DEV_URL, headers={"User-Agent": "cmsc723/1.0"})
    with urllib.request.urlopen(request, timeout=600) as response, open(tmp, "wb") as out:
        size = 0
        next_report = 10 << 20
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            size += len(chunk)
            if size >= next_report:
                print(f"  {size // (1 << 20)} MB")
                next_report += 10 << 20
    tmp.rename(dest)
    return dest


def load_dev(path: Path) -> list[dict]:
    """Every row of the dev parquet as a dict."""
    return pq.read_table(path).to_pylist()


def gold_titles(item: dict) -> list[str] | None:
    """The two Gold paragraph titles, or None if the question is not usable.

    HotpotQA questions are annotated with two Gold paragraphs, but the number
    of supporting-fact *sentences* varies (some questions mark several
    sentences of the same Gold paragraph). We keep the distinct titles in the
    dataset's order: every question then has exactly two gold titles, and the
    Oracle's title matching never sees duplicates.
    """
    facts = item["supporting_facts"]
    titles: list[str] = []
    for title in facts["title"]:
        if title not in titles:
            titles.append(title)
    if len(titles) != 2:
        return None
    return titles


def to_record(item: dict, titles: list[str]) -> dict:
    """One jsonl line: everything downstream needs, nothing more."""
    return {
        "id": item["id"],
        "question": item["question"],
        "answer": item["answer"],
        "type": item["type"],
        "level": item["level"],
        "gold_titles": titles,
    }


def sample_ids(
    ids: list[str],
    seed: int,
    pilot_size: int = PILOT_SIZE,
    test_size: int = TEST_SIZE,
) -> tuple[list[str], list[str]]:
    """Deterministic disjoint sample: the first `pilot_size` ids go to the Pilot set.

    Sampling from sorted ids, so the same seed gives the same ids on any machine
    regardless of the order the source data arrives in.
    """
    rng = random.Random(seed)
    picked = sorted(rng.sample(sorted(ids), pilot_size + test_size))
    return picked[:pilot_size], picked[pilot_size:]


def write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_id_lists(results_dir: Path, pilot_ids: list[str], test_ids: list[str]) -> None:
    """The id lists are what gets committed: they pin the sample for everyone."""
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
    """Download (or use `items`), sample, and write the question files and id lists.

    Returns a small summary. Overwrites the output files, so running it twice
    gives identical files.
    """
    cfg = load_config()
    seed = cfg["seeds"][0] if seed is None else seed
    data_dir = Path(data_dir) if data_dir is not None else REPO_ROOT / cfg["paths"]["data_dir"] / "hotpotqa"
    results_dir = Path(results_dir) if results_dir is not None else REPO_ROOT / cfg["paths"]["results_dir"]

    if items is None:
        items = load_dev(download_dev(data_dir))

    usable: dict[str, dict] = {}
    dropped = 0
    for item in items:
        titles = gold_titles(item)
        if titles is None:
            dropped += 1
            continue
        usable[item["id"]] = to_record(item, titles)
    if len(usable) != len(items) - dropped:
        raise ValueError("duplicate question ids in the source data")

    pilot_ids, test_ids = sample_ids(list(usable), seed, pilot_size, test_size)

    data_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(data_dir / "pilot.jsonl", [usable[i] for i in pilot_ids])
    write_jsonl(data_dir / "test.jsonl", [usable[i] for i in test_ids])
    write_id_lists(results_dir, pilot_ids, test_ids)

    return {
        "seed": seed,
        "source_total": len(items),
        "dropped": dropped,
        "usable": len(usable),
        "pilot": len(pilot_ids),
        "test": len(test_ids),
    }


def main() -> None:
    summary = build()
    print(
        f"seed {summary['seed']}: {summary['usable']} of {summary['source_total']} questions usable"
        f" ({summary['dropped']} dropped for not having two distinct gold titles)"
    )
    print(f"pilot: {summary['pilot']} questions -> data/hotpotqa/pilot.jsonl")
    print(f"test:  {summary['test']} questions -> data/hotpotqa/test.jsonl")
    print("id lists (committed): results/question_ids/pilot_ids.txt, test_ids.txt")


if __name__ == "__main__":
    main()
