"""Small checks for the pipeline. Run with `pytest`."""
import configparser

from datasets import load_dataset
import subprocess
import sys
import json

from pipeline import REPO_ROOT, load_config


def test_config_loads():
    cfg = load_config()
    assert cfg["loop"]["rounds"] == 3
    assert set(cfg["models"]) == {"judge", "rewriter", "answerer"}

def test_question_sets_reproducible():
    """T02: running the script twice gives identical files."""
    cfg = load_config()
    files = [
        REPO_ROOT / cfg["paths"]["data_dir"] / "hotpotqa" / "pilot.jsonl",
        REPO_ROOT / cfg["paths"]["data_dir"] / "hotpotqa" / "test.jsonl",
        REPO_ROOT / cfg["paths"]["results_dir"] / "question_ids" / "hotpotqa_pilot.txt",
        REPO_ROOT / cfg["paths"]["results_dir"] / "question_ids" / "hotpotqa_test.txt",
    ]

    def run_and_read():
        subprocess.run([sys.executable, "-m", "pipeline.dataset"], cwd=REPO_ROOT, check=True)
        return [f.read_bytes() for f in files]

    assert run_and_read() == run_and_read()



def test_question_sets_disjoint():
    """T02: Pilot and Test share no ids, and every question has 2 gold_titles."""
    cfg = load_config()
    d = REPO_ROOT / cfg["paths"]["data_dir"] / "hotpotqa"
    pilot = [json.loads(line) for line in open(d / "pilot.jsonl")]
    test = [json.loads(line) for line in open(d / "test.jsonl")]

    assert len(pilot) == 100
    assert len(test) == 1000
    assert not {q["id"] for q in pilot} & {q["id"] for q in test}
    assert all(len(q["gold_titles"]) == 2 for q in pilot + test)