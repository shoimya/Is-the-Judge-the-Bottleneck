"""Where the project's settings and folders are. Every other file gets them from here.

    Press Debug on this file to print the settings and folders.
"""

from pathlib import Path

import yaml

# Here we find the repo's top folder: two levels up from this file (pipeline/config.py -> repo).
REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = REPO_ROOT / "config.yaml"

# Large files that stay on this machine (git-ignored).
DATA_DIR = REPO_ROOT / "data"
QUESTIONS_DIR = DATA_DIR / "hotpotqa"                    # pilot.jsonl, test.jsonl, raw/ download
INDEX_DIR = DATA_DIR / "wiki" / "bm25_index"             # the built BM25 search index
WIKI_DUMP_DIR = DATA_DIR / "wiki" / "dump"               # Wikipedia download, only needed to rebuild the index
HUGGING_FACE_CACHE_DIR = DATA_DIR / "cache" / "huggingface"

# Run logs (git-ignored) and the small result files that go on GitHub.
RUNS_DIR = REPO_ROOT / "runs"
RESULTS_DIR = REPO_ROOT / "results"
QUESTION_IDS_DIR = RESULTS_DIR / "question_ids"
RECALL_CSV = RESULTS_DIR / "pilot" / "bm25_recall.csv"


def load_config(config_file: Path = CONFIG_FILE) -> dict:
    """Read config.yaml and return its settings as a dictionary."""
    with open(config_file) as opened_file:
        return yaml.safe_load(opened_file)


if __name__ == "__main__":
    # Here we show what the rest of the code will see.
    print("Settings from", CONFIG_FILE)
    for setting_name, setting_value in load_config().items():
        print(f"  {setting_name}: {setting_value}")
    print("Data folder:   ", DATA_DIR)
    print("Run logs:      ", RUNS_DIR)
    print("Results folder:", RESULTS_DIR)
