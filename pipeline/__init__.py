"""The pipeline: questions -> retriever -> judge -> rewriter -> answerer.

`load_config()` lives here so every piece reads settings the same way.
"""

import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config.yaml"


def load_config(path: str | Path = CONFIG_PATH) -> dict:
    with open(path) as config_file:
        return yaml.safe_load(config_file)


DATA_DIR = REPO_ROOT / load_config()["paths"]["data_dir"]

# Downloads and caches stay on the project drive, not in the home folder.
os.environ.setdefault("HF_HOME", str(DATA_DIR / "cache" / "huggingface"))
