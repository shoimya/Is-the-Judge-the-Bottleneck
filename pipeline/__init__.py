"""The pipeline: questions -> retriever -> judge -> rewriter -> answerer.

`load_config()` lives here so every piece reads settings the same way.
"""

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config.yaml"


def load_config(path: str | Path = CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
