"""The pipeline: questions -> retriever -> judge -> rewriter -> answerer.

`load_config()` lives here so every piece reads settings the same way, and `pipeline_number()` so every piece reads
the pipeline setting (Rewriter on or off) the same way.
"""

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config.yaml"


def load_config(path: str | Path = CONFIG_PATH) -> dict:
    """Read config.yaml (or another settings file) fresh from disk, so a change to it is seen straight away."""
    with open(path) as f:
        return yaml.safe_load(f)


def pipeline_number() -> int:
    """Which pipeline config.yaml asks for: 1 (Rewriter on) or 2 (Rewriter off). Anything else stops with a message."""
    pipeline = load_config()["loop"]["pipeline"]
    if pipeline not in (1, 2):
        raise ValueError(f"config.yaml's loop.pipeline is {pipeline!r}; "
                         "it must be 1 (Rewriter on) or 2 (Rewriter off).")
    return pipeline
