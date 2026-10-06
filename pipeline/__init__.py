"""The pipeline: questions -> retriever -> judge -> rewriter -> answerer.

`load_config()` lives here so every piece reads settings the same way.
"""

import os
from pathlib import Path

import yaml

# Here we find the repo's top folder (two levels up from this file) and the settings file inside it.
REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config.yaml"


def load_config(path: str | Path = CONFIG_PATH) -> dict:
    """Read config.yaml (or another settings file) and return its settings as a dictionary."""
    # Here we open the YAML file and turn it into plain Python dictionaries and lists.
    with open(path) as config_file:
        return yaml.safe_load(config_file)


# Here we decide, once, where all large files (datasets, index, model files) are stored: the data/ folder
# named in config.yaml. Every other file imports DATA_DIR from here instead of working it out again.
DATA_DIR = REPO_ROOT / load_config()["paths"]["data_dir"]

# Here we tell Hugging Face to save its downloads inside data/, not in the home folder (the Mac is short on space).
os.environ.setdefault("HF_HOME", str(DATA_DIR / "cache" / "huggingface"))

# Here we keep JAX (preinstalled on Kaggle, picked up by the BM25 library) off the GPU. Left alone, it reserves
# 75% of GPU memory as soon as the search index loads, and vLLM then has no room for the model.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
