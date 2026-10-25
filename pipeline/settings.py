"""Facts about the machine this code runs on. Everything that differs between the Mac and Kaggle lives here.

    Press Run to see what this machine uses.

The experiment's own choices (the model for each role, k, Rounds, seeds) stay in config.yaml.
"""

import os
from pathlib import Path

from pipeline import REPO_ROOT, load_config

# Here we tell the two machines apart: Kaggle sets KAGGLE_KERNEL_RUN_TYPE in every notebook session; the Mac never does.
RUNNING_ON_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ

# Here we give this project's Ollama server a port of its own. It is not Ollama's default (11434), which the Ollama
# app uses and which may keep its models in the home folder, and not the other project's (11500).
OLLAMA_PORT = 11600
OLLAMA_ADDRESS = f"http://127.0.0.1:{OLLAMA_PORT}"

# Here we choose where Ollama keeps its model files. On Kaggle we use /tmp: everything in /kaggle/working is saved
# with the notebook, and that saved output is capped at about 20 GB.
if RUNNING_ON_KAGGLE:
    MODELS_FOLDER = Path("/tmp/ollama")
else:
    MODELS_FOLDER = REPO_ROOT / load_config()["paths"]["data_dir"] / "ollama"

# Here we cap the context window (prompt + reply, in tokens) that any one call may ask for, to what this machine's
# memory holds: the Mac has 16 GB of RAM shared with everything else, Kaggle's T4 GPU has 16 GB to itself.
OLLAMA_CONTEXT_WINDOW = 32768 if RUNNING_ON_KAGGLE else 16384


def ollama_environment() -> dict:
    """The settings an Ollama program needs to use this project's server: which folder its models live in, and
    which port it listens on. Starting the server and downloading a model both use it, so they always agree."""
    # Here we start from this program's own environment and add Ollama's two settings to it.
    return dict(os.environ, OLLAMA_MODELS=str(MODELS_FOLDER), OLLAMA_HOST=f"127.0.0.1:{OLLAMA_PORT}")


def model_is_downloaded(model_name: str) -> bool:
    """True if a model from Ollama's library, such as "qwen3:8b", is already in the models folder."""
    # Here we look for the small "manifest" file Ollama writes for every model it downloads.
    # A name without a tag means Ollama's "latest" tag, as `ollama pull` treats it.
    name, _, tag = model_name.partition(":")
    manifest_file = MODELS_FOLDER / "manifests" / "registry.ollama.ai" / "library" / name / (tag or "latest")
    return manifest_file.exists()


if __name__ == "__main__":
    print(f"Running on Kaggle:      {RUNNING_ON_KAGGLE}")
    print(f"Ollama address:         {OLLAMA_ADDRESS}")
    print(f"Models folder:          {MODELS_FOLDER}")
    print(f"Largest context window: {OLLAMA_CONTEXT_WINDOW:,} tokens")
    for role, model_name in load_config()["models"].items():
        print(f"{role:>9}'s model {model_name}: {'downloaded' if model_is_downloaded(model_name) else 'MISSING'}")
