"""Set up this machine for the project in one go. Run it after cloning, and again whenever you like.

    Open this file and press Run, with any Python 3 selected as the interpreter.

Each step checks first and only installs or downloads what is missing, so a second run changes nothing:
Python 3.12+, the .venv, the libraries, the Pilot and Test sets, the search index, Ollama and its model.
Steps it can't do for you (installing Python or Ollama, the Kaggle download) are printed as instructions.
Every run is logged to runs/project_set_up/<date>/.

It runs in two halves. The first half uses only Python's own library, because nothing is installed yet.
It then restarts itself with the .venv's Python, so the second half can use the project's own code.
"""

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# The repo's top folder: this file sits in it.
REPO_ROOT = Path(__file__).resolve().parent
VENV_DIR = REPO_ROOT / ".venv"
VENV_PYTHON = VENV_DIR / "bin" / "python"
REQUIREMENTS_FILE = REPO_ROOT / "requirements.txt"
SET_UP_LOGS_DIR = REPO_ROOT / "runs" / "project_set_up"

OLDEST_PYTHON = (3, 12)
PYTHONS_TO_LOOK_FOR = ["python3.13", "python3.12"]      # used when the Python running this file is too old
PROJECT_PACKAGE_NAME = "judge-bottleneck"               # what the `-e .` line installs (see pyproject.toml)

# The two halves share one log file and one start time; the first half hands them over in these variables.
LOG_FILE_VARIABLE = "PROJECT_SET_UP_LOG_FILE"
START_TIME_VARIABLE = "PROJECT_SET_UP_STARTED_AT"


# ---------------------------------------------------------------------------
# Logging: every line goes to the screen and to this run's log file
# ---------------------------------------------------------------------------

def new_log_file() -> Path:
    """A fresh log file, runs/project_set_up/<date>/project_set_up_<time>.log, so reruns never overwrite it."""
    now = datetime.now()
    log_folder = SET_UP_LOGS_DIR / now.strftime("%Y-%m-%d")
    log_folder.mkdir(parents=True, exist_ok=True)
    return log_folder / f"project_set_up_{now.strftime('%H%M%S')}.log"


def log_file_for_this_run() -> Path:
    """The log file the first half started, or a new one if this is the first half."""
    if LOG_FILE_VARIABLE not in os.environ:
        os.environ[LOG_FILE_VARIABLE] = str(new_log_file())
        os.environ[START_TIME_VARIABLE] = str(time.time())
    return Path(os.environ[LOG_FILE_VARIABLE])


def report(message: str) -> None:
    """Show a message and add it, with the time, to this run's log."""
    # Here flush=True shows the message straight away, before the output of any command run next.
    print(message, flush=True)
    with open(log_file_for_this_run(), "a") as log_file:
        log_file.write(f"{datetime.now().strftime('%H:%M:%S')}  {message}\n")


def run_and_log(command: list) -> None:
    """Run a command from the repo folder, showing its output; stop the set-up if it fails."""
    report("  running: " + " ".join(str(part) for part in command))
    finished = subprocess.run(command, cwd=REPO_ROOT)
    if finished.returncode != 0:
        report(f"WARNING  that command failed (exit code {finished.returncode}); fix the error above and rerun.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# First half: only Python's own library
# ---------------------------------------------------------------------------

def python_is_new_enough(version: tuple) -> bool:
    """True if a Python version, such as (3, 12, 4), is one the project supports."""
    return tuple(version[:2]) >= OLDEST_PYTHON


def find_new_enough_python() -> str:
    """The path of a Python 3.12+ on this Mac: the one running this file, or one on the PATH ("" if none)."""
    if python_is_new_enough(sys.version_info):
        return sys.executable
    for python_name in PYTHONS_TO_LOOK_FOR:
        python_path = shutil.which(python_name)
        if python_path:
            return python_path
    return ""


def make_virtual_environment() -> None:
    """Create .venv with Python 3.12+ if it isn't there yet."""
    if VENV_PYTHON.exists():
        report("OK       .venv already exists")
        return
    python_path = find_new_enough_python()
    if not python_path:
        report("WARNING  the project needs Python 3.12 or newer, and none was found. Install it with:\n"
               "             brew install python@3.12\n"
               "         then run this file again.")
        sys.exit(1)
    report(f"MAKING   .venv with {python_path}")
    run_and_log([python_path, "-m", "venv", VENV_DIR])


def library_name(requirement: str) -> str:
    """The plain name of a pinned library, in pip's spelling: "bm25s[core]==0.3.11" -> "bm25s"."""
    name = requirement.split("==")[0].split("[")[0].strip()
    # Here we match pip's spelling: case, "_" and "." don't matter in library names.
    return name.lower().replace("_", "-").replace(".", "-")


def missing_libraries(requirement_lines: list, installed_versions: dict) -> list:
    """The requirement lines whose library is not installed at exactly that version (comments ignored)."""
    missing = []
    for line in requirement_lines:
        requirement = line.split("#")[0].strip()
        if not requirement:
            continue
        # Here the `-e .` line stands for this project's own package, which any version satisfies.
        if requirement == "-e .":
            if library_name(PROJECT_PACKAGE_NAME) not in installed_versions:
                missing.append(requirement)
            continue
        wanted_version = requirement.split("==")[1].strip()
        if installed_versions.get(library_name(requirement)) != wanted_version:
            missing.append(requirement)
    return missing


def installed_versions_in_venv() -> dict:
    """Every library installed in .venv, as {name in pip's spelling: version}."""
    pip_list = subprocess.run([VENV_PYTHON, "-m", "pip", "list", "--format=json"],
                              capture_output=True, text=True, check=True)
    return {library_name(library["name"]): library["version"] for library in json.loads(pip_list.stdout)}


def install_libraries() -> None:
    """Install the exact library versions in requirements.txt into .venv, if any are missing."""
    requirement_lines = REQUIREMENTS_FILE.read_text().splitlines()
    missing = missing_libraries(requirement_lines, installed_versions_in_venv())
    if not missing:
        report("OK       every library in requirements.txt is installed")
        return
    report(f"INSTALL  {len(missing)} missing libraries: {', '.join(missing)}")
    # Here --no-cache-dir stops pip from keeping a copy of every download in the home folder.
    run_and_log([VENV_PYTHON, "-m", "pip", "install", "--no-cache-dir", "-r", REQUIREMENTS_FILE])


def running_in_venv() -> bool:
    """True if this file is being run by the .venv's Python."""
    return Path(sys.prefix).resolve() == VENV_DIR.resolve()


def first_half() -> None:
    """Get .venv and its libraries in place, then restart this file with the .venv's Python."""
    report(f"Setting up {REPO_ROOT.name} (log: {log_file_for_this_run()})")
    make_virtual_environment()
    install_libraries()
    if not running_in_venv():
        # Here we swap this process for the .venv's Python running this same file; the log file and start time
        # travel with it in the environment variables.
        os.execv(VENV_PYTHON, [str(VENV_PYTHON), __file__])


# ---------------------------------------------------------------------------
# Second half: runs with the .venv's Python, so it can use the project's code
# ---------------------------------------------------------------------------

def question_sets_exist(questions_dir: Path) -> bool:
    """True if both the Pilot set and the Test set files are there."""
    return (questions_dir / "pilot.jsonl").exists() and (questions_dir / "test.jsonl").exists()


def make_question_sets() -> list:
    """Build the Pilot and Test sets if they're missing. Returns the steps left for the user (none here)."""
    from pipeline.config import QUESTIONS_DIR
    if question_sets_exist(QUESTIONS_DIR):
        report("OK       Pilot and Test sets are in data/hotpotqa/")
        return []
    report("MAKING   the Pilot and Test sets (downloads HotpotQA into data/)")
    run_and_log([sys.executable, REPO_ROOT / "setup" / "get_questions.py"])
    return []


def check_search_index() -> list:
    """Check the BM25 search index is in data/wiki/bm25_index/; if not, say how to get it."""
    from pipeline.config import INDEX_DIR
    # Here we look for the settings file every saved bm25s index has.
    if (INDEX_DIR / "params.index.json").exists():
        report("OK       the search index is in data/wiki/bm25_index/")
        return []
    report("WARNING  the search index is missing. It needs your Kaggle account, so do this by hand:")
    steps = ("Download the search index (~2.8 GB):\n"
             "    1. On kaggle.com, open Your Work > Datasets > hotpotqa-bm25-index, and click Download.\n"
             f"    2. Unzip it so its files (params.index.json, ...) sit directly in\n"
             f"       {INDEX_DIR}\n"
             "    3. Run this file again; it checks the index is found.")
    report(steps)
    return [steps]


def ollama_model_is_downloaded(models_dir: Path, ollama_model_name: str) -> bool:
    """True if Ollama's models folder holds this model, e.g. "qwen3:4b-instruct"."""
    # Here a name without a tag means Ollama's "latest" tag, as `ollama pull` treats it.
    model_name, _, tag = ollama_model_name.partition(":")
    manifest_file = models_dir / "manifests" / "registry.ollama.ai" / "library" / model_name / (tag or "latest")
    return manifest_file.exists()


def ollama_models_the_project_uses() -> list:
    """The Ollama names of the models config.yaml asks for, for every role (usually just one)."""
    from pipeline.config import load_config
    from pipeline.llm import model_for, ollama_name_for
    config = load_config()
    model_names = {model_for(role, config) for role in ["judge", "rewriter", "answerer"]}
    return sorted(ollama_name_for(model_name) for model_name in model_names)


def get_ollama_models() -> list:
    """Make sure Ollama is installed and the project's models are in data/ollama/; download the models if needed."""
    from pipeline.llm import ollama_is_running
    from pipeline.server_launcher import OLLAMA_MODELS_DIR, start_ollama_server, stop_ollama_server
    if not shutil.which("ollama"):
        report("WARNING  Ollama isn't installed. Install it yourself (it's system software):")
        steps = ("Install Ollama, the program that runs the model on the Mac:\n"
                 "    1. In a terminal, run:  brew install ollama\n"
                 "       (no Homebrew? download it from https://ollama.com/download instead)\n"
                 "    2. Run this file again; it then downloads the model into data/ollama/.")
        report(steps)
        return [steps]
    missing_models = [name for name in ollama_models_the_project_uses()
                      if not ollama_model_is_downloaded(OLLAMA_MODELS_DIR, name)]
    if not missing_models:
        report("OK       Ollama is installed and the model is in data/ollama/")
        return []
    if ollama_is_running():
        # Here an Ollama someone else started may keep its models in the home folder, so we don't pull through it.
        report("WARNING  Ollama is already running, maybe storing models in your home folder. Stop it, then rerun.")
        return ["Stop the running Ollama (quit the app, or Ctrl+C in its terminal), then run this file again."]
    report(f"DOWNLOAD Ollama models into data/ollama/: {', '.join(missing_models)}")
    start_ollama_server()
    try:
        for model_name in missing_models:
            run_and_log(["ollama", "pull", model_name])
    finally:
        # Here we close the server this script opened, and nothing else.
        stop_ollama_server()
    return []


def second_half() -> None:
    """Get the data and the model in place, check the setup, and list what is left to do by hand."""
    steps_left = []
    steps_left += make_question_sets()
    steps_left += check_search_index()
    steps_left += get_ollama_models()
    report("CHECK    run.py's setup check")
    from run import check_setup
    check_setup()
    minutes_taken = (time.time() - float(os.environ[START_TIME_VARIABLE])) / 60
    report(f"\nFinished in {minutes_taken:.1f} minutes. Steps left to do by hand: {len(steps_left)}")
    for step_number, step in enumerate(steps_left, start=1):
        report(f"\n{step_number}. {step}")
    if not steps_left:
        report("Everything is set up. Select .venv as the Python interpreter in VS Code, then see the README.")


if __name__ == "__main__":
    # Here the log file variable is only set once the first half has run, so after the restart it's skipped.
    if LOG_FILE_VARIABLE not in os.environ:
        first_half()
    second_half()
