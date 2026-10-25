"""Set up this machine for the project in one go. Run it after cloning, and again whenever you like.

    Open this file and press Run. Any Python 3 can start it.

Each step checks first and only installs or downloads what is missing, so a second run changes nothing:
    1. Python at the version Kaggle uses      4. the Pilot and Test sets
    2. the .venv                              5. the search index
    3. the libraries in requirements.txt      6. Ollama and the models named in config.yaml
Steps it can't do for you (installing Python or Ollama, downloading the index from Kaggle) are listed at the end.
Nothing is logged yet: it only prints to the screen.

It runs in two halves. The first half uses only Python's own library, because nothing is installed yet.
It then restarts itself with the .venv's Python, so the second half can use the project's own code.
To step through the second half with breakpoints, select .venv as the interpreter and press Debug:
the restart is then skipped.
"""

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# The repo's top folder: this file sits directly in it.
REPO_ROOT = Path(__file__).resolve().parent
VENV_DIR = REPO_ROOT / ".venv"
VENV_PYTHON = VENV_DIR / "bin" / "python"
REQUIREMENTS_FILE = REPO_ROOT / "requirements.txt"
PROJECT_PACKAGE_NAME = "judge-bottleneck"         # the name pyproject.toml gives this repo's own code

KAGGLE_PYTHON_VERSION = (3, 13)                   # the Python Kaggle runs; change it here if Kaggle moves
OLLAMA_MODELS_DIR = REPO_ROOT / "data" / "ollama"  # Ollama keeps its models here, never in the home folder
OLLAMA_ADDRESS = "http://127.0.0.1:11434"         # where a running Ollama server answers
OLLAMA_START_WAIT_SECONDS = 30                    # how long to wait for the Ollama server to start

# Set just before the restart, so the second run of this file knows the first half is done.
RESTARTED_VARIABLE = "PROJECT_SET_UP_RESTARTED"


def run_command(command: list) -> None:
    """Run a command from the repo folder, showing its output; stop the set-up if it fails."""
    print("  running: " + " ".join(str(part) for part in command), flush=True)
    finished = subprocess.run(command, cwd=REPO_ROOT)
    if finished.returncode != 0:
        print(f"WARNING  that command failed (exit code {finished.returncode}); fix the error above and rerun.")
        sys.exit(1)


def version_text(version: tuple) -> str:
    """A Python version as people write it: (3, 13) -> "3.13"."""
    return ".".join(str(part) for part in version)


# ---------------------------------------------------------------------------
# First half: only Python's own library
# ---------------------------------------------------------------------------

def find_kaggle_python() -> str:
    """The path of a Python at Kaggle's version: the one running this file, or one on the PATH ("" if none)."""
    if tuple(sys.version_info[:2]) == KAGGLE_PYTHON_VERSION:
        return sys.executable
    # Here we look for Homebrew-style names such as "python3.13".
    found_path = shutil.which("python" + version_text(KAGGLE_PYTHON_VERSION))
    return found_path or ""


def venv_python_version() -> tuple:
    """The (major, minor) Python version the existing .venv was made with."""
    version_check = subprocess.run(
        [VENV_PYTHON, "-c", "import sys; print(sys.version_info[0], sys.version_info[1])"],
        capture_output=True, text=True, check=True,
    )
    major, minor = version_check.stdout.split()
    return (int(major), int(minor))


def make_virtual_environment() -> None:
    """Create .venv with Kaggle's Python version if it isn't there; stop if an old one has the wrong version."""
    wanted_version = version_text(KAGGLE_PYTHON_VERSION)
    if VENV_PYTHON.exists():
        existing_version = venv_python_version()
        if existing_version == KAGGLE_PYTHON_VERSION:
            print(f"OK       .venv already exists, with Python {wanted_version}")
            return
        # Here we don't delete the old .venv ourselves: deleting is the owner's call.
        print(f"WARNING  .venv uses Python {version_text(existing_version)}, but Kaggle uses {wanted_version}.\n"
              f"         Delete the .venv folder, then run this file again.")
        sys.exit(1)

    python_path = find_kaggle_python()
    if not python_path:
        print(f"WARNING  the project uses Python {wanted_version} (Kaggle's version), and none was found.\n"
              f"         Install it with:  brew install python@{wanted_version}\n"
              f"         then run this file again.")
        sys.exit(1)
    print(f"MAKING   .venv with {python_path}")
    run_command([python_path, "-m", "venv", VENV_DIR])


def library_name(requirement: str) -> str:
    """The plain name of a pinned library, in pip's spelling: "bm25s[core]==0.3.11" -> "bm25s"."""
    name = requirement.split("==")[0].split("[")[0].strip()
    # Here we match pip's spelling: case, "_" and "." don't matter in library names.
    return name.lower().replace("_", "-").replace(".", "-")


def wanted_library_versions() -> dict:
    """Every pinned library in requirements.txt, as {name in pip's spelling: version} (comments skipped)."""
    wanted_versions = {}
    for line in REQUIREMENTS_FILE.read_text().splitlines():
        requirement = line.split("#")[0].strip()
        # Here we skip the `-e .` line: it is this repo's own code, with no pinned version.
        if not requirement or requirement == "-e .":
            continue
        wanted_versions[library_name(requirement)] = requirement.split("==")[1].strip()
    return wanted_versions


def installed_library_versions() -> dict:
    """Every library installed in .venv, as {name in pip's spelling: version}."""
    pip_list = subprocess.run([VENV_PYTHON, "-m", "pip", "list", "--format=json"],
                              capture_output=True, text=True, check=True)
    installed_libraries = json.loads(pip_list.stdout)
    return {library_name(library["name"]): library["version"] for library in installed_libraries}


def project_code_is_installed(installed_versions: dict) -> bool:
    """True if this repo's own code (the `-e .` line) is installed in .venv, so `import pipeline` works anywhere."""
    return library_name(PROJECT_PACKAGE_NAME) in installed_versions


def install_libraries() -> None:
    """Install the exact library versions from requirements.txt into .venv, if any are missing or different."""
    wanted_versions = wanted_library_versions()
    installed_versions = installed_library_versions()
    missing_names = [name for name, version in wanted_versions.items() if installed_versions.get(name) != version]
    if not project_code_is_installed(installed_versions):
        missing_names.append(PROJECT_PACKAGE_NAME)
    if not missing_names:
        print("OK       every library in requirements.txt is installed")
        return
    print(f"INSTALL  {len(missing_names)} libraries missing or at another version: {', '.join(missing_names)}")
    # Here --no-cache-dir stops pip from keeping a copy of every download in the home folder.
    run_command([VENV_PYTHON, "-m", "pip", "install", "--no-cache-dir", "-r", REQUIREMENTS_FILE])


def running_in_venv() -> bool:
    """True if this file is being run by the .venv's Python."""
    return Path(sys.prefix).resolve() == VENV_DIR.resolve()


def first_half() -> None:
    """Get Python, .venv and the libraries in place, then restart this file with the .venv's Python."""
    print(f"Setting up {REPO_ROOT.name}")
    make_virtual_environment()
    install_libraries()
    if not running_in_venv():
        os.environ[RESTARTED_VARIABLE] = "yes"
        # Here we swap this process for the .venv's Python running this same file.
        os.execv(VENV_PYTHON, [str(VENV_PYTHON), __file__])


# ---------------------------------------------------------------------------
# Second half: runs with the .venv's Python, so it can use the project's code
# ---------------------------------------------------------------------------

def make_question_sets(config: dict) -> list:
    """Build the Pilot and Test sets if either file is missing. Returns the steps left by hand (none here)."""
    from pipeline import dataset
    pilot_file = dataset.questions_dir() / "pilot.jsonl"
    test_file = dataset.questions_dir() / "test.jsonl"
    if pilot_file.exists() and test_file.exists():
        print("OK       the Pilot and Test sets are in data/hotpotqa/")
        return []
    print("MAKING   the Pilot and Test sets (downloads HotpotQA dev, ~28 MB, into data/hotpotqa/)")
    # Here the project's own sampling code picks the same questions on every machine.
    summary = dataset.build()
    print(f"OK       {summary['pilot']} Pilot and {summary['test']} Test questions written")
    return []


def check_search_index() -> list:
    """Check the BM25 search index is where the code looks; if not, return the steps to get it from Kaggle."""
    from pipeline.retriever import index_dir
    index_folder = index_dir()
    # Here we look for the settings file every saved bm25s index has.
    if (index_folder / "params.index.json").exists():
        print(f"OK       the search index is in {index_folder.relative_to(REPO_ROOT)}/")
        return []
    print("WARNING  the search index is missing. It needs your Kaggle account, so it's a step for you.")
    return [
        "Download the search index (~2.8 GB):\n"
        "    1. On kaggle.com, open Your Work > Datasets > hotpotqa-bm25-index, and click Download.\n"
        "    2. Unzip it so its files (params.index.json, corpus.jsonl, ...) sit directly in\n"
        f"       {index_folder}\n"
        "    3. Run this file again; it checks the index is found."
    ]


def ollama_model_is_downloaded(model_name: str) -> bool:
    """True if data/ollama/ already holds this model, e.g. "qwen3:8b"."""
    # Here a name without a tag means Ollama's "latest" tag, as `ollama pull` treats it.
    name, _, tag = model_name.partition(":")
    manifest_file = OLLAMA_MODELS_DIR / "manifests" / "registry.ollama.ai" / "library" / name / (tag or "latest")
    return manifest_file.exists()


def ollama_server_is_running() -> bool:
    """True if an Ollama server is already answering on this machine."""
    try:
        urllib.request.urlopen(OLLAMA_ADDRESS + "/api/version", timeout=2)
        return True
    except OSError:
        return False


def start_ollama_server() -> subprocess.Popen:
    """Start an Ollama server that keeps its models in data/ollama/, and wait until it answers."""
    OLLAMA_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    # Here OLLAMA_MODELS points the server at the repo, so downloads never land in the home folder.
    server_settings = dict(os.environ, OLLAMA_MODELS=str(OLLAMA_MODELS_DIR))
    server = subprocess.Popen(["ollama", "serve"], env=server_settings,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    give_up_at = time.time() + OLLAMA_START_WAIT_SECONDS
    while time.time() < give_up_at:
        if ollama_server_is_running():
            return server
        time.sleep(1)
    server.terminate()
    print(f"WARNING  the Ollama server didn't start within {OLLAMA_START_WAIT_SECONDS} seconds; rerun this file.")
    sys.exit(1)


def stop_ollama_server(server: subprocess.Popen) -> None:
    """Stop the Ollama server this script started (and no other)."""
    server.terminate()
    server.wait()


def get_ollama_models(model_names: list) -> list:
    """Download every model in model_names into data/ollama/ if it isn't there. Returns the steps left by hand."""
    if not shutil.which("ollama"):
        print("WARNING  Ollama isn't installed. It's system software, so it's a step for you.")
        return [
            "Install Ollama, the program that runs the model on the Mac:\n"
            "    1. In a terminal, run:  brew install ollama\n"
            "       (no Homebrew? download it from https://ollama.com/download instead)\n"
            "    2. Run this file again; it then downloads the model into data/ollama/."
        ]
    missing_models = [name for name in model_names if not ollama_model_is_downloaded(name)]
    if not missing_models:
        print(f"OK       Ollama is installed and {', '.join(model_names)} is in data/ollama/")
        return []
    if ollama_server_is_running():
        # Here a server someone else started may keep its models in the home folder, so we don't download through it.
        print("WARNING  an Ollama server is already running, maybe storing models in your home folder.")
        return ["Quit the running Ollama (the menu-bar app, or Ctrl+C in its terminal), then run this file again."]

    print(f"DOWNLOAD the Ollama model(s) {', '.join(missing_models)} into data/ollama/")
    server = start_ollama_server()
    try:
        for model_name in missing_models:
            run_command(["ollama", "pull", model_name])
    finally:
        # Here we close the server this script opened, even if a download failed.
        stop_ollama_server(server)
    return []


def print_steps_left(steps_left: list) -> None:
    """List what the user still has to do by hand, or say everything is ready."""
    print(f"\nSteps left to do by hand: {len(steps_left)}")
    for step_number, step in enumerate(steps_left, start=1):
        print(f"\n{step_number}. {step}")
    if not steps_left:
        print("Everything is set up. In VS Code, select .venv as the Python interpreter.")


def second_half() -> None:
    """Get the question sets, the search index and the models in place, then list what's left by hand."""
    from pipeline import load_config
    config = load_config()
    # Here the models to download come from config.yaml, so editing the config is all it takes to change them.
    model_names = sorted(set(config["models"].values()))
    steps_left = []
    steps_left += make_question_sets(config)
    steps_left += check_search_index()
    steps_left += get_ollama_models(model_names)
    print_steps_left(steps_left)


if __name__ == "__main__":
    # Here the first half is skipped after the restart, because it has already run.
    if os.environ.get(RESTARTED_VARIABLE) != "yes":
        first_half()
    second_half()
    print("To turn the .venv on and off in a terminal:\n"
        f"    on:   cd \"{REPO_ROOT}\" && source .venv/bin/activate\n"
        "    off:  deactivate\n"
        "In VS Code, pick it once with Cmd+Shift+P > Python: Select Interpreter > .venv;\n"
        "every Run and Debug then uses it.")
    