"""Get this machine ready to run the project. The first script to run after cloning the repo. (T01)

    python3 setup_project.py        (or press Run / Debug on this file)

For everything the project needs, it first checks whether it already exists and only installs what is missing:
Python 3.12 or newer, the project folders, pytest's settings file, a virtual environment (.venv, on the Mac),
the libraries in requirements.txt (plus requirements-gpu.txt on a GPU machine, once that file exists), the
project's own pipeline/ folder as an installed package, the Pilot set and Test set (made by
pipeline/question_sets.py), and the search index (linked on Kaggle; elsewhere it explains how to download it).
It never deletes anything, so it is safe to run again. Everything it did is logged in runs/setup/<date>/.

It uses only Python's built-in modules and starts even on the Mac's built-in Python 3.9, because nothing is
installed yet when it runs.
"""

from __future__ import annotations   # lets the type hints below work on the Mac's built-in Python 3.9

import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# pipeline/shared.py uses only built-in modules too, so this script can import it before anything is installed.
from pipeline.shared import INDEX_DIR, PROJECT_ROOT, note, question_file_path, warn, write_run_log

# Settings for this script.
OLDEST_SUPPORTED_PYTHON = (3, 12)
PYTHON_COMMANDS_TO_TRY = ["python3.13", "python3.12"]   # newest first
HOMEBREW_PYTHON_PACKAGE = "python@3.12"
PROJECT_FOLDERS = ["data", "data/cache", "runs", "results"]
VIRTUAL_ENVIRONMENT_DIR = PROJECT_ROOT / ".venv"
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"
GPU_REQUIREMENTS_FILE = PROJECT_ROOT / "requirements-gpu.txt"
PIP_DOWNLOAD_CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "pip"
SETUP_LOGS_DIR = PROJECT_ROOT / "runs" / "setup"
PYTEST_SETTINGS_FILE = PROJECT_ROOT / "pytest.ini"
# pytest's settings: tests live in test/, and they may import the scripts at the top of the repo
# (setup_project.py and build_index.py; pipeline/ is found because it's installed).
PYTEST_SETTINGS = "[pytest]\ntestpaths = test\npythonpath = .\n"
# The project's own code: pipeline/ is installed as a package, so every file finds it wherever it's run from.
PIPELINE_PACKAGE_DIR = PROJECT_ROOT / "pipeline"
# T02: the script that makes the Pilot set and Test set if they're missing.
QUESTION_SETS_SCRIPT = PIPELINE_PACKAGE_DIR / "question_sets.py"
# T03: the Kaggle dataset that holds a copy of the search index (attached under /kaggle/input/).
INDEX_KAGGLE_DATASET = "hotpotqa-bm25-index"
KAGGLE_INPUT_DIR = Path("/kaggle/input")


# --- Step 1: Python ---

def is_supported_python(version: tuple[int, int]) -> bool:
    """True if a Python version (major, minor) is new enough for the project."""
    return version >= OLDEST_SUPPORTED_PYTHON


def python_version_of(python_command: str) -> tuple[int, int]:
    """Ask a Python program which version it is, as (major, minor)."""
    version_text = subprocess.run([python_command, "-c", "import sys; print(sys.version_info[0], sys.version_info[1])"],
                                  capture_output=True, text=True, check=True).stdout
    major, minor = version_text.split()
    return int(major), int(minor)


def install_python_with_homebrew(log_lines: list[str]) -> str:
    """Install Python 3.12 with Homebrew (Mac only) and return the command that runs it."""
    if shutil.which("brew") is None:
        raise SystemExit("Python 3.12 or newer is needed and Homebrew isn't installed to get it. "
                         "Install Homebrew from https://brew.sh, then run this script again.")
    note(log_lines, f"Installing {HOMEBREW_PYTHON_PACKAGE} with Homebrew ...")
    subprocess.run(["brew", "install", HOMEBREW_PYTHON_PACKAGE], check=True)
    return shutil.which("python3.12")


def find_project_python(log_lines: list[str]) -> str:
    """The Python the project will use: this one if new enough, else an installed newer one, else install one."""
    current_version = sys.version_info[:2]
    if is_supported_python(current_version):
        note(log_lines, f"Python: using {sys.executable} ({current_version[0]}.{current_version[1]}), already installed")
        return sys.executable

    # Here we look for a newer Python that is already installed before installing anything.
    for python_command in PYTHON_COMMANDS_TO_TRY:
        python_path = shutil.which(python_command)
        if python_path is not None:
            note(log_lines, f"Python: using {python_path}, already installed "
                            f"(this script was started with {current_version[0]}.{current_version[1]})")
            return python_path

    if platform.system() != "Darwin":
        raise SystemExit("Python 3.12 or newer is needed. Install it, then run this script again.")
    return install_python_with_homebrew(log_lines)


# --- Step 2: the machine ---

def detect_machine() -> dict:
    """Which machine this is (Kaggle or this computer) and whether it has an NVIDIA GPU."""
    return {
        "name": "kaggle" if "KAGGLE_KERNEL_RUN_TYPE" in os.environ else "local",
        "system": platform.system(),
        "has_gpu": shutil.which("nvidia-smi") is not None,   # this tool only exists where there is a GPU
    }


# --- Step 3: folders ---

def create_project_folders(project_root: Path = PROJECT_ROOT) -> list[Path]:
    """Create any project folder that doesn't exist yet; return the ones created."""
    created_folders = []
    for folder_name in PROJECT_FOLDERS:
        folder = project_root / folder_name
        if not folder.is_dir():
            folder.mkdir(parents=True)
            created_folders.append(folder)
    return created_folders


def folder_is_writable(folder: Path) -> bool:
    """True if a file can be written in this folder (checked with a small file this function then removes)."""
    test_file = folder / ".write_test"
    try:
        test_file.write_text("ok")
        test_file.unlink()
        return True
    except OSError:
        return False


def create_pytest_settings(log_lines: list[str], settings_file: Path = PYTEST_SETTINGS_FILE) -> None:
    """Write pytest.ini unless it exists, so `pytest` finds the tests in test/ and they can import the scripts."""
    if settings_file.exists():
        note(log_lines, f"{settings_file.name}: already exists")
        return
    settings_file.write_text(PYTEST_SETTINGS)
    note(log_lines, f"{settings_file.name}: created")


# --- Step 4: the virtual environment ---

def create_virtual_environment(python_command: str, log_lines: list[str]) -> str:
    """Create .venv with the project's Python unless it exists; return the Python inside it."""
    venv_python = VIRTUAL_ENVIRONMENT_DIR / "bin" / "python"
    if venv_python.exists():
        note(log_lines, f".venv: already exists ({VIRTUAL_ENVIRONMENT_DIR})")
    else:
        note(log_lines, f".venv: creating it with {python_command} ...")
        subprocess.run([python_command, "-m", "venv", str(VIRTUAL_ENVIRONMENT_DIR)], check=True)

    venv_version = python_version_of(str(venv_python))
    if not is_supported_python(venv_version):
        warn(log_lines, f".venv runs Python {venv_version[0]}.{venv_version[1]}, older than the project needs. "
                        f"Delete the .venv folder and run this script again.")
    return str(venv_python)


# --- Steps 5 and 6: libraries ---

def read_pinned_requirements(requirements_file: Path) -> dict[str, str]:
    """Read a requirements file as {library name: exact version}, skipping comments and blank lines."""
    pinned_versions = {}
    for line in requirements_file.read_text().splitlines():
        requirement = line.split("#")[0].strip()   # here we drop comments at the end of a line
        if requirement:
            library_name, version = requirement.split("==")
            pinned_versions[library_name.strip()] = version.strip()
    return pinned_versions


def installed_version(python_command: str, library_name: str) -> str | None:
    """The version of a library installed for that Python, or None if it isn't installed."""
    name_without_extras = library_name.split("[")[0]   # e.g. "bm25s[core]" is installed as "bm25s"
    version_check = subprocess.run(
        [python_command, "-c", f"import importlib.metadata as metadata; print(metadata.version('{name_without_extras}'))"],
        capture_output=True, text=True)
    return version_check.stdout.strip() if version_check.returncode == 0 else None


def missing_requirements(python_command: str, requirements_file: Path) -> list[str]:
    """The libraries in a requirements file that aren't installed for that Python, or are at another version."""
    missing_libraries = []
    for library_name, wanted_version in read_pinned_requirements(requirements_file).items():
        if installed_version(python_command, library_name) != wanted_version:
            missing_libraries.append(f"{library_name}=={wanted_version}")
    return missing_libraries


def install_missing_requirements(python_command: str, requirements_file: Path, log_lines: list[str]) -> None:
    """Install only the libraries from a requirements file that are missing, keeping pip's downloads in data/cache."""
    libraries_to_install = missing_requirements(python_command, requirements_file)
    if not libraries_to_install:
        note(log_lines, f"{requirements_file.name}: everything already installed")
        return

    note(log_lines, f"{requirements_file.name}: installing {', '.join(libraries_to_install)} ...")
    run_pip_install(python_command, libraries_to_install)


def run_pip_install(python_command: str, pip_arguments: list[str]) -> None:
    """Run `pip install` with those arguments in that Python, keeping pip's downloads inside the project."""
    # Here we point pip's download cache into the project, so nothing lands in the home folder.
    pip_environment = {**os.environ, "PIP_CACHE_DIR": str(PIP_DOWNLOAD_CACHE_DIR)}
    subprocess.run([python_command, "-m", "pip", "install", *pip_arguments], check=True, env=pip_environment)


def install_gpu_requirements(python_command: str, machine: dict, log_lines: list[str]) -> None:
    """Install the GPU-only libraries, but only on a GPU machine and only once requirements-gpu.txt exists."""
    if not machine["has_gpu"]:
        note(log_lines, "GPU libraries: skipped (no GPU on this machine)")
    elif not GPU_REQUIREMENTS_FILE.exists():
        note(log_lines, "GPU libraries: none yet (requirements-gpu.txt arrives in T04)")
    else:
        install_missing_requirements(python_command, GPU_REQUIREMENTS_FILE, log_lines)


def where_python_finds_pipeline(python_command: str) -> str | None:
    """The folder that Python imports `pipeline` from, or None if it can't find it."""
    # We run the check from the root folder "/", so Python can't find pipeline/ just because we're standing next to it.
    find_pipeline = "import pipeline, os; print(os.path.dirname(pipeline.__file__))"
    check = subprocess.run([python_command, "-c", find_pipeline], capture_output=True, text=True, cwd="/")
    return check.stdout.strip() if check.returncode == 0 else None


def install_pipeline_package(python_command: str, log_lines: list[str]) -> None:
    """Install pipeline/ as a package (in editable mode), so every file can import it from any folder."""
    if where_python_finds_pipeline(python_command) == str(PIPELINE_PACKAGE_DIR):
        note(log_lines, "pipeline package: already installed")
        return

    note(log_lines, "pipeline package: installing it from pyproject.toml ...")
    # "Editable" (-e) means Python reads the files in pipeline/ directly, so edits work without reinstalling.
    run_pip_install(python_command, ["-e", str(PROJECT_ROOT)])


# --- Step 7: data ---

def build_question_sets_if_missing(python_command: str, log_lines: list[str]) -> None:
    """Make the Pilot set and Test set with question_sets.py, unless both question files already exist. (T02)"""
    question_set_files = [question_file_path("pilot"), question_file_path("test")]
    if all(question_file.exists() for question_file in question_set_files):
        note(log_lines, "Question sets: already exist (data/hotpotqa/)")
        return

    note(log_lines, "Question sets: making them with pipeline/question_sets.py ...")
    # Here we run the script with the project's Python, because it needs libraries this script can't import.
    subprocess.run([python_command, str(QUESTION_SETS_SCRIPT)], check=True)


def find_attached_kaggle_index() -> Path | None:
    """On Kaggle: the folder of the attached search index dataset, or None if it isn't attached."""
    # Kaggle puts attached datasets somewhere under /kaggle/input/, so we look for the index's settings file.
    for settings_file in KAGGLE_INPUT_DIR.glob("**/params.index.json"):
        return settings_file.parent
    return None


def explain_how_to_get_the_index(machine: dict, log_lines: list[str]) -> None:
    """Warn that the search index is missing, and say how to get it on this machine."""
    warn(log_lines, f"the search index isn't in {INDEX_DIR.relative_to(PROJECT_ROOT)}/, so search won't work yet.")
    if machine["name"] == "kaggle":
        note(log_lines, f"  On Kaggle: Add Input -> Your Datasets -> {INDEX_KAGGLE_DATASET}, then run this script again.")
        note(log_lines, "  If that dataset is lost, `python build_index.py` rebuilds it (about 12 minutes).")
    else:
        note(log_lines, f"  1. Download the Kaggle dataset {INDEX_KAGGLE_DATASET} (about 2.8 GB). It's private: ask the")
        note(log_lines, "     repo owner to share it with your Kaggle account. On its Kaggle page, click Download.")
        note(log_lines, f"  2. Unzip it so its 7 files sit directly in {INDEX_DIR.relative_to(PROJECT_ROOT)}/")
        note(log_lines, "  3. Run this script again: it will find the index.")


def get_search_index(machine: dict, log_lines: list[str]) -> None:
    """Make sure the search index is in data/wiki/bm25_index/: link Kaggle's attached copy, or explain how to get it. (T03)"""
    if (INDEX_DIR / "params.index.json").exists():
        note(log_lines, f"Search index: already in {INDEX_DIR.relative_to(PROJECT_ROOT)}/")
        return

    attached_index_dir = find_attached_kaggle_index() if machine["name"] == "kaggle" else None
    if attached_index_dir is None:
        explain_how_to_get_the_index(machine, log_lines)
        return

    # Kaggle's attached datasets are read-only, so instead of copying 2.8 GB we make a shortcut (a symlink) to it.
    INDEX_DIR.parent.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.symlink_to(attached_index_dir)
    note(log_lines, f"Search index: linked {INDEX_DIR.relative_to(PROJECT_ROOT)} to the attached dataset {attached_index_dir}")


# --- Step 8: final check ---

def check_setup(python_command: str, machine: dict, log_lines: list[str]) -> None:
    """Confirm every project folder can be written to, record installed versions, and warn about anything off."""
    for folder_name in PROJECT_FOLDERS:
        if not folder_is_writable(PROJECT_ROOT / folder_name):
            warn(log_lines, f"{folder_name}/ can't be written to")
    if machine["name"] == "kaggle" and not machine["has_gpu"]:
        warn(log_lines, "Kaggle without a GPU: set Accelerator to GPU T4 x2 before running the model.")
    for library_name in read_pinned_requirements(REQUIREMENTS_FILE):
        note(log_lines, f"  installed: {library_name} {installed_version(python_command, library_name)}")


# --- All steps, in order ---

def set_up_project(log_lines: list[str]) -> str:
    """Run every setup step in order; return the Python the project now uses."""
    note(log_lines, f"Setup started {datetime.now().isoformat(timespec='seconds')} in {PROJECT_ROOT}")
    project_python = find_project_python(log_lines)

    machine = detect_machine()
    note(log_lines, f"Machine: {machine['name']} ({machine['system']}), GPU: {'yes' if machine['has_gpu'] else 'no'}")

    created_folders = create_project_folders()
    if created_folders:
        created_folder_names = [str(folder.relative_to(PROJECT_ROOT)) for folder in created_folders]
        note(log_lines, f"Folders: created {', '.join(created_folder_names)}")
    else:
        note(log_lines, "Folders: all already exist")
    create_pytest_settings(log_lines)

    # Here we use .venv on this computer; Kaggle starts fresh every session, so it uses Kaggle's own Python.
    if machine["name"] == "local":
        project_python = create_virtual_environment(project_python, log_lines)

    install_missing_requirements(project_python, REQUIREMENTS_FILE, log_lines)
    install_gpu_requirements(project_python, machine, log_lines)
    install_pipeline_package(project_python, log_lines)
    build_question_sets_if_missing(project_python, log_lines)
    get_search_index(machine, log_lines)
    check_setup(project_python, machine, log_lines)
    return project_python


if __name__ == "__main__":
    setup_log_lines = []
    try:
        project_python = set_up_project(setup_log_lines)
        note(setup_log_lines, "Setup OK.")
        if project_python.startswith(str(VIRTUAL_ENVIRONMENT_DIR)):
            note(setup_log_lines, "Next: in VS Code pick the .venv interpreter, or in a terminal run "
                                  "`source .venv/bin/activate`.")
    except BaseException as setup_error:
        # Here we record a failed setup in the log too, then let the error show as usual.
        warn(setup_log_lines, f"setup failed: {setup_error!r}")
        raise
    finally:
        print("Log saved in", write_run_log(setup_log_lines, SETUP_LOGS_DIR, "setup"))
