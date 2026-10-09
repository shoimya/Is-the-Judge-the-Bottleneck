"""Get this machine ready to run the project. The first script to run after cloning the repo. (T01)

    python3 setup_project.py        (or press Run / Debug on this file)

For everything the project needs, it first checks whether it already exists and only installs what is missing:
Python 3.12 or newer, the project folders, pytest's settings file, a virtual environment (.venv, on the Mac), and
the libraries in requirements.txt (plus requirements-gpu.txt on a GPU machine, once that file exists). It never deletes anything,
so it is safe to run again. Everything it did is logged in runs/setup/<date>/setup_<time>.log.

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

# Settings for this script.
PROJECT_ROOT = Path(__file__).resolve().parent
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
# pytest's settings: tests live in test/, and they may import the scripts at the top of the repo.
PYTEST_SETTINGS = "[pytest]\ntestpaths = test\npythonpath = .\n"


# --- The setup log ---

def note(log_lines: list[str], message: str) -> None:
    """Print a step's result and keep it for the log file."""
    print(message, flush=True)   # flush: show it now, in order with pip's output (matters on Kaggle)
    log_lines.append(message)


def warn(log_lines: list[str], message: str) -> None:
    """Print a warning in capitals and keep it for the log file, so it can't be missed."""
    note(log_lines, f"WARNING: {message}")


def write_setup_log(log_lines: list[str], logs_dir: Path = SETUP_LOGS_DIR) -> Path:
    """Save the log to <logs_dir>/<date>/setup_<time>.log, so every run of setup gets its own file."""
    started_at = datetime.now()
    log_folder = logs_dir / started_at.strftime("%Y-%m-%d")
    log_folder.mkdir(parents=True, exist_ok=True)
    log_file = log_folder / f"setup_{started_at.strftime('%H%M%S')}.log"
    log_file.write_text("\n".join(log_lines) + "\n")
    return log_file


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
    # Here we point pip's download cache into the project, so nothing lands in the home folder.
    pip_environment = {**os.environ, "PIP_CACHE_DIR": str(PIP_DOWNLOAD_CACHE_DIR)}
    subprocess.run([python_command, "-m", "pip", "install", *libraries_to_install], check=True, env=pip_environment)


def install_gpu_requirements(python_command: str, machine: dict, log_lines: list[str]) -> None:
    """Install the GPU-only libraries, but only on a GPU machine and only once requirements-gpu.txt exists."""
    if not machine["has_gpu"]:
        note(log_lines, "GPU libraries: skipped (no GPU on this machine)")
    elif not GPU_REQUIREMENTS_FILE.exists():
        note(log_lines, "GPU libraries: none yet (requirements-gpu.txt arrives in T04)")
    else:
        install_missing_requirements(python_command, GPU_REQUIREMENTS_FILE, log_lines)


# --- Step 7: final check ---

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
        print("Log saved in", write_setup_log(setup_log_lines))
