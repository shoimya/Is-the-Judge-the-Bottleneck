"""What every file in the project shares: the project's folders, and the run log helpers.

    python pipeline/shared.py        (or press Run / Debug on this file)

Only paths used by more than one file live here; a path one file uses sits at the top of that file.
setup_project.py imports this file before anything is installed, so it uses only Python's built-in modules
and must start on the Mac's built-in Python 3.9.
"""

from __future__ import annotations   # lets the type hints below work on the Mac's built-in Python 3.9

from datetime import datetime
from pathlib import Path

# The project's folders.
PROJECT_ROOT = Path(__file__).resolve().parent.parent   # this file is in pipeline/, one folder below the project
QUESTIONS_DIR = PROJECT_ROOT / "data" / "hotpotqa"      # the Pilot set and Test set (T02)
INDEX_DIR = PROJECT_ROOT / "data" / "wiki" / "bm25_index"   # the Wikipedia search index (T03)


def question_file_path(set_name: str) -> Path:
    """Where a question set ("pilot" or "test") is saved, one question per line."""
    return QUESTIONS_DIR / f"{set_name}.jsonl"


# --- The run log ---

def note(log_lines: list[str], message: str) -> None:
    """Print a step's result and keep it for the log file."""
    print(message, flush=True)   # flush: show it now, in order with other programs' output (matters on Kaggle)
    log_lines.append(message)


def warn(log_lines: list[str], message: str) -> None:
    """Print a warning that starts with WARNING and keep it for the log file, so it can't be missed."""
    note(log_lines, f"WARNING: {message}")


def write_run_log(log_lines: list[str], logs_dir: Path, log_name: str) -> Path:
    """Save a log to <logs_dir>/<date>/<log_name>_<time>.log, so every run gets its own file."""
    started_at = datetime.now()
    log_folder = logs_dir / started_at.strftime("%Y-%m-%d")
    log_folder.mkdir(parents=True, exist_ok=True)
    log_file = log_folder / f"{log_name}_{started_at.strftime('%H%M%S')}.log"
    log_file.write_text("\n".join(log_lines) + "\n")
    return log_file


if __name__ == "__main__":
    # A short demo: show the shared folders, and what a note and a warning look like.
    demo_log_lines = []
    note(demo_log_lines, f"Project folder: {PROJECT_ROOT}")
    note(demo_log_lines, f"Pilot set file: {question_file_path('pilot').relative_to(PROJECT_ROOT)}")
    note(demo_log_lines, f"Search index:   {INDEX_DIR.relative_to(PROJECT_ROOT)}")
    warn(demo_log_lines, "this is what a warning looks like in a log")
