"""Check this machine has everything the project needs. It only looks: nothing is installed, downloaded or logged.

    Select .venv as the interpreter, open this file and press Run.

Each check prints PASS or FAIL, and every FAIL says what is wrong:
    1. .venv uses the Python version Kaggle uses
    2. every library in requirements.txt is installed at its pinned version, and so is this repo's own code
    3. the search index is in data/wiki/bm25_index/
    4. the HotpotQA Pilot and Test sets are complete and match the committed id lists
If something is missing, run project_set_up.py, then run this file again.
"""

import json
import sys

from project_set_up import (
    KAGGLE_PYTHON_VERSION,
    REPO_ROOT,
    installed_library_versions,
    project_code_is_installed,
    running_in_venv,
    version_text,
    wanted_library_versions,
)

# The files every saved bm25s index has; if one is missing or empty, the index can't load.
INDEX_FILES = [
    "params.index.json",
    "vocab.index.json",
    "data.csc.index.npy",
    "indices.csc.index.npy",
    "indptr.csc.index.npy",
    "corpus.jsonl",
]

# The fields every question line needs.
QUESTION_FIELDS = ["id", "question", "answer", "type", "level", "gold_titles"]


def check_python_version() -> list:
    """Problems with the Python version .venv runs (an empty list means it passes)."""
    running_version = tuple(sys.version_info[:2])
    if running_version == KAGGLE_PYTHON_VERSION:
        return []
    return [f".venv runs Python {version_text(running_version)}, but Kaggle uses "
            f"{version_text(KAGGLE_PYTHON_VERSION)}. Delete .venv and run project_set_up.py."]


def check_libraries() -> list:
    """Problems with the libraries: each one missing from .venv, or at a version other than the pinned one."""
    wanted_versions = wanted_library_versions()
    installed_versions = installed_library_versions()
    problems = []
    for name, wanted_version in wanted_versions.items():
        installed_version = installed_versions.get(name)
        if installed_version is None:
            problems.append(f"{name}: want {wanted_version}, not installed")
        elif installed_version != wanted_version:
            problems.append(f"{name}: want {wanted_version}, found {installed_version}")
    if not project_code_is_installed(installed_versions):
        problems.append("this repo's own code (the `-e .` line) is not installed, so files in pipeline/ can't run")
    return problems


def check_search_index() -> list:
    """Problems with the search index: each file that is missing or empty in the folder the code reads."""
    from pipeline.retriever import index_dir
    index_folder = index_dir()
    problems = []
    for file_name in INDEX_FILES:
        index_file = index_folder / file_name
        if not index_file.exists():
            problems.append(f"{file_name} is missing from {index_folder}")
        elif index_file.stat().st_size == 0:
            problems.append(f"{file_name} is empty in {index_folder}")
    return problems


def questions_file_for(split_name: str):
    """Where a split's questions live: data/hotpotqa/<split>.jsonl."""
    from pipeline.dataset import questions_dir
    return questions_dir() / f"{split_name}.jsonl"


def read_questions(questions_file) -> list:
    """Every question in a .jsonl file, one dict per line."""
    with open(questions_file, encoding="utf-8") as opened_file:
        return [json.loads(line) for line in opened_file if line.strip()]


def read_committed_ids(id_list_file) -> list:
    """The question ids in a committed id list, one per line, in order."""
    lines = id_list_file.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


def count_incomplete_questions(questions: list) -> int:
    """How many questions are missing a needed field or don't have exactly two Gold titles."""
    incomplete_count = 0
    for question in questions:
        has_every_field = all(field in question for field in QUESTION_FIELDS)
        if not has_every_field or len(question["gold_titles"]) != 2:
            incomplete_count += 1
    return incomplete_count


def check_one_split(split_name: str, wanted_size: int, config: dict) -> list:
    """Problems with one split ("pilot" or "test"): missing, wrong size, wrong ids, or incomplete questions."""
    questions_file = questions_file_for(split_name)
    id_list_file = REPO_ROOT / config["paths"]["results_dir"] / "question_ids" / f"{split_name}_ids.txt"
    if not questions_file.exists():
        return [f"{questions_file.name} is missing from {questions_file.parent}"]

    questions = read_questions(questions_file)
    question_ids = [question["id"] for question in questions]
    problems = []
    if len(questions) != wanted_size:
        problems.append(f"{split_name}: want {wanted_size} questions, found {len(questions)}")
    # Here we compare in order too, so a different sample of the same size still fails.
    if question_ids != read_committed_ids(id_list_file):
        problems.append(f"{split_name}: the questions don't match {id_list_file.name}")
    incomplete_count = count_incomplete_questions(questions)
    if incomplete_count:
        problems.append(f"{split_name}: {incomplete_count} questions miss a field or don't have two Gold titles")
    return problems


def check_question_splits(config: dict) -> list:
    """Problems with the HotpotQA Pilot and Test sets, including any question that sits in both."""
    from pipeline.dataset import PILOT_SIZE, TEST_SIZE
    problems = check_one_split("pilot", PILOT_SIZE, config) + check_one_split("test", TEST_SIZE, config)
    if problems:
        return problems
    pilot_ids = {question["id"] for question in read_questions(questions_file_for("pilot"))}
    test_ids = {question["id"] for question in read_questions(questions_file_for("test"))}
    shared_ids = pilot_ids & test_ids
    if shared_ids:
        problems.append(f"{len(shared_ids)} questions are in both the Pilot and Test sets")
    return problems


def report(check_name: str, problems: list) -> bool:
    """Print PASS, or FAIL with each problem underneath. Returns True if the check passed."""
    if not problems:
        print(f"PASS  {check_name}")
        return True
    print(f"FAIL  {check_name}")
    for problem in problems:
        print(f"      - {problem}")
    return False


def run_all_checks() -> None:
    """Run every check in order and print how many passed."""
    if not running_in_venv():
        # Here we stop early: without .venv, the project's libraries can't be imported.
        print("This file needs the .venv's Python. In VS Code: Cmd+Shift+P > Python: Select Interpreter > .venv,\n"
              "then press Run again. (No .venv yet? Run project_set_up.py first.)")
        return

    from pipeline import load_config
    config = load_config()
    results = [
        report("Python version", check_python_version()),
        report("libraries in requirements.txt", check_libraries()),
        report("search index", check_search_index()),
        report("HotpotQA Pilot and Test sets", check_question_splits(config)),
    ]
    passed_count = sum(results)
    print(f"\n{passed_count} of {len(results)} checks passed.")
    if passed_count < len(results):
        print("Run project_set_up.py to fix what's missing, then run this file again.")


if __name__ == "__main__":
    run_all_checks()
