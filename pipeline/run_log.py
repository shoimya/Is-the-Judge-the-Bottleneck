"""The run log: every run that uses the pipeline is saved here, so nothing has to be rerun to be studied. (T05)

    runs/<scenario folder>/<run id>/meta.json          what was run: scenario, questions, models, config, git commit, times
    runs/<scenario folder>/<run id>/questions.jsonl    one line per question, written as soon as that question finishes

Only runs that use the pipeline are logged (run.py, loop.py's demo). Tests, setup/ scripts and `run.py --check`
write nothing here. A run that stops partway (e.g. Kaggle ends the session) continues where it stopped when the
same scenario is run again on the same questions with the same config.

    Press Debug on this file to print every scenario's log folder.
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path

from pipeline.answerer import PROMPT_VERSION as ANSWER_PROMPT_VERSION
from pipeline.config import REPO_ROOT, RUNS_DIR
from pipeline.llm import model_for

META_FILE_NAME = "meta.json"
QUESTIONS_FILE_NAME = "questions.jsonl"

# The folder each scenario's logs go in, by the scenario names run.py accepts (see pipeline/loop.py).
SCENARIO_FOLDER_NAMES = {
    "closed-book": "closed-book",
    "single-turn": "single-turn",
    "always-loop": "always-loop",
    "coin-flip": "coin-flip",
    "A": "A_llm-stops_llm-steers",
    "B": "B_llm-stops_oracle-steers",
    "C": "C_oracle-stops_llm-steers",
    "D": "D_oracle-stops_oracle-steers",
}


# --- Small helpers ---

def current_time() -> str:
    """The time now, to the second, e.g. 2026-10-14T15:30:12."""
    return datetime.now().isoformat(timespec="seconds")


def git_output(*git_arguments: str) -> str:
    """Run a git command in the repo and return what it prints ("unknown" if git isn't available)."""
    try:
        return subprocess.run(["git", "-C", str(REPO_ROOT), *git_arguments],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


# --- Folders ---

def scenario_log_folder(setting_name: str, runs_dir: Path = RUNS_DIR) -> Path:
    """The folder a scenario's logs go in; created the first time that scenario runs."""
    log_folder = runs_dir / SCENARIO_FOLDER_NAMES[setting_name]
    log_folder.mkdir(parents=True, exist_ok=True)
    return log_folder


def make_new_run_folder(log_folder: Path) -> Path:
    """A new, empty folder for one run, named by its start time (with -2, -3... if two runs start in the same second)."""
    run_id = datetime.now().strftime("%Y-%m-%dT%H%M%S")
    run_folder = log_folder / run_id
    copy_number = 2
    while run_folder.exists():
        run_folder = log_folder / f"{run_id}-{copy_number}"
        copy_number += 1
    run_folder.mkdir()
    return run_folder


# --- meta.json ---

def build_meta(run_folder: Path, setting_name: str, question_set_name: str, question_ids: list[str], config: dict) -> dict:
    """Everything needed to know later exactly what this run was."""
    return {
        "run_id": run_folder.name,
        "scenario": setting_name,
        "dataset": "hotpotqa",
        "question_set": question_set_name,
        "question_ids": question_ids,
        "started_at": current_time(),
        "finished_at": None,   # filled in when the last question is done
        "backend": config["backend"],
        "models": {role: model_for(role, config) for role in ("judge", "rewriter", "answerer")},
        "prompt_versions": {"answerer": ANSWER_PROMPT_VERSION},   # the Judge and Rewriter are added in T06
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_uncommitted_changes": git_output("status", "--porcelain") not in ("", "unknown"),
        "config": config,
    }


def write_meta(run_folder: Path, meta: dict) -> None:
    """Save the run's description as meta.json (readable, indented)."""
    (run_folder / META_FILE_NAME).write_text(json.dumps(meta, indent=2) + "\n")


def read_meta(run_folder: Path) -> dict:
    """Read a run's meta.json."""
    return json.loads((run_folder / META_FILE_NAME).read_text())


def mark_run_finished(run_folder: Path) -> None:
    """Record the finish time in meta.json, so this run is never resumed."""
    meta = read_meta(run_folder)
    meta["finished_at"] = current_time()
    write_meta(run_folder, meta)


# --- questions.jsonl ---

def append_question_record(run_folder: Path, record: dict) -> None:
    """Add one finished question to questions.jsonl, as one line of JSON."""
    with open(run_folder / QUESTIONS_FILE_NAME, "a") as questions_file:
        questions_file.write(json.dumps(record) + "\n")


def read_question_records(run_folder: Path) -> list[dict]:
    """Read every question record in a run, in the order they were written."""
    questions_file_path = run_folder / QUESTIONS_FILE_NAME
    if not questions_file_path.exists():
        return []
    return [json.loads(line) for line in questions_file_path.read_text().splitlines()]


def drop_half_written_last_line(run_folder: Path) -> None:
    """Remove a last line cut off mid-write by a crash, so the file reads cleanly and resuming can add after it."""
    questions_file_path = run_folder / QUESTIONS_FILE_NAME
    if not questions_file_path.exists():
        return
    file_text = questions_file_path.read_text()
    if file_text and not file_text.endswith("\n"):
        # Here we keep everything up to the last complete line (each complete line ends with a line break).
        questions_file_path.write_text(file_text[:file_text.rfind("\n") + 1])


# --- Starting, resuming and running ---

def find_unfinished_run(log_folder: Path, setting_name: str, question_set_name: str,
                        question_ids: list[str], config: dict) -> Path | None:
    """The newest run of this scenario that stopped partway on the same questions with the same config, if any."""
    for run_folder in sorted(log_folder.iterdir(), reverse=True):
        if not (run_folder / META_FILE_NAME).exists():
            continue
        meta = read_meta(run_folder)
        same_run = (meta["finished_at"] is None and meta["scenario"] == setting_name
                    and meta["question_set"] == question_set_name and meta["question_ids"] == question_ids
                    and meta["config"] == config)
        if same_run:
            return run_folder
    return None


def start_or_resume_run(setting_name: str, question_set_name: str, question_ids: list[str],
                        config: dict, runs_dir: Path = RUNS_DIR) -> Path:
    """Continue an unfinished matching run, or start a new one; return its folder."""
    log_folder = scenario_log_folder(setting_name, runs_dir)
    unfinished_run = find_unfinished_run(log_folder, setting_name, question_set_name, question_ids, config)
    if unfinished_run is not None:
        drop_half_written_last_line(unfinished_run)
        return unfinished_run
    run_folder = make_new_run_folder(log_folder)
    write_meta(run_folder, build_meta(run_folder, setting_name, question_set_name, question_ids, config))
    return run_folder


def run_scenario(setting_name: str, scenario, questions: list[dict], question_set_name: str, config: dict,
                 retriever, runs_dir: Path = RUNS_DIR) -> Path:
    """Run one scenario on a list of questions, logging each question as it finishes; return the run's folder.

    `scenario` is the function from pipeline/loop.py that answers one question.
    """
    question_ids = [question["id"] for question in questions]
    run_folder = start_or_resume_run(setting_name, question_set_name, question_ids, config, runs_dir)
    # Here we skip questions an earlier, interrupted attempt already finished.
    already_done = {record["question_id"] for record in read_question_records(run_folder)}
    if already_done:
        print(f"Resuming {run_folder}: {len(already_done)} of {len(questions)} questions already done.")

    for question_number, question in enumerate(questions, start=1):
        if question["id"] in already_done:
            continue
        started_at = current_time()
        record = scenario(question, retriever, config)   # <- step into this line to follow one question
        # Here we put the start and finish times right after the question, then save the record straight away.
        timed_record = {**{key: record[key] for key in ("question_id", "question", "gold_answer", "gold_titles")},
                        "started_at": started_at, "finished_at": current_time(), **record}
        append_question_record(run_folder, timed_record)
        print(f"[{question_number}/{len(questions)}] {question['id']}: {record['final_answer']!r}")

    mark_run_finished(run_folder)
    return run_folder


if __name__ == "__main__":
    for setting_name, folder_name in SCENARIO_FOLDER_NAMES.items():
        print(f"{setting_name:<12} -> {RUNS_DIR / folder_name}")
