"""The launcher: check the setup, or run a scenario on one question or a whole question set. Every run is logged.

    Open this file, choose what to run in the settings just below, and press Run (or Debug).

Scenarios: closed-book, single-turn (working now); always-loop, coin-flip, A, B, C, D (T08).
Logs go to runs/<scenario>/<run id>/ (see pipeline/run_log.py). Running the same settings again after a crash
continues where it stopped. To follow the code, put breakpoints in pipeline/loop.py and press Debug on this file.
"""

import platform
import sys

from pipeline.config import RUNS_DIR, load_config
from pipeline.dataset import find_question, load_question_set
from pipeline.llm import START_OLLAMA_HINT, ollama_is_running, shut_down_vllm_engines
from pipeline.loop import SCENARIOS
from pipeline.retriever import Retriever
from pipeline.run_log import read_question_records, run_scenario

# What pressing Run does. Change these, then press Run.
CHECK_SETUP_ONLY = False      # True: only check this machine is ready (Python, config.yaml, runs/), run nothing
SCENARIO = "single-turn"      # "closed-book" or "single-turn" (working now); "always-loop", "coin-flip", "A"-"D" (T08)
QUESTION_SET = "pilot"        # "pilot" (100 questions) or "test" (1,000; only after the T10 freeze)
QUESTION_ID = None            # one question's id from QUESTION_SET, or None for the set's first question
RUN_WHOLE_SET = False         # True: run every question in QUESTION_SET (QUESTION_ID is then ignored)
BACKEND = None                # None: use config.yaml's backend ("ollama" on the Mac); "vllm" on Kaggle


def check_python_version() -> None:
    """Print the Python version; warn if it is older than the project supports (the Mac runs 3.12, Kaggle 3.13)."""
    print(f"Python   {platform.python_version()} on {platform.system()}")
    if sys.version_info < (3, 12):
        print("WARNING  the project needs Python 3.12 or newer")


def check_runs_folder_is_writable() -> None:
    """Prove run logs can be saved, by writing a small file in runs/ and deleting it again."""
    RUNS_DIR.mkdir(exist_ok=True)
    test_file = RUNS_DIR / ".write_test"
    test_file.write_text("ok")
    test_file.unlink()
    print(f"runs/    writable ({RUNS_DIR.resolve()})")


def check_setup() -> None:
    """Check that this machine is ready: Python version, settings file, and a writable runs/ folder."""
    check_python_version()
    config = load_config()
    print(f"Config   backend={config['backend']} model={config['model']} rounds={config['rounds']}")
    check_runs_folder_is_writable()
    print("Setup OK.")


def choose_questions(question_set_name: str, question_id: str | None, whole_set: bool) -> list[dict]:
    """The questions to run: the whole set, one chosen question, or (by default) the set's first question."""
    question_set = load_question_set(question_set_name)
    if whole_set:
        return question_set
    if question_id:
        return [find_question(question_id, question_set)]
    return question_set[:1]


def print_one_question(record: dict) -> None:
    """Show what happened to one question: what was searched, what came back, and the answer."""
    print("Question:    ", record["question"])
    print("Gold answer: ", record["gold_answer"], "| Gold titles:", record["gold_titles"])
    for single_round in record["rounds"]:
        if single_round["retriever"] is not None:
            retrieved_titles = [paragraph["title"] for paragraph in single_round["retriever"]["retrieved"]]
            print(f"Round {single_round['round']} search:", single_round["retriever"]["query"], "->", retrieved_titles)
    print("Answer:      ", record["final_answer"])


def run_and_log(setting_name: str, questions: list[dict], question_set_name: str, config: dict) -> None:
    """Run one scenario on the chosen questions, logging every question; then say where the log is."""
    if config["backend"] == "ollama" and not ollama_is_running():
        sys.exit(START_OLLAMA_HINT)
    run_folder = run_scenario(setting_name, SCENARIOS[setting_name], questions, question_set_name, config,
                              retriever=Retriever.load())   # <- step into this line to follow the pipeline
    if len(questions) == 1:
        print_one_question(read_question_records(run_folder)[0])
    print("Log saved in", run_folder)


def run_one_scenario(scenario_name: str, question_set_name: str, question_id: str | None,
                     run_whole_set: bool, backend: str | None) -> None:
    """Run one scenario on the chosen questions with the chosen backend, and log it."""
    config = load_config()
    # Here a backend chosen at the top of this file (or by the Kaggle notebook) replaces config.yaml's.
    if backend:
        config["backend"] = backend
    questions = choose_questions(question_set_name, question_id, run_whole_set)
    run_and_log(scenario_name, questions, question_set_name, config)
    shut_down_vllm_engines()   # here we free the GPU on Kaggle, so the program can exit


if __name__ == "__main__":
    if CHECK_SETUP_ONLY:
        check_setup()
    else:
        run_one_scenario(SCENARIO, QUESTION_SET, QUESTION_ID, RUN_WHOLE_SET, BACKEND)
