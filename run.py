"""The launcher: check the setup, or run a scenario on one question or a whole question set. Every run is logged.

    python run.py --check                                   is this machine ready? (T01)
    python run.py --setting single-turn                     the first Pilot question
    python run.py --setting closed-book --question <id>     a chosen Pilot question
    python run.py --setting single-turn --set pilot         all 100 Pilot questions
    python run.py --setting single-turn --set test --backend vllm     all Test questions, on Kaggle

Scenarios: closed-book, single-turn (working now); always-loop, coin-flip, A, B, C, D (T08).
Logs go to runs/<scenario>/<run id>/ (see pipeline/run_log.py). Running the same command again after a crash
continues where it stopped. To follow the code, put breakpoints in pipeline/loop.py and press Debug on this file.
"""

import argparse
import platform
import sys

from pipeline.config import RUNS_DIR, load_config
from pipeline.dataset import find_question, load_question_set
from pipeline.llm import START_OLLAMA_HINT, ollama_is_running, shut_down_vllm_engines
from pipeline.loop import SCENARIOS
from pipeline.retriever import Retriever
from pipeline.run_log import read_question_records, run_scenario


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


def read_arguments() -> argparse.Namespace:
    """Read the command-line options."""
    argument_parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    argument_parser.add_argument("--check", action="store_true", help="check the setup and exit")
    argument_parser.add_argument("--setting", choices=list(SCENARIOS), default="single-turn",
                                 help="which scenario to run (default: single-turn)")
    argument_parser.add_argument("--question", help="id of one question to run (default: the set's first question)")
    argument_parser.add_argument("--set", choices=["pilot", "test"], help="run every question in this set")
    argument_parser.add_argument("--backend", choices=["ollama", "vllm"], help="use this instead of config.yaml's backend")
    return argument_parser.parse_args()


if __name__ == "__main__":
    arguments = read_arguments()
    if arguments.check:
        check_setup()
    else:
        config = load_config()
        if arguments.backend:
            config["backend"] = arguments.backend
        question_set_name = arguments.set or "pilot"
        questions = choose_questions(question_set_name, arguments.question, whole_set=bool(arguments.set))
        run_and_log(arguments.setting, questions, question_set_name, config)
        shut_down_vllm_engines()   # here we free the GPU on Kaggle, so the program can exit
