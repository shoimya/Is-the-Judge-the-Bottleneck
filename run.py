"""The launcher: check the setup, or follow one question through one scenario.

    python run.py --check                                   is this machine ready? (T01)
    python run.py --setting single-turn                     first Pilot question, Single-turn
    python run.py --setting closed-book --question <id>     a chosen Pilot question
    python run.py --setting single-turn --backend vllm      on Kaggle

Scenarios: closed-book, single-turn (working now); always-loop, coin-flip, A, B, C, D (T08).
Running a whole question set and writing the run log come in T05 and T08.
To follow the code, put breakpoints in pipeline/loop.py and press Debug on this file (VS Code: Run and Debug).
"""

import argparse
import platform
import sys

from pipeline.config import RUNS_DIR, load_config
from pipeline.dataset import find_question, load_question_set
from pipeline.llm import START_OLLAMA_HINT, ollama_is_running, shut_down_vllm_engines
from pipeline.loop import SCENARIOS
from pipeline.retriever import Retriever


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


def run_one_question(setting_name: str, question_id: str | None, config: dict) -> None:
    """Run one Pilot question (the first, unless an id is given) through one scenario and print the result."""
    if config["backend"] == "ollama" and not ollama_is_running():
        sys.exit(START_OLLAMA_HINT)

    pilot_set = load_question_set("pilot")
    question = find_question(question_id, pilot_set) if question_id else pilot_set[0]
    scenario = SCENARIOS[setting_name]
    result = scenario(question, Retriever.load(), config)   # <- step into this line to follow the pipeline

    print("Question:    ", question["question"])
    print("Gold answer: ", question["answer"])
    for field_name, field_value in result.items():
        print(f"{field_name + ':':<14}{field_value}")


def read_arguments() -> argparse.Namespace:
    """Read the command-line options."""
    argument_parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    argument_parser.add_argument("--check", action="store_true", help="check the setup and exit")
    argument_parser.add_argument("--setting", choices=list(SCENARIOS), default="single-turn",
                                 help="which scenario to run (default: single-turn)")
    argument_parser.add_argument("--question", help="id of a Pilot question (default: the first one)")
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
        run_one_question(arguments.setting, arguments.question, config)
        shut_down_vllm_engines()   # here we free the GPU on Kaggle, so the program can exit
