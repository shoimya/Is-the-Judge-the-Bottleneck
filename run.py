"""Run one setting on one question set and write its run log.

    python run.py --check     # setup test: Python version, config, runs/ writable

Running settings (--setting A --set pilot) is added in T08.
"""

import argparse
import platform
import sys

from pipeline import REPO_ROOT, load_config


def check_setup() -> None:
    print(f"Python   {platform.python_version()} on {platform.system()}")
    if sys.version_info[:2] != (3, 12):
        print("WARNING  project targets Python 3.12")

    config = load_config()
    print(f"Config   rounds={config['loop']['rounds']} seeds={config['seeds']}")

    runs_dir = REPO_ROOT / config["paths"]["runs_dir"]
    write_test_file = runs_dir / ".write_test"
    write_test_file.write_text("ok")
    write_test_file.unlink()
    print(f"runs/    writable ({runs_dir.resolve()})")

    print("Setup OK.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="verify the setup and exit")
    arguments = parser.parse_args()

    if arguments.check:
        check_setup()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
