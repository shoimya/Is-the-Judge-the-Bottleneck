"""Run one setting on one question set and write its run log.

    python run.py --check     # setup test: Python version, config, runs/ writable

Running settings (--setting A --set pilot) is added in T08.
"""

import argparse
import platform
import sys

from pipeline import REPO_ROOT, load_config


def check_setup() -> None:
    """Check that this machine is ready: the Python version, the settings file, and a writable runs/ folder."""
    # Here we print the Python version and warn if it is not the one the project targets.
    print(f"Python   {platform.python_version()} on {platform.system()}")
    if sys.version_info[:2] != (3, 12):
        print("WARNING  project targets Python 3.12")

    # Here we read config.yaml and print two settings, which proves the file loads.
    config = load_config()
    print(f"Config   rounds={config['loop']['rounds']} seeds={config['seeds']}")

    # Here we prove the runs/ folder is writable by creating a small file there and deleting it again.
    runs_dir = REPO_ROOT / config["paths"]["runs_dir"]
    write_test_file = runs_dir / ".write_test"
    write_test_file.write_text("ok")
    write_test_file.unlink()
    print(f"runs/    writable ({runs_dir.resolve()})")

    print("Setup OK.")


def main() -> None:
    # Here we read the command-line options; with no option, we print the help text.
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="verify the setup and exit")
    arguments = parser.parse_args()

    if arguments.check:
        check_setup()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
