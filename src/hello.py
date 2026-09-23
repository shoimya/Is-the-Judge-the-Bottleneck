"""Smoke test for setup: `python -m src.hello`.

Checks the Python version, that the config loads, and that runs/ is writable.
"""

import platform
import sys

from src.config import REPO_ROOT, load_config


def main() -> None:
    print(f"Python   {platform.python_version()} on {platform.system()}")
    if sys.version_info[:2] != (3, 11):
        print("WARNING  project targets Python 3.11")

    cfg = load_config()
    print(f"Config   rounds={cfg['loop']['rounds']} seeds={cfg['seeds']}")

    runs_dir = REPO_ROOT / cfg["paths"]["runs_dir"]
    probe = runs_dir / ".write_test"
    probe.write_text("ok")
    probe.unlink()
    print(f"runs/    writable ({runs_dir.resolve()})")

    print("Hello from Is-the-Judge-the-Bottleneck. Setup OK.")


if __name__ == "__main__":
    main()
