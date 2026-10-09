"""Open and close the Ollama server on the Mac, so the model can be used without a terminal left open.

    Open this file, set ACTION below to "start" or "stop", and press Run (or Debug).

The server keeps its models in data/ollama/ and keeps running after this file ends, until it is closed.
Closing only stops a server this file opened; one started by hand in a terminal is left alone (Ctrl+C there).
Each server's output is logged to runs/server_launcher/<date>/.
"""

import json
import os
import signal
import subprocess
import time
from datetime import datetime
from pathlib import Path

from pipeline.config import DATA_DIR, RUNS_DIR
from pipeline.llm import ollama_is_running

# What pressing Run does. Change it, then press Run.
ACTION = "start"         # "start" opens the server; "stop" closes it

OLLAMA_MODELS_DIR = DATA_DIR / "ollama"
# Here we note the server this file opened (its process id and log), so `stop` knows which one to close.
SERVER_FILE = OLLAMA_MODELS_DIR / "server.json"
SERVER_LOGS_DIR = RUNS_DIR / "server_launcher"
SECONDS_TO_WAIT = 30     # how long to wait for the server to open or close


def wait_until_server_is(running: bool) -> bool:
    """Check once a second until the server is running (or stopped); False if it takes too long."""
    for _ in range(SECONDS_TO_WAIT):
        if ollama_is_running() == running:
            return True
        time.sleep(1)
    return ollama_is_running() == running


def start_ollama_server(server_file: Path = SERVER_FILE, logs_dir: Path = SERVER_LOGS_DIR) -> bool:
    """Open the Ollama server with its models in data/ollama/. Returns True if this call opened it."""
    if ollama_is_running():
        print("The Ollama server is already open.")
        return False

    now = datetime.now()
    log_file_path = logs_dir / now.strftime("%Y-%m-%d") / f"ollama_server_{now.strftime('%H%M%S')}.log"
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file_path, "a") as log_file:
        # Here start_new_session=True lets the server keep running after this file ends.
        server = subprocess.Popen(["ollama", "serve"], env={**os.environ, "OLLAMA_MODELS": str(OLLAMA_MODELS_DIR)},
                                  stdout=log_file, stderr=log_file, start_new_session=True)

    server_details = {"process_id": server.pid, "log_file": str(log_file_path), "started_at": now.isoformat()}
    server_file.parent.mkdir(parents=True, exist_ok=True)
    server_file.write_text(json.dumps(server_details, indent=2) + "\n")

    if not wait_until_server_is(running=True):
        print(f"WARNING: the Ollama server didn't answer within {SECONDS_TO_WAIT} s. See its log: {log_file_path}")
        return True
    print(f"Opened the Ollama server (models in {OLLAMA_MODELS_DIR}). Log: {log_file_path}")
    print('Close it with: set ACTION = "stop" in pipeline/server_launcher.py and press Run.')
    return True


def stop_ollama_server(server_file: Path = SERVER_FILE) -> bool:
    """Close the Ollama server this file opened. Returns True if a server was closed."""
    if not server_file.exists():
        if ollama_is_running():
            print("The running Ollama server wasn't opened by this file; close it with Ctrl+C in its terminal.")
        else:
            print("No Ollama server is open.")
        return False

    server_details = json.loads(server_file.read_text())
    try:
        # Here we ask the server to shut down cleanly (the same as pressing Ctrl+C in its terminal).
        os.kill(server_details["process_id"], signal.SIGTERM)
    except ProcessLookupError:
        print("The server this file opened had already stopped.")
        server_file.unlink()
        return False

    if not wait_until_server_is(running=False):
        print(f"WARNING: the Ollama server is still answering after {SECONDS_TO_WAIT} s. "
              f"Its log: {server_details['log_file']}")
        return False
    server_file.unlink()
    print("Closed the Ollama server.")
    return True


if __name__ == "__main__":
    if ACTION == "start":
        start_ollama_server()
    elif ACTION == "stop":
        stop_ollama_server()
    else:
        print(f'ACTION must be "start" or "stop", not {ACTION!r}.')
