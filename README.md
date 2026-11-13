# Is the Judge the Bottleneck?

In a self-correcting retrieval loop for multi-hop QA, how much does an LLM sufficiency judge lose compared with an oracle judge, and is the loss from **stopping** ("enough, answer now") or **steering** ("here's what's missing")?

Terms (Round, Evidence, Stop decision, Steer signal, …) are defined in [CONTEXT.md](CONTEXT.md). Every decision is in [specification.md](specification.md); tickets and their status are in [backlog/](backlog/README.md).

## Layout

The code follows the pipeline: questions → retriever → judge → rewriter → answerer. Every file runs on its own: open it and press Run (or Debug) to see a short demo.

```
pipeline/
  dataset.py         HotpotQA → Pilot set (100) and Test set (1,000) questions
  retriever.py       BM25 search over HotpotQA's 2017 Wikipedia; also builds the search index
  answerer.py        question + Evidence → short answer
  judge.py           LLM judge (Stop decision, Steer signal, p_yes) + Oracle judge
  rewriter.py        Steer signal → next search query
  llm2.py            the one place that talks to the model (Ollama); starts and stops our Ollama server
  llm.py             earlier model code, still used by the roles' older functions
  scoring.py         answer vs gold answer: exact match, precision, recall, F1, cosine similarity (embedding model)
  settings.py        what differs between the Mac and Kaggle (models folder, port, context window)
  Methods/
    closed_book.py   answer from memory, no search
    single_turn.py   one search with the question, then answer
    always_loop.py   3 Rounds of search steered by the Judge, then answer
    condition_a.py   the LLM judge stops and steers (the deployed system); the Oracle labels each Stop decision
    condition_b.py   the LLM judge stops, the Oracle steers (B − A = the cost of steering)
    condition_c.py   the Oracle stops, the LLM judge steers (C minus A = the cost of stopping)
    condition_d.py   the Oracle stops and steers (the best case)
project_set_up.py    sets this machine up in one run (Python, .venv, libraries, questions, model)
project_check.py     checks the setup without changing anything
config.yaml          experiment settings: model per role, k, Rounds, seeds, reply limits
pyproject.toml       makes pipeline/ importable from any file, so every file runs with Run
requirements.txt     pinned libraries
requirements-gpu.txt vLLM pins (not used since the move to Ollama)
Notebooks/
  kaggle_methods.ipynb     runs one method (Closed-book, Single-turn or Always-loop) on Kaggle with Ollama
  kaggle_condition_a.ipynb runs Condition A (LLM judge stops and steers) on Kaggle with Ollama
  kaggle_condition_b.ipynb runs Condition B (LLM judge stops, Oracle steers) on Kaggle with Ollama
  kaggle_condition_c.ipynb runs Condition C (Oracle stops, LLM judge steers) on Kaggle with Ollama
  kaggle_condition_d.ipynb runs Condition D (Oracle stops and steers) on Kaggle with Ollama
  kaggle_WikiIndex.ipynb   builds the HotpotQA BM25 index on Kaggle (only needed to rebuild it)
NOTES.md             pilot choices, run sheet, throughput numbers
results/             committed question id lists and BM25 recall checks
Knowledge-Resources/ the Sep 25 specification and glossary (background reading)
Archive/             earlier code no longer used (run.py, loop.py, evaluate.py, tests, …)
backlog/             build tickets T01–T19
data/, runs/         questions, search index, model files and run logs (not on GitHub)
```

Don't hard-code settings. Experiment settings go in `config.yaml`, read with `pipeline.load_config()`; anything that differs between the Mac and Kaggle goes in `pipeline/settings.py`. Each method writes one log per run to `runs/<method>/<date>/<HHMMSS>.txt`.

## Setup on Windows

1. Install Python **3.13** from [python.org](https://www.python.org/downloads/windows/), including the Python launcher, and install [Ollama for Windows](https://ollama.com/download/windows).
2. Open the repo in VS Code. Restart VS Code after installing Ollama so its PATH is refreshed.
3. Open `project_set_up.py` and press Run. The same script supports Windows and macOS: on Windows it creates `.venv/Scripts/python.exe`, installs the pinned requirements, prepares the question sets, and downloads the configured chat and embedding models into `data/ollama/`.
4. If `.venv` was copied from another operating system, rename it before running setup; virtual environments cannot be shared between operating systems.
5. Download the private Kaggle dataset `hotpotqa-bm25-index` (ask the repo owner for access). Extract its files directly into `data/wiki/bm25_index/`.
6. Press Ctrl+Shift+P > Python: Select Interpreter and choose `.venv/Scripts/python.exe`. Open `project_check.py` and press Run, then try a method in `pipeline/Methods/`.

Terminal activation is optional. In PowerShell use `.\.venv\Scripts\Activate.ps1`; in Command Prompt use `.venv\Scripts\activate.bat`. If PowerShell blocks activation, select the interpreter in VS Code instead. Setup uses a separate Ollama server on port 11600 and stops only the server it started.

## Setup on Mac

1. **Get the code** (the current work is on the `SC.V3` branch):
   ```bash
   git clone -b SC.V3 https://github.com/shoimya/Is-the-Judge-the-Bottleneck.git
   ```
2. **Install the two programs setup can't install for you:** Python 3.13 (the version Kaggle uses) and Ollama (runs the model on the Mac).
   ```bash
   brew install python@3.13 ollama
   ```
3. **Run the setup:** open the repo in VS Code, open `project_set_up.py` and press Run (any Python 3 can start it). Quit the Ollama app first. In one run, it:
   - makes `.venv` with Python 3.13 and installs `requirements.txt`
   - builds the Pilot and Test sets in `data/hotpotqa/` (they match `results/question_ids/`)
   - downloads the model named in `config.yaml` (`qwen3:8b`, ~5 GB) into `data/ollama/`
   - checks that the search index is there

   Each step checks first, so running it again changes nothing. Anything it can't do is listed at the end.
4. **Get the search index** (~2.8 GB; setup prints these steps if it's missing): on kaggle.com, download the private dataset `hotpotqa-bm25-index` (ask the repo owner for access) and unzip it so its files sit directly in `data/wiki/bm25_index/`.
5. **Select the interpreter:** in VS Code, Cmd+Shift+P > Python: Select Interpreter > `.venv`.
6. **Check everything:** open `project_check.py` and press Run. It should say 4 of 4 checks passed.
7. **Try it out:** press Run on `pipeline/retriever.py` (a search), `pipeline/llm2.py` (one model call), then on a method in `pipeline/Methods/`. Each method's settings (`QUESTION_SET`, `HOW_MANY_QUESTIONS`) are at the top of its file.

Everything stays inside the repo on the SSD, never in the home folder.

## Setup on Kaggle / Colab

Open `kaggle.ipynb` and run all cells at the start of every session. On Kaggle, turn on Internet and pick the GPU T4 x2 accelerator. The notebook clones the repo, installs requirements, keeps `runs/` in persistent storage, and runs `python run.py --check`.

Every number in the paper comes from Kaggle. The Mac is for development.

## Adding a library

Pin the exact version in `requirements.txt` (`name==x.y.z`), and check it installs on the Mac and on Kaggle before pushing.

To run Condition C, open `pipeline/Methods/condition_c.py`, choose `QUESTION_SET` and `HOW_MANY_QUESTIONS` at the top, and press Run. It uses the existing setup and writes logs to `runs/condition_c/<date>/<HHMMSS>.txt`.
