# Is the Judge the Bottleneck?

In a self-correcting retrieval loop for multi-hop QA, how much does an LLM sufficiency judge lose compared with an oracle judge, and is the loss from **stopping** ("enough, answer now") or **steering** ("here's what's missing")?

What the project studies, every decision and the glossary (Round, Evidence, Stop decision, Steer signal, …) are in [CONTEXT.md](CONTEXT.md). Tickets and their status are in [backlog/](backlog/README.md).

## Layout

The code follows the pipeline: question → retriever → judge → rewriter → answerer. Every file in `pipeline/` and `setup/` runs on its own: open it and press Debug (or `python <file>`) to run a short demo, with breakpoints wherever you like.

```
pipeline/            what runs for every question
  config.py          load_config() + the project's folders
  dataset.py         read the Pilot set / Test set
  retriever.py       load the BM25 index, search
  llm.py             the one place that talks to the model (Ollama on the Mac, vLLM on Kaggle)
  judge.py           LLM judge + Oracle judge (T06, T07)
  rewriter.py        Steer signal → next query (T06)
  answerer.py        Evidence → answer
  loop.py            one function per scenario: Closed-book, Single-turn, Always-loop, Coin-flip, A–D
setup/               one-time jobs
  get_questions.py   download HotpotQA, pick the Pilot and Test sets
  build_index.py     download Wikipedia and build the search index (already built; only if lost)
  check_index.py     how often search finds the Gold paragraphs
  check_model.py     model smoke test + speed benchmark
run.py               launcher: --check, or follow one question through one scenario
evaluate.py          scores, statistics, tables, figures
test_pipeline.py     checks for every piece, in pipeline order (pytest)
kaggle.ipynb         setup + run on Kaggle / Colab
config.yaml          the choices an experiment can change
pyproject.toml       lets every file `import pipeline` from anywhere
requirements.txt      libraries for every machine (exact versions)
requirements-gpu.txt  vLLM, Kaggle only (doesn't install on a Mac)
.python-version      3.12, picked up by Python tools on the Mac
NOTES.md             pilot choices, run sheet, throughput numbers
CONTEXT.md           what the project studies, every decision, and the glossary
results/             final tables + figures for the paper
backlog/             build tickets T01–T19, worked in order
data/, runs/         datasets and run logs (not on GitHub)
LICENSE              MIT, for the code
```

`config.yaml` holds only the choices an experiment can change; read it with `pipeline.config.load_config()`. Fixed facts (links, folders, decoding rules) are named constants at the top of the file that uses them.

## Setup on Mac

Requires Python 3.12 or newer (`brew install python@3.12`). The Mac uses 3.12; Kaggle runs 3.13, and both give identical results.

```bash
git clone https://github.com/shoimya/Is-the-Judge-the-Bottleneck.git
cd Is-the-Judge-the-Bottleneck
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt      # also installs this repo's pipeline/ (the `-e .` line)
python run.py --check                # Setup OK.
pytest                               # model tests are skipped until Ollama is running
```

Then, once per machine (the search index is copied into `data/wiki/bm25_index/` from the Kaggle dataset `hotpotqa-bm25-index`):

```bash
python setup/get_questions.py        # the Pilot and Test sets; ids must match results/question_ids/
python setup/check_index.py          # search quality: 11/24/34/44 % all Gold paragraphs found
```

Follow one question through the pipeline (needs Ollama, below):

```bash
python run.py --setting single-turn  # or closed-book; add --question <id> for a chosen Pilot question
```

To debug in VS Code, select the `.venv` interpreter, open any file in `pipeline/`, `setup/` or `run.py`, and press F5 (Python Debugger: Current File).

## Ollama on the Mac (development only)

Install once with `brew install ollama` (no background app). Start it from the repo folder whenever you need a model, so its models are stored in `data/ollama/` (git-ignored), not in your home folder:

```bash
OLLAMA_MODELS="$PWD/data/ollama" ollama serve     # leave this terminal open
ollama pull qwen3:4b-instruct                     # in a second terminal, once: ~2.5 GB
```

Ollama only runs while that terminal is open. Reported numbers never come from Ollama; they come from vLLM on Kaggle.

## Setup on Kaggle / Colab

Open `kaggle.ipynb` and run all cells at the start of every session. Before running, in the notebook's settings panel:

1. Accelerator: **GPU T4 x2**.
2. Internet: **On** (needs a phone-verified Kaggle account).
3. Add Input > Your Datasets > **`hotpotqa-bm25-index`** (the saved search index).

The notebook has 11 numbered sections, each with a plain-language note. It clones the repo at the branch set in section 1, installs the requirements, keeps `runs/` in storage that survives the session, checks the setup, rebuilds the question sets and checks they match GitHub, links the search index and checks its results match the Mac's, then installs vLLM and runs the model smoke test and speed benchmark. Model files download to `/tmp`, so they don't fill the ~20 GB working folder.

Every number in the paper comes from Kaggle. The Mac is for development.

## Adding a library

Pin the exact version (`name==x.y.z`) in `requirements.txt`, or in `requirements-gpu.txt` if it only runs on Kaggle's GPU. Use the version pip actually installs on Kaggle (where every reported number comes from), and check it installs on the Mac too before pushing.

## License

Code: [MIT](LICENSE). The datasets (HotpotQA, MuSiQue) keep their own licenses and are not redistributed here.
