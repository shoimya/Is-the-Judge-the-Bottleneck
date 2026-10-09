# Is the Judge the Bottleneck?

Some question-answering systems search several times before answering. After each search, a language model acts as a judge: it decides whether the evidence is enough to answer, and if not, what is still missing. This project measures how much accuracy and cost that judge loses compared with a perfect judge that knows which paragraphs are needed, and whether the loss comes from stopping at the wrong time or from searching for the wrong thing.

The research questions, the design and every decision are in [CONTEXT.md](CONTEXT.md). The build plan and each ticket's status are in [backlog/](backlog/README.md).

## Getting started

You need macOS with [Homebrew](https://brew.sh), or a Kaggle notebook, and an internet connection.

**1. Clone the repository.**

```bash
git clone https://github.com/shoimya/Is-the-Judge-the-Bottleneck.git
cd Is-the-Judge-the-Bottleneck
```

**2. Run the setup script.** Any Python 3 can start it, including the one built into macOS.

```bash
python3 setup_project.py
```

For everything the project needs, the script first checks whether it's already there and only installs what's missing. Running it again is safe and installs nothing new. It:

- finds Python 3.12 or newer, and installs `python@3.12` with Homebrew only if none is found
- creates the folders `data/`, `data/cache/`, `runs/` and `results/`
- creates `pytest.ini`, so `pytest` finds the tests
- creates a virtual environment in `.venv/` (on the Mac; Kaggle uses its own Python)
- installs the libraries in `requirements.txt` that aren't installed yet, keeping pip's downloads in `data/cache/pip`
- makes the Pilot set and Test set with `question_sets.py`, if `data/hotpotqa/` doesn't have them yet (see below)
- finds the Wikipedia search index in `data/wiki/bm25_index/`. On Kaggle it links the attached dataset; on a Mac it prints how to download it (see below)
- writes what it did to `runs/setup/<date>/setup_<time>.log`

It finishes with `Setup OK.` If something goes wrong, the error is in that log too.

**3. Use the virtual environment.**

- In a terminal: `source .venv/bin/activate`
- In VS Code: Command Palette → **Python: Select Interpreter** → the one in `.venv`

**4. Run the tests.**

```bash
pytest
```

### On Kaggle

Turn on **Internet** in the notebook's settings panel (it needs a phone-verified account). Attach the search index with **Add Input → Your Datasets → hotpotqa-bm25-index**. Then run in a cell:

```python
!git clone --branch SC.V3 https://github.com/shoimya/Is-the-Judge-the-Bottleneck.git
%cd Is-the-Judge-the-Bottleneck
!python setup_project.py
```

Kaggle resets every session, so run these again at the start of each one.

## The question sets

Every experiment runs on two fixed sets of questions from HotpotQA's dev split, fullwiki setting. The Pilot set has 100 questions and is for tuning prompts and settings. The Test set has a separate 1,000 questions, and nobody runs it until the settings are frozen. Setup makes both sets for you. To make them by hand, with the virtual environment active (step 3):

```bash
python question_sets.py
```

It downloads HotpotQA from Hugging Face into `data/hotpotqa/raw/` and picks the questions with a fixed seed, so every machine gets the same ones. It writes the questions to `data/hotpotqa/pilot.jsonl` and `test.jsonl`, one per line, with the fields `id`, `question`, `answer`, `type`, `level` and `gold_titles`. Only the id lists in `results/question_ids/` go on GitHub. If a rerun would change those lists, the log at `runs/question_sets/<date>/` shows a warning. The script ends by printing the first Pilot question.

## Searching Wikipedia

The search uses BM25, a keyword search, over HotpotQA's 2017 Wikipedia: about 5.2 million intro paragraphs. The index is about 2.8 GB and lives in `data/wiki/bm25_index/`.

**Getting the index.** It's already built and saved as the private Kaggle dataset `hotpotqa-bm25-index`. Ask the repo owner to share it with your Kaggle account.
- On Kaggle, attach the dataset and run setup, which links it into place.
- On a Mac, download the dataset from its Kaggle page and unzip its 7 files into `data/wiki/bm25_index/`. Run setup again to check it's found.
- Only if the dataset is lost: `python build_index.py` rebuilds it from scratch. Run it on Kaggle. It takes about 12 minutes and more memory than a laptop has.

**Trying the search**, with the virtual environment active:

```bash
python retriever.py
```

It searches with the first Pilot question and prints the top 5 paragraphs, marking the Gold ones. Then it checks search quality on all 100 Pilot questions: for the top 2, 5, 10 and 20 results, how often all Gold paragraphs were found, and how often at least one was. The table is saved to `results/pilot/bm25_recall.csv`, and the log goes to `runs/retriever/<date>/`. If the numbers ever differ from the saved table, the log warns.

Other code searches with two functions from `retriever.py`: `load_index()` opens the index once, and `search(index, query, k, exclude_titles)` returns the `k` best paragraphs as `{title, text, score}`, skipping titles already found.

## What's in the repository

```
setup_project.py   gets a machine ready to run the project (run this first)
question_sets.py   makes the Pilot set and Test set from HotpotQA, and reads them back
retriever.py       searches Wikipedia; running it checks search quality on the Pilot set
build_index.py     builds the Wikipedia search index (Kaggle only, and only if the saved copy is lost)
requirements.txt   libraries the project needs, at exact versions
test/              tests, run with `pytest`
CONTEXT.md         what the project studies, every decision, and the glossary
backlog/           build tickets T01-T19 and their status
LICENSE            MIT, for the code
```

`setup_project.py` creates these, and git ignores all of them except `results/` and `pytest.ini`:

```
.venv/             the project's Python and libraries (Mac)
data/              datasets, the search index, downloads and caches
runs/              logs of every run, including setup's
results/           the question id lists, the search-quality table, and tables and figures for the paper
pytest.ini         tells pytest where the tests are
```

## Adding a library

Add it to `requirements.txt` at an exact version (`name==x.y.z`) and run `python3 setup_project.py` again; it installs only what's new. Use the version pip installs on Kaggle, where every reported number comes from.

## License

The code is [MIT](LICENSE). The datasets (HotpotQA, MuSiQue) keep their own licenses and aren't redistributed here.
