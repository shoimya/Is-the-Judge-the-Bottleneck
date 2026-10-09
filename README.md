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

Turn on **Internet** in the notebook's settings panel (it needs a phone-verified account), then run in a cell:

```python
!git clone --branch SC.V3 https://github.com/shoimya/Is-the-Judge-the-Bottleneck.git
%cd Is-the-Judge-the-Bottleneck
!python setup_project.py
```

Kaggle resets every session, so run these again at the start of each one.

## What's in the repository

```
setup_project.py   gets a machine ready to run the project (run this first)
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
results/           tables and figures for the paper
pytest.ini         tells pytest where the tests are
```

## Adding a library

Add it to `requirements.txt` at an exact version (`name==x.y.z`) and run `python3 setup_project.py` again; it installs only what's new. Use the version pip installs on Kaggle, where every reported number comes from.

## License

The code is [MIT](LICENSE). The datasets (HotpotQA, MuSiQue) keep their own licenses and aren't redistributed here.
