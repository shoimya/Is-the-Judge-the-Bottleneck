# T01 · Repo and environment

Tier: 1 · Owner: _ · Depends: none

## What

`setup_project.py` at the top of the repo: the first script anyone runs after cloning. With no arguments it gets the machine ready to run the project, on the Mac and on Kaggle. **For everything it needs, it first checks whether it already exists and only downloads or installs what is missing.** Later tickets add their own steps the same way. Standard library only, and it must start even on the Mac's built-in Python 3.9 (it runs before anything is installed). Safe to run again; it never deletes anything.

Steps, one function each:
1. **Find Python 3.12 or newer:** the Python running the script if it is new enough, otherwise an installed `python3.12`/`python3.13`, otherwise install `python@3.12` with Homebrew on the Mac. If none of that works, stop and say how to install it.
2. **Detect the machine:** Mac or Kaggle, GPU or not.
3. **Create the folders** the project needs: `data/`, `data/cache/`, `runs/`, `results/` (skipping any that exist), and **`pytest.ini`** if missing, so `pytest` finds the tests in `test/` and they can import the scripts at the top of the repo.
4. **Create `.venv`** on the Mac with that Python, if it doesn't exist yet (Kaggle starts fresh each session, so it uses Kaggle's Python).
5. **Install what `requirements.txt` lists but is missing** (or at the wrong version) in that Python, with pip's download cache in `data/cache/pip` (never the home folder).
6. **Install `requirements-gpu.txt`** only on a GPU machine and only once that file exists (it arrives with vLLM in T04).
7. **Check the setup:** every folder writable; list the installed library versions.
8. **Log it** to `runs/setup/<date>/setup_<time>.log`: machine, Python, what already existed, what was installed, installed versions, and loud warnings (e.g. no GPU on Kaggle). A failed setup is logged too.

It ends with **Setup OK** and the next step. Datasets, the search index and the model are added as steps by the tickets that need them (T02, T03, T04), each checking first.

Also in this ticket:
- `requirements.txt` with exact versions (`name==x.y.z`): only `pytest` for now; each ticket adds what it needs.
- `.gitignore` for `data/`, `runs/`, `.venv/`, `__pycache__/`, `.DS_Store`.

## Done when

A teammate can clone the repo, run `python3 setup_project.py` (any Python 3, even the Mac's built-in 3.9), and see **Setup OK**, on the Mac and on Kaggle. Running it a second time installs nothing new.

## Description

Everyone runs the same Python, the same exact library versions, and the same folders, set up by one script instead of a list of manual steps. Kaggle resets every session, so the same script is simply run again there.

## Notes

- Oct 9: rebuilt from scratch under the owner's coding rules (CLAUDE.md). Decided with the owner: the script is `setup_project.py` (not `setup.py`, which pip treats as a packaging script); logs go in `runs/`; folders `data/`, `data/cache/`, `runs/`, `results/`; the script creates `.venv` itself. Replaces the earlier `run.py --check`, stub files and `config.yaml`.
- Python: the Mac runs 3.12, Kaggle 3.13; both are supported.
- Kaggle needs **Internet** switched on in the notebook settings (requires a phone-verified account).
- Oct 9: Done. Verified on the Mac (started from the built-in Python 3.9; found 3.12, built `.venv`, installed pytest; a second run installed nothing; 2 tests pass) and on Kaggle (two runs, both Setup OK, branch `SC.V3`).
