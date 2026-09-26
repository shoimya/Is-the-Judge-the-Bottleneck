# T01 · Repo and environment

Tier: 1 · Owner: _ · Depends: none · Status: done (Sep 23: Mac on Python 3.12.14 and Kaggle both pass `run.py --check`)

## What

Set up this layout. The folders follow the pipeline: questions → retriever → judge → rewriter → answerer.

```
Is-the-Judge-the-Bottleneck/
├── pipeline/
│   ├── dataset.py     # HotpotQA / MuSiQue → Pilot set & Test set questions
│   ├── retriever.py   # BM25 search
│   ├── judge.py       # LLM judge + Oracle judge (Stop decision, Steer signal)
│   ├── rewriter.py    # Steer signal → next query
│   ├── answerer.py    # Evidence → answer
│   ├── llm.py         # the one place that talks to the model
│   └── loop.py        # wires the pieces together; Conditions A–D, reference points
├── run.py             # run one setting on one question set → writes the run log
├── evaluate.py        # scores, statistics, tables, figures
├── test_pipeline.py   # small checks, run with `pytest`
├── kaggle.ipynb       # setup + run on Kaggle
├── config.yaml        # every setting
├── requirements.txt
├── NOTES.md           # pilot choices, run sheet, throughput numbers
├── README.md
├── results/           # final tables + figures for the paper (committed)
├── data/              # git-ignored
└── runs/              # git-ignored
```

For this ticket, the pipeline files can be empty stubs with one line saying what goes there. Later tickets fill them in.

- **Python 3.12** (matches Kaggle's 3.12.12 and Colab).
- `requirements.txt` with exact versions (`name==x.y.z`). Only `pyyaml` for now; each later ticket adds what it needs.
- `config.yaml` with every setting, starting with `rounds: 3`, `seeds`, and `models.judge / rewriter / answerer`. Values not decided yet are `null` with a comment.
- `.gitignore` for `data/`, `runs/`, `.venv/`, `__pycache__/` and `.DS_Store`.
- `kaggle.ipynb`: clones the repo, installs requirements, points `runs/` at persistent storage (`/kaggle/working`, or Google Drive on Colab), and runs `python run.py --check`.
- `run.py --check`: prints the Python version, loads `config.yaml`, and confirms `runs/` is writable.

## Done when

A teammate can clone, install, and run `python run.py --check` on the Mac and on Kaggle.

## Description

1. **Folders follow the pipeline.** Each piece of the pipeline is one file in `pipeline/`. `data/` and `runs/` stay off GitHub because they're large and change constantly.
2. **Same software for everyone.** Same Python version, exact library versions, and one settings file. No settings buried in code.
3. **Plug-and-play roles.** `config.yaml` names a model per role, and `pipeline/llm.py` is the only file that loads models. Swapping the Judge is a one-line config change.
4. **Kaggle notebook.** Kaggle and Colab reset every session, so the notebook redoes the setup each time and keeps `runs/` somewhere that survives a disconnect.

## Notes

- Python 3.12 was chosen on Sep 23 because Kaggle, where every reported number is produced, runs 3.12.12. Mac: `brew install python@3.12`.
- Generate the final exact pins on Kaggle (`pip freeze` after install) and reuse them everywhere, so a version that only exists for the Mac never gets pinned.
- Kaggle needs **Internet** switched on in the notebook settings (requires a phone-verified account). `kaggle.ipynb` stops with that hint if it's off.
