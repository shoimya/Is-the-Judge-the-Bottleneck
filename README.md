# Is the Judge the Bottleneck?

In a self-correcting retrieval loop for multi-hop QA, how much does an LLM sufficiency judge lose compared with an oracle judge, and is the loss from **stopping** ("enough, answer now") or **steering** ("here's what's missing")?

Terms (Round, Evidence, Stop decision, Steer signal, …) are defined in [CONTEXT.md](CONTEXT.md). Every decision is in [specification.md](specification.md); tickets and their status are in [backlog/](backlog/README.md).

## Layout

The code follows the pipeline: questions → retriever → judge → rewriter → answerer.

```
pipeline/
  dataset.py     HotpotQA / MuSiQue → Pilot set & Test set questions
  retriever.py   BM25 search
  judge.py       LLM judge + Oracle judge (Stop decision, Steer signal)
  rewriter.py    Steer signal → next query
  answerer.py    Evidence → answer
  llm.py         the one place that talks to the model
  loop.py        wires the pieces together; Conditions A–D, reference points
run.py           run one setting on one question set → writes the run log
evaluate.py      scores, statistics, tables, figures
test_pipeline.py small checks (pytest)
kaggle.ipynb     setup + run on Kaggle / Colab
config.yaml      every setting
NOTES.md         pilot choices, run sheet, throughput numbers
results/         final tables + figures for the paper
backlog/         build tickets T01–T19, worked in order
data/, runs/     datasets and run logs (not on GitHub)
```

Don't hard-code settings. Add them to `config.yaml` and read them with `pipeline.load_config()`.

## Setup on Mac

Requires Python 3.12 (`brew install python@3.12`).

```bash
git clone https://github.com/shoimya/Is-the-Judge-the-Bottleneck.git
cd Is-the-Judge-the-Bottleneck
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run.py --check
pytest
```

## Setup on Kaggle / Colab

Open `kaggle.ipynb` and run all cells at the start of every session. On Kaggle, turn on Internet and pick the GPU T4 x2 accelerator. The notebook clones the repo, installs requirements, keeps `runs/` in persistent storage, and runs `python run.py --check`.

Every number in the paper comes from Kaggle. The Mac is for development.

## Adding a library

Pin the exact version in `requirements.txt` (`name==x.y.z`), and check it installs on the Mac and on Kaggle before pushing.
