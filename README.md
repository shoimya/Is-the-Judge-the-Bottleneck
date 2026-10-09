# Is the Judge the Bottleneck?

When a QA system searches several times and an LLM decides when to stop and what to search for next, we measure how much accuracy that LLM judge loses against a perfect oracle judge, and whether stopping or steering causes the loss. Decisions and the glossary live in [CONTEXT.md](CONTEXT.md); ticket status lives in [backlog/](backlog/README.md).

## Layout

Every file runs on its own. Open it in VS Code and press Run to see a short demo, or Debug to step through it. Where a file lets you choose something, the choice is a variable at the top of the file.

```
pipeline/              what runs for every question
  config.py            load_config() + the project's folders
  dataset.py           read the Pilot set / Test set
  retriever.py         load the BM25 index, search
  llm.py               the one place that talks to the model (Ollama on the Mac, vLLM on Kaggle)
  judge.py             LLM judge + Oracle judge (T06, T07)
  rewriter.py          Steer signal -> next query (T06)
  answerer.py          Evidence -> answer
  loop.py              one function per scenario: Closed-book, Single-turn, Always-loop, Coin-flip, A-D
  run_log.py           where each scenario's logs go: runs/<scenario>/
  server_launcher.py   open / close the Ollama server on the Mac
setup/                 one-time jobs
  get_questions.py     download HotpotQA, pick the Pilot and Test sets
  build_index.py       build the search index from Wikipedia (Kaggle only; already built)
  check_index.py       how often search finds the Gold paragraphs
  check_model.py       model smoke test + speed benchmark
project_set_up.py      sets up a machine; run it first
run.py                 check the setup, or run a scenario and log it
evaluate.py            scores, statistics, tables, figures (T09, T12)
test_pipeline.py       every test, in pipeline order
kaggle.ipynb           setup + runs on Kaggle's GPUs
config.yaml            experiment choices that can change a reported number
pyproject.toml         lets every file `import pipeline`
requirements.txt       Python libraries for every machine, exact versions
requirements-gpu.txt   vLLM, Kaggle only
NOTES.md               pilot choices, run sheet, throughput numbers
CONTEXT.md             what the project studies, every decision, the glossary, the coding rules
results/               tables + figures for the paper
backlog/               tickets T01-T19, worked in order
data/, runs/           datasets, the model, run logs (not on GitHub)
LICENSE                MIT for the code; HotpotQA and MuSiQue keep their own licenses
```

## How to run it

### On the Mac, for development

Install these once, if you don't have them:
- VS Code with Microsoft's Python extension.
- Python 3.12 or newer, from python.org or with `brew install python@3.12`.
- Ollama, from ollama.com/download or with `brew install ollama`. It runs the model on the Mac.

Get the code and the search index:
- In VS Code, open the Command Palette, choose **Git: Clone**, paste `https://github.com/shoimya/Is-the-Judge-the-Bottleneck.git`, and open the folder.
- Switch to the work branch `SC.V2` with the branch name in the bottom-left corner of VS Code.
- Download the search index, about 2.8 GB. On kaggle.com, open the dataset `hotpotqa-bm25-index` and click Download. You need to be added to it first, because it's private.
- Unzip it into `data/wiki/bm25_index/` inside the repo, so that `data/wiki/bm25_index/params.index.json` exists. Don't let it end up one folder deeper.

Set up the project:
- Open `project_set_up.py` and press Run. Any Python 3 works as the interpreter for this first run.
- It creates `.venv`, installs the Python libraries, builds the Pilot and Test question sets, and downloads the model into `data/ollama/`. It skips anything already there, so running it again is safe.
- The first run takes a few minutes, mostly the 2.5 GB model download. It ends with "Steps left to do by hand" and lists each one. Do them, then run it again until the count is 0.
- Choose `.venv` as the interpreter: Command Palette, then **Python: Select Interpreter**, then `.venv`. Every other file runs with it.

Run things. For each one, open the file, set the variable shown, and press Run.
- Open the Ollama server with `pipeline/server_launcher.py` and `ACTION = "start"`. Anything that asks the model needs it. It keeps running in the background after the file finishes.
- Check everything works with `test_pipeline.py`. Expect all tests to pass. Four of them are skipped while the server is closed.
- Check search quality with `setup/check_index.py`. It should print 11 / 24 / 34 / 44 % of questions with every Gold paragraph found in the top 2 / 5 / 10 / 20.
- Check the model with `setup/check_model.py`. Set `RUN_BENCHMARK = True` to also time it, which takes about 9 minutes on a Mac.
- Follow one question through the pipeline with `run.py`. Set `SCENARIO` to `"single-turn"` or `"closed-book"`. The other scenarios arrive with T06-T08. Set `QUESTION_ID` to pick a Pilot question, or `RUN_WHOLE_SET = True` for all 100. Each run is logged in `runs/<scenario>/<start time>/`. If a whole-set run stops partway, press Run again and it carries on.
- See one piece on its own by running any file in `pipeline/`.
- Close the server when you're done with `pipeline/server_launcher.py` and `ACTION = "stop"`. While open it holds about 3 GB of memory.

### On Kaggle, for every reported number

- Create a new Kaggle notebook and import `kaggle.ipynb` from this repo with File, then Import Notebook.
- In the notebook's Session options, set Accelerator to **GPU T4 x2** and Internet to **On**. Internet needs a phone-verified Kaggle account.
- Choose Add Input, then Your Datasets, then `hotpotqa-bm25-index`.
- In section 1 of the notebook, choose the branch, the scenario and the question set.
- Press Run All at the start of every session, because Kaggle forgets everything in between. Each of the 11 sections says what it does and stops with a clear message if something fails.
- Click Save Version when it finishes. That keeps the run logs in `/kaggle/working/runs/`.

## What gets installed

Software on the Mac:
- Python 3.12 or newer. Kaggle uses 3.13, and both give identical question sets and search results.
- Ollama, which runs the model on the Mac for development only.
- VS Code with the Python extension, to run and debug the files.

Python libraries in `requirements.txt`, installed into `.venv` at exact versions:
- `PyYAML` 6.0.3 reads `config.yaml`.
- `bm25s[core]` 0.3.11 is the BM25 search engine. `numba` 0.65.0 and `numpy` 2.4.6 come with it, pinned to the versions vLLM installs on Kaggle.
- `PyStemmer` 3.1.0 cuts words to their root, so "directed" matches "directing".
- `huggingface_hub` 1.33.0 downloads HotpotQA. It stays below 2.0 because vLLM needs that.
- `pyarrow` 25.0.1 reads the HotpotQA file.
- `pytest` 9.1.1 runs the tests.
- The repo's own `pipeline/` folder, so every file can import it.

Kaggle only, in `requirements-gpu.txt`:
- `vllm` 0.30.0 runs the model on Kaggle's GPUs. The notebook also uninstalls TorchAudio, because Kaggle's copy crashes vLLM.

The model:
- Qwen3-4B-Instruct-2507 plays all three roles: the Judge, the Rewriter and the Answerer. `config.yaml` sets it. Today only the Answerer and the model check use it; the Judge and the Rewriter arrive in T06.
- On the Mac, Ollama runs a compressed copy called `qwen3:4b-instruct`, about 2.5 GB in `data/ollama/`. It's only for writing and testing code, because the compressed weights can answer differently.
- On Kaggle, vLLM runs the full fp16 model, about 8 GB, downloaded into `/tmp` each session. Every number in the paper comes from these runs.
- The pilot in T10 picks the final model, either this one or Qwen3-8B-AWQ. Tier 3 adds a larger judge, Qwen3-14B-AWQ, and one from another model family.

Data:
- HotpotQA dev questions, 28 MB, downloaded from Hugging Face into `data/hotpotqa/`.
- The BM25 search index over 5.23 million Wikipedia intro paragraphs, about 2.8 GB, from the private Kaggle dataset `hotpotqa-bm25-index`.
- The Wikipedia dump it was built from, 1.55 GB. You only need it if the index is lost, and then only on Kaggle.

To add a library, pin the exact version that pip installs on Kaggle in `requirements.txt`, or in `requirements-gpu.txt` if it only runs on the GPU. Check that it installs on the Mac too before pushing.
