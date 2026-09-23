# Is the Judge the Bottleneck?

## Layout

| Folder | Contents | In git? |
|---|---|---|
| `src/` | Project code | Yes |
| `configs/` | Settings; `default.yaml` holds every tunable value | Yes |
| `data/` | Datasets and Wikipedia (several GB, each person downloads) | No |
| `runs/` | Raw experiment logs | No |
| `results/` | Final tables and figures | Yes |
| `notebooks/` | Colab/Kaggle notebooks | Yes |
| `docs/` | Notes, memos, decisions | Yes |

Don't hard-code settings in `src/`. Add them to `configs/default.yaml` and read them with `src.config.load_config()`.

## Setup on Mac

Requires Python 3.11 (`brew install python@3.11`).

```bash
git clone https://github.com/shoimya/Is-the-Judge-the-Bottleneck.git
cd Is-the-Judge-the-Bottleneck
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.hello
```

## Setup on Colab / Kaggle

Open `notebooks/00_setup.ipynb` and run all cells at the start of every session. It clones the repo, installs requirements, links `runs/` to Google Drive (Colab) or `/kaggle/working` (Kaggle), and runs `src.hello`.

## Adding a library

Pin the exact version in `requirements.txt` (`name==x.y.z`), and check that it installs on both Mac and Colab before pushing.
