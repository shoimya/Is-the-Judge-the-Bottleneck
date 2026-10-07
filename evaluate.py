"""Scores, statistics, tables and figures, computed from run logs only.

`load_run` is built in T05, `gold_coverage` in T08 (its done-when compares
Condition D against A); scoring and summaries in T09, analysis and figures
in T12.
"""

import json
from pathlib import Path

import pandas as pd

from pipeline import REPO_ROOT, judge, load_config


def load_run(run_id: str | Path, *, runs_dir: Path | None = None) -> pd.DataFrame:
    """One row per question from runs/<run_id>/questions.jsonl.

    `run_id` is the folder name (e.g. 2026-10-14T1530_A) or any path to a run
    folder. meta.json is attached as `df.attrs["meta"]`; the `rounds` column
    holds each question's per-Round list.
    """
    if runs_dir is None:
        cfg = load_config()
        runs_dir = REPO_ROOT / cfg["paths"]["runs_dir"]
    folder = Path(run_id) if Path(run_id).is_dir() else runs_dir / run_id
    meta_path = folder / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"no run log at {folder}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    records = [json.loads(line) for line in
               (folder / "questions.jsonl").read_text(encoding="utf-8").splitlines()
               if line.strip()]
    df = pd.DataFrame(records)
    df.attrs["meta"] = meta
    return df


def gold_coverage(df: pd.DataFrame) -> float:
    """The fraction of questions whose Evidence held every Gold title at some
    Round (the T08 done-when check: Condition D vs Condition A)."""
    if df.empty:
        return 0.0
    covered = 0
    for gold_titles, rounds in zip(df["gold_titles"], df["rounds"]):
        titles: set[str] = set()
        found = False
        for round_log in rounds:
            titles.update(round_log["retrieved_titles"])
            if judge.oracle_stop(titles, gold_titles):
                found = True
                break
        covered += found
    return covered / len(df)
