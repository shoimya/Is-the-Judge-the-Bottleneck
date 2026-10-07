"""Run one setting on one question set and write its run log (T05, T08).

    python run.py --check     # setup test: Python version, config, runs/ writable
    python run.py --fake      # fake run of 5 dummy questions (the T05 checks)
    python run.py --fake --fail-after 2   # die after 2 new questions, then rerun to resume
    python run.py --setting A --set pilot  # one of the seven T08 settings

Every run lands in runs/<run_id>/ with run_id = date-time + condition
(e.g. 2026-10-14T1530_A):

- meta.json: date and time, condition, dataset, question set, model per role,
  backend, git commit, prompt versions, and a full copy of the config used.
- questions.jsonl: one line per question (the T05 field list). A dead session
  (Kaggle sessions die) resumes the same folder: ids already logged are the
  checkpoint, so nothing is lost and nothing is duplicated.
"""

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from pipeline import REPO_ROOT, load_config


def check() -> None:
    print(f"Python   {platform.python_version()} on {platform.system()}")
    if sys.version_info[:2] != (3, 12):
        print("WARNING  project targets Python 3.12")

    cfg = load_config()
    print(f"Config   rounds={cfg['loop']['rounds']} seeds={cfg['seeds']}")

    runs_dir = REPO_ROOT / cfg["paths"]["runs_dir"]
    probe = runs_dir / ".write_test"
    probe.write_text("ok")
    probe.unlink()
    print(f"runs/    writable ({runs_dir.resolve()})")

    print("Setup OK.")


# ------------------------------------------------------------- run folders


def make_run_id(condition: str, now: datetime | None = None) -> str:
    """The folder name: 2026-10-14T1530_A (run start time + condition)."""
    return f"{(now or datetime.now()):%Y-%m-%dT%H%M}_{condition}"


def git_commit() -> str:
    """The commit this code is running from, for reproducibility."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=REPO_ROOT, check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"  # no git where the run happens (e.g. a bare copy)


def start_run(condition: str, *, dataset: str = "pilot", question_set: str | Path | None = None,
              runs_dir: Path | None = None, config: dict | None = None,
              now: datetime | None = None) -> Path:
    """Create runs/<run_id>/ with meta.json and an empty questions.jsonl."""
    cfg = load_config() if config is None else config
    runs_dir = runs_dir or (REPO_ROOT / cfg["paths"]["runs_dir"])
    if question_set is None:
        question_set = REPO_ROOT / cfg["paths"]["data_dir"] / "hotpotqa" / f"{dataset}.jsonl"
    folder = runs_dir / make_run_id(condition, now=now)
    if folder.exists():
        # a fresh run must never merge into an existing one (two runs of the
        # same condition in the same minute): disambiguate the id
        n = 2
        while (runs_dir / f"{folder.name}-{n}").exists():
            n += 1
        folder = runs_dir / f"{folder.name}-{n}"
    folder.mkdir(parents=True, exist_ok=True)
    meta = {
        "run_id": folder.name,
        "datetime": (now or datetime.now()).isoformat(timespec="seconds"),
        "condition": condition,
        "dataset": dataset,
        "question_set": str(question_set),
        "models": cfg["models"],
        "backend": cfg["llm"]["backend"],
        "git_commit": git_commit(),
        "prompt_versions": prompt_versions(),
        "config": cfg,
    }
    (folder / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (folder / "questions.jsonl").touch()
    return folder


def prompt_versions() -> dict:
    """The prompt version each role file is on (written into meta.json)."""
    from pipeline import answerer, judge, rewriter

    return {
        "judge": getattr(judge, "PROMPT_VERSION", "unset"),
        "rewriter": getattr(rewriter, "PROMPT_VERSION", "unset"),
        "answerer": getattr(answerer, "PROMPT_VERSION", "unset"),
    }


def latest_run(runs_dir: Path, condition: str, dataset: str) -> Path | None:
    """The newest run folder whose meta.json matches, or None.

    Folder names sort by start time, so the max is the most recent run.
    """
    if not runs_dir.is_dir():
        return None
    candidates = []
    for folder in runs_dir.iterdir():
        meta_path = folder / "meta.json"
        if not meta_path.is_file():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if meta.get("condition") == condition and meta.get("dataset") == dataset:
            candidates.append(folder)
    return max(candidates) if candidates else None


def load_done_ids(run_dir: Path) -> set[str]:
    """Question ids already logged: the resume checkpoint."""
    done = set()
    path = run_dir / "questions.jsonl"
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["id"])
    return done


def append_question(run_dir: Path, record: dict) -> None:
    """Append one question line (the log is append-only on purpose)."""
    with open(run_dir / "questions.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_questions(questions, step, *, condition: str, dataset: str = "pilot",
                  question_set: str | Path | None = None, run_dir: Path | None = None,
                  config: dict | None = None, resume: bool = True) -> dict:
    """Run step(question) -> record over questions, one line per result.

    Resumable by default: when a run with the same condition+dataset exists,
    it resumes that folder and skips ids already in questions.jsonl — a dead
    session loses nothing and duplicates nothing. resume=False starts a fresh
    folder. A step that raises KeyboardInterrupt propagates; restart to resume.
    """
    cfg = load_config() if config is None else config
    runs_dir = run_dir or (REPO_ROOT / cfg["paths"]["runs_dir"])
    folder = latest_run(runs_dir, condition, dataset) if resume else None
    if folder is None:
        folder = start_run(condition, dataset=dataset, question_set=question_set,
                           runs_dir=runs_dir, config=cfg)
    done = load_done_ids(folder)
    new = 0
    for question in questions:
        if question["id"] in done:
            continue
        append_question(folder, step(question))
        done.add(question["id"])
        new += 1
    return {"run_id": folder.name, "run_dir": folder, "new": new,
            "skipped": len(done) - new}


# ------------------------------------------------------------- the fake run


def step_fake(question: dict) -> dict:
    """One question's worth of dummy values, in the full questions.jsonl schema.

    The T05 done-when fake run; the real step arrives with the T08 loop. The
    shape is real: 3 Rounds, stop only at the last, the Answerer called every
    Round (Rounds 1-2 count as shadow), the Rewriter only while continuing.
    total_*_tokens count main calls only — shadow answers are excluded from
    cost (specification 8.2) and stay recoverable in rounds[].tokens.
    """
    qid = question["id"]
    rounds = []
    for round_i in range(1, 4):
        stop = "no" if round_i < 3 else "yes"
        rounds.append({
            "round": round_i,
            "query": f"fake query {round_i}",
            "retrieved_titles": [f"Fake title {round_i}a", f"Fake title {round_i}b"],
            "judge_raw": (f"REASONING: fake reasoning for {qid}.\n"
                          f"ENOUGH: {stop}\nMISSING: fake missing information"),
            "stop": stop,
            "p_yes": 0.5,
            "steer": "fake missing information",
            "shadow_answer": f"fake answer for {qid}",
            "shadow_em": 0,
            "tokens": {"judge": {"prompt": 100, "completion": 20},
                       "rewriter": {"prompt": 50, "completion": 5},
                       "answerer": {"prompt": 80, "completion": 4}},
        })
    return {
        "id": qid,
        "question": question["question"],
        "gold_answer": question["answer"],
        "gold_titles": question["gold_titles"],
        "rounds_used": 3,
        "final_answer": f"fake answer for {qid}",
        "em": 0,
        "f1": 0.0,
        "total_prompt_tokens": 3 * 100 + 2 * 50 + 80,   # main calls only
        "total_completion_tokens": 3 * 20 + 2 * 5 + 4,
        "llm_calls": {"main": 6, "shadow": 2},
        "rounds": rounds,
    }


def fake_run(args) -> None:
    """The T05 fake run: --fake [--questions N] [--fail-after N]."""
    cfg = load_config()
    pilot_path = REPO_ROOT / cfg["paths"]["data_dir"] / "hotpotqa" / "pilot.jsonl"
    questions = [json.loads(line) for line in
                 pilot_path.read_text(encoding="utf-8").splitlines()][: args.questions or 5]

    processed = 0

    def step(question):
        nonlocal processed
        processed += 1
        if args.fail_after is not None and processed > args.fail_after:
            raise KeyboardInterrupt(f"fake session death after {args.fail_after} new questions")
        return step_fake(question)

    try:
        summary = run_questions(questions, step, condition=args.condition,
                                dataset="pilot")
    except KeyboardInterrupt:
        print(f"Interrupted (fake session death) after {processed} new questions; "
              "rerun --fake to resume where it stopped.")
        raise
    print(f"run {summary['run_id']}: {summary['new']} new questions, "
          f"{summary['skipped']} skipped (already logged)")
    print(f"log: {summary['run_dir'] / 'questions.jsonl'}")


def main() -> None:
    from pipeline import loop

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true",
                        help="verify the setup and exit")
    parser.add_argument("--fake", action="store_true",
                        help="fake run with dummy values (the T05 checks)")
    parser.add_argument("--questions", type=int, default=None, metavar="N",
                        help="run size: cap the question set (fake default 5)")
    parser.add_argument("--fail-after", type=int, default=None, metavar="N",
                        help="fake only: die after N new questions, to test resumability")
    parser.add_argument("--condition", default="fake",
                        help="condition suffix in the run id (fake runs default to 'fake')")
    parser.add_argument("--setting", choices=sorted(loop.SETTINGS),
                        help="one of the seven T08 settings (condition = the setting name)")
    parser.add_argument("--set", choices=["pilot", "test"], default="pilot",
                        help="question set (default pilot)")
    parser.add_argument("--fresh", action="store_true",
                        help="start a new run folder instead of resuming the latest one")
    args = parser.parse_args()

    if args.check:
        check()
    elif args.setting:
        run_setting(args)
    elif args.fake:
        fake_run(args)
    else:
        parser.print_help()


def run_setting(args) -> None:
    """Run one setting over a question set (T08): the loop as the step."""
    from pipeline import loop
    from pipeline.retriever import index_dir, load_retriever

    cfg = load_config()
    set_path = REPO_ROOT / cfg["paths"]["data_dir"] / "hotpotqa" / f"{args.set}.jsonl"
    questions = [json.loads(line) for line in
                 set_path.read_text(encoding="utf-8").splitlines()]
    if args.questions:
        questions = questions[: args.questions]
    r = load_retriever(index_dir())

    def step(question):
        return loop.run_question(question, args.setting, r, config=cfg)

    summary = run_questions(questions, step, condition=args.setting,
                            dataset=args.set, config=cfg, resume=not args.fresh)
    print(f"run {summary['run_id']} (setting {args.setting}, set {args.set}): "
          f"{summary['new']} new questions, {summary['skipped']} skipped (already logged)")
    print(f"log: {summary['run_dir'] / 'questions.jsonl'}")


if __name__ == "__main__":
    main()
