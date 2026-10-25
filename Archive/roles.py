"""Parse-rate check for the three roles (T06's done-when).

    python -m pipeline.roles --questions 20

For each question: retrieve the top-2 paragraphs as stand-in Evidence, call
the Judge and the Answerer on it, and the Rewriter on the Judge's steer
signal. Reports how often each role's output parses, and the Judge's p_yes
range. The official 95%-parse-rate check runs on vLLM (kaggle.ipynb cell 7);
the Mac/Ollama numbers are dev sanity only.
"""

import argparse
import json

from pipeline import REPO_ROOT, answerer, judge, load_config, retriever, rewriter


def check(questions, r) -> dict:
    """One call per role per question; returns parse counts and p_yes stats."""
    counts = {"judge": 0, "rewriter": 0, "answerer": 0}
    p_yes_values = []
    print(f"{'id':<26} {'judge':<8} {'p_yes':<8} {'rewriter':<8} {'answerer':<8}")
    for q in questions:
        evidence = retriever.search(q["question"], k=2, retriever=r)
        j = judge.ask(q["question"], evidence)
        w = rewriter.ask(q["question"], [q["question"]], j["missing"])
        a = answerer.ask(q["question"], evidence)
        counts["judge"] += not j["parse_error"]
        counts["rewriter"] += not w["parse_error"]
        counts["answerer"] += not a["parse_error"]
        if j["p_yes"] is not None:
            p_yes_values.append(j["p_yes"])
        p_yes_display = f"{j['p_yes']:.3f}" if j["p_yes"] is not None else "-"
        print(f"{q['id']:<26} {j['enough']:<8} {p_yes_display:<8} "
              f"{w['query'] or 'PARSE ERROR':<8} {a['answer'] or 'PARSE ERROR':<8}")
    return {"counts": counts, "total": len(questions), "p_yes_values": p_yes_values}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", type=int, default=20, metavar="N",
                        help="how many questions to check (default 20)")
    parser.add_argument("--set", choices=["pilot", "test"], default="pilot",
                        help="which question set (default pilot)")
    args = parser.parse_args()

    cfg = load_config()
    set_path = REPO_ROOT / cfg["paths"]["data_dir"] / "hotpotqa" / f"{args.set}.jsonl"
    questions = [json.loads(line) for line in
                 set_path.read_text(encoding="utf-8").splitlines()][: args.questions]
    r = retriever.load_retriever(retriever.index_dir())
    result = check(questions, r)

    print()
    for role in ("judge", "rewriter", "answerer"):
        rate = result["counts"][role] / result["total"]
        print(f"{role:<9} parsed {result['counts'][role]}/{result['total']} "
              f"({rate:.1%})")
    if result["p_yes_values"]:
        print(f"judge p_yes: min {min(result['p_yes_values']):.3f}, "
              f"max {max(result['p_yes_values']):.3f}, "
              f"missing {result['total'] - len(result['p_yes_values'])}")


if __name__ == "__main__":
    main()
