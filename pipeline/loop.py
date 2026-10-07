"""The retrieval loop: wires the pieces together for the seven settings. Built in T08.

Each setting is the same loop with two swappable parts (specification 6.2):
who makes the Stop decision and who produces the Steer signal. The retriever,
Rewriter and Answerer are identical across settings; the LLM judge is called
in every looped setting (even where its output is not used) so its verdict is
always logged.

    A  stop: LLM    steer: LLM        B  stop: oracle  steer: LLM
    C  stop: LLM    steer: oracle     D  stop: oracle  steer: oracle
    single-turn   stop: always, no steer    always-loop  stop: never, LLM steer
    closed-book   no retrieval, one Answerer call

run_question() returns the T05 questions.jsonl record. Evidence trimming to
the token budget lands in T10 (`loop.token_budget` is null until then).
"""

from pipeline import answerer as answerer_role
from pipeline import judge
from pipeline import retriever as retriever_mod
from pipeline import rewriter as rewriter_role
from pipeline import load_config

SETTINGS = {
    "A": {"stop": "llm", "steer": "llm"},
    "B": {"stop": "oracle", "steer": "llm"},
    "C": {"stop": "llm", "steer": "oracle"},
    "D": {"stop": "oracle", "steer": "oracle"},
    "single-turn": {"stop": "always", "steer": "none"},
    "always-loop": {"stop": "never", "steer": "llm"},
    "closed-book": {"stop": None, "steer": None},
}


def _em(predicted: str, gold: str) -> int:
    """Exact match, case- and whitespace-insensitive. T09 generalizes scoring;
    this only fills the log's em / shadow_em fields."""
    return int(" ".join(predicted.split()).casefold() == " ".join(gold.split()).casefold())


def _decide_stop(source: str, judge_result: dict, evidence_titles, gold_titles) -> bool:
    if source == "llm":
        return judge_result["enough"] == "yes"
    if source == "oracle":
        return judge.oracle_stop(evidence_titles, gold_titles)
    if source == "always":
        return True
    if source == "never":
        return False
    raise ValueError(f"unknown stop source {source!r}")


def _decide_steer(source: str, judge_result: dict, evidence_titles, gold_titles):
    if source == "llm":
        return judge_result["missing"]
    if source == "oracle":
        return judge.oracle_steer(evidence_titles, gold_titles)
    if source == "none":
        return None
    raise ValueError(f"unknown steer source {source!r}")


def _role_tokens(result: dict) -> dict:
    return {"prompt": result["prompt_tokens"], "completion": result["completion_tokens"]}


def run_question(question: dict, setting: str, r, *, config: dict | None = None) -> dict:
    """One setting's run over one question: the questions.jsonl record.

    `question` is a set line (id, question, answer, gold_titles); `r` is the
    retriever (bm25s.BM25). The record's token totals and llm_calls count
    main calls only — shadow answers (the Answerer at every Round before the
    stopping one) are excluded from cost, specification 8.2.
    """
    cfg = load_config() if config is None else config
    if setting not in SETTINGS:
        raise ValueError(f"unknown setting {setting!r}; expected one of {sorted(SETTINGS)}")
    spec = SETTINGS[setting]
    max_rounds = cfg["loop"]["rounds"]

    if setting == "closed-book":
        answer = answerer_role.ask(question["question"], [], config=cfg)
        final = answer["answer"]
        return {
            "id": question["id"],
            "question": question["question"],
            "gold_answer": question["answer"],
            "gold_titles": question["gold_titles"],
            "rounds_used": 0,
            "final_answer": final,
            "em": _em(final, question["answer"]),
            "f1": 0.0,  # computed in T09
            "total_prompt_tokens": answer["prompt_tokens"],
            "total_completion_tokens": answer["completion_tokens"],
            "llm_calls": {"main": 1, "shadow": 0},
            "rounds": [],
        }

    k = cfg["retrieval"]["k"]
    if not k:
        raise ValueError("retrieval.k is not set in config.yaml (decided in T10)")

    evidence, past_queries, rounds = [], [], []
    steer = None
    for round_i in range(1, max_rounds + 1):
        rewriter_result = None
        if round_i == 1:
            query = question["question"]  # the raw question opens every run
        else:
            rewriter_result = rewriter_role.ask(
                question["question"], past_queries, steer, config=cfg)
            query = rewriter_result["query"]
        past_queries.append(query)

        hits = retriever_mod.search(
            query, k=k, exclude_titles=[p["title"] for p in evidence], retriever=r)
        evidence += hits
        titles = [p["title"] for p in evidence]

        judge_result = judge.ask(question["question"], evidence, config=cfg)
        shadow = answerer_role.ask(question["question"], evidence, config=cfg)

        stop = _decide_stop(spec["stop"], judge_result, titles, question["gold_titles"])
        stop = stop or round_i == max_rounds  # Round 3 answers regardless
        steer = _decide_steer(spec["steer"], judge_result, titles, question["gold_titles"])

        rounds.append({
            "round": round_i,
            "query": query,
            "retrieved_titles": [hit["title"] for hit in hits],
            "judge_raw": judge_result["raw"],
            "stop": "yes" if stop else "no",
            "p_yes": judge_result["p_yes"],
            "steer": steer,
            "shadow_answer": shadow["answer"],
            "shadow_em": _em(shadow["answer"], question["answer"]),
            "tokens": {"judge": _role_tokens(judge_result),
                       "rewriter": _role_tokens(rewriter_result) if rewriter_result else None,
                       "answerer": _role_tokens(shadow)},
        })
        if stop:
            break

    final = rounds[-1]["shadow_answer"]  # the Answerer at the stopping Round
    return _finish(rounds, final, question)


def _finish(rounds: list, final: str, question: dict) -> dict:
    """Assemble the record from the per-Round log (shared token math)."""
    rewriters = sum(1 for r in rounds if r["tokens"]["rewriter"])
    main_prompt = sum(r["tokens"]["judge"]["prompt"] for r in rounds) \
        + sum(r["tokens"]["rewriter"]["prompt"] for r in rounds if r["tokens"]["rewriter"]) \
        + rounds[-1]["tokens"]["answerer"]["prompt"]
    main_completion = sum(r["tokens"]["judge"]["completion"] for r in rounds) \
        + sum(r["tokens"]["rewriter"]["completion"] for r in rounds if r["tokens"]["rewriter"]) \
        + rounds[-1]["tokens"]["answerer"]["completion"]
    return {
        "id": question["id"],
        "question": question["question"],
        "gold_answer": question["answer"],
        "gold_titles": question["gold_titles"],
        "rounds_used": len(rounds),
        "final_answer": final,
        "em": _em(final, question["answer"]),
        "f1": 0.0,  # computed in T09
        "total_prompt_tokens": main_prompt,
        "total_completion_tokens": main_completion,
        "llm_calls": {"main": len(rounds) + rewriters + 1,  # judges + rewriters + final answerer
                      "shadow": len(rounds) - 1},
        "rounds": rounds,
    }
