"""The retrieval loop: one function per scenario, each answering ONE question. (Skeleton now; T08 completes it.)

    Scenario       Who decides to stop       Who says what is missing     Function
    Closed-book    no search                 -                            closed_book
    Single-turn    stops after one search    -                            single_turn
    Always-loop    never stops early         LLM judge                    always_loop
    Coin-flip      a random Round            LLM judge                    coin_flip (reads Always-loop's log)
    A              LLM judge                 LLM judge                    llm_stops_llm_steers
    B              LLM judge                 Oracle judge                 llm_stops_oracle_steers
    C              Oracle judge              LLM judge                    oracle_stops_llm_steers
    D              Oracle judge              Oracle judge                 oracle_stops_oracle_steers

The 2x2 (A-D) follows the proposal: B - A is the cost of the LLM judge's steering, C - A the cost of its stopping.

Every scenario takes (question, retriever, config) and returns the question's run-log record (T05):
    question_id, question, gold_answer, gold_titles, final_answer, rounds_used, token totals, llm_calls, and
    rounds: one entry per Round with what the retriever, judge, answerer and rewriter each did.
The looping scenarios share run_rounds, so the Round steps are written once.

    Press Debug on this file to run Single-turn on the first Pilot question (open the Ollama server first: pipeline/server_launcher.py).
"""

from pipeline.answerer import PROMPT_VERSION as ANSWER_PROMPT_VERSION
from pipeline.answerer import answer
from pipeline.judge import llm_judge_missing_text, llm_judge_says_enough, oracle_missing_title, oracle_says_enough
from pipeline.retriever import Retriever


# --- Building one question's record ---

def make_retriever_entry(query: str, retrieved: list[dict]) -> dict:
    """What the retriever did in a Round: the query it was sent and every paragraph it brought back, in full."""
    return {"query": query, "retrieved": retrieved}


def make_answerer_entry(evidence: list[dict], reply: dict) -> dict:
    """What the Answerer did: which paragraphs it saw, which prompt version, what it answered, and the tokens used.

    Its full prompt isn't stored: build_answer_prompt rebuilds it exactly from these paragraphs and that version.
    """
    return {
        "paragraphs_seen": [paragraph["title"] for paragraph in evidence],
        "prompt_version": ANSWER_PROMPT_VERSION,
        "answer": reply["text"].strip(),
        "prompt_tokens": reply["prompt_tokens"],
        "completion_tokens": reply["completion_tokens"],
    }


def make_round(round_number: int, retriever_entry: dict | None, answerer_entry: dict,
               judge_entry: dict | None = None, rewriter_entry: dict | None = None) -> dict:
    """One Round, in the order things happen. A role that didn't take part is None (e.g. no Judge in Single-turn)."""
    return {
        "round": round_number,
        "retriever": retriever_entry,
        "judge": judge_entry,
        "answerer": answerer_entry,
        "rewriter": rewriter_entry,
    }


def make_result(question: dict, rounds: list[dict], final_answer: str) -> dict:
    """One question's run-log record: the question, the answer that counts, totals, and every Round."""
    # Here we gather every model call made for this question (Judge, Answerer, Rewriter) to add up its cost.
    model_calls = [single_round[role] for single_round in rounds
                   for role in ("judge", "answerer", "rewriter") if single_round[role] is not None]
    return {
        "question_id": question["id"],
        "question": question["question"],
        "gold_answer": question["answer"],
        "gold_titles": question["gold_titles"],
        "final_answer": final_answer,
        "rounds_used": sum(single_round["retriever"] is not None for single_round in rounds),   # Rounds with a search
        "total_prompt_tokens": sum(model_call["prompt_tokens"] for model_call in model_calls),
        "total_completion_tokens": sum(model_call["completion_tokens"] for model_call in model_calls),
        "llm_calls": len(model_calls),
        "rounds": rounds,
    }


# --- Scenarios that don't loop ---

def closed_book(question: dict, retriever: Retriever, config: dict) -> dict:
    """Closed-book: the Answerer answers from memory, with no search and no Judge (flags memorized answers)."""
    reply = answer(question["question"], evidence=[], config=config)
    # Round 0: no search, so the Answerer sees no paragraphs.
    rounds = [make_round(0, retriever_entry=None, answerer_entry=make_answerer_entry([], reply))]
    return make_result(question, rounds, final_answer=reply["text"].strip())


def single_turn(question: dict, retriever: Retriever, config: dict) -> dict:
    """Single-turn: search once with the question, then the Answerer answers from whatever came back. No Judge."""
    query = question["question"]
    evidence = retriever.search(query, k=config["paragraphs_per_round"])
    reply = answer(question["question"], evidence, config)
    rounds = [make_round(1, make_retriever_entry(query, evidence), make_answerer_entry(evidence, reply))]
    return make_result(question, rounds, final_answer=reply["text"].strip())


# --- Scenarios that loop: they differ only in who decides to stop and who says what is missing ---

def run_rounds(question: dict, retriever: Retriever, config: dict, decide_to_stop, write_steer_signal) -> dict:
    """Run up to config["rounds"] Rounds for one question and return its run-log record. (T08)

    Each Round, top to bottom:
      1. Here we ask the LLM judge (always, so its verdict is logged in every looping scenario).
      2. Here we record a shadow answer from the Evidence so far.
      3. Here we ask decide_to_stop; at the last Round we stop regardless. The shadow answer becomes the answer.
      4. Otherwise here we ask write_steer_signal what is missing, the Rewriter writes the next query,
         and the search adds new paragraphs (skipping titles already held).
    """
    raise NotImplementedError("run_rounds is built in T08, once the Judges (T06, T07) and the Rewriter (T06) exist.")


def never_stop(question: dict, evidence: list[dict], config: dict) -> bool:
    """Stop decision for Always-loop: always "keep going", so every question gets every Round."""
    return False


def always_loop(question: dict, retriever: Retriever, config: dict) -> dict:
    """Always-loop: never stop early; the LLM judge says what is missing."""
    return run_rounds(question, retriever, config,
                      decide_to_stop=never_stop, write_steer_signal=llm_judge_missing_text)


def llm_stops_llm_steers(question: dict, retriever: Retriever, config: dict) -> dict:
    """Condition A (the deployed system): the LLM judge decides when to stop and says what is missing."""
    return run_rounds(question, retriever, config,
                      decide_to_stop=llm_judge_says_enough, write_steer_signal=llm_judge_missing_text)


def llm_stops_oracle_steers(question: dict, retriever: Retriever, config: dict) -> dict:
    """Condition B: the LLM judge decides when to stop; the Oracle judge says what is missing."""
    return run_rounds(question, retriever, config,
                      decide_to_stop=llm_judge_says_enough, write_steer_signal=oracle_missing_title)


def oracle_stops_llm_steers(question: dict, retriever: Retriever, config: dict) -> dict:
    """Condition C: the Oracle judge decides when to stop; the LLM judge says what is missing."""
    return run_rounds(question, retriever, config,
                      decide_to_stop=oracle_says_enough, write_steer_signal=llm_judge_missing_text)


def oracle_stops_oracle_steers(question: dict, retriever: Retriever, config: dict) -> dict:
    """Condition D (best case): the Oracle judge decides when to stop and says what is missing."""
    return run_rounds(question, retriever, config,
                      decide_to_stop=oracle_says_enough, write_steer_signal=oracle_missing_title)


def coin_flip(question: dict, retriever: Retriever, config: dict) -> dict:
    """Coin-flip: stop at a random Round. Reads Always-loop's logged shadow answers; no model calls. (T08/T09)"""
    raise NotImplementedError("Coin-flip reads Always-loop's run log, which exists from T05/T08.")


# The scenario names run.py accepts, and the function each one runs.
SCENARIOS = {
    "closed-book": closed_book,
    "single-turn": single_turn,
    "always-loop": always_loop,
    "coin-flip": coin_flip,
    "A": llm_stops_llm_steers,
    "B": llm_stops_oracle_steers,
    "C": oracle_stops_llm_steers,
    "D": oracle_stops_oracle_steers,
}


if __name__ == "__main__":
    from pipeline.config import load_config
    from pipeline.dataset import load_question_set
    from pipeline.run_log import run_scenario

    # Here we run Single-turn on the first Pilot question; like every pipeline run, it is logged in runs/single-turn/.
    first_question = load_question_set("pilot")[:1]
    run_folder = run_scenario("single-turn", single_turn, first_question, "pilot", load_config(), retriever=Retriever.load())
    print("Log saved in", run_folder)
