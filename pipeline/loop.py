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

Every scenario takes (question, retriever, config) and returns a result:
    {"setting", "question_id", "answer", "rounds_used", "evidence_titles"}
(T05 adds the full run log.) The looping scenarios share run_rounds, so the Round steps are written once.

    Press Debug on this file to run Single-turn on the first Pilot question (needs `ollama serve` on the Mac).
"""

from pipeline.answerer import answer
from pipeline.judge import llm_judge_missing_text, llm_judge_says_enough, oracle_missing_title, oracle_says_enough
from pipeline.retriever import Retriever


def make_result(setting: str, question: dict, answer_text: str, rounds_used: int, evidence: list[dict]) -> dict:
    """What a scenario returns for one question."""
    return {
        "setting": setting,
        "question_id": question["id"],
        "answer": answer_text,
        "rounds_used": rounds_used,
        "evidence_titles": [paragraph["title"] for paragraph in evidence],
    }


# --- Scenarios that don't loop ---

def closed_book(question: dict, retriever: Retriever, config: dict) -> dict:
    """Closed-book: the Answerer answers from memory, with no search and no Judge (flags memorized answers)."""
    reply = answer(question["question"], evidence=[], config=config)
    return make_result("closed-book", question, reply["text"].strip(), rounds_used=0, evidence=[])


def single_turn(question: dict, retriever: Retriever, config: dict) -> dict:
    """Single-turn: search once with the question, then the Answerer answers from whatever came back. No Judge."""
    evidence = retriever.search(question["question"], k=config["paragraphs_per_round"])
    reply = answer(question["question"], evidence, config)
    return make_result("single-turn", question, reply["text"].strip(), rounds_used=1, evidence=evidence)


# --- Scenarios that loop: they differ only in who decides to stop and who says what is missing ---

def run_rounds(question: dict, retriever: Retriever, config: dict, setting: str,
               decide_to_stop, write_steer_signal) -> dict:
    """Run up to config["rounds"] Rounds for one question and return the result. (T08)

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
    return run_rounds(question, retriever, config, "always-loop",
                      decide_to_stop=never_stop, write_steer_signal=llm_judge_missing_text)


def llm_stops_llm_steers(question: dict, retriever: Retriever, config: dict) -> dict:
    """Condition A (the deployed system): the LLM judge decides when to stop and says what is missing."""
    return run_rounds(question, retriever, config, "A",
                      decide_to_stop=llm_judge_says_enough, write_steer_signal=llm_judge_missing_text)


def llm_stops_oracle_steers(question: dict, retriever: Retriever, config: dict) -> dict:
    """Condition B: the LLM judge decides when to stop; the Oracle judge says what is missing."""
    return run_rounds(question, retriever, config, "B",
                      decide_to_stop=llm_judge_says_enough, write_steer_signal=oracle_missing_title)


def oracle_stops_llm_steers(question: dict, retriever: Retriever, config: dict) -> dict:
    """Condition C: the Oracle judge decides when to stop; the LLM judge says what is missing."""
    return run_rounds(question, retriever, config, "C",
                      decide_to_stop=oracle_says_enough, write_steer_signal=llm_judge_missing_text)


def oracle_stops_oracle_steers(question: dict, retriever: Retriever, config: dict) -> dict:
    """Condition D (best case): the Oracle judge decides when to stop and says what is missing."""
    return run_rounds(question, retriever, config, "D",
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

    config = load_config()
    first_question = load_question_set("pilot")[0]
    result = single_turn(first_question, Retriever.load(), config)
    print("Question:     ", first_question["question"])
    print("Gold answer:  ", first_question["answer"])
    print("Model answer: ", result["answer"])
    print("Evidence:     ", result["evidence_titles"])
