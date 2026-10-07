"""The two Judges. Each gives a Stop decision ("is the Evidence enough?") and a Steer signal ("what is missing?").

    LLM judge     asks the model (built in T06)
    Oracle judge  looks at the Gold paragraphs, no model (built in T07)

loop.py's scenarios choose one Judge for stopping and one for steering. Until T06/T07 these functions only
describe what they will do.
"""


def llm_judge_says_enough(question: dict, evidence: list[dict], config: dict) -> bool:
    """Stop decision from the LLM judge: True if the model says ENOUGH: yes. (T06)"""
    raise NotImplementedError("The LLM judge is built in T06.")


def llm_judge_missing_text(question: dict, evidence: list[dict], config: dict) -> str:
    """Steer signal from the LLM judge: the model's MISSING: line. (T06)"""
    raise NotImplementedError("The LLM judge is built in T06.")


def oracle_says_enough(question: dict, evidence: list[dict], config: dict) -> bool:
    """Stop decision from the Oracle judge: True once every Gold paragraph is in the Evidence. (T07)"""
    raise NotImplementedError("The Oracle judge is built in T07.")


def oracle_missing_title(question: dict, evidence: list[dict], config: dict) -> str:
    """Steer signal from the Oracle judge: "Missing information about: <first missing Gold title>". (T07)"""
    raise NotImplementedError("The Oracle judge is built in T07.")
