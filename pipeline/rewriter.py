"""Rewriter: question + queries already tried + Steer signal -> the next search query. (Built in T06.)"""


def write_next_query(question: dict, past_queries: list[str], steer_signal: str, config: dict) -> str:
    """Ask the model for one new search query aimed at what the Steer signal says is missing. (T06)"""
    raise NotImplementedError("The Rewriter is built in T06.")
