"""Answerer: question + Evidence -> short answer. Built in T06.

The Answerer sees the question and the Evidence and outputs a short answer
only (a name, a date, a number, or yes/no) — the format exact match needs.
ask() fills the prompt, calls llm.generate("answerer", prompt), and cleans
the answer (strip, drop one trailing period); an empty output is flagged
parse_error.
"""

from pipeline import llm
from pipeline.judge import render_evidence

PROMPT_VERSION = "v0"

ANSWERER_PROMPT = """Question: {question}

Evidence:

{evidence}

Answer with a short answer only (a name, a date, a number, or yes/no). Do not explain."""


def parse(out: dict) -> dict:
    """The answer is the output with whitespace and a trailing period removed."""
    answer = out["text"].strip()
    if answer.endswith(".") and not answer.endswith(".."):
        answer = answer[:-1].rstrip()
    return {"answer": answer, "parse_error": not answer}


def ask(question: str, evidence, *, config: dict | None = None) -> dict:
    """One Answerer call: the short answer."""
    prompt = ANSWERER_PROMPT.format(question=question, evidence=render_evidence(evidence))
    out = llm.generate("answerer", prompt, config=config)
    return {
        "prompt_version": PROMPT_VERSION,
        "prompt": prompt,
        "raw": out["text"],
        "prompt_tokens": out["prompt_tokens"],
        "completion_tokens": out["completion_tokens"],
        **parse(out),
    }
