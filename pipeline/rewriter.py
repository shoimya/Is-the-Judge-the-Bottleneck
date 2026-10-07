"""Rewriter: question + Steer signal -> next search query. Built in T06.

The Rewriter sees the question, the queries already tried, and the Steer
signal (the Judge's MISSING text — only the Rewriter reads it), and outputs
one search query. ask() fills the prompt, calls llm.generate("rewriter",
prompt), and parses the query; an empty output is flagged parse_error.
"""

from pipeline import llm

PROMPT_VERSION = "v0"

REWRITER_PROMPT = """Question: {question}

Search queries already tried:
{queries}

What the evidence is still missing:
{steer}

Write one short search query (a few words, no punctuation) for a Wikipedia keyword search that finds the missing information."""


def render_queries(queries) -> str:
    """The past queries as the Rewriter reads them."""
    if not queries:
        return "(none yet)"
    return "\n".join(f"- {query}" for query in queries)


def parse(out: dict) -> dict:
    """The query is the first non-empty line of the output, cleaned."""
    for line in out["text"].splitlines():
        query = line.strip().strip('"').strip()
        if query:
            return {"query": query, "parse_error": False}
    return {"query": "", "parse_error": True}


def ask(question: str, past_queries, steer: str, *, config: dict | None = None) -> dict:
    """One Rewriter call: the next search query."""
    prompt = REWRITER_PROMPT.format(
        question=question,
        queries=render_queries(past_queries),
        steer=steer or "nothing",
    )
    out = llm.generate("rewriter", prompt, config=config)
    return {
        "prompt_version": PROMPT_VERSION,
        "prompt": prompt,
        "raw": out["text"],
        "prompt_tokens": out["prompt_tokens"],
        "completion_tokens": out["completion_tokens"],
        **parse(out),
    }
