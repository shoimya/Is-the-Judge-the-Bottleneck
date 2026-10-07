"""LLM judge and Oracle judge: each gives a Stop decision and a Steer signal.

Built in T06 (LLM judge) and T07 (Oracle judge).

The LLM judge sees the question and the Evidence, and answers in the fixed
format (specification 6.3):

    REASONING: <a few sentences>
    ENOUGH: yes|no
    MISSING: <what information is still needed, or "nothing">

ask() fills the prompt, calls llm.generate("judge", prompt), and parses the
three fields. ENOUGH is the Stop decision; MISSING is the Steer signal (only
the Rewriter reads it). p_yes is P("yes") / (P("yes") + P("no")) from the
token probabilities at the ENOUGH position. Malformed output counts as "no"
and is flagged parse_error (the run log keeps judge_raw either way).
"""

import re

from pipeline import llm

PROMPT_VERSION = "v0"

JUDGE_PROMPT = """Question: {question}

Evidence:

{evidence}

Answer in exactly this format:
REASONING: <a few sentences about what the evidence supports and what it does not>
ENOUGH: yes|no
MISSING: <what information is still needed to answer, or "nothing">"""

_REASONING_RE = re.compile(r"REASONING\s*:\s*(.*?)(?=\s*(?:ENOUGH|MISSING)\s*:|\Z)",
                           re.IGNORECASE | re.DOTALL)
_ENOUGH_RE = re.compile(r"ENOUGH\s*:\s*(yes|no)\b", re.IGNORECASE)
_MISSING_RE = re.compile(r"MISSING\s*:\s*(.*?)\s*\Z", re.IGNORECASE | re.DOTALL)


def render_evidence(evidence) -> str:
    """The Evidence as the Judge and Answerer read it: numbered paragraphs."""
    blocks = []
    for i, paragraph in enumerate(evidence, 1):
        blocks.append(f"[{i}] {paragraph['title']}\n{paragraph['text']}")
    return "\n\n".join(blocks)


def parse(out: dict) -> dict:
    """Pull REASONING, ENOUGH, MISSING and p_yes out of one generate() result.

    p_yes comes from the token logprob table at the ENOUGH position (found by
    rejoining the sampled tokens; the table's yes/no spellings are normalized).
    If ENOUGH cannot be parsed the verdict counts as "no" with parse_error.
    """
    text = out["text"]
    reasoning = _REASONING_RE.search(text)
    enough = _ENOUGH_RE.search(text)
    missing = _MISSING_RE.search(text)
    return {
        "reasoning": reasoning.group(1).strip() if reasoning else "",
        "enough": enough.group(1).lower() if enough else "no",
        "missing": missing.group(1).strip() if missing else "",
        "p_yes": p_yes_at_enough(out),
        "parse_error": reasoning is None or enough is None,
    }


def p_yes_at_enough(out: dict) -> float | None:
    """P(yes) / (P(yes) + P(no)) from the table at the ENOUGH position.

    The position is found by rejoining the sampled tokens: the first yes/no
    token at or after the "ENOUGH" header. Fallback (odd tokenizations): the
    last yes/no token anywhere. None when the backend gave no usable tables.
    """
    tables = out.get("logprobs") or []
    sampled = []
    for table in tables:
        token = next((entry["token"] for entry in table if entry.get("sampled")), None)
        if token is not None:
            sampled.append((token, table))
    text = "".join(token for token, _ in sampled)
    anchor = text.find("ENOUGH")
    if anchor >= 0:
        offset = 0
        for token, table in sampled:
            start, offset = offset, offset + len(token)
            if start >= anchor and token in llm.YES_NO_TOKENS:
                return llm.p_yes_normalized(table)
    for token, table in reversed(sampled):
        if token in llm.YES_NO_TOKENS:
            return llm.p_yes_normalized(table)
    return None


def ask(question: str, evidence, *, config: dict | None = None) -> dict:
    """One Judge call: the run log's judge_raw, stop, steer and p_yes."""
    prompt = JUDGE_PROMPT.format(question=question, evidence=render_evidence(evidence))
    out = llm.generate("judge", prompt, config=config)
    return {
        "prompt_version": PROMPT_VERSION,
        "prompt": prompt,
        "raw": out["text"],
        "prompt_tokens": out["prompt_tokens"],
        "completion_tokens": out["completion_tokens"],
        **parse(out),
    }


# ------------------------------------------------------------- oracle judge
#
# The "perfect judge" the LLM judge is compared against (T07, specification
# 6.5). Pure Python, no model calls: it looks at the answer key (the Gold
# titles). For steering it only names WHICH paragraph is missing, never its
# text, so the Rewriter still has to write a query and BM25 still has to find
# it — that keeps the steering comparison fair.


def _normalize_title(title: str) -> str:
    """Case- and whitespace-insensitive: "  yoruba   People " == "yoruba people"."""
    return " ".join(title.split()).casefold()


def oracle_stop(evidence_titles, gold_titles) -> bool:
    """True when every Gold paragraph title is in the Evidence."""
    evidence = {_normalize_title(title) for title in evidence_titles}
    return all(_normalize_title(title) in evidence for title in gold_titles)


def oracle_steer(evidence_titles, gold_titles) -> str:
    """The first missing Gold title, in the dataset's order, or "nothing"."""
    evidence = {_normalize_title(title) for title in evidence_titles}
    for title in gold_titles:
        if _normalize_title(title) not in evidence:
            return f"Missing information about: {title}"
    return "nothing"
