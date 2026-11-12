"""LLM judge and Oracle judge: each gives a Stop decision and a Steer signal.

Built in T06 (LLM judge) and T07 (Oracle judge).

The LLM judge sees the question and the Evidence, and answers in the fixed
format (specification 6.3):

    REASONING: <a few sentences>
    ENOUGH: yes|no
    MISSING: <what information is still needed, or "nothing">

ask() fills the prompt, calls llm.generate("judge", prompt), and parses the
three fields. ENOUGH is the Stop decision; MISSING is the Steer signal (the
next Round's search query, as written). p_yes is P("yes") / (P("yes") + P("no")) from the
token probabilities at the ENOUGH position. Malformed output counts as "no"
and is flagged parse_error (the run log keeps judge_raw either way).
"""

import re

from pipeline import llm, llm2, load_config

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
# titles). For steering it only names WHICH paragraph is missing (its title),
# never its text, so BM25 still has to find it — that keeps the steering
# comparison fair. Since Nov 12 there is no Rewriter: the steer goes to BM25
# as written, so the Oracle's steer is the bare title (the old "Missing
# information about:" prefix only added noise words to the search).


def _normalize_title(title: str) -> str:
    """Case- and whitespace-insensitive: "  yoruba   People " == "yoruba people"."""
    return " ".join(title.split()).casefold()


def oracle_stop(evidence_titles, gold_titles) -> bool:
    """True when every Gold paragraph title is in the Evidence."""
    evidence = {_normalize_title(title) for title in evidence_titles}
    return all(_normalize_title(title) in evidence for title in gold_titles)


def oracle_steer(evidence_titles, gold_titles) -> str:
    """The first missing Gold title, in the dataset's order, or "nothing". It is also the next search query."""
    evidence = {_normalize_title(title) for title in evidence_titles}
    for title in gold_titles:
        if _normalize_title(title) not in evidence:
            return title
    return "nothing"

# -----
# The LLM judge on llm2: press Run to judge the Evidence for Pilot question 1, next to the Oracle judge's verdict
# ---------------------------------------------------------------------------

# Here are the Judge's rules, sent as the system prompt of every call.
JUDGE_SYSTEM_PROMPT = ("You are the Judge in a search loop. Decide whether the evidence is enough to answer the "
                       "question. Reason briefly first, then decide. If it is not enough, name exactly which fact "
                       "or entity is still missing.")

# Here is the user prompt: the question, the Evidence as numbered paragraphs, and what to answer.
JUDGE_PROMPT_JSON = ("Question: {question}\n\nEvidence:\n\n{evidence}\n\n"
                     "Answer as JSON with keys reasoning (a few sentences about what the evidence supports and "
                     "what it does not), enough (yes or no), and missing (what is still needed, or nothing).")

# Here is the JSON shape every judgment must have. Reasoning comes first, so the Judge thinks before it decides.
JUDGMENT_FORMAT = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "enough": {"type": "string", "enum": ["yes", "no"]},
        "missing": {"type": "string"},
    },
    "required": ["reasoning", "enough", "missing"],
}


def judge_prompt(question: str, evidence: list[dict]) -> str:
    """The prompt the Judge reads: the question, then the Evidence as numbered paragraphs."""
    return JUDGE_PROMPT_JSON.format(question=question, evidence=render_evidence(evidence))


def judge_evidence(question: str, evidence: list[dict]) -> dict:
    """Ask the LLM judge whether the Evidence is enough to answer the question, and what is missing if not.

    Returns a dictionary with:
        reasoning         the Judge's reasoning
        enough            the Stop decision: "yes" or "no" ("no" if the reply never fitted its JSON shape)
        missing           the Steer signal: what is still needed (the next Round searches with it as written)
        p_yes             how sure the Judge was that the Evidence is enough (None if the odds weren't there)
        reply             the model's full reply, exactly as it wrote it (the last attempt's)
        prompt            exactly what the Judge read (on a retry, the model also saw a correction note)
        prompt_version    which version of the prompts was used
        attempts          how many calls it took (send_to_ollama_with_retries)
        answer_problems   what is still wrong with the reply ([] = it fits the JSON shape)
        tokens_read, tokens_written, seconds, prompt_may_be_cut, reply_was_cut   the call's costs and warnings
    """
    config = load_config()
    prompt = judge_prompt(question, evidence)
    # Here the model, reply limit, temperature and seed all come from config.yaml, the same for every method.
    result = llm2.send_to_ollama_with_retries(
        llm2.model_for_role("judge"), JUDGE_SYSTEM_PROMPT, prompt, JUDGMENT_FORMAT,
        temperature=config["llm"]["temperature"], seed=config["seeds"][0],
        reply_limit_in_tokens=llm2.reply_limit_for_role("judge"),
    )
    # Here a reply that never fitted its shape (after every attempt) counts as "no" with no confidence,
    # as CONTEXT.md decides, and answer_problems says why.
    reply_fits = not result["answer_problems"]
    judgment = result["answer"] if reply_fits else {}
    # Here we read the odds the model gave "yes" and "no" at the word after "enough" (only from a reply that fits).
    p_yes = llm2.p_yes_normalized(llm2.odds_at_field(result["word_odds"], "enough")) if reply_fits else None
    return {
        "reasoning": judgment.get("reasoning", ""),
        "enough": judgment.get("enough", "no"),
        "missing": judgment.get("missing", ""),
        "p_yes": p_yes,
        "reply": result["reply"],
        "prompt": prompt,
        "prompt_version": PROMPT_VERSION,
        "attempts": result["attempts"],
        "answer_problems": result["answer_problems"],
        "tokens_read": result["tokens_read"],
        "tokens_written": result["tokens_written"],
        "seconds": result["seconds"],
        "prompt_may_be_cut": result["prompt_may_be_cut"],
        "reply_was_cut": result["reply_was_cut"],
    }


if __name__ == "__main__":
    from pipeline.dataset import load_questions
    from pipeline.retriever import Retriever

    first_question = load_questions("pilot")[0]
    # Here we search once with the question, as Round 1 of every looping method does.
    demo_evidence = Retriever.load().search(first_question["question"], k=load_config()["retrieval"]["k"])
    evidence_titles = [paragraph["title"] for paragraph in demo_evidence]

    server_started_here = llm2.start_ollama_server()
    try:
        judgment_result = judge_evidence(first_question["question"], demo_evidence)
        print(f"\nQuestion:    {first_question['question']}")
        print(f"Evidence:    {', '.join(evidence_titles)}")
        print(f"Gold titles: {', '.join(first_question['gold_titles'])}")
        print("\nLLM judge:")
        print(f"  reasoning: {judgment_result['reasoning']}")
        print(f"  enough:    {judgment_result['enough']}   (p_yes = {judgment_result['p_yes']})")
        print(f"  missing:   {judgment_result['missing']}")
        print(f"  Attempts: {judgment_result['attempts']}   Tokens read: {judgment_result['tokens_read']}, "
              f"written: {judgment_result['tokens_written']}, in {judgment_result['seconds']} s")
        # Here the Oracle judge looks at the Gold titles, for comparison.
        print("\nOracle judge:")
        print(f"  enough:    {'yes' if oracle_stop(evidence_titles, first_question['gold_titles']) else 'no'}")
        print(f"  missing:   {oracle_steer(evidence_titles, first_question['gold_titles'])}\n")
    finally:
        # Here we stop the server only if this demo started it.
        if server_started_here:
            llm2.stop_ollama_server()