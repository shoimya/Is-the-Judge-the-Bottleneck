"""Rewriter: question + Steer signal -> next search query. Built in T06.

The Rewriter sees the question, the queries already tried, and the Steer
signal (the Judge's MISSING text), and outputs one search query. It runs only
under Pipeline 1 (config.yaml's loop.pipeline); under Pipeline 2 the steer is
the next query as written, and this file is not called. ask() fills the prompt,
calls llm.generate("rewriter", prompt), and parses the query; an empty output
is flagged parse_error.
"""

from pipeline import llm, llm2, load_config

# Here v1 = the few-shot Rewriter prompt (Nov 10), used again from Nov 13: v2 (Nov 12) also showed the Rewriter the
# earlier Rounds' steers next to the queries it wrote for them, and that changed nothing, so it was dropped.
# v0 = the earlier prompt without examples.
PROMPT_VERSION = "v1"

REWRITER_PROMPT = """Question: {question}

Search queries already tried:
{queries}

What the evidence is still missing:
{steer}

Write one short search query (a few words, no punctuation) for a Wikipedia keyword search that finds the missing information. Name the missing person, place or thing directly, and never repeat a query already tried.

Examples (question / still missing -> query):
- Which gaming console was both Yakuza Kiwami and Yakuza 0 released on? / which consoles Yakuza 0 was released on -> Yakuza 0 release consoles
- In the NASA mission where Moon trees were taken into space, what was the nickname of the Command Module? / the nickname of the Apollo 14 Command Module -> Apollo 14 Command Module
- Which airport is located further north in the United States, San Angelo Regional Airport or Grand Junction Regional Airport? / where Grand Junction Regional Airport is located -> Grand Junction Regional Airport location
- Are both American Foxhound and Löwchen types of Foxhounds? / what kind of dog the Löwchen is -> Löwchen dog breed"""


def render_queries(queries) -> str:
    """The queries already tried, one per line, as the Rewriter reads them."""
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
# ------
# The Rewriter on llm2: press Run to rewrite a made-up Steer signal into a search query, then search with it
# ---------------------------------------------------------------------------

# Here are the Rewriter's rules, sent as the system prompt of every call.
REWRITER_SYSTEM_PROMPT = ("You are the Rewriter in a search loop. Write one short search query (a few words, "
                          "no punctuation) that finds the missing information in a Wikipedia keyword search.")

# Here is the JSON shape every reply must have: one "query" text.
QUERY_FORMAT = {
    "type": "object",
    "properties": {"query": {"type": "string"}},
    "required": ["query"],
}


def rewriter_prompt(question: str, past_queries: list[str], steer: str) -> str:
    """The prompt the Rewriter reads: the question, the queries already tried, and what the Evidence is missing."""
    # Here an empty Steer signal is written as "nothing", as the Judge writes it when nothing is missing.
    return REWRITER_PROMPT.format(question=question, queries=render_queries(past_queries), steer=steer or "nothing")


def rewrite_query(question: str, past_queries: list[str], steer: str) -> dict:
    """Turn the question, the queries already tried and the Steer signal into the next search query.

    Returns a dictionary with:
        query             the next search query ("" if the model never gave one in the right format)
        reply             the model's full reply, exactly as it wrote it (the last attempt's)
        prompt            exactly what the Rewriter read (on a retry, the model also saw a correction note)
        prompt_version    which version of the prompts was used
        attempts          how many calls it took (send_to_ollama_with_retries)
        answer_problems   what is still wrong with the reply ([] = it fits the JSON shape)
        tokens_read, tokens_written, seconds, prompt_may_be_cut, reply_was_cut   the call's costs and warnings
    """
    config = load_config()
    prompt = rewriter_prompt(question, past_queries, steer)
    # Here the model, reply limit, temperature and seed all come from config.yaml, the same for every method.
    result = llm2.send_to_ollama_with_retries(
        llm2.model_for_role("rewriter"), REWRITER_SYSTEM_PROMPT, prompt, QUERY_FORMAT,
        temperature=config["llm"]["temperature"], seed=config["seeds"][0],
        reply_limit_in_tokens=llm2.reply_limit_for_role("rewriter"),
    )
    # Here a reply that never fitted its shape (after every attempt) counts as an empty query.
    query_text = (result["answer"] or {}).get("query", "")
    return {
        # Here we drop surrounding spaces and quotes, which the search would only treat as noise.
        "query": query_text.strip().strip('"').strip(),
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
    from pipeline.retriever import Retriever

    demo_question = "Which river flows through the capital city of Hungary?"
    demo_past_queries = [demo_question]
    demo_steer = "the name of the river that flows through Budapest"

    server_started_here = llm2.start_ollama_server()
    try:
        rewrite_result = rewrite_query(demo_question, demo_past_queries, demo_steer)
        print(f"\nQuestion:     {demo_question}")
        print(f"Steer signal: {demo_steer}")
        print(f"New query:    {rewrite_result['query']}")
        print(f"Attempts: {rewrite_result['attempts']}   Tokens read: {rewrite_result['tokens_read']}, "
              f"written: {rewrite_result['tokens_written']}, in {rewrite_result['seconds']} s")

        # Here we search with the new query, to see whether it finds better pages than the question did.
        print("\nTop 3 pages for the new query:")
        for rank, paragraph in enumerate(Retriever.load().search(rewrite_result["query"], k=3), start=1):
            print(f"  {rank}. [{paragraph['score']:.1f}] {paragraph['title']}")
        print()
    finally:
        # Here we stop the server only if this demo started it.
        if server_started_here:
            llm2.stop_ollama_server()