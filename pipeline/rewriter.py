"""Rewriter: question + Steer signal -> next search query. Built in T06.

The Rewriter sees the question, the queries already tried, and the Steer
signal (the Judge's MISSING text — only the Rewriter reads it), and outputs
one search query. ask() fills the prompt, calls llm.generate("rewriter",
prompt), and parses the query; an empty output is flagged parse_error.
"""

from pipeline import llm, llm2, load_config

# Here v1 = the few-shot Rewriter prompt (Nov 10); v0 = the earlier prompt without examples.
PROMPT_VERSION = "v1"

REWRITER_PROMPT = """You are the Rewriter in a search loop over Wikipedia paragraphs. Each round, a Judge says what the evidence is still missing. Your job is to turn that into the next keyword search query.

Rules, in order:
1. Find the new thing the Judge named: a person, place, work, organisation or event that appears in "still missing" but in none of the queries already tried. Write it in "new_thing". If the Judge named nothing new, write "none".
2. Build the query around that new thing. The best query looks like the title of a Wikipedia article: the name itself, plus at most two words saying which fact is wanted.
3. Never reuse the question's own keywords as the query. They were already tried in Round 1 and found the paragraphs you already have.
4. Never write a query that is the same as, or a shorter form of, a query already tried. If your query would repeat one, add the new thing's name to it or choose the other entity the question mentions.
5. Ignore words like "confirmation that", "explicit", "specific" and "evidence that" in the Judge's text. Keep the names.
6. If "still missing" is "nothing", search for the entity the question mentions that no tried query has named yet.
Question: {question}
Search queries already tried:
{queries}
What the evidence is still missing:
{steer}
Examples:
Question: Who wrote the novel that the 2004 film Troy was based on?
Tried: the question as written; "Troy 2004 film novel"
Still missing: explicit confirmation that 'The Iliad' is the work the film was based on, and who wrote it
Reply: The Iliad author
Question: What record label is owned by one of the singers on the 1993 album Honky Tonk Angels?
Tried: the question as written
Still missing: ownership of a record label by one of Dolly Parton, Loretta Lynn or Tammy Wynette
Reply: Dolly Parton record label"""

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