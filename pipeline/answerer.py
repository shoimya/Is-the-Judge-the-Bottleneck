"""Answerer: question + Evidence -> short answer. Built in T06.

The Answerer sees the question and the Evidence and outputs a short answer
only (a name, a date, a number, or yes/no) — the format exact match needs.
ask() fills the prompt, calls llm.generate("answerer", prompt), and cleans
the answer (strip, drop one trailing period); an empty output is flagged
parse_error.
"""

from pipeline import llm, llm2, load_config
from pipeline.dataset import load_questions
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


# ---------------------------------------------------------------------------
# The Answerer on llm2: press Run to answer Pilot question 1 with no Evidence (as Closed-book does)
# ---------------------------------------------------------------------------

# Here are the Answerer's rules, sent as the system prompt of every call.
ANSWERER_SYSTEM_PROMPT = ("You are the Answerer in a question-answering system, think deeply before answering. Give a short answer only "
                          "(a name, a date, a number, or yes/no). Do not explain.")

# Here are the two prompts: one for when there is Evidence, one for when there is none (Closed-book).
PROMPT_WITH_EVIDENCE = "Question: {question}\n\nEvidence:\n\n{evidence}\n\nAnswer the question using the evidence."
PROMPT_WITHOUT_EVIDENCE = "Question: {question}\n\nAnswer the question from what you know."

# Here is the JSON shape every answer must have: one short "answer" text.
ANSWER_FORMAT = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}


def answerer_prompt(question: str, evidence: list[dict]) -> str:
    """The prompt the Answerer reads: the question, then the Evidence as numbered paragraphs (if there is any)."""
    if not evidence:
        return PROMPT_WITHOUT_EVIDENCE.format(question=question)
    return PROMPT_WITH_EVIDENCE.format(question=question, evidence=render_evidence(evidence))


def clean_answer(answer_text: str) -> str:
    """The answer without surrounding spaces or one trailing period, so "Paris." and "Paris" score the same."""
    answer_text = answer_text.strip()
    # Here we drop a single final period, but keep "..." (which may be part of a title).
    if answer_text.endswith(".") and not answer_text.endswith(".."):
        answer_text = answer_text[:-1].rstrip()
    return answer_text


def answer_question(question: str, evidence: list[dict]) -> dict:
    """Answer a question from the Evidence (pass [] for no Evidence), and return the answer and the call's details.

    Returns a dictionary with:
        answer            the short answer ("" if the model never gave one in the right format)
        reply             the model's full reply, exactly as it wrote it (the last attempt's)
        prompt            exactly what the Answerer read (on a retry, the model also saw a correction note)
        prompt_version    which version of the prompts was used
        attempts          how many calls it took (send_to_ollama_with_retries)
        answer_problems   what is still wrong with the reply ([] = it fits the JSON shape)
        tokens_read, tokens_written, seconds, prompt_may_be_cut, reply_was_cut   the call's costs and warnings
    """
    config = load_config()
    prompt = answerer_prompt(question, evidence)
    # Here the model, reply limit, temperature and seed all come from config.yaml, the same for every method.
    result = llm2.send_to_ollama_with_retries(
        llm2.model_for_role("answerer"), ANSWERER_SYSTEM_PROMPT, prompt, ANSWER_FORMAT,
        temperature=config["llm"]["temperature"], seed=config["seeds"][0],
        reply_limit_in_tokens=llm2.reply_limit_for_role("answerer"),
    )
    # Here an answer that never fitted its shape (after every attempt) counts as an empty answer.
    answer_text = (result["answer"] or {}).get("answer", "")
    return {
        "answer": clean_answer(answer_text),
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
    from pipeline.scoring import exact_match

    first_question = load_questions("pilot")[0]
    server_started_here = llm2.start_ollama_server()
    try:
        # Here we answer with no Evidence, as Closed-book does.
        answer_result = answer_question(first_question["question"], evidence=[])
        print(f"\nQuestion:     {first_question['question']}")
        print(f"Gold answer:  {first_question['answer']}")
        print(f"Model answer: {answer_result['answer']}")
        print(f"Correct:      {exact_match(answer_result['answer'], first_question['answer'])}")
        print(f"Attempts: {answer_result['attempts']}   Tokens read: {answer_result['tokens_read']}, "
              f"written: {answer_result['tokens_written']}, in {answer_result['seconds']} s\n")
    finally:
        # Here we stop the server only if this demo started it.
        if server_started_here:
            llm2.stop_ollama_server()
# --------------------------------------------------------------------------
