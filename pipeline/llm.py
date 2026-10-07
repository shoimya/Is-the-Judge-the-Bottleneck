"""One door for every LLM call: generate(role, prompt). Built in T04.

    from pipeline import llm
    out = llm.generate("judge", "Is this evidence enough? ...")
    # -> {"text": ..., "prompt_tokens": ..., "completion_tokens": ..., "logprobs": ...}

Two backends behind the door, picked by config.yaml `llm.backend`:

- `ollama` -- the Mac, development only (fast iteration; never a reported number)
- `vllm`   -- Kaggle (2x T4). Every number in the paper comes from here.

The model each role uses comes from `models.{role}` in the config (Ollama
tags on the Mac, Hugging Face ids under vLLM); greedy decoding and per-role
max output tokens live under `llm`. Swapping a model -- e.g. only the Judge
in Tier 3 -- is a config change, never a code change.

`logprobs` is one table per generated token, most likely first: a list of
[{"token": str, "logprob": float, "sampled": bool}, ...]. "sampled" marks the
token the model actually generated at that position. Under vLLM the tables
carry the yes/no token ids (research 02), so the Judge's P("yes") at the
ENOUGH position can be read later (pipeline/judge.py, T06); p_yes() and
p_yes_normalized() fold one table's yes/no spellings.

    python -m pipeline.llm   # the 3 test prompts, one per role

vLLM facts baked in (research 02): T4 is fp16 only (no bf16); one model copy
per T4 (data parallel) for the 8B AWQ model, so this engine stays tp=1; no
FP8 KV cache and no act-order GPTQ checkpoints; Qwen3's thinking is disabled
via the chat template (the Judge's reasoning goes in its REASONING field).
"""

import argparse
import math

from pipeline import load_config

ROLES = ("judge", "rewriter", "answerer")

# The spellings of yes/no a Judge might start its answer with (research 02);
# p_yes() folds the yes spellings into one probability.
YES_TOKENS = ("yes", "Yes", " yes")
NO_TOKENS = ("no", "No", " no")
YES_NO_TOKENS = YES_TOKENS + NO_TOKENS

TEST_PROMPTS = {
    "judge": ("Is the statement 'Paris is the capital of France' supported by "
              "the evidence 'Paris is the capital of France'? Answer with yes or no."),
    "rewriter": "Turn this question into a short search query: Which film won the most Oscars in 1997?",
    "answerer": "What is the capital of France?",
}


def plan(role: str, config: dict | None = None) -> tuple[str, str, int, float]:
    """(backend, model, max_tokens, temperature) for `role`, from the config."""
    cfg = load_config() if config is None else config
    if role not in ROLES:
        raise ValueError(f"unknown role {role!r}; expected one of {ROLES}")
    backend = cfg["llm"]["backend"]
    if backend not in _BACKENDS:
        raise ValueError(f"unknown llm.backend {backend!r}; expected one of {sorted(_BACKENDS)}")
    model = cfg["models"][role]
    if not model:
        raise ValueError(f"models.{role} is not set in config.yaml (decided in T10)")
    return backend, model, cfg["llm"]["max_tokens"][role], cfg["llm"]["temperature"]


def generate(role: str, prompt: str, *, config: dict | None = None) -> dict:
    """One completion for `role`. Returns text, token counts and logprobs."""
    backend, model, max_tokens, temperature = plan(role, config)
    return _BACKENDS[backend](model, prompt, max_tokens, temperature)


# ---------------------------------------------------------------- logprobs


def probability_of(token: str, logprobs: list[dict] | None) -> float | None:
    """P(first generated token == `token`), from a generate() logprobs table."""
    if not logprobs:
        return None
    for entry in logprobs:
        if entry["token"] == token:
            return math.exp(entry["logprob"])
    return None


def p_yes(logprobs: list[dict] | None) -> float | None:
    """P(one position's generated token is yes, Yes or " yes"), summed.

    None when the table has no yes entry at all (e.g. the backend gave none).
    """
    probabilities = [p for p in (probability_of(t, logprobs) for t in YES_TOKENS) if p is not None]
    return sum(probabilities) if probabilities else None


def p_yes_normalized(table: list[dict] | None) -> float | None:
    """P(yes) / (P(yes) + P(no)) over one position's table (specification 6.3).

    The Judge's confidence: None when the table holds no yes and no no entry.
    """
    if not table:
        return None
    yes = sum(p for p in (probability_of(t, table) for t in YES_TOKENS) if p is not None)
    no = sum(p for p in (probability_of(t, table) for t in NO_TOKENS) if p is not None)
    total = yes + no
    return yes / total if total else None


# ---------------------------------------------------------------- ollama


def _ollama_generate(model: str, prompt: str, max_tokens: int, temperature: float) -> dict:
    """Ollama (Mac, development only)."""
    import ollama  # in requirements.txt; never imported for vLLM runs

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        # think=False: Qwen3 would otherwise spend the whole budget thinking;
        # the Judge's reasoning belongs in its REASONING field (research 02).
        think=False,
        options={"temperature": temperature, "num_predict": max_tokens},
        logprobs=True,
        top_logprobs=10,
    )
    return {
        "text": response.message.content.strip(),
        "prompt_tokens": response.prompt_eval_count,
        "completion_tokens": response.eval_count,
        "logprobs": _ollama_position_tables(response.logprobs),
    }


def _ollama_position_tables(entries) -> list[list[dict]] | None:
    """One table per generated token, most likely first (SDK >= 0.6.1).

    `entries` is Ollama's per-token logprobs list; each table starts with the
    sampled token (sampled=True), then the candidate spellings.
    """
    if not entries:
        return None
    tables = []
    for entry in entries:
        table = [{"token": entry.token, "logprob": entry.logprob, "sampled": True}]
        table += [{"token": candidate.token, "logprob": candidate.logprob,
                   "sampled": False}
                  for candidate in entry.top_logprobs or []
                  if candidate.token != entry.token]
        tables.append(table)
    return tables


# ---------------------------------------------------------------- vllm


_engines: dict[str, tuple] = {}  # model -> (engine, yes_no_ids)


def _get_engine(model: str):
    """The process's vLLM engine for `model`, loaded lazily (first call is slow).

    One engine per model so Tier 3 can run the Judge on a different model than
    the other roles. fp16 only (T4 has no bf16), tp=1: one engine serves one
    T4. The main runs launch one process per T4 (data parallel) in the
    notebook; two engines at once need more GPU than one T4 has.
    """
    if model not in _engines:
        from vllm import LLM

        engine = LLM(model=model, dtype="float16")
        _engines[model] = (engine, _encode_yes_no(engine))
    return _engines[model]


def _encode_yes_no(engine) -> list[int]:
    """Token ids of the yes/no spellings that are a single token each."""
    tokenizer = engine.get_tokenizer()
    ids = []
    for spelling in YES_NO_TOKENS:
        try:
            encoded = tokenizer.encode(spelling, add_special_tokens=False)
        except TypeError:  # TokenizerGroup without kwargs passthrough
            encoded = tokenizer.encode(spelling)[-1:]
        if len(encoded) == 1:
            ids.append(encoded[0])
    return ids


def _vllm_generate(model: str, prompt: str, max_tokens: int, temperature: float) -> dict:
    from vllm import SamplingParams

    engine, yes_no_ids = _get_engine(model)
    messages = [{"role": "user", "content": prompt}]
    tokenizer = engine.get_tokenizer()
    # enable_thinking=False: Qwen3 would otherwise think out loud; the Judge's
    # reasoning belongs in its REASONING field (research 02).
    try:
        prompt_ids = tokenizer.apply_chat_template(
            messages, enable_thinking=False, add_generation_prompt=True
        )
    except TypeError:  # TokenizerGroup without kwargs passthrough
        prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    sampling = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        logprob_token_ids=yes_no_ids,
    )
    outputs = engine.generate({"prompt_token_ids": prompt_ids}, sampling)
    out = outputs[0].outputs[0]
    return {
        "text": out.text.strip(),
        "prompt_tokens": len(outputs[0].prompt_token_ids),
        "completion_tokens": len(out.token_ids),
        "logprobs": _vllm_position_tables(out, tokenizer),
    }


def _vllm_position_tables(out, tokenizer) -> list[list[dict]] | None:
    """One table per generated token: the sampled token (sampled=True) plus
    the requested yes/no ids (logprob_token_ids, research 02)."""
    if not out.logprobs:
        return None
    tables = []
    for position, table in enumerate(out.logprobs):
        entries = []
        for token_id, item in table.items():
            if token_id is None:
                continue
            entries.append({
                "token": item.decoded_token or tokenizer.decode([token_id]),
                "logprob": float(item.logprob),
                "sampled": token_id == out.token_ids[position],
            })
        entries.sort(key=lambda entry: entry["logprob"], reverse=True)
        tables.append(entries)
    return tables


_BACKENDS = {"ollama": _ollama_generate, "vllm": _vllm_generate}


# ---------------------------------------------------------------- CLI


def main() -> None:
    """Run the 3 test prompts, one per role (T04's backends check)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--role", choices=[*ROLES, "all"], default="all",
                        help="run one role's prompt instead of all three")
    args = parser.parse_args()

    roles = ROLES if args.role == "all" else (args.role,)
    for role in roles:
        out = generate(role, TEST_PROMPTS[role])
        print(f"--- {role} ---")
        print(out["text"])
        print(f"prompt {out['prompt_tokens']} tokens, completion {out['completion_tokens']} tokens")
        first = out["logprobs"][0] if out["logprobs"] else None
        if first:
            top = ", ".join(f"{entry['token']!r}: {math.exp(entry['logprob']):.3f}"
                            for entry in first[:3])
            print(f"first-token logprobs: {top}")
        if role == "judge":
            print(f"P(yes) = {p_yes(first)}")
            print(f"P(yes)/(P(yes)+P(no)) = {p_yes_normalized(first)}")
        print()


if __name__ == "__main__":
    main()
