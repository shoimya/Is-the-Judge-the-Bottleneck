"""The one place that talks to a model. Built in T04.

Every role calls generate(role, prompt); config.yaml decides which backend (Ollama on the
Mac, vLLM on Kaggle) and which model each role uses. Both backends return the same shape:

    {"text": str, "prompt_tokens": int, "completion_tokens": int,
     "logprobs": [{"token": str, "logprob": float, "top": [{"token": str, "logprob": float}, ...]}, ...]}

with one logprobs entry per generated token (the Judge's P("yes") is read from these in T06).
completion_tokens is the cost count and includes the end-of-answer token, which has no logprobs entry
on either backend (vLLM returns one; we drop it so both backends give the same format).
Ollama's prompt_tokens counts only prompt tokens it had to process (a cached prompt can count 0);
vLLM counts the whole prompt. Ollama is for development only, so only vLLM counts are reported.

    python -m pipeline.llm smoke --backend vllm        # Kaggle: 3 test prompts, answers and probabilities
    python -m pipeline.llm benchmark --backend vllm    # Kaggle: speed on 20 Pilot questions, 3 Rounds each
"""

import argparse
import json
import os
import time
import urllib.request

from pipeline import DATA_DIR, load_config
from pipeline.dataset import load_question_set

# Here we name the address of the Ollama server running on the Mac (started with `ollama serve`).
OLLAMA_URL = "http://localhost:11434/api/chat"


def model_for(role: str, config: dict) -> str:
    """The model name this role uses on the active backend."""
    # Here we look up, e.g., models -> judge -> vllm in config.yaml.
    return config["models"][role][config["backend"]]


def generate(role: str, prompt: str, config: dict | None = None) -> dict:
    """Send one prompt to the model for this role and return its text, token counts and per-token probabilities.

    A shortcut for generate_many with a batch of one.
    """
    return generate_many(role, [prompt], config)[0]


def generate_many(role: str, prompts: list[str], config: dict | None = None) -> list[dict]:
    """Send a batch of prompts for one role; one generation per prompt, in order. vLLM runs them together, which is much faster."""
    # Here we read config.yaml unless the caller passed settings in.
    config = config or load_config()
    # Here we send the prompts to whichever backend config.yaml picks.
    if config["backend"] == "ollama":
        return [generate_with_ollama(role, prompt, config) for prompt in prompts]
    if config["backend"] == "vllm":
        return generate_with_vllm(role, prompts, config)
    raise ValueError(f"Unknown backend: {config['backend']}")


def generate_with_ollama(role: str, prompt: str, config: dict) -> dict:
    """Send one prompt to the Ollama server on the Mac and return the reply in our shared shape."""
    generation_config = config["generation"]
    # Here we build the request: which model, the prompt as a chat message, and how to generate.
    request_body = {
        "model": model_for(role, config),
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,   # one complete reply instead of word-by-word pieces
        "think": False,    # Qwen3: answer directly; the Judge writes its reasoning in its REASONING field instead
        "logprobs": True,  # also return how likely each generated token was
        "top_logprobs": generation_config["top_logprobs"],   # and the top alternatives at each position
        "options": {
            "temperature": generation_config["temperature"],   # 0 = always pick the most likely token
            "num_predict": generation_config["max_tokens"][role],   # longest reply allowed for this role
        },
    }
    # Here we send the request to the Ollama server and wait (up to 5 minutes) for the reply.
    request = urllib.request.Request(OLLAMA_URL, data=json.dumps(request_body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=300) as response:
        reply = json.load(response)

    # Here we copy the parts we need from Ollama's reply into our shared shape (see the top of this file).
    return {
        "text": reply["message"]["content"],
        "prompt_tokens": reply.get("prompt_eval_count", 0),   # missing when Ollama reused a cached prompt
        "completion_tokens": reply["eval_count"],
        "logprobs": [
            {
                "token": token_entry["token"],
                "logprob": token_entry["logprob"],
                "top": [{"token": alternative["token"], "logprob": alternative["logprob"]}
                        for alternative in token_entry["top_logprobs"]],
            }
            for token_entry in reply["logprobs"]
        ],
    }


# Here we keep one loaded vLLM engine per model name, so roles that share a model share the engine
# (loading a model takes minutes, so we only want to do it once).
loaded_vllm_engines: dict = {}


def generate_with_vllm(role: str, prompts: list[str], config: dict) -> list[dict]:
    """Run a batch of prompts through vLLM on Kaggle's GPU and return the replies in our shared shape."""
    # Here we make vLLM start its worker as a fresh process: copying (forking) a program that already touched
    # the GPU, e.g. through JAX after loading the BM25 index, fails with "CUDA driver initialization failed".
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    # Imported here: vLLM only installs on Kaggle's GPUs, never on the Mac.
    from vllm import LLM, SamplingParams

    # Here we load the model onto the GPU the first time it is needed, and reuse it after that.
    model_name = model_for(role, config)
    if model_name not in loaded_vllm_engines:
        vllm_config = config["vllm"]
        loaded_vllm_engines[model_name] = LLM(
            model=model_name,
            dtype="float16",   # the T4 has no bfloat16
            max_model_len=vllm_config["max_model_len"],   # longest prompt + reply the model will accept
            gpu_memory_utilization=vllm_config["gpu_memory_utilization"],   # share of GPU memory vLLM may use
        )
    engine = loaded_vllm_engines[model_name]

    # Here we set how to generate: same settings as the Ollama request above.
    generation_config = config["generation"]
    sampling_params = SamplingParams(
        temperature=generation_config["temperature"],
        max_tokens=generation_config["max_tokens"][role],
        logprobs=generation_config["top_logprobs"],
    )
    # Here we wrap each prompt as a one-message chat and send the whole batch to the GPU at once.
    conversations = [[{"role": "user", "content": prompt}] for prompt in prompts]
    outputs = engine.chat(conversations, sampling_params, use_tqdm=False,
                          chat_template_kwargs={"enable_thinking": False})   # Qwen3: answer directly

    # Here we convert each vLLM reply into our shared shape (see the top of this file).
    generations = []
    for output in outputs:
        completion = output.outputs[0]

        # Here we build one logprobs entry per generated token. vLLM gives, for each position, a dictionary
        # from token id to that token's probability info: the chosen token plus the top alternatives.
        logprobs = []
        for token_id, logprobs_by_token_id in zip(completion.token_ids, completion.logprobs):
            alternatives_by_rank = sorted(logprobs_by_token_id.values(), key=lambda alternative: alternative.rank)
            chosen_token = logprobs_by_token_id[token_id]
            logprobs.append({
                "token": chosen_token.decoded_token,
                "logprob": chosen_token.logprob,
                "top": [{"token": alternative.decoded_token, "logprob": alternative.logprob}
                        for alternative in alternatives_by_rank[:generation_config["top_logprobs"]]],
            })
        # Here we drop the entry for the end-of-answer token: Ollama has no entry for it, so neither do we.
        if completion.finish_reason == "stop" and logprobs:
            logprobs.pop()

        generations.append({
            "text": completion.text,
            "prompt_tokens": len(output.prompt_token_ids),
            "completion_tokens": len(completion.token_ids),
            "logprobs": logprobs,
        })
    return generations


# Here we list three tiny test prompts, one per role, used to check that a model loads and answers sensibly.
SMOKE_PROMPTS = [
    ("answerer", "Answer with one word: what is the capital of France?"),
    ("rewriter", "Write one short search query to find out where Scott Derrickson was born."),
    ("judge", "Evidence: Ed Wood was an American filmmaker.\n"
              "Question: Were Scott Derrickson and Ed Wood of the same nationality?\n"
              "Is the evidence enough to answer? Reply in exactly this format:\nENOUGH: yes or no"),
]


def run_smoke_test(config: dict) -> None:
    """Three prompts, one per role: print each answer, its token counts, and the top alternatives for its last token.

    For the Judge prompt the last token is the yes/no verdict; turning these into P("yes") is T06's job (pipeline/judge.py).
    """
    for role, prompt in SMOKE_PROMPTS:
        # Here we send the test prompt and print the answer with its token counts.
        generation = generate(role, prompt, config)
        print(f"[{role}] model={model_for(role, config)}  prompt_tokens={generation['prompt_tokens']}  "
              f"completion_tokens={generation['completion_tokens']}  logprob_entries={len(generation['logprobs'])}")
        print(f"  answer: {generation['text'].strip()!r}")

        # Here we find the last token that is not just a space or line break, and print the model's
        # top alternatives at that position (for the Judge: how sure it was of "no" versus "yes").
        visible_entries = [entry for entry in generation["logprobs"] if entry["token"].strip()]
        if visible_entries:
            last_visible_token = visible_entries[-1]
            alternatives_text = ", ".join(f"{alternative['token']!r} {alternative['logprob']:.2f}"
                                          for alternative in last_visible_token["top"])
            print(f"  top alternatives at {last_visible_token['token']!r} (log probability): {alternatives_text}")


def build_evidence_prompt(question: dict, evidence: list[str]) -> str:
    """The Evidence paragraphs followed by the question, as the benchmark's prompts start."""
    return "Evidence:\n" + "\n".join(evidence) + f"\n\nQuestion: {question['question']}\n"


def run_benchmark(config: dict, question_count: int = 20, paragraphs_per_round: int = 5) -> None:
    """Mock 3-Round loop on Pilot questions with real BM25 Evidence; report tokens per second and a GPU-hour estimate."""
    from pipeline.retriever import Retriever   # needs the BM25 index in data/wiki/bm25_index

    # Here we take the first Pilot questions and search once for each, fetching enough paragraphs
    # for every Round (5 per Round x 3 Rounds = 15). Round 1 shows the first 5, Round 2 the first 10, ...
    pilot_set = load_question_set("pilot")[:question_count]
    retriever = Retriever.load(DATA_DIR / config["wiki"]["index_dir"])
    rounds = config["loop"]["rounds"]
    evidence_by_question = [
        [f"{result['title']}: {result['text']}"
         for result in retriever.search(question["question"], k=paragraphs_per_round * rounds)]
        for question in pilot_set
    ]
    # Here we send one throwaway prompt so the model is loaded before the timer starts.
    generate("answerer", "Say OK.", config)

    # Here we time the practice loop: every Round, the Judge, Answerer and Rewriter each answer every question.
    prompt_tokens = completion_tokens = 0
    started = time.perf_counter()
    for round_number in range(1, rounds + 1):
        # Here we give each question the Evidence it would have by this Round.
        evidence_prompts = [
            build_evidence_prompt(question, evidence[:paragraphs_per_round * round_number])
            for question, evidence in zip(pilot_set, evidence_by_question)
        ]
        # Here we write one batch of prompts per role, shaped like the real prompts (T06).
        batches = [
            ("judge", [evidence_prompt + "Explain briefly whether the evidence is enough, then write ENOUGH: yes or no, "
                                         "then MISSING: what is still needed." for evidence_prompt in evidence_prompts]),
            ("answerer", [evidence_prompt + "Answer with a short phrase only." for evidence_prompt in evidence_prompts]),
            ("rewriter", [f"Question: {question['question']}\nWrite one search query for the missing information."
                          for question in pilot_set]),
        ]
        # Here we run each batch and add up how many tokens the model read and wrote.
        for role, prompts in batches:
            for generation in generate_many(role, prompts, config):
                prompt_tokens += generation["prompt_tokens"]
                completion_tokens += generation["completion_tokens"]
    elapsed_seconds = time.perf_counter() - started

    # Here we print the speed, and scale the time per question up to the full Tier 1 runs.
    hours_per_question = elapsed_seconds / len(pilot_set) / 3600
    print(f"{len(pilot_set)} questions x {rounds} Rounds in {elapsed_seconds:.0f} s")
    print(f"prompt tokens: {prompt_tokens}  ({prompt_tokens / elapsed_seconds:.0f}/s)")
    print(f"generated tokens: {completion_tokens}  ({completion_tokens / elapsed_seconds:.0f}/s)")
    print(f"estimate for Tier 1 main runs (1,000 questions x 5 loop settings): {hours_per_question * 1000 * 5:.1f} GPU-hours")


def main() -> None:
    # Here we read the command ("smoke" or "benchmark") and an optional backend that overrides config.yaml.
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["smoke", "benchmark"])
    parser.add_argument("--backend", choices=["ollama", "vllm"], help="override config.yaml's backend for this run")
    arguments = parser.parse_args()

    config = load_config()
    if arguments.backend:
        config["backend"] = arguments.backend
    if arguments.command == "smoke":
        run_smoke_test(config)
    if arguments.command == "benchmark":
        run_benchmark(config)


if __name__ == "__main__":
    main()
