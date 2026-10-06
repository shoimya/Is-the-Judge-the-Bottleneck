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

OLLAMA_URL = "http://localhost:11434/api/chat"


def model_for(role: str, config: dict) -> str:
    """The model name this role uses on the active backend."""
    return config["models"][role][config["backend"]]


def generate(role: str, prompt: str, config: dict | None = None) -> dict:
    """Send one prompt to the model for this role and return its text, token counts and per-token probabilities."""
    return generate_many(role, [prompt], config)[0]


def generate_many(role: str, prompts: list[str], config: dict | None = None) -> list[dict]:
    """Send a batch of prompts for one role; one generation per prompt, in order. vLLM runs them together, which is much faster."""
    config = config or load_config()
    if config["backend"] == "ollama":
        return [generate_with_ollama(role, prompt, config) for prompt in prompts]
    if config["backend"] == "vllm":
        return generate_with_vllm(role, prompts, config)
    raise ValueError(f"Unknown backend: {config['backend']}")


def generate_with_ollama(role: str, prompt: str, config: dict) -> dict:
    generation_config = config["generation"]
    request_body = {
        "model": model_for(role, config),
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,   # Qwen3: answer directly; the Judge writes its reasoning in its REASONING field instead
        "logprobs": True,
        "top_logprobs": generation_config["top_logprobs"],
        "options": {
            "temperature": generation_config["temperature"],
            "num_predict": generation_config["max_tokens"][role],
        },
    }
    request = urllib.request.Request(OLLAMA_URL, data=json.dumps(request_body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=300) as response:
        reply = json.load(response)
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


# One loaded vLLM engine per model name, so roles that share a model share the engine.
loaded_vllm_engines: dict = {}


def generate_with_vllm(role: str, prompts: list[str], config: dict) -> list[dict]:
    # Start vLLM's worker as a fresh process: copying (forking) a program that already touched the GPU,
    # e.g. through JAX after loading the BM25 index, fails with "CUDA driver initialization failed".
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    # Imported here: vLLM only installs on Kaggle's GPUs, never on the Mac.
    from vllm import LLM, SamplingParams

    model_name = model_for(role, config)
    if model_name not in loaded_vllm_engines:
        vllm_config = config["vllm"]
        loaded_vllm_engines[model_name] = LLM(
            model=model_name,
            dtype="float16",   # the T4 has no bfloat16
            max_model_len=vllm_config["max_model_len"],
            gpu_memory_utilization=vllm_config["gpu_memory_utilization"],
        )
    engine = loaded_vllm_engines[model_name]

    generation_config = config["generation"]
    sampling = SamplingParams(
        temperature=generation_config["temperature"],
        max_tokens=generation_config["max_tokens"][role],
        logprobs=generation_config["top_logprobs"],
    )
    conversations = [[{"role": "user", "content": prompt}] for prompt in prompts]
    outputs = engine.chat(conversations, sampling, use_tqdm=False,
                          chat_template_kwargs={"enable_thinking": False})   # Qwen3: answer directly

    generations = []
    for output in outputs:
        completion = output.outputs[0]
        logprobs = []
        for token_id, alternatives in zip(completion.token_ids, completion.logprobs):
            ranked = sorted(alternatives.values(), key=lambda alternative: alternative.rank)
            logprobs.append({
                "token": alternatives[token_id].decoded_token,
                "logprob": alternatives[token_id].logprob,
                "top": [{"token": alternative.decoded_token, "logprob": alternative.logprob}
                        for alternative in ranked[:generation_config["top_logprobs"]]],
            })
        if completion.finish_reason == "stop" and logprobs:
            logprobs.pop()   # the end-of-answer token: Ollama has no entry for it, so neither do we
        generations.append({
            "text": completion.text,
            "prompt_tokens": len(output.prompt_token_ids),
            "completion_tokens": len(completion.token_ids),
            "logprobs": logprobs,
        })
    return generations


SMOKE_PROMPTS = [
    ("answerer", "Answer with one word: what is the capital of France?"),
    ("rewriter", "Write one short search query to find out where Scott Derrickson was born."),
    ("judge", "Evidence: Ed Wood was an American filmmaker.\n"
              "Question: Were Scott Derrickson and Ed Wood of the same nationality?\n"
              "Is the evidence enough to answer? Reply in exactly this format:\nENOUGH: yes or no"),
]


def run_smoke_test(config: dict) -> None:
    """Three prompts, one per role: print each answer, its token counts, and the top alternatives for its last word.

    For the Judge prompt the last word is the yes/no verdict; turning these into P("yes") is T06's job (pipeline/judge.py).
    """
    for role, prompt in SMOKE_PROMPTS:
        generation = generate(role, prompt, config)
        print(f"[{role}] model={model_for(role, config)}  prompt_tokens={generation['prompt_tokens']}  "
              f"completion_tokens={generation['completion_tokens']}  logprob_entries={len(generation['logprobs'])}")
        print(f"  answer: {generation['text'].strip()!r}")
        visible_entries = [entry for entry in generation["logprobs"] if entry["token"].strip()]
        if visible_entries:
            last_word = visible_entries[-1]
            alternatives = ", ".join(f"{alternative['token']!r} {alternative['logprob']:.2f}" for alternative in last_word["top"])
            print(f"  top alternatives at {last_word['token']!r} (log probability): {alternatives}")


def run_benchmark(config: dict, question_count: int = 20, paragraphs_per_round: int = 5) -> None:
    """Mock 3-Round loop on Pilot questions with real BM25 Evidence; report tokens per second and a GPU-hour estimate."""
    from pipeline.retriever import Retriever   # needs the BM25 index in data/wiki/bm25_index

    pilot_set = load_question_set("pilot")[:question_count]
    retriever = Retriever.load(DATA_DIR / config["wiki"]["index_dir"])
    evidence_by_question = [
        [f"{result['title']}: {result['text']}" for result in
         retriever.search(question["question"], k=paragraphs_per_round * config["loop"]["rounds"])]
        for question in pilot_set
    ]
    generate("answerer", "Say OK.", config)   # load the model before timing

    prompt_tokens = completion_tokens = 0
    started = time.perf_counter()
    for round_number in range(1, config["loop"]["rounds"] + 1):
        evidence_prompts = [
            "Evidence:\n" + "\n".join(evidence[:paragraphs_per_round * round_number]) + f"\n\nQuestion: {question['question']}\n"
            for question, evidence in zip(pilot_set, evidence_by_question)
        ]
        batches = [
            ("judge", [evidence_prompt + "Explain briefly whether the evidence is enough, then write ENOUGH: yes or no, "
                                         "then MISSING: what is still needed." for evidence_prompt in evidence_prompts]),
            ("answerer", [evidence_prompt + "Answer with a short phrase only." for evidence_prompt in evidence_prompts]),
            ("rewriter", [f"Question: {question['question']}\nWrite one search query for the missing information."
                          for question in pilot_set]),
        ]
        for role, prompts in batches:
            for generation in generate_many(role, prompts, config):
                prompt_tokens += generation["prompt_tokens"]
                completion_tokens += generation["completion_tokens"]
    seconds = time.perf_counter() - started

    per_question_hours = seconds / len(pilot_set) / 3600
    print(f"{len(pilot_set)} questions x {config['loop']['rounds']} Rounds in {seconds:.0f} s")
    print(f"prompt tokens: {prompt_tokens}  ({prompt_tokens / seconds:.0f}/s)")
    print(f"generated tokens: {completion_tokens}  ({completion_tokens / seconds:.0f}/s)")
    print(f"estimate for Tier 1 main runs (1,000 questions x 5 loop settings): {per_question_hours * 1000 * 5:.1f} GPU-hours")


def main() -> None:
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
