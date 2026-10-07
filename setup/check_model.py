"""Check the model: a smoke test (3 tiny prompts), then a speed benchmark. (T04)

    python setup/check_model.py                         Mac: Ollama (start `ollama serve` first)
    python setup/check_model.py --backend vllm          Kaggle: vLLM on the GPU
    python setup/check_model.py --smoke-only            skip the benchmark (it takes minutes on a Mac)

Smoke test: one prompt per role; prints the answer, token counts and the top alternatives for its last token.
Benchmark: 20 Pilot questions through a practice 3-Round loop with real search results; prints the speed and an
upper-bound GPU-hour estimate for the Tier 1 main runs. Only Kaggle's numbers go in NOTES.md.
"""

import argparse
import time

from pipeline.config import load_config
from pipeline.dataset import load_question_set
from pipeline.llm import generate, generate_many, model_for, shut_down_vllm_engines
from pipeline.retriever import Retriever

# Three tiny prompts, one per role, with answers we know.
SMOKE_PROMPTS = [
    ("answerer", "Answer with one word: what is the capital of France?"),
    ("rewriter", "Write one short search query to find out where Scott Derrickson was born."),
    ("judge", "Evidence: Ed Wood was an American filmmaker.\n"
              "Question: Were Scott Derrickson and Ed Wood of the same nationality?\n"
              "Is the evidence enough to answer? Reply in exactly this format:\nENOUGH: yes or no"),
]

# Benchmark size, and how many settings the Tier 1 main runs have (A-D, Closed-book, Single-turn, Always-loop).
BENCHMARK_QUESTION_COUNT = 20
BENCHMARK_PARAGRAPHS_PER_ROUND = 5
MAIN_RUN_SETTINGS = 7


# --- Smoke test ---

def print_top_alternatives(generation: dict) -> None:
    """Print the model's top alternatives at the last real token (for the Judge: how sure it was of no vs yes)."""
    # Here we skip tokens that are only spaces or line breaks.
    visible_entries = [entry for entry in generation["logprobs"] if entry["token"].strip()]
    if not visible_entries:
        return
    last_visible_token = visible_entries[-1]
    alternatives_text = ", ".join(f"{alternative['token']!r} {alternative['logprob']:.2f}"
                                  for alternative in last_visible_token["top"])
    print(f"  top alternatives at {last_visible_token['token']!r} (log probability): {alternatives_text}")


def run_smoke_test(config: dict) -> None:
    """Send each smoke prompt to its role's model and print what came back."""
    for role, prompt in SMOKE_PROMPTS:
        generation = generate(role, prompt, config)
        print(f"[{role}] model={model_for(role, config)}  prompt_tokens={generation['prompt_tokens']}  "
              f"completion_tokens={generation['completion_tokens']}  logprob_entries={len(generation['logprobs'])}")
        print(f"  answer: {generation['text'].strip()!r}")
        print_top_alternatives(generation)


# --- Benchmark ---

def search_benchmark_evidence(questions: list[dict], retriever: Retriever, paragraphs_needed: int) -> list[list[str]]:
    """Search once per question for every paragraph all Rounds will show, as "title: text" lines."""
    return [[f"{result['title']}: {result['text']}" for result in retriever.search(question["question"], k=paragraphs_needed)]
            for question in questions]


def build_evidence_prompt(question: dict, evidence_lines: list[str]) -> str:
    """The Evidence paragraphs followed by the question: how the benchmark's Judge and Answerer prompts start."""
    return "Evidence:\n" + "\n".join(evidence_lines) + f"\n\nQuestion: {question['question']}\n"


def prompts_for_one_round(questions: list[dict], evidence_by_question: list[list[str]],
                          paragraphs_shown: int) -> list[tuple[str, list[str]]]:
    """One batch of prompts per role for one Round, each question seeing the first `paragraphs_shown` paragraphs."""
    evidence_prompts = [build_evidence_prompt(question, evidence_lines[:paragraphs_shown])
                        for question, evidence_lines in zip(questions, evidence_by_question)]
    return [
        ("judge", [evidence_prompt + "Explain briefly whether the evidence is enough, then write ENOUGH: yes or no, "
                                     "then MISSING: what is still needed." for evidence_prompt in evidence_prompts]),
        ("answerer", [evidence_prompt + "Answer with a short phrase only." for evidence_prompt in evidence_prompts]),
        ("rewriter", [f"Question: {question['question']}\nWrite one search query for the missing information."
                      for question in questions]),
    ]


def time_practice_loop(questions: list[dict], evidence_by_question: list[list[str]], config: dict) -> dict:
    """Run every Round (Judge, Answerer and Rewriter on every question) and measure time and tokens."""
    prompt_tokens = 0
    completion_tokens = 0
    started_at = time.perf_counter()
    for round_number in range(1, config["rounds"] + 1):
        paragraphs_shown = BENCHMARK_PARAGRAPHS_PER_ROUND * round_number   # Evidence grows each Round
        for role, prompts in prompts_for_one_round(questions, evidence_by_question, paragraphs_shown):
            for generation in generate_many(role, prompts, config):
                prompt_tokens += generation["prompt_tokens"]
                completion_tokens += generation["completion_tokens"]
    return {"elapsed_seconds": time.perf_counter() - started_at,
            "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}


def print_benchmark_results(timing: dict, question_count: int, config: dict) -> None:
    """Print the speed, and scale the time per question up to every Test question in every main-run setting."""
    elapsed_seconds = timing["elapsed_seconds"]
    hours_per_question = elapsed_seconds / question_count / 3600
    print(f"{question_count} questions x {config['rounds']} Rounds in {elapsed_seconds:.0f} s")
    print(f"prompt tokens: {timing['prompt_tokens']}  ({timing['prompt_tokens'] / elapsed_seconds:.0f}/s)")
    print(f"generated tokens: {timing['completion_tokens']}  ({timing['completion_tokens'] / elapsed_seconds:.0f}/s)")
    # Upper bound: every question here ran every Round.
    print(f"upper-bound estimate for Tier 1 main runs ({config['test_size']:,} questions x {MAIN_RUN_SETTINGS} settings): "
          f"{hours_per_question * config['test_size'] * MAIN_RUN_SETTINGS:.1f} GPU-hours")


def run_benchmark(config: dict) -> None:
    """Time a practice 3-Round loop on the first Pilot questions, using real search results as Evidence."""
    questions = load_question_set("pilot")[:BENCHMARK_QUESTION_COUNT]
    evidence_by_question = search_benchmark_evidence(questions, Retriever.load(),
                                                     paragraphs_needed=BENCHMARK_PARAGRAPHS_PER_ROUND * config["rounds"])
    generate("answerer", "Say OK.", config)   # here we load the model before the timer starts
    timing = time_practice_loop(questions, evidence_by_question, config)
    print_benchmark_results(timing, len(questions), config)


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    argument_parser.add_argument("--backend", choices=["ollama", "vllm"], help="use this instead of config.yaml's backend")
    argument_parser.add_argument("--smoke-only", action="store_true", help="run the smoke test and skip the benchmark")
    arguments = argument_parser.parse_args()

    config = load_config()
    if arguments.backend:
        config["backend"] = arguments.backend
    run_smoke_test(config)
    if not arguments.smoke_only:
        run_benchmark(config)
    # Here we stop the model on the GPU; without this, the Kaggle cell can hang after printing.
    shut_down_vllm_engines()
