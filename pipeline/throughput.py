"""Fake 3-Round loop throughput for one backend (T04).

    python -m pipeline.throughput

Each of 20 Pilot questions goes through a fake loop: Round 1..3, and in each
Round the Judge, Rewriter and Answerer get one completion whose prompt is
padded with filler paragraphs to a realistic Evidence length (the real loop
arrives in T08). Prints prompt tok/s, generated tok/s and wall time, plus a
markdown row for NOTES.md > Throughput.

Run it on Kaggle with the vLLM backend; the Mac/Ollama numbers are
development sanity only and are never reported.
"""

import argparse
import json
import time

from pipeline import REPO_ROOT, load_config
from pipeline import llm

# Filler for the fake Evidence: one real-feeling paragraph, repeated until the
# prompt reaches the target length. T10 sets the real Evidence token budget;
# this target only makes the T04 measurement realistic.
FILLER = (
    "The Normandy landings were the landing operations and associated airborne "
    "operations on 6 June 1944 of the Allied invasion of Normandy in Operation "
    "Overlord during the Second World War. Codenamed Operation Neptune and often "
    "referred to as D-Day, it is the largest seaborne invasion in history. The "
    "operation began the liberation of France, and the rest of Western Europe, "
    "and laid the foundations of the Allied victory on the Western Front. "
    "Planning for the operation began in 1943. In the months leading up to the "
    "invasion, the Allies conducted a substantial military deception, codenamed "
    "Operation Bodyguard, to mislead the Germans as to the date and location of "
    "the main Allied landings. The weather on D-Day was far from ideal, and the "
    "operation had to be delayed 24 hours; a further postponement would have "
    "meant a delay of at least two weeks, as the invasion planners had "
    "requirements for the phase of the moon, the tides, and the time of day "
    "that meant only a few days each month were deemed suitable. "
)
TARGET_WORDS = 1500  # ~2000 tokens; the real Evidence budget is set in T10


def fake_prompt(role: str, question: str, round_number: int) -> str:
    """One fake Round prompt: role line + question + padded Evidence."""
    prefix = f"[{role} round {round_number}] Question: {question}\n\nEvidence paragraphs:\n"
    repeats = (TARGET_WORDS - len(prefix.split())) // len(FILLER.split()) + 1
    return prefix + FILLER * repeats


def measure(questions: list[dict], rounds: int = 3) -> dict:
    """Run the fake loop over `questions` and return the throughput numbers."""
    t0 = time.time()
    prompt_tokens = completion_tokens = 0
    for i, q in enumerate(questions):
        for round_number in range(1, rounds + 1):
            for role in llm.ROLES:
                out = llm.generate(role, fake_prompt(role, q["question"], round_number))
                prompt_tokens += out["prompt_tokens"]
                completion_tokens += out["completion_tokens"]
        print(f"  question {i + 1}/{len(questions)} done", flush=True)
    wall = time.time() - t0
    return {
        "wall_s": wall,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "prompt_tok_s": prompt_tokens / wall,
        "generated_tok_s": completion_tokens / wall,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", type=int, default=20,
                        help="Pilot questions to run (default 20)")
    args = parser.parse_args()

    cfg = load_config()
    pilot = REPO_ROOT / cfg["paths"]["data_dir"] / "hotpotqa" / "pilot.jsonl"
    questions = [json.loads(line) for line in open(pilot, encoding="utf-8")][: args.questions]

    backend, model, _, _ = llm.plan("judge")
    print(f"backend {backend}, judge model {model}, {len(questions)} questions x "
          f"{cfg['loop']['rounds']} rounds x 3 roles")

    result = measure(questions, rounds=cfg["loop"]["rounds"])
    print()
    print(f"wall              {result['wall_s'] / 60:.1f} min")
    print(f"prompt tok/s      {result['prompt_tok_s']:.1f}")
    print(f"generated tok/s   {result['generated_tok_s']:.1f}")
    print(f"prompt tokens     {result['prompt_tokens']}")
    print(f"completion tokens {result['completion_tokens']}")
    print()
    print("| backend | model | prompt tok/s | generated tok/s | wall (20 q) | date | who | notes |")
    print(f"| {backend} | {model} | {result['prompt_tok_s']:.1f} | "
          f"{result['generated_tok_s']:.1f} | {result['wall_s'] / 60:.1f} min | | | |")


if __name__ == "__main__":
    main()
