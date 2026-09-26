# T10 · Pilot and freeze

Tier: 1 · Owner: _ · Depends: T09, research [02](../issues/02-models-on-free-gpu.md)

## What

On the **Pilot set only**:

1. **Pick the Tier 1 model** from research 02's top 2 candidates, **Qwen3-8B-AWQ** and **Qwen3-4B-Instruct-2507**: run Condition A and Single-turn with each. Pick the one with a higher parse rate and higher EM. Tie → the faster one.
2. **Tune k** (paragraphs per Round), trying 2, 5 and 10: pick the smallest k where Condition D's Gold coverage after 3 Rounds is clearly above Single-turn's.
3. **Tune the token budget** so 3 Rounds of Evidence fit in the model's context without trimming in most questions.
4. **Refine prompts** (v0 → v1) where parsing fails or answers are too long for EM. Fix only format problems; don't tune to make the Judge look better or worse.
5. **Freeze:** write the final values into `config.yaml`, commit, and tag the commit `v1-frozen`.
6. Write the Pilot section of `NOTES.md`: what was tried, what was chosen, and why (a few paragraphs plus the small tables). This becomes the paper's setup section.

## Done when

- `config.yaml` has no TODO values left for Tier 1.
- The commit is tagged `v1-frozen`.
- the Pilot section of `NOTES.md` explains every chosen value.

## Description

The pilot is where every "we'll decide later" gets decided, on questions that will never be reported. After the freeze, nobody changes prompts or settings for Tier 1. That's what makes the Test set numbers honest.
