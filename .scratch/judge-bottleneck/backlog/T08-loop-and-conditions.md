# T08 · The loop and all seven settings

Tier: 1 · Owner: _ · Depends: T03, T05, T06, T07

## What

`pipeline/loop.py`, one loop engine where the Stop source and Steer source are swappable:

```
evidence = search(question, k)
for round in 1..3:
    judge = LLM judge (always run, so its verdict is logged in every setting)
    shadow_answer = Answerer(question, evidence)            # logged, not counted
    stop  = stop_source(judge, oracle)                      # LLM, oracle, never, or always
    if stop or round == 3: final answer = shadow_answer; break
    steer = steer_source(judge, oracle)                     # LLM or oracle
    query = Rewriter(question, past queries, steer)
    evidence += search(query, k, exclude=evidence titles)   # trimmed to token budget
```

`run.py --setting <name> --set pilot|test` runs one setting over a question set and writes a run log (T05). Settings:

| Setting | Stop from | Steer from |
|---|---|---|
| A | LLM judge | LLM judge |
| B | Oracle judge | LLM judge |
| C | LLM judge | Oracle judge |
| D | Oracle judge | Oracle judge |
| Single-turn | always stop after Round 1 | none |
| Always-loop | never stop before Round 3 | LLM judge |
| Closed-book | no retrieval, Answerer only | none |

Answer-oracle is not a run: it's computed from the shadow answers in the Always-loop log.

## Done when

- All seven settings run end-to-end on 10 Pilot set questions on vLLM and write valid run logs.
- Condition D reaches all Gold paragraphs at least as often as Condition A (a quick sanity check that the oracle wiring is right).
- A setting that dies halfway resumes cleanly.

## Description

This is the core of the study. The 2×2 comes from swapping only two things, who decides to stop and who says what's missing, while the retriever, the Rewriter and the Answerer stay identical. The LLM judge is called in every setting, even where its output isn't used, so we always have its verdict to score against the answer keys.
