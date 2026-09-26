# T06 · Role prompts and output parsing

Tier: 1 · Owner: _ · Depends: T03, T04

## What

- One prompt per role, kept as a string at the top of its own file: `pipeline/judge.py`, `pipeline/rewriter.py`, `pipeline/answerer.py`.
  - **Judge:** sees the question and the Evidence. Must answer in the fixed format:
    ```
    REASONING: <a few sentences>
    ENOUGH: yes|no
    MISSING: <what information is still needed, or "nothing">
    ```
  - **Rewriter:** sees the question, the queries already tried, and the Steer signal. Outputs one search query.
  - **Answerer:** sees the question and the Evidence. Outputs a short answer only (a name, date, yes/no…), which is the format EM needs.
- Each of those files has one function that fills the prompt, calls `llm.generate("<role>", prompt)`, and parses the output.
- **Judge parser:** reads ENOUGH and MISSING, and computes `p_yes` = P("yes") / (P("yes") + P("no")) from the token probabilities at the ENOUGH position. If the output is malformed, it counts as "no" and is flagged `parse_error: true` in the log.

## Done when

- On 20 Pilot set questions (vLLM), each role's output parses at least 95% of the time.
- `p_yes` is between 0 and 1 for every Judge call.
- The prompts are version-numbered (`v0`), and the version is written into the run log.

## Description

These are first drafts. T10 tunes them on the Pilot set. Fixing the output format now keeps the Judge's reasoning, verdict and "what's missing" in separate fields, so each can be logged and scored on its own.
