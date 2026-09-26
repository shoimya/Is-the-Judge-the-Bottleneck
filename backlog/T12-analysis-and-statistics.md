# T12 · Analysis, statistics and figures

Tier: 1 · Owner: _ · Depends: T11

## What

Extend `evaluate.py` (Mac is fine, no GPU) so `python evaluate.py --report` reads the Test set logs and produces:

1. **Main table (RQ1, RQ2):** EM, F1, tokens and calls for all seven settings plus Answer-oracle, each with a 95% **paired bootstrap** CI (10,000 resamples, seed from config).
2. **Gap decomposition:** B − A (cost of bad stopping), C − A (cost of bad steering), D − A (total gap), each with a CI and a **McNemar** test on per-question correctness.
3. **Memorization control:** the same table restricted to questions Closed-book got wrong.
4. **Judge quality:** precision and recall of the LLM judge's stop verdicts against the Coverage key and the Answerability key, plus early-stop and late-stop rates.
5. **Calibration:** a reliability diagram and ECE of `p_yes` against each key.
6. **Accuracy vs cost curve:** EM against mean tokens for every setting, plus a curve from sweeping a stop threshold on `p_yes` offline (possible because shadow answers exist for every Round of Always-loop).
7. Save every table as CSV and LaTeX in `results/test/` and every figure as PDF in `results/figures/`.

## Done when

Every table and figure above exists in `results/`, and `python evaluate.py --report` rebuilds all of it from the logs.

## Description

This produces the paper's results section. B − A and C − A answer RQ2 directly: whichever is larger is the bigger bottleneck. The memorization subset stops the model's prior knowledge from hiding the effect of retrieval.
