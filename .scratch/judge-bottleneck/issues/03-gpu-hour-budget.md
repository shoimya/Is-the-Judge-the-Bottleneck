# Do the Tier 1 runs fit in the free GPU hours before Dec 1?

Type: research
Status: resolved
Blocked by: 02

## Question

Using the throughput of the chosen model on Kaggle's free GPU (from ticket 02), estimate the GPU-hours for 1,000 Test set questions × (Conditions A–D + Single-turn + Always-loop + closed-book), with up to 3 Rounds, a Judge call and a shadow answer per Round, and a Rewriter call per non-final Round. Answer-oracle needs no extra run: it is computed from shadow answers. Does it fit in ~30 GPU-h/week alongside the pilot, Tier 2 and Tier 3, within the 10 weeks to Dec 1? If not, which lever is cheapest: fewer Test set questions, sharing runs between conditions (for example, Conditions A and C share every Round up to the first stop), or batching?

Feeds backlog: [T11](../../../backlog/T11-main-runs.md).

## Answer

**It fits comfortably, even if the throughput estimate is 3× too optimistic.** No cost-cutting lever is needed; the question count stays at 1,000.

Worst-case sizing per question in a loop setting (all 3 Rounds, k≈5 paragraphs of ~100 tokens): the Judge and the shadow answer each read ~0.7k, ~1.2k and ~1.7k prompt tokens in Rounds 1–3, and the Rewriter ~0.2k twice, so about **7.6k prompt tokens and ~450 generated tokens** per question.

| | Prompt tokens | Generated tokens |
|---|---|---|
| 5 loop settings (A, B, C, D, Always-loop) × 1,000 | ~38M | ~2.3M |
| Single-turn + Closed-book × 1,000 | ~1.5M | ~0.1M |

At research 02's estimate for 2× T4 (one model copy per card: ~3k prompt tok/s and ~500 generated tok/s combined), that is about **3.7 h of prompt processing + 1.3 h of generation ≈ 5 GPU-hours** for all of Tier 1's main runs. At 3× slower it's ~15 h, still half of one week's ~30 h quota. The whole project (pilot, Tier 1, Tier 2, Tier 3 reruns) comes to roughly **15–50 GPU-hours** against ~300 available before Dec 1.

vLLM's prefix caching should make it cheaper still: the Judge call and the shadow answer read the same Evidence.

**Caveat:** these are estimates, not measurements. T04 measures real throughput on 20 questions, and T11 recomputes the budget from it before starting. If a measurement comes in more than 5× slower, reopen this ticket.
