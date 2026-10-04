# Backlog

Build tickets, worked **in order**. Each one depends on the one before it (see its `Depends:` line). Set `Owner:` in the ticket when someone picks it up, and update the Status column below when its "Done when" check passes.

This is the **only place ticket status is tracked**. The research answers behind the tickets are in [specification.md §12](../specification.md#12-research-findings-behind-the-decisions).

## Tier 1: RQ1 + RQ2 on HotpotQA (must-have)

| # | Ticket | Target week | Status |
|---|---|---|---|
| T01 | [Repo and environment](T01-repo-and-environment.md) | Sep 23 – 29 | Done |
| T02 | [Pilot set and Test set](T02-question-sets.md) | Sep 23 – 29 | Done |
| T03 | [Wikipedia corpus and BM25 search](T03-corpus-and-bm25.md) | Sep 30 – Oct 6 | In progress: code done, Kaggle build next |
| T04 | [Model backend (plug-and-play roles)](T04-model-backend.md) | Sep 30 – Oct 6 |  |
| T05 | [Run log](T05-run-log.md) | Oct 7 – 13 |  |
| T06 | [Role prompts and output parsing](T06-role-prompts.md) | Oct 7 – 13 |  |
| T07 | [Oracle judge](T07-oracle-judge.md) | Oct 7 – 13 |  |
| T08 | [The loop and all seven settings](T08-loop-and-conditions.md) | Oct 14 – 20 |  |
| T09 | [Scoring and per-run summary](T09-evaluation.md) | Oct 14 – 20 |  |
| T10 | [Pilot and freeze](T10-pilot-and-freeze.md) | Oct 21 – 27 |  |
| T11 | [Main runs on the Test set](T11-main-runs.md) | Oct 28 – Nov 6 |  |
| T12 | [Analysis, statistics and figures](T12-analysis-and-statistics.md) | Nov 3 – 10 |  |
| T13 | [Error analysis of the Judge](T13-error-analysis.md) | Nov 7 – 12 |  |

**Tier 1 finish line (~Nov 12):** a complete paper's worth of results.

## Tier 2: MuSiQue

| # | Ticket | Target week | Status |
|---|---|---|---|
| T14 | [MuSiQue data and search setting](T14-musique-data-and-corpus.md) | Nov 11 – 14 |  |
| T15 | [MuSiQue pilot check and main runs](T15-musique-runs.md) | Nov 14 – 19 |  |
| T16 | [MuSiQue analysis and cross-dataset table](T16-musique-analysis.md) | Nov 19 – 21 |  |

## Tier 3: RQ3 (bigger or different judge)

| # | Ticket | Target week | Status |
|---|---|---|---|
| T17 | [Larger judge](T17-larger-judge.md) | Nov 19 – 23 |  |
| T18 | [Judge from a different model family](T18-other-family-judge.md) | Nov 22 – 25 |  |
| T19 | [Hosted 70B judge (optional, ≤ $40)](T19-hosted-70b-judge.md) | Nov 24 – 26 |  |

**Nov 26 – Dec 1:** buffer and paper polish. If Tier 1 slips, drop Tier 3 first, then Tier 2.

## Where teammates can overlap

The chain is sequential, but a few tickets don't touch each other's files and can go to different people in the same week: T03 ∥ T04, T05 ∥ T07, and T13 ∥ T14.
