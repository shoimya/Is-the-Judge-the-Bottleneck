# T16 · MuSiQue analysis and cross-dataset table

Tier: 2 · Owner: _ · Depends: T15

## What

- Rerun `python evaluate.py --report` on the MuSiQue logs (it should only need a different run list).
- Add a **cross-dataset table**: B − A, C − A and D − A for HotpotQA and MuSiQue side by side, with CIs.
- Break MuSiQue results down by hop count (2, 3, 4) to see whether the stop or steer gap grows with more hops.

## Done when

The MuSiQue tables and figures, the cross-dataset table and the by-hop table are in `results/`.

## Description

This answers "does the answer to RQ2 change when questions need more steps?" If steering gets worse with more hops, that's a finding in itself.
