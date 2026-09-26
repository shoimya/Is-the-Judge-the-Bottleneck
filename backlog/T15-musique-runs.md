# T15 · MuSiQue pilot check and main runs

Tier: 2 · Owner: _ · Depends: T14

## What

- **Pilot check:** run Condition A on 20 MuSiQue Pilot set questions with the frozen prompts. Only confirm that parsing works. Don't retune.
- **Main runs:** the same seven settings as T11 on the MuSiQue Test set, same frozen commit, logged in the Runs section of `NOTES.md`.

## Done when

All seven MuSiQue logs are complete and `results/test_musique/summary.csv` exists.

## Description

The same experiment on a second dataset. Keeping the prompts frozen means any difference comes from the dataset, not from retuning.
