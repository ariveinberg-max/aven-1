# ADR-0009: Locked holdout selection rule

- **Status:** Proposed. The rule is accepted; the dataset choice is **pending** the WP-1.4 audit on downloaded data.
- **Date:** 2026-09-26

## Context
Gate 2 of CAP-1 is decided on a **locked holdout** dataset that is never used for development, tuning or model selection. Choosing it after looking at results would bias the gate, so the rule is fixed now, before any real data is loaded.

## Decision
Run `neurolayer data audit stieger2021 dreyer2023 --out docs/results/dataset-audit.md` (requires downloads). Then pick the locked holdout by these criteria, in order:

1. The license is verified for `benchmark` use (WP-1.1).
2. It has at least 50 subjects.
3. The common channels cover the `neurosity_crown` montage, so regime R3 is possible on the holdout.
4. `min_trials_per_class` is at least 30, which supports budgets up to k=20 with at least 10 test trials per class.
5. **Tie-break:** prefer the dataset *not* used to pretrain the third-party foundation-model baselines (MIRepNet, LaBraM, CBraMod), per their papers. A dataset seen in pretraining would inflate those baselines on the holdout.

The chosen dataset's card gets `roles: [locked_holdout]`. Every holdout evaluation is logged as an official run and listed in the results ledger. Each model version gets at most one holdout evaluation.

## Consequences
The holdout dataset is excluded from the development pool, hyperparameter search and ablations. If neither candidate meets criteria 1–4, write a follow-up ADR choosing another dataset from the MOABB list before Stage 4 starts.

## Alternatives considered
- A held-out subset of subjects from the development datasets: this cannot test dataset shift (R2), which is the capability's point.
- Choosing after seeing baseline results: biased.
