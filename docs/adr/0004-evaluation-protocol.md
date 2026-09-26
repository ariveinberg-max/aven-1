# ADR-0004: CAP-1 evaluation protocol

- **Status:** Accepted
- **Date:** 2026-09-26

## Context
The 2026 literature shows that EEG results are easily inflated in several ways:

- subject or dataset identity leakage (the "identity trap")
- random calibration splits that interleave with test trials
- per-subject normalization using test data
- hyperparameter tuning on targets

Our capability claim must survive scrutiny from partners and investors.

## Decision
Adopt the protocol specified in `docs/architecture/evaluation-protocol.md`:

- subject-disjoint (R1) and dataset-disjoint (R2) folds
- consumer-montage restriction (R3)
- chronological calibration with a fixed test window shared across the budget grid
- labels stripped before `predict`
- a deep-copied decoder per (subject, k)
- metrics: BA@k, AUCEC over log2(1 + k), UUR@k (70%), TTC70
- bootstrap CIs over subjects
- paired Wilcoxon + Holm for comparisons
- leakage controls: label shuffle, identity probe, dataset-ID probe

The harness lives in `neurolayer.evaluation` and is a protected module.

## Consequences
Numbers will look lower than those in papers that use random splits. That is intended. Numeric thresholds for CAP-1 are fixed only after Gate 1, before the locked holdout is touched.

## Alternatives considered
- Random k-fold within subject: optimistic and not deployment-realistic.
- Accuracy only: hides the users for whom the system fails. UUR and TTC expose them.
