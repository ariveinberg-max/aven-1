# ADR-0010: The label-shuffle control is evaluated over multiple seeds

- **Status:** Accepted
- **Date:** 2026-09-26

## Context
WP-4.3 specified the label-shuffle control as "BA ≈ chance, CI includes 0.5". While running it on synthetic data (B3, regime R2), a single-seed run showed BA@20 = 0.554 with a subject-bootstrap CI of [0.521, 0.590], excluding 0.5.

Repeating the control over 8 seeds gave across-seed means of 0.494 to 0.502 at every budget (SD ≈ 0.03). The pipeline does not leak. A single shuffled classifier is **one random direction shared by all target subjects**, so the subject bootstrap cannot see that source of variability, and single-seed CIs are too narrow for this control.

## Decision
- The shuffle control is run with at least 5 seeds: `neurolayer control CONFIG --seeds 5`.
- It **passes** if, at every budget k, |across-seed mean BA − 0.5| ≤ max(2 × across-seed standard error, 0.02).
- Gate reports include the per-seed table.
- `docs/architecture/evaluation-protocol.md` §7 is updated accordingly.

## Consequences
The control costs about 5× one run, which is acceptable for gate decisions. The same shared-randomness caveat applies to any single-seed deep-learning result, so gate claims for stochastic models also report variability across ≥ 3 training seeds (already required in `reproducibility.md`).
