# ADR-0011: CAP-1 Gate 2 margins (set at Gate 1)

- **Status:** Proposed. **Pending real baseline results** (WP-4.2 on the development pool).
- **Date:** 2026-09-26

## Context
CAP-1 Gate 2 compares the proprietary model with the strongest baseline on the locked holdout (spec §6). The margin Δ and the usable-user-rate improvement must be fixed **before** any proprietary model touches the holdout, and **after** real baselines are measured. Otherwise they are guesses.

## Decision procedure (fixed now)
1. Run `configs/experiments/baselines/devpool_b{0..4}_r{1,2,3}.yaml` as official runs. Add B5/B6 once weights are pinned.
2. `neurolayer report compare` per regime, plus `neurolayer control` (ADR-0010) and `neurolayer probe`.
3. For each regime and k ∈ {0 (U), 10, 20}, take the best baseline's BA@k and its 95% CI half-width h.
4. **Δ = max(5 pp, 2h)** per (regime, k): the proprietary model must exceed the best baseline by more than the baselines' own uncertainty.
5. **UUR@20 improvement ≥ 10 pp** absolute on R2 and R3, unless the best baseline's UUR@20 is already ≥ 0.9, in which case require a TTC70 improvement instead.
6. Record the resulting numbers in this ADR, change its status to Accepted, and only then run any proprietary model on the locked holdout (ADR-0009).

## Consequences
Thresholds adapt to how noisy the baselines are, and they cannot be tuned after seeing the proprietary model's numbers.
