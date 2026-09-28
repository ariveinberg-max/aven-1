# EXP-20260926-h1-subset-consistency

- **Status:** done (synthetic)
- **Work package / hypothesis:** Stage 5 / H1 (montage-agnostic encoding + invariance objective)
- **Configs:** `configs/experiments/stage5/h1_base_r3.yaml` vs `h1_consistency_r3.yaml`
- **Data:** synthetic two-lab pool (not a capability claim; see caveats)

## Question
Does a subset-consistency loss (features from two random channel subsets of the same trial should agree) make the spatial-field decoder more robust when the target has far fewer channels than the source (R3)?

## Hypothesis and prediction (written before running)
- Hypothesis: the consistency loss pushes the learned spatial fields toward montage-invariant features.
- **Prediction:** AUCEC on synthetic R3 (3-channel target) improves by **≥ 1 pp** with `consistency_weight = 0.5` vs 0.
- Falsified if the improvement is ≤ 0.

## Setup
Two synthetic labs × 8 subjects × 70 trials/class, leave-dataset-out, target montage `bci_iv_2b` (C3, Cz, C4), k ∈ {0, 5, 10, 20, 40}, u = 20.

## Results

Runs: `20260926T071420Z-stage5-h1-base-r3-2b7f6d` (λ = 0) vs `20260926T071437Z-stage5-h1-consistency-r3-15b74b` (λ = 0.5).

| k/class | BA, λ = 0 | BA, λ = 0.5 | Δ |
|---|---|---|---|
| 0 (u=20) | 0.809 | 0.792 | −0.017 |
| 5 | 0.805 | 0.795 | −0.010 |
| 10 | 0.807 | 0.821 | +0.014 |
| 20 | 0.813 | 0.822 | +0.009 |
| 40 | 0.822 | 0.834 | +0.012 |
| **AUCEC** | **0.809** | **0.807** | **−0.002** |

## Conclusion
**Refuted.** The prediction was ≥ +1 pp AUCEC; the result is −0.2 pp. The loss trades small-budget accuracy for large-budget accuracy, which is the wrong direction for a calibration-efficiency product. It stays **off by default** (`consistency_weight = 0`).

Next: channel dropout alone may already provide the invariance. Test the λ sweep {0.05, 0.1} and a dropout-rate sweep on real R3 data before discarding the idea.

**Caveat:** synthetic data only.
