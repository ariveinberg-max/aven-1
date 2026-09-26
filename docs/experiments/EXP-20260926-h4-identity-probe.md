# EXP-20260926-h4-identity-probe

- **Status:** done (synthetic)
- **Work package / hypothesis:** Stage 5 / H4 (learn intent, not identity)
- **Script:** `scripts/stage5_identity_probe.py`

## Question
Is subject identity less decodable from the spatial-field decoder's learned features (with per-subject alignment) than from the tangent-space representation?

## Hypothesis and prediction (written before running)
- Hypothesis: per-subject Euclidean alignment plus a task-trained, low-dimensional feature space removes much of the identity signal.
- **Prediction:** the subject-probe accuracy above chance is **≥ 10 pp lower** for the learned features than for tangent space.
- Falsified if it is not lower.

## Results

`uv run python scripts/stage5_identity_probe.py` (16 synthetic subjects, 5-fold linear probe):

| representation | dims | subject-probe accuracy | chance | above chance |
|---|---|---|---|---|
| tangent space (no alignment) | 45 | 1.000 | 0.062 | +0.938 |
| spatial-field features (per-subject aligned) | 48 | 0.695 | 0.062 | **+0.632** |
| spatial-field features (unaligned) | 48 | 0.916 | 0.062 | +0.853 |

## Conclusion
**Supported:** 30.6 pp lower than tangent space (predicted ≥ 10 pp). Most of the reduction comes from per-subject alignment (unaligned learned features: +0.85).

Identity is still **clearly decodable** (+0.63 above chance), so the representation is *less* identity-bearing, not identity-free. For privacy (PRIV-7), learned features must still be treated as personal data.

Next: an adversarial or LEACE-style identity-erasure ablation, checking whether task BA survives.
