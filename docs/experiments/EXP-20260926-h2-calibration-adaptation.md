# EXP-20260926-h2-calibration-adaptation

- **Status:** done (synthetic)
- **Work package / hypothesis:** Stage 5 / H2 (calibration-time adaptation)
- **Configs:** `configs/experiments/stage5/h2_noadapt_r2.yaml` vs `h2_adapt_r2.yaml`

## Question
How much do target alignment (from the calibration window) and head fine-tuning contribute on a new lab (R2)?

## Hypothesis and prediction (written before running)
- Hypothesis: most of the cross-subject gap is a per-subject covariance shift, which target alignment removes; fine-tuning adds subject-specific decision boundaries.
- **Prediction:** with adaptation, BA@0 (u = 20 unlabeled) is **≥ 3 pp** higher than without, and BA@20 is **≥ 2 pp** higher.
- Falsified if either difference is ≤ 0.

## Results

Runs: `20260926T071452Z-stage5-h2-noadapt-r2-09c461` (no target alignment, no fine-tuning) vs `20260926T071508Z-stage5-h2-adapt-r2-4ec85d` (default).

| k/class | no adaptation | adaptation | Δ |
|---|---|---|---|
| 0 (u=20) | 0.681 | 0.865 | **+0.184** |
| 5 | 0.681 | 0.861 | +0.180 |
| 10 | 0.681 | 0.869 | +0.188 |
| 20 | 0.681 | 0.870 | **+0.189** |
| 40 | 0.681 | 0.881 | +0.200 |
| UUR@40 | 0.44 | 1.00 | +0.56 |

## Conclusion
**Supported, much more strongly than predicted** (predicted ≥ 3 pp at k=0 and ≥ 2 pp at k=20).

Almost all of the gain comes from **target alignment using only 20 unlabeled trials**: BA@0 already jumps by 18 pp before any label is used. Head fine-tuning adds 1–2 pp at larger k. The no-adaptation curve is flat because, with alignment and fine-tuning both off, calibration data is unused.

This is the core of the product thesis: a new user produces unlabeled signal while wearing the device, and that alone moves most users past the 70% usability line (UUR 0.44 → 1.00 here).

**Caveat:** synthetic subject differences are mostly linear mixing changes, exactly what Euclidean alignment undoes. Expect a smaller (but likely still large) effect on real EEG.
