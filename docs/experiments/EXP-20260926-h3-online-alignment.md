# EXP-20260926-h3-online-alignment

- **Status:** done (synthetic)
- **Work package / hypothesis:** Stage 5 / H3 (causal online adaptation)
- **Configs:** `configs/experiments/stage5/h3_offline_r2_strict.yaml` vs `h3_online_r2_strict.yaml`

## Question
In **strict zero-shot** use (no calibration, no unlabeled window: u = 0, k = 0), can causal online re-alignment during use recover performance?

## Hypothesis and prediction (written before running)
- Hypothesis: after a few trials of use, the running alignment approaches the calibration-window alignment.
- **Prediction:** BA@0 with online alignment is **≥ 3 pp** higher than offline strict zero-shot, and BA@20 changes by no more than −1 pp.
- Falsified if BA@0 does not improve.
- Note: online alignment uses only past trials (causal), as the protocol permits for declared online adaptation.

## Results

Runs: `20260926T071524Z-stage5-h3-offline-r2-strict-4c8976` (offline) vs `20260926T071552Z-stage5-h3-online-r2-strict-721137` (causal online alignment). Both use u = 0.

| k/class | offline | online (causal) | Δ |
|---|---|---|---|
| 0 (strict zero-shot) | 0.681 | 0.843 | **+0.162** |
| 5 | 0.853 | 0.858 | +0.005 |
| 10 | 0.872 | 0.866 | −0.006 |
| 20 | 0.872 | 0.874 | +0.002 |
| 40 | 0.883 | 0.883 | 0.000 |

## Conclusion
**Supported** (predicted ≥ +3 pp at k=0 and no worse than −1 pp at k=20; observed +16.2 pp and +0.2 pp).

With online alignment, a user with **no calibration at all** gets most of the benefit of a calibration window after a handful of trials of normal use. The first 1–2 trials are decoded unaligned. This is the path to the "put it on and it works" experience. Because it uses past trials only, it is a declared causal adaptation under protocol §7.

**Caveat:** synthetic data. On real data, check how many trials the running alignment needs to stabilize (the per-trial learning curve is a follow-up metric).
