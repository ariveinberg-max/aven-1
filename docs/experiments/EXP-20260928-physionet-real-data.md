# EXP-20260928-physionet-real-data

- **Status:** running (predictions below were committed before any result of the proprietary model existed)
- **Owner:** agent session, for the repository owner
- **Work package / hypothesis:** WP-4.2 (baselines, real data), Stage 5 real-data check / H-real-1, H-real-2
- **Configs:** `configs/experiments/baselines/physionet_b{0..4}_{r1,r1crown}.yaml`, `configs/experiments/physionet/nl_{r1,r1crown}.yaml`
- **Official run ids:** filled in under Results

## Question
Does the proprietary spatial-field decoder v0, whose advantage so far exists only on synthetic data, still beat the strongest classical baseline on real EEG from new people? That includes a consumer-headset montage (the Neurosity Crown's 8 sites).

## Why this is not Gate 2
Gate 2 needs the locked holdout (ADR-0009) and the R2/R3 regimes across datasets. Only PhysioNet MI is licensed and reachable from this environment, so this experiment uses one dataset. Its regimes are:
- **R1:** 5-fold cross-subject within PhysioNet, 64 channels.
- **R1-Crown:** the same folds with test subjects restricted to the Crown's 8 sites (CP3, C3, F5, PO3, PO4, F6, C4, CP4). This is a within-dataset montage shift.

PhysioNet's roughly 22 trials per class allow budgets k ∈ {0, 5, 8} with ≥ 10 test trials per class (audit). The gap analysis (§6.5) warns that **gains on R1 alone would suggest dataset or identity artifacts**. This experiment can therefore *refute* the synthetic-data story, but cannot *confirm* the business case on its own.

## Hypotheses and predictions (written before running)
- **H-real-1 (R1-Crown):** `nl_spatial_field` beats the strongest baseline among B1–B4 on R1-Crown. Prediction: mean BA@0 at least **+3 pp** higher, and not worse at k = 5 and 8. Rationale: position-based spatial filters should transfer to a sparse montage better than models fitted on a fixed channel set.
- **H-real-2 (R1):** on the full 64-channel montage the advantage is smaller. Prediction: |ΔBA| ≤ 3 pp at k = 5 and 8. At k = 0 the prediction is positive but under 5 pp.
- **Expected absolute level:** cross-subject left/right imagery on PhysioNet is hard. We expect BA@0 of about 0.55–0.65 for all methods, and a usable-user rate (≥ 70%) below 30% at k = 8.
- **Falsified if:** `nl_spatial_field` is **worse** than the best baseline by more than 3 pp at k = 0 on R1-Crown, or worse at every budget on both regimes. Then the synthetic results do not carry over, and the Stage 5 model needs rework before any Gate 2 attempt.

## Setup
- Data: `physionet_mi`, license ODC-By 1.0, verified by the owner on 2026-09-28; purpose `benchmark`. 109 subjects, fetched from PhysioNet's AWS mirror with SHA-256 verification.
- Preprocessing: default pipeline (unit check, 1–40 Hz band-pass, resample to 128 Hz), window 0.5–2.5 s after the cue.
- Protocol: CAP-1 harness unchanged; chronological calibration, fixed test window, `n_unlabeled` = 10, seed 0, bootstrap CIs over subjects.
- Decoders: B0 chance, B1 CSP+LDA per subject, B2 tangent space + LR per subject, B3 tangent space + LR pooled with re-centering, B4 EEGNet (braindecode), and `nl_spatial_field` v0 with default parameters. No hyperparameter was tuned on PhysioNet.
- All runs are `--official`, from a clean checkout of commit `70c0582`.

## Results
*Pending.*

## Leakage controls
*Pending: multi-seed shuffle control on the proprietary model (ADR-0010).*

## Conclusion
*Pending.*
