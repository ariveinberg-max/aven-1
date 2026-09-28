# Results ledger

Only **official** runs (`neurolayer run --official`: clean git tree, license gate passed) may be listed here. Each row must link its run id, config hash and experiment card. Synthetic smoke runs are never listed.

| Date | Experiment card | Regime | Decoder | Datasets | BA@0 | BA@10 | BA@20 | UUR@20 | AUCEC | Run id | Config hash |
|------|-----------------|--------|---------|----------|------|-------|-------|--------|-------|--------|-------------|
| – | – | – | – | – | – | – | – | – | – | – | – |

## PhysioNet MI (real data, 109 subjects; budgets k = 0/5/8 because of ~22 trials per class)

All rows are official runs (clean tree, license verified 2026-09-28). R1-Crown means new people with only the Neurosity Crown's 8 sites. **Caution:** PhysioNet shows the target on the left or right of the screen, and a decoder can exploit that cue (see the confound rows and [EXP-20260928-physionet-real-data](../experiments/EXP-20260928-physionet-real-data.md)). None of these numbers is a CAP-1 capability claim.

| Date | Experiment card | Regime / montage | Decoder | BA@0 | BA@5 | BA@8 | UUR@8 | AUCEC | Run id | Config hash |
|------|-----------------|------------------|---------|------|------|------|-------|-------|--------|-------------|
| 2026-09-28 | [EXP-physionet](../experiments/EXP-20260928-physionet-real-data.md) | R1 / 64 ch | B0 chance | 0.495 | 0.495 | 0.495 | 0.00 | 0.495 | `20260928T012419Z-physionet-b0-r1-b23620` | `32798405ad36` |
| 2026-09-28 | EXP-physionet | R1-Crown | B0 chance | 0.495 | 0.495 | 0.495 | 0.00 | 0.495 | `20260928T031340Z-physionet-b0-r1crown-0441f4` | `c667803238d5` |
| 2026-09-28 | EXP-physionet | R1-Crown | B1 CSP+LDA (per subject) | 0.500 | 0.550 | 0.582 | 0.18 | 0.533 | `20260928T031258Z-physionet-b1-r1crown-516f62` | `6bfa6cafd96c` |
| 2026-09-28 | EXP-physionet | R1-Crown | B2 TS+LR (per subject) | 0.500 | 0.569 | 0.582 | 0.19 | 0.542 | `20260928T031322Z-physionet-b2-r1crown-cfd78e` | `4d5e4ce978f9` |
| 2026-09-28 | EXP-physionet | R1-Crown | B3 TS+LR pooled | 0.626 | 0.612 | 0.629 | 0.27 | 0.619 | `20260928T015255Z-physionet-b3-r1crown-5bc0b8` | `7b7b5c2da5ee` |
| 2026-09-28 | EXP-physionet | R1-Crown | B4 EEGNet ⚠️ cue-confounded | 0.756 | 0.760 | 0.763 | 0.64 | 0.759 | `20260928T031833Z-physionet-b4-r1crown-179f8a` | `88076018f3ea` |
| 2026-09-28 | EXP-physionet | R1-Crown | nl_spatial_field v0 | 0.645 | 0.618 | 0.630 | 0.31 | 0.630 | `20260928T031229Z-physionet-nl-r1crown-a8d4e7` | `3f4838526166` |
| 2026-09-28 | EXP-physionet (confound) | Crown motor only | B3 | 0.590 | 0.584 | 0.584 | 0.16 | 0.586 | `20260928T033852Z-physionet-confound-b3-motor-5d54a5` | `d77f770e133d` |
| 2026-09-28 | EXP-physionet (confound) | Crown non-motor only | B3 | 0.577 | 0.586 | 0.596 | 0.19 | 0.583 | `20260928T033615Z-physionet-confound-b3-nonmotor-071995` | `6c5af543d2d1` |
| 2026-09-28 | EXP-physionet (confound) | Crown motor only | B4 EEGNet | 0.677 | 0.684 | 0.689 | 0.50 | 0.682 | `20260928T033322Z-physionet-confound-b4-motor-2f386f` | `ee5d40788574` |
| 2026-09-28 | EXP-physionet (confound) | Crown non-motor only | B4 EEGNet | 0.719 | 0.718 | 0.722 | 0.59 | 0.719 | `20260928T032637Z-physionet-confound-b4-nonmotor-ab72a4` | `2a555da228df` |

Details: [comparison R1-Crown](physionet/compare-r1crown.md), [confound ablation](physionet/compare-confound.md), [shuffle control and identity probe](physionet/controls-r1crown.md), [audit and QA](qa-physionet_mi.md).

## Gate reports
- Gate 0 (pipeline reproduces MOABB): **passed on PhysioNet MI** (20 subjects; see [gate0-physionet.md](gate0-physionet.md)). Cho2017 and Lee2019 are pending (hosts unreachable, licenses unverified).
- Gate 1 (baselines + CAP-1 thresholds): *pending WP-4.2*
- Gate 2 (CAP-1 on locked holdout): *pending Stage 5*
- Gate 3 (real device): *pending Stage 8*
