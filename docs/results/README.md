# Results ledger

Only **official** runs (`neurolayer run --official`: clean git tree, license gate passed) may be listed here. Each row must link its run id, config hash and experiment card. Synthetic smoke runs are never listed.

| Date | Experiment card | Regime | Decoder | Datasets | BA@0 | BA@10 | BA@20 | UUR@20 | AUCEC | Run id | Config hash |
|------|-----------------|--------|---------|----------|------|-------|-------|--------|-------|--------|-------------|
| – | – | – | – | – | – | – | – | – | – | – | – |

## PhysioNet MI (real data, 109 subjects; budgets k = 0/5/8 because of ~22 trials per class)

All rows are official runs (clean tree, license verified 2026-09-28), with 109 subjects scored and 0 skipped. R1 means new people with the 64-channel cap. R1-Crown means new people with only the Neurosity Crown's 8 sites. The last blocks are channel ablations for the cue-confound check. **Caution:** PhysioNet shows the target on the left or right of the screen, and decoders can exploit that cue (see [EXP-20260928-physionet-real-data](../experiments/EXP-20260928-physionet-real-data.md)). None of these numbers is a CAP-1 capability claim.

| Date | Regime / channels | Decoder | BA@0 | BA@5 | BA@8 | UUR@8 | AUCEC | Run id | Config hash |
|------|-------------------|---------|------|------|------|-------|-------|--------|-------------|
| 2026-09-28 | R1 / 64 ch | B0 chance | 0.495 | 0.495 | 0.495 | 0.00 | 0.495 | `20260928T012419Z-physionet-b0-r1-b23620` | `32798405ad36` |
| 2026-09-28 | R1 / 64 ch | B1 CSP+LDA (per subject) | 0.500 | 0.546 | 0.553 | 0.10 | 0.528 | `20260928T045018Z-physionet-b1-r1-b90798` | `ebd034db2b26` |
| 2026-09-28 | R1 / 64 ch | B2 TS+LR (per subject) | 0.500 | 0.561 | 0.572 | 0.15 | 0.537 | `20260928T045122Z-physionet-b2-r1-bee681` | `71a97e61ed6d` |
| 2026-09-28 | R1 / 64 ch | B3 TS+LR pooled | 0.607 | 0.613 | 0.619 | 0.26 | 0.611 | `20260928T050420Z-physionet-b3-r1-41b7a1` | `6177896f5d69` |
| 2026-09-28 | R1 / 64 ch | B4 EEGNet ⚠️ cue-confounded | 0.760 | 0.764 | 0.772 | 0.67 | 0.763 | `20260928T050550Z-physionet-b4-r1-0b37e4` | `41f7c2b9d443` |
| 2026-09-28 | R1 / 64 ch | **nl_spatial_field v0** | 0.698 | 0.666 | 0.695 | 0.48 | 0.682 | `20260928T040417Z-physionet-nl-r1-b80175` | `09d13b66564b` |
| 2026-09-28 | R1-Crown | B0 chance | 0.495 | 0.495 | 0.495 | 0.00 | 0.495 | `20260928T031340Z-physionet-b0-r1crown-0441f4` | `c667803238d5` |
| 2026-09-28 | R1-Crown | B1 CSP+LDA (per subject) | 0.500 | 0.550 | 0.582 | 0.18 | 0.533 | `20260928T031258Z-physionet-b1-r1crown-516f62` | `6bfa6cafd96c` |
| 2026-09-28 | R1-Crown | B2 TS+LR (per subject) | 0.500 | 0.569 | 0.582 | 0.19 | 0.542 | `20260928T031322Z-physionet-b2-r1crown-cfd78e` | `4d5e4ce978f9` |
| 2026-09-28 | R1-Crown | B3 TS+LR pooled | 0.626 | 0.612 | 0.629 | 0.27 | 0.619 | `20260928T015255Z-physionet-b3-r1crown-5bc0b8` | `7b7b5c2da5ee` |
| 2026-09-28 | R1-Crown | B3 (cached re-run, identical per subject) | 0.626 | 0.612 | 0.629 | 0.27 | 0.619 | `20260928T045605Z-physionet-b3-r1crown-d3ae81` | `7b7b5c2da5ee` |
| 2026-09-28 | R1-Crown | B4 EEGNet ⚠️ cue-confounded | 0.756 | 0.760 | 0.763 | 0.64 | 0.759 | `20260928T031833Z-physionet-b4-r1crown-179f8a` | `88076018f3ea` |
| 2026-09-28 | R1-Crown | **nl_spatial_field v0** | 0.645 | 0.618 | 0.630 | 0.31 | 0.630 | `20260928T031229Z-physionet-nl-r1crown-a8d4e7` | `3f4838526166` |
| 2026-09-28 | Crown motor only | B3 | 0.590 | 0.584 | 0.584 | 0.16 | 0.586 | `20260928T033852Z-physionet-confound-b3-motor-5d54a5` | `d77f770e133d` |
| 2026-09-28 | Crown non-motor only | B3 | 0.577 | 0.586 | 0.596 | 0.19 | 0.583 | `20260928T033615Z-physionet-confound-b3-nonmotor-071995` | `6c5af543d2d1` |
| 2026-09-28 | Crown motor only | B4 EEGNet | 0.677 | 0.684 | 0.689 | 0.50 | 0.682 | `20260928T033322Z-physionet-confound-b4-motor-2f386f` | `ee5d40788574` |
| 2026-09-28 | Crown non-motor only | B4 EEGNet | 0.719 | 0.718 | 0.722 | 0.59 | 0.719 | `20260928T032637Z-physionet-confound-b4-nonmotor-ab72a4` | `2a555da228df` |
| 2026-09-28 | Crown motor only | nl_spatial_field | 0.590 | 0.568 | 0.589 | 0.17 | 0.579 | `20260928T044616Z-physionet-confound-nl-motor-876885` | `74578403dabc` |
| 2026-09-28 | Crown non-motor only | nl_spatial_field | 0.587 | 0.577 | 0.591 | 0.15 | 0.582 | `20260928T043401Z-physionet-confound-nl-nonmotor-f52ac1` | `919ffdeed9f2` |
| 2026-09-28 | Motor strip (21 ch) | B3 | 0.632 | 0.620 | 0.626 | 0.29 | 0.626 | `20260928T050700Z-physionet-confound64-b3-motor-67ec73` | `9740df0ca34f` |
| 2026-09-28 | Non-motor (43 ch) | B3 | 0.589 | 0.591 | 0.594 | 0.20 | 0.590 | `20260928T051141Z-physionet-confound64-b3-nonmotor-b9d130` | `d81ff7af01ce` |
| 2026-09-28 | Motor strip (21 ch) | nl_spatial_field | 0.664 | 0.645 | 0.662 | 0.39 | 0.654 | `20260928T052326Z-physionet-confound64-nl-motor-bdc55d` | `031b35c8865b` |
| 2026-09-28 | Non-motor (43 ch) | nl_spatial_field | 0.634 | 0.618 | 0.636 | 0.32 | 0.626 | `20260928T053536Z-physionet-confound64-nl-nonmotor-be0ea1` | `d393e4d9ab24` |

Details: [R1 comparison](physionet/compare-r1.md), [R1-Crown comparison](physionet/compare-r1crown.md), [Crown ablation](physionet/compare-confound.md), [full-cap ablation](physionet/compare-confound64.md), [shuffle control and identity probe](physionet/controls-r1crown.md), [audit and QA](qa-physionet_mi.md).

## Gate reports
- Gate 0 (pipeline reproduces MOABB): **passed on PhysioNet MI** (40 subjects, Δ −0.008 / −0.006 AUC; see [gate0-physionet.md](gate0-physionet.md)). Cho2017 and Lee2019 are pending (hosts unreachable, licenses unverified).
- Gate 1 (baselines + CAP-1 thresholds): *pending WP-4.2*
- Gate 2 (CAP-1 on locked holdout): *pending Stage 5*
- Gate 3 (real device): *pending Stage 8*
