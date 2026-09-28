# PhysioNet MI: download, audit and QA (real data, exploration only)

**Date:** 2026-09-28 · **Purpose:** exploration (license not yet verified by a human, so no benchmark or training use yet).

## Provenance
- Source: PhysioNet EEGMMIDB v1.0.0, fetched from PhysioNet's official AWS Open Data mirror (`physionet-open` bucket) with `scripts/fetch_physionet_mirror.py`, because physionet.org is not reachable from the cloud session.
- 109 subjects, runs 1, 2, 4, 6, 8, 10, 12 and 14: **872 files, 1.9 GB, every file matching PhysioNet's published SHA-256**. `neurolayer data fetch physionet_mi --no-bids` then ran offline and wrote `data/raw/physionet_mi/1.0.0/checksums.sha256` (git-ignored).

## Audit (`neurolayer data audit`; full table in [audit-physionet_mi.md](audit-physionet_mi.md))
- All **109 subjects load**: 1 session and 6 imagery runs each, about 22 left-hand and 22 right-hand trials (minimum 18 per class).
- **Sampling rates: 160 Hz, and 128 Hz for sub-088, sub-092 and sub-100.** Pipelines must resample. Mixed rates are an error by design.
- **Maximum calibration budget with ≥ 10 test trials per class: k = 8.** On this dataset, CAP-1 can report k = 0 and 5 only. k = 10/20/40 need Cho2017, Lee2019 or Stieger2021.
- Simulable consumer montages: Neurosity Crown, OpenBCI Cyton motor-8, BCI-IV-2b, Emotiv EPOC X and Insight. Not Muse S (no TP9/TP10).

## QA (`neurolayer data qa`, 654 recordings)
| Check | Result |
|-------|--------|
| Flat channels | none |
| Implausible amplitude | none (median absolute amplitude per recording 6–126 µV, median 28 µV) |
| Noisy channels | 151 flags in 56 subjects, almost all frontopolar (Fp2 93, Fp1 87, AF7 67, Fpz 64, AF8 54): eye blinks, outside the motor channels CAP-1 uses |
| Line noise (60 Hz) | flagged in 357 recordings (68 subjects); median ratio 14×. Removed by the 8–30 Hz band-pass (and the notch transform where configured) |

## What this unblocks
Once a human verifies the license (WP-1.1, see [owner actions](../plan/owner-actions.md)), the next steps are Gate 0 (MOABB reproduction), official baselines on R1, and an exploratory run of the proprietary model.
