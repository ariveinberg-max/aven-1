# Gate 0 on PhysioNet MI (WP-2.6)

**Question:** does our ingestion, preprocessing and epoching reproduce MOABB's own within-session results on the same subjects? If not, nothing downstream can be trusted.

**Command:** `neurolayer gate0 physionet_mi --max-subjects 40` (default tolerance 0.03 AUC), from a clean checkout of `70c0582`; a first run used 20 subjects. MOABB's reference evaluation (`WithinSessionEvaluation`, `LeftRightImagery`) ran on the same SHA-256-verified files (`MNE_DATA` pointed at the local cache, since physionet.org is unreachable).

| run | pipeline | sessions | ours (AUC) | MOABB (AUC) | Δ | pass |
|---|---|---|---|---|---|---|
| **40 subjects, clean `70c0582`** | csp_lda | 40 | 0.658 | 0.666 | **−0.008** | yes |
| **40 subjects, clean `70c0582`** | ts_lr | 40 | 0.671 | 0.677 | **−0.006** | yes |
| 20 subjects (first run) | csp_lda | 20 | 0.650 | 0.668 | −0.019 | yes |
| 20 subjects (first run) | ts_lr | 20 | 0.665 | 0.655 | +0.010 | yes |

**Result: passed** for PhysioNet MI. The 5-fold CV splits differ between the two implementations, so per-session differences are noise around zero; the means agree within tolerance for both pipelines.

**Provenance:** the 40-subject run used a clean checkout of `70c0582`. The first 20-subject run used commit `4b10074` with the license-card change (later committed as `70c0582`) present but uncommitted; the card only affects the license gate, not processing. Gate 0 prints a table and writes no run manifest.

**Scope:** WP-2.6 also asks for Cho2017 and Lee2019. Both are pending: their hosts are unreachable from cloud sessions, and their licenses are not yet verified (WP-1.1).
