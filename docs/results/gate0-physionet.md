# Gate 0 on PhysioNet MI (WP-2.6)

**Question:** does our ingestion, preprocessing and epoching reproduce MOABB's own within-session results on the same subjects? If not, nothing downstream can be trusted.

**Command:** `neurolayer gate0 physionet_mi --max-subjects 20` (default tolerance 0.03 AUC). MOABB's reference evaluation (`WithinSessionEvaluation`, `LeftRightImagery`) ran on the same SHA-256-verified files (`MNE_DATA` pointed at the local cache, since physionet.org is unreachable).

| pipeline | sessions | ours (AUC) | MOABB (AUC) | Δ | pass |
|---|---|---|---|---|---|
| csp_lda | 20 | 0.650 | 0.668 | −0.019 | yes |
| ts_lr | 20 | 0.665 | 0.655 | +0.010 | yes |

**Result: passed** for PhysioNet MI. The 5-fold CV splits differ between the two implementations, so per-session differences are noise around zero; the means agree within tolerance for both pipelines.

**Provenance:** run on 2026-09-28 at commit `4b10074`, with the license-card change that was committed as `70c0582` present but uncommitted (the card only affects the license gate, not processing). Gate 0 prints a table and writes no run manifest. A re-run on 40 subjects from a clean checkout is queued and will be added here.

**Scope:** WP-2.6 also asks for Cho2017 and Lee2019. Both are pending: their hosts are unreachable from cloud sessions, and their licenses are not yet verified (WP-1.1).
