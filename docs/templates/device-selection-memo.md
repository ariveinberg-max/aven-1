# Device selection memo: TEMPLATE (WP-8.1)

**Decision:** which consumer EEG device we use for the pilot (Gate 3) and the first product.
**Date / author / reviewers:** `TODO`
**Status:** template. Fill in from real R3 results, never from synthetic ones.

## 1. Candidates
| Device | Channels over sensorimotor cortex | Sampling rate | SDK / streaming | Raw-data access and license terms | Price | Availability |
|--------|-----------------------------------|---------------|-----------------|-----------------------------------|-------|--------------|
| Neurosity Crown | `TODO(verify)` | `TODO(verify)` | LSL / Neurosity SDK | `TODO(verify)` | | |
| OpenBCI Cyton (+ Daisy) with a motor-cortex layout | user-placed | 250 Hz | BrainFlow | open hardware | | |
| Muse S / Athena | `TODO(verify)`: frontal/temporal only | | BrainFlow | `TODO(verify)` | | |
| Other: | | | | | | |

## 2. Evidence from CAP-1 regime R3
For each candidate montage: BA@k, UUR@70% and TTC70 with 95% CIs from `configs/experiments/baselines/*_r3*.yaml` and the proprietary model (`configs/experiments/stage5/*r3*`), **on real data** after WP-1.1. Link the run ids.

| Montage | Best baseline BA@10 | Proprietary BA@10 | UUR@70% @20 | Run ids |
|---------|---------------------|-------------------|-------------|---------|
| | | | | |

## 3. Practical criteria
- Setup time and comfort over a 45-minute session (dogfooding notes).
- Signal quality: fraction of flat or noisy channels in `neurolayer data qa` over 5 test sessions.
- Timing latency and jitter (photodiode test, pilot protocol §5).
- Data rights: can we store raw data and train commercial models on it under the vendor's terms? Get counsel's read.
- Supply: can we buy 5–10 units within the pilot timeline?

## 4. Recommendation
`TODO`: device, the reasons in two or three sentences, and the risks accepted.
