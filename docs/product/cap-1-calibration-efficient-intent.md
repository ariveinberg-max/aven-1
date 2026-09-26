# CAP-1: Calibration-efficient motor-intent decoding

**Status:** Defined (v1.0) · **Owner:** founders · **Harness:** `src/neurolayer/evaluation/` · **Protocol spec:** [evaluation-protocol.md](../architecture/evaluation-protocol.md) · **Why this capability:** [gap analysis](../research/06-gap-analysis.md)

## 1. One-sentence definition

> A person we have never seen, wearing an electrode layout we did not train on, gets left-vs-right motor-intent control from their EEG **with as few labeled calibration trials as possible**. We measure this on held-out subjects and held-out datasets against a reproduced baseline suite.

## 2. The user story it enables

> "I put on a headset for the first time, play a 2-minute calibration game, and then I can steer left and right in a game, a wheelchair simulator, or a robot, with no neuroscience knowledge required."

## 3. What exactly is measured

### 3.1 Task
- **Input:** EEG epochs time-locked to a motor-intent cue (for example 0.5–2.5 s after the cue), from any channel layout. Canonical channel names follow the 10-05 system.
- **Output:** one of `left_hand`, `right_hand`.
- **v2 extensions** (not in v1): add `feet` and `rest`; add continuous (asynchronous) control; add cross-session drift.

### 3.2 Generalization regimes

| Regime | New subject | New dataset / amplifier / montage | Target channels | Question answered |
|--------|:-----------:|:---------------------------------:|-----------------|-------------------|
| **R0** (reference) | no (within-subject) | no | full | What does a fully calibrated per-user model achieve? (upper reference) |
| **R1** | ✔ | no | full | Does it transfer across people? |
| **R2** | ✔ | ✔ (leave-one-dataset-out) | full | Does it transfer across labs, amplifiers and montages? |
| **R3** | ✔ | ✔ | **consumer layout** (`neurosity_crown` 8-ch primary; `bci_iv_2b` 3-ch stress test) | Does it work with the channels a consumer device actually has? |

R3 subsets research-cap channels to simulate a consumer device. That is a *proxy* and does not model dry-electrode noise. The real-device check is Stage 8 (§7).

### 3.3 Calibration budgets (the x-axis)

For each held-out subject, the decoder may use:

- **k labeled trials per class**, k ∈ {0, 5, 10, 20, 40}. These are the **chronologically first** trials of each class. Calibration always happens before use in real life, so it does here.
- Optionally, **u unlabeled trials** from the same early calibration window (condition `U`, default u = 20). Real systems can observe a new user's signals before any labels exist.
- **Never** anything from the subject's test window. The test window is identical for every k, so points on the curve are comparable.

Conditions reported:

- `Z`: strict zero-shot. k = 0, u = 0; no target data at all.
- `U`: unsupervised adaptation. k = 0, u > 0.
- `S(k)`: supervised calibration with k per class (plus u unlabeled).

### 3.4 Metrics

| Metric | Definition | Why |
|--------|------------|-----|
| **BA@k** | Balanced accuracy on the subject's fixed test window at budget k; mean, median and 95% bootstrap CI over subjects | Main quantity; robust to class imbalance |
| **CEC** | The calibration-efficiency curve: BA@k plotted against k, per regime | The capability at a glance |
| **AUCEC** | Area under the CEC with x = log2(1 + k), normalized to [0, 1] | One number; weights small budgets more, because that is what users feel |
| **UUR@k** | *Usable-user rate*: share of held-out subjects with BA ≥ 0.70 at budget k | Product reality; exposes BCI inefficiency instead of hiding it in a mean |
| **TTC70** | *Trials-to-criterion*: smallest k at which a subject reaches BA ≥ 0.70 (∞ if never); median over subjects | "How long until it works for me" |
| ITR | Wolpaw information transfer rate (bits/min) at the protocol's trial length | Comparability with the BCI literature |
| Cohen's κ | Chance-corrected agreement | Comparability |

## 4. Baseline suite (must be reproduced before any proprietary claim)

| ID | Baseline | Adaptation behavior |
|----|----------|---------------------|
| B0 | Chance and majority class | none |
| B1 | CSP + LDA, trained **per subject** on calibration only | Supervised only (undefined at k=0) |
| B2 | Riemannian tangent space + logistic regression, **per subject** | Supervised only |
| B3 | Riemannian tangent space + logistic regression, **pooled across source subjects**, with target re-centering (Riemannian or Euclidean alignment) | U and S(k) |
| B4 | EEGNet pooled, then fine-tuned on calibration | S(k); U with alignment |
| B5 | MIRepNet (motor-imagery foundation model, MIT) fine-tuned | S(k) |
| B6 | One general EEG foundation model (LaBraM or CBraMod), fine-tuned, **evaluation only** unless its weights pass license review (ADR-0007) | S(k) |

Gate 0 is passed when B1 and B2 reproduce published MOABB within-session results on the development datasets within a stated tolerance (±3 pp mean accuracy). This proves our pipeline is correct before we trust anything it says.

## 5. Datasets

- **Development pool** (tuning, ablations): PhysioNet MI, Cho2017, Lee2019-MI (license verification pending, WP-1.1).
- **Locked holdout** (touched **only** for gate decisions, max once per model version, every access logged): chosen in WP-1.4 after a channel audit. Candidates are Dreyer2023 (87 subjects) and Stieger2021 (62 subjects, longitudinal).
- **Benchmark-only** (never trained on): BCI Competition IV 2a/2b (no-derivatives terms).

## 6. Pass/fail gates

| Gate | Criterion | Unlocks |
|------|-----------|---------|
| **Gate 0: pipeline correct** | Reproduce MOABB-reported B1/B2 within ±3 pp on development datasets | Stage 4 baselines |
| **Gate 1: baselines measured** | Full CEC for B0–B6 on R1–R3 (development pool); results in the [results ledger](../results/README.md). **Numeric CAP-1 thresholds are then fixed in an ADR, before any proprietary model touches the locked holdout.** | Stage 5 |
| **Gate 2: CAP-1 achieved (simulated)** | On the **locked holdout**, R2 and R3:<br>(a) proprietary model's BA@k > the best baseline's BA@k at each k ∈ {0 (U), 10, 20}, by a margin Δ fixed at Gate 1 (initial proposal Δ = 5 pp), paired Wilcoxon across subjects, Holm-corrected, α = 0.05;<br>(b) UUR@20 improves by ≥ 10 pp absolute;<br>(c) all leakage controls pass (§8). | API and product stages; real-device data collection |
| **Gate 3: CAP-1 achieved (real device)** | Same criteria on ≥ 20 new participants recorded with a consumer device (Stage 8), plus the product milestone: **median TTC70 ≤ 20 trials/class (~2–3 minutes)** | External claims, partner pilots |

*Why relative thresholds?* Absolute numbers for this exact protocol do not exist in the literature. That is part of the gap. Guessing them now would either be sandbagged or fantasy. Measuring baselines first (Gate 1) and then fixing thresholds before testing our own model is the honest order.

## 7. Real-world validation (Stage 8)

- **Device:** Neurosity Crown (8 channels over the sensorimotor strip) and/or OpenBCI Cyton with a sensorimotor layout. Final pick after the R3 results.
- **Protocol:** identical cue timing and trial structure to our calibration game, so the lab-to-product gap is minimized.
- **Consent:** covers product, research and commercial model training separately (PRIV-2).

## 8. Anti-self-deception rules (enforced in code where possible)

1. **Subject-disjoint and dataset-disjoint folds.** The harness asserts disjointness.
2. **Decoders never see test labels.** Test epochs are passed with labels stripped.
3. **Chronological calibration and a fixed test window**, identical across k.
4. **No normalization statistics from the test window** (for example, per-subject z-scoring must use only calibration or unlabeled-window data).
5. **Hyperparameters are tuned on source subjects only** (inner cross-validation), never on target subjects or the locked holdout.
6. **Leakage controls reported with every Gate result:**
   - *Label-shuffle control:* retrain with shuffled source labels. BA must be ≈ 0.5.
   - *Identity probe:* how well subject identity can be decoded from the representation (reported; large values flag the identity trap).
   - *Dataset-ID probe:* same for dataset identity (flags dataset shortcuts in R2).
7. **All subjects are reported**, including inefficient ones. No cherry-picking of subjects.
8. **Every official number is backed by a run manifest:** git commit (clean tree), config hash, dataset card versions, and seeds.
