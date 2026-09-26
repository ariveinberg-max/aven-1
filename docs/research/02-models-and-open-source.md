# 2. Models, open-source tooling, and what the 2026 evidence says

## 2.1 Open-source toolkits we will build on

These are all permissively licensed, so commercial use is allowed. Versions were checked on PyPI on 2026-09-26.

| Package | Version | License | Role in our stack |
|---------|---------|---------|-------------------|
| MNE-Python | 1.13 | BSD-3 | Reading EEG formats, filtering, epoching, montages |
| mne-bids | 0.19 | BSD-3 | Writing and reading BIDS-EEG (our on-disk standard for raw data) |
| MOABB | 1.7 | BSD-3 | Uniform access to dozens of BCI datasets, plus a reference benchmark |
| Braindecode | 1.8 | BSD-3 | Reference deep models (EEGNet, ShallowFBCSPNet, Deep4, EEG-Conformer, ATCNet) and wrappers for some foundation models (including REVE) |
| pyRiemann | 0.12 | BSD-3 | Covariance and Riemannian methods. This is the strongest classical family for motor imagery. |
| scikit-learn | 1.9 | BSD-3 | Classical baselines, metrics |
| PyTorch | 2.14 | Apache-2.0 | Deep learning |
| BrainFlow | 5.23 | MIT | Device SDK (OpenBCI, Muse, Neurosity, and others) for live streaming (Stage 7) |
| pylsl | 1.18 | MIT | Lab Streaming Layer for device-agnostic streaming |

**Implication.** The *plumbing* is solved and free. Our proprietary value is in:

- the harmonization choices
- the evaluation protocol
- the models trained on top
- the data

It is not in re-implementing loaders or filters. Wrap these libraries; do not fork them.

## 2.2 EEG foundation models

| Model | Venue | Key idea | Code / weights license | Notes |
|-------|-------|----------|------------------------|-------|
| BENDR | 2021 | Contrastive pretraining on clinical EEG (TUEG) | open | Early; still used as a baseline |
| BIOT | NeurIPS 2023 | Biosignal tokenization across varying channels | open | Handles heterogeneous inputs |
| **LaBraM** | ICLR 2024 (spotlight) | Neural tokenizer + masked modeling on ~2,500 h from ~20 datasets | **Code MIT**; weights available | Strong cross-subject average in 2026 benchmarks. Weights inherit pretraining-data terms (see §2.4). |
| EEGPT | NeurIPS 2024 | Pretrained transformer with dual self-supervision | open | |
| **CBraMod** | ICLR 2025 | Criss-cross (spatial and temporal) attention | **Code MIT**; weights on Hugging Face | Pretraining corpus: clinical EEG *(reported: TUEG; verify)* |
| BrainOmni | 2025 | Unified EEG + MEG model | open | Best average rank in one 2026 linear-probing benchmark |
| LUNA | 2025 | Topology-agnostic (any electrode layout) | open | Relevant to cross-device work |
| **REVE** | NeurIPS 2025 | 4D positional encoding for any montage; **60k+ h, 92 datasets, 25k subjects** | **Custom license: no redistribution of weights or derivatives.** Prohibits biometric identification and profiling. **Access may be revoked if an underlying dataset becomes restricted.** | Largest pretraining effort. The license makes it unusable as the base of a product we would license to OEMs. |
| **MIRepNet** | 2025 (Knowledge-Based Systems 2026) | First **motor-imagery-specific** foundation model. Neurophysiological channel template; adapts with fewer than 30 trials per class. | **MIT** | Our closest academic neighbor for CAP-1, and a mandatory baseline |
| PARS (Apple) | NeurIPS 2025 workshop | Self-supervised "pairwise relative shift" pretraining; ear-EEG | paper | Signals Apple's interest in ear-EEG |

Intracortical and spiking models (NDT2/NDT3, POYO/POYO+, SPINT) and Meta's sEMG generic models are adjacent. They matter for the architecture's modality-agnostic design, not for CAP-1.

## 2.3 What the 2026 evaluations actually show

This is the most important section of the research. The literature's headline claims ("foundation models generalize across subjects") do not survive careful evaluation cleanly.

1. **EEG-FM-Bench / EEG-FM-Compass** (ICML 2026; *National Science Review*, Jul 2026)
   - Compared 12 open foundation models against specialist deep models and classical pipelines, on 13 datasets and 9 paradigms, under leave-one-subject-out and few-shot protocols.
   - **Linear probing frequently underperforms.** Frozen foundation-model features are not a general-purpose EEG representation yet.
   - **Specialist models remain competitive.**
   - **Larger models do not reliably generalize better.**
   - Average cross-subject balanced accuracy is reported at about **64.6% for LaBraM, 62.7% for CBraMod, and 58.1% for the best traditional pipeline** *(reported, secondary summary; verify)*. That is a real but modest gain.
2. **OmniEEG-Bench** (Jun 2026): 54 datasets and 6 task families (including "motor and interaction"). Pretraining-data **diversity** and model size are associated with better rank. For us, data diversity is a lever.
3. **"The Identity Trap in EEG Foundation Models"** (Lin, Wu, Jung; Jun 2026)
   - In LaBraM, CBraMod and REVE, **subject identity dominates the learned representations**.
   - High accuracy under subject-disjoint splits can come from identity features that happen to correlate with the label, rather than from the target signal.
   - Related 2026 audits reach similar conclusions:
     - "Pretrained, Frozen, Still Leaking": attribute leakage.
     - "Stress-testing EEG Foundation Models for Clinical Decoding": dataset identity, with negative controls.
     - "Foundation Models for EEG Are Blind to Long-Range Temporal Correlations": cross-population fragility.
4. **Channel adaptation benchmark** (Apr 2026)
   - Compares ways of feeding a new electrode layout into a pretrained model: convolutional projection, spherical-spline interpolation, source-space decomposition, and Riemannian re-centering.
   - Cross-montage transfer remains an **open design choice** rather than a solved component.
   - We found **no public benchmark on consumer 4–8-channel dry-electrode layouts**.
5. **MOABB reproducibility study** (2024; 30 pipelines, 36 datasets)
   - **Riemannian pipelines are the strongest family for motor imagery.**
   - Deep-learning pipelines needed substantially more trials per class to become competitive *(reported: >150 trials/class)*.
   - Any claimed gain must beat a well-tuned Riemannian baseline, not just EEGNet.
6. **NeurIPS 2025 EEG Foundation Challenge** (Healthy Brain Network, 3,000+ participants; cross-task and cross-subject)
   - Won by KU Leuven with a large pretrained model.
   - Follow-up work reports only small zero-shot cross-subject gains on the challenge's regression targets (normalized RMSE ~0.999 → ~0.980).
7. **Cross-subject survey** (May 2026)
   - Covers alignment, adversarial, disentanglement and contrastive families.
   - **Domain generalization with no target data remains challenging.**
   - Progress depends on "more rigorous evaluation practices".
8. **BCI inefficiency**
   - 15–30% of users typically cannot reach 70% two-class motor-imagery accuracy; some cohorts report ~50% below 70% in a first session.
   - The **70% criterion** is the conventional threshold for usable binary BCI communication.
9. **Subject-level heterogeneity** (Jul 2026; Cho2017 + PhysionetMI + Zhou2016 through MOABB)
   - The best pipeline differs from subject to subject.
   - A small *portfolio* of pipelines chosen per subject beats any single pipeline.
   - This points to calibration-time model *selection* as a lever.

### Implications for our design

- **Evaluation discipline is itself a competitive advantage.** Our protocol has three leakage defenses:
  - held-out subjects **and** held-out datasets
  - chronological calibration and a fixed test window
  - identity and dataset-ID probes plus label-shuffle negative controls

  See [evaluation protocol](../architecture/evaluation-protocol.md).
- **Mandatory baselines:**
  - Riemannian (tangent space + logistic regression, with re-centering)
  - CSP+LDA
  - EEGNet
  - MIRepNet
  - one general foundation model (LaBraM or CBraMod), as license permits
- **Do not assume a foundation model solves CAP-1.** Treat pretrained weights as candidate *initializations*, and use them only where their license and pretraining data are clean (ADR-0007).
- **Calibration-time adaptation and model selection** (alignment, test-time adaptation, portfolio selection) are under-exploited levers. That is where our proprietary modeling program starts (Stage 5).

## 2.4 License traps

A model's weights are shaped by its training data. Even if the **code** is MIT:

- **LaBraM and CBraMod** weights were pretrained largely on clinical corpora obtained under data-use agreements. Whether commercial use of derived weights is permitted depends on those agreements. **Do not ship products on these weights without legal review.**
- **REVE** weights explicitly forbid redistribution and can be revoked.
- **Meta's EMG data and code** are CC-BY-NC-4.0 (non-commercial).

This is why our catalog records the license of **every** dataset, and why the license gate refuses to train shippable models on anything that is not verified as commercially usable. See [datasets](03-datasets.md) and ADR-0005.

## Benchmarks that matter to us

| Benchmark | Why |
|-----------|-----|
| MOABB | Reproduce classical and deep baselines on motor-imagery datasets (Gate 0 of CAP-1) |
| EEG-FM-Bench / EEG-FM-Compass (MIT) | Reference numbers for foundation-model baselines |
| OmniEEG-Bench | Task-card format for standardized evaluation |
| FALCON (NeurIPS 2024) | Model for a "few-shot stability" benchmark. It is intracortical, but its design (calibration data, few-shot recalibration data, private evaluation data) is the template for our calibration-efficiency benchmark. |
