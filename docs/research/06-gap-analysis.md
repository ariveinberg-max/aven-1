# 6. Gap analysis: the technical opportunity

## 6.1 Testing the founding hypothesis

> *Hypothesis:* "Neural signals differ substantially between people, so a model trained on one person may not work on another without calibration. Solving generalization is the opportunity."

**Verdict: true and still open for non-invasive EEG, but not specific enough to act on.** "Generalization" is really several distinct problems. Some are being worked on by well-funded groups; others are not.

| Sub-problem | Who is working on it | State in 2026 | Open for us? |
|-------------|----------------------|---------------|--------------|
| **Cross-subject**, same device and montage | Academia (alignment methods, foundation models like LaBraM, CBraMod, REVE, MIRepNet) | Modest gains over classical baselines (a few points). "Identity trap" audits question some reported gains. | Partly. Crowded academically, but no one has productized it |
| **Cross-device / cross-montage**, especially **low-channel dry consumer headsets** | Channel-adaptation benchmark (2026), LUNA and REVE (any-montage claims), ALPHA (SSVEP cross-headset) | Montage handling is an open design choice. **No benchmark on consumer 4–8-channel layouts.** Pretraining uses research and clinical caps. | **Yes: main technical gap** |
| **Cross-session drift** (same person, different day) | FALCON (intracortical); longitudinal EEG data (Stieger2021); online test-time adaptation papers | Recognized, and benchmarked only for implants | Yes (secondary, CAP-1 v2) |
| **Calibration efficiency as the target metric** (how much user data reaches usable control) | Nobody standardizes it for non-invasive, cross-device use. Papers report accuracy at fixed data sizes. | Not measured the way a product needs it | **Yes: the metric itself is a gap** |
| **Commercially clean models** | Nobody, visibly. Academic models inherit data-use-agreement, non-commercial or no-derivatives terms; REVE's license is restrictive. | Business gap, not a research gap | **Yes: business moat** |
| **Honest evaluation** (identity and dataset leakage controls) | Emerging critiques (2026 audits) | Not standard practice | **Yes: discipline moat** |
| Cross-user generalization via **scale** (thousands of users, one device) | Meta (EMG), done | Solved for wrist EMG | No. Copy the *playbook*, not the product |

## 6.2 The opportunity statement

> **A device-agnostic neural decoding layer that gets a new person on a new consumer device to usable control with the least calibration, measured by a calibration-efficiency benchmark we define (and eventually publish), trained only on commercially clean data.**

Why this is a company and not just a paper:

- **It is the adoption bottleneck for every product line you listed.** Gaming, AR/VR, accessibility and robotics all need "put on device → works in two minutes". You want customers who do not need to care about neuroscience. That is only possible if calibration is invisible.
- **It sells to the customers the market already has.** Hardware makers license brain-sensing software today (Neurable and Arctop prove the model), and none offers active intent control that works across devices with minimal calibration. Developers get one API across headsets.
- **It compounds.** Every user who calibrates produces exactly the data that improves the generic model (with consent). That is the Meta EMG flywheel, started from public data.
- **It is modality-agnostic.** The same calibration-efficiency framing applies to EEG today and to EMG, fNIRS or ultrasound signals later. Better sensors make our layer more valuable instead of obsolete.

## 6.3 Candidate capabilities, ranked

| Rank | Capability | Product fit | Data today | Competition | Decision |
|------|------------|-------------|------------|-------------|----------|
| 1 | **CAP-1: calibration-efficient motor intent (L/R), cross-subject + cross-dataset + consumer montage** | Gaming, robotics, accessibility, AR/VR | **Strong:** hundreds of subjects across 8+ public datasets with a shared label | Academic only; no product | **Build first** |
| 2 | CAP-2: reactive selection (SSVEP/c-VEP) on dry electrodes, cross-device | AR/VR menus, accessibility | Good (SSVEP benchmarks, BETA) | Cognixion (clinical), NextMind (discontinued); calibration-free methods exist | Next. Faster to a usable demo, less differentiated |
| 3 | CAP-3: error-related potentials as implicit feedback for AI agents | "Everyday AI", agent alignment | **Weak** (small ErrP datasets) | Zander Labs, academic reinforcement learning from implicit human feedback | Research track, differentiator |
| – | Passive states (focus, workload, stress) | Wellness | Moderate | **Crowded** and regulated | Not a priority |
| – | EEG text or speech decoding | Everyday AI | Weak | Meta FAIR, Sabi (claims) | Not viable yet (EEG character error rate 67%) |

### Why not start with CAP-2 (SSVEP), which works better?

SSVEP already reaches high accuracy with calibration-free methods (CCA/FBCCA) on good electrodes, and cross-headset transfer has been published (ALPHA). It would give an impressive demo sooner. But our differentiation there is thinner, and it does not stress-test the core claim ("our representation generalizes"). Motor intent is the hardest common case. If the architecture handles it, SSVEP becomes a new task head on the same pipeline. The plan keeps CAP-2 as the first follow-on, and the architecture is paradigm-agnostic from the start.

## 6.4 Honest constraints we design around

1. **Low information rate.** Two-class motor imagery at ~70–85% accuracy yields only a few bits per minute. Products must combine EEG intent with other channels (gaze, EMG, voice, AI context), letting EEG do what it uniquely can: hands-free, silent, implicit.
2. **BCI inefficiency.** Some people will not reach 70%. We report the *usable-user rate* as a first-class metric, never hide it, and design fallbacks (CAP-2 reactive paradigms work for many people who struggle with motor imagery).
3. **Proxy gap.** Simulating consumer montages by subsetting research-cap channels ignores dry-electrode noise, fit and motion. CAP-1 therefore has an explicit real-device phase (Stage 8) before any product claim.
4. **Scale gap vs Meta.** We start with hundreds of public subjects, not thousands. The data strategy in [document 3](03-datasets.md#35-the-data-gap-is-also-the-proprietary-data-opportunity) closes this over time.

## 6.5 Kill and pivot criteria (decide in advance to avoid sunk-cost drift)

Evaluated at the end of Stage 5 (proprietary model v1):

- **Continue CAP-1 as lead** if our model beats the strongest reproduced baseline on **R2 (new dataset) and R3 (consumer montage)** at k ∈ {0, 10, 20} trials per class. Thresholds are fixed in ADR after Gate 1, before any proprietary model is run on the locked holdout.
- **Pivot lead to CAP-2** (keeping CAP-1 as research) if gains are not significant on R2/R3, **or** if the usable-user rate at k=20 on R3 stays below 50% for all methods, meaning the physics does not support a product in this setting.
- **Re-scope the data strategy** if gains appear only on R1 (same dataset). That pattern means we are learning dataset or identity artifacts, not transferable structure.

## 6.6 What we are building, in one picture

```
 public datasets ──┐                               ┌──► evaluation harness (CAP-1 calibration-efficiency curve)
 (license-gated)   │                               │        ▲ fixed, leakage-controlled, chronological
                   ▼                               │        │
 consumer devices ─► canonical data ─► signal ─► neural ──► task heads + calibration-time adaptation ──► API ──► apps / SDK / OS input (HID)
 (Stage 7-8)         model (BIDS)     processing  representation   (proprietary)
                                                  (proprietary)
```

The **neural representation plus calibration-time adaptation** is the moat. Everything else is replaceable infrastructure, built to be rigorous.
