# 1. Company landscape (September 2026)

This document groups the neural-interface companies by what they actually own. For each group it ends with what that means for us. Funding and trial figures are as publicly reported; see [sources](sources.md).

## A. Implanted BCIs: clinical, high-bandwidth, capital-intensive

| Company | Approach | Status (2026) | What they own |
|---------|----------|---------------|---------------|
| **Neuralink** | Intracortical threads, robot surgery | 21 participants by Jan 2026 (26 reported later). PRIME (computer/robot-arm control) and VOICE (speech) trials in US, UK, Canada, UAE. Blindsight has Breakthrough Device designation. | Hardware, surgery, per-patient decoders |
| **Synchron** | Stentrode (endovascular, no open-brain surgery) | $200M Series D (Nov 2025) to fund the 2026 pivotal trial for the first implantable-BCI premarket approval. First native integration with **Apple BCI HID**. **Chiral** "cognitive AI" brain foundation model with NVIDIA. | Minimally invasive access, Apple integration, early foundation-model narrative |
| **Paradromics** | Connexus high-density microelectrode array | First human implant in the Connect-One Early Feasibility Study (17 Jun 2026). Real-time speech with the first participant (Sept 2026). | High-channel-count hardware, speech decoding |
| **Precision Neuroscience** | Layer 7 thin-film cortical surface array | FDA 510(k) clearance (2025) for up to 30 days of implantation. Medtronic partnership (Jan 2026). $250M Series D reported. | Surface arrays, surgical workflow integration |
| **Science Corp** | PRIMA retinal implant; biohybrid research | $230M Series C (Mar 2026), ~$1.5B valuation. PRIMA restored functional central vision in 38 geographic-atrophy patients. | Vision restoration |
| **Blackrock Neurotech** | Utah array (the academic workhorse) | Long-running research implants | Legacy research hardware |

**What this means for us.** These companies use per-patient decoders that are recalibrated frequently. Decoder drift across days is a recognized open problem; the [FALCON benchmark](02-models-and-open-source.md#benchmarks-that-matter-to-us) exists because of it. Their market is paralysis and ALS, reached through FDA premarket approval. That is not a market for a software startup without hardware.

They do validate two things we care about:

1. The long-term value accrues in the decoding and AI layer (Synchron's Chiral).
2. Operating systems now accept neural input as a first-class device class (Apple BCI HID).

## B. Next-generation non-invasive hardware: well-funded, pre-product

| Company | Approach | Status |
|---------|----------|--------|
| **Merge Labs** | Ultrasound and molecular interfaces, no electrodes | Out of stealth Jan 2026 with a **$252M seed** (OpenAI, Bain Capital, Gabe Newell). Co-founded by Sam Altman with Forest Neurotech leadership. |
| **Kernel** | Time-domain fNIRS (Kernel Flow) | Research-grade wearable optical imaging |
| **Sabi** | High-density EEG cap claiming inner-speech-to-text | Announced plans for a 2026 launch, backed by Khosla. Claims (30 wpm, "brain foundation model on 100k hours") are **unverified**, and independent academic results say EEG speech decoding remains very hard. |
| **Meta FAIR (research)** | Brain2Qwerty: decoding typed sentences from MEG/EEG | Nature Neuroscience 2026. Character error rate **32% with MEG vs 67% with EEG**. |

**What this means for us.** High-bandwidth non-invasive decoding needs new sensor physics, and those bets cost hundreds of millions. Our software should be **modality-agnostic** (EEG today; fNIRS, ultrasound or MEG-like signals later), so that we benefit from better sensors instead of being made obsolete by them. That is why the data model in this repository is not EEG-specific.

## C. Neuromotor (wrist EMG): Meta has solved cross-user generalization at scale

- **Meta Neural Band** ships with Meta Ray-Ban Display ($799, since 30 Sept 2025). A developer preview for display glasses opened in May 2026 and supports Neural Band gestures.
- *Nature* (2025), "A generic non-invasive neuromotor interface for human-computer interaction":
  - Meta collected sEMG from **thousands of consenting participants** with a standardized wristband.
  - The resulting generic models work on new users **with no calibration**: 0.66 target acquisitions/s, 0.88 gesture detections/s, and 20.9 wpm handwriting.
  - Personalization adds about 16% on handwriting.
- The released dataset (100 participants per task) and code are **CC-BY-NC-4.0**, so we cannot use them commercially.
- Meta/CTRL-labs reportedly filed about 238 patents on EMG sensing and decoding between 2018 and 2026.

**What this means for us.**

1. Cross-user generalization is solvable when you control the device and collect data at scale. This is the playbook to copy for EEG.
2. Do not compete on wrist EMG.
3. A hybrid future (EEG + EMG + gaze) is plausible. Our intent-decoding API should expose modality-agnostic "intents", so that an EMG source could plug in later through licensed hardware.

## D. Consumer EEG hardware and cognitive-state software: our most direct competitors

| Company | Product | Decodes | Developer and licensing story |
|---------|---------|---------|-------------------------------|
| **Neurable** | MW75 Neuro headphones; HyperX gaming headset (CES 2026 award) | Focus, fatigue, mental load | $35M Series A (Dec 2025). **Pivoted to OEM licensing in Apr 2026.** Partners include HP/HyperX, iMotions, AFRL. |
| **Arctop** | NeuOS SDK for headphones, earbuds, glasses | Cognitive load, attention, enjoyment | $10M Series A (2023). Licenses its SDK to enterprises and hardware makers. Patents include "Brain ID" authentication. |
| **Interaxon (Muse)** | Muse S Athena: 4 EEG channels (TP9, AF7, AF8, TP10) at 256 Hz, plus fNIRS and PPG | Meditation, sleep | Consumer app plus subscription. Raw data is limited. |
| **Emotiv** | EPOC X (14 ch), Insight (5 ch), MN8 earbuds | Performance metrics; **Mental Commands require per-user training** | Cortex API. Raw EEG requires a paid license. |
| **Neurosity** | Crown: 8 channels (CP3, C3, F5, PO3, PO4, F6, C4, CP4) at 256 Hz | Focus, calm, "Kinesis" (trained motor imagery) | JS/Python SDK. Ships an **MCP server** for AI-assistant integration. |
| **OpenBCI** | Cyton (8 ch, 250 Hz, ~$1,250); Galea for XR (~$36–43k) | Raw signals (open hardware) | BrainFlow and LSL ecosystem; research community |
| **IDUN, BrainBit, others** | In-ear and headband EEG | Sleep and wellness | SDKs |

**What this means for us.**

- **The "license our models to hardware companies" line already has competitors** (Neurable, Arctop). So does the "developer SDK" line (Emotiv, Neurosity, BrainFlow).
- What they sell is **passive-state metrics**. These are becoming commodities. They are hard to validate scientifically, and they are increasingly regulated (EU AI Act emotion-inference ban in workplaces and schools; neural-data laws).
- **Nobody offers active intent decoding that works across devices with minimal calibration.** Emotiv and Neurosity both require per-user training for motor commands. This is the gap CAP-1 targets.
- The Neurosity Crown's 8 channels cover the motor cortex (C3, C4, CP3, CP4). That makes it a realistic first real-world test device for CAP-1. The Muse's forehead and ear sites do not cover the motor cortex, so it serves as a stress test.

## E. Assistive and visual-attention BCIs

- **Cognixion** (Axon-R EEG + AR): feasibility trial pairing its EEG headband with **Apple Vision Pro** for ALS, spinal cord injury, stroke and TBI patients. About 10 participants, running through Apr 2026, with a pivotal trial planned.
- **NextMind** (visual-attention EEG for AR; acquired by Snap in Mar 2022): the standalone developer kit was discontinued after the acquisition.

**What this means for us.** Visual-evoked paradigms (SSVEP, c-VEP) work non-invasively and pair naturally with headsets. That makes them the **CAP-2** candidate. NextMind is a warning that a standalone accessory is a hard business. Integrate into platforms (Vision Pro, Quest, Android XR) and OS input profiles instead.

## F. Passive BCI and neuroadaptive AI ("everyday AI")

- **Zander Labs** (Netherlands/Germany): passive BCI for neuroadaptive systems. It has a **€30M contract with Germany's Cyber Agency** (the NAFAS project) and is expanding to the US in 2026.
- **Academic work** on reinforcement learning from implicit human feedback uses error-related potentials (ErrPs) as a reward signal for agents and robots (2025 papers).

**What this means for us.** "Brain signals as implicit feedback for AI" is a real direction, but it is still research-stage. It is our **CAP-3** differentiator for the "everyday AI" vision. It fits an AI-native company better than focus scores do.

## G. Brain foundation-model companies

- **Piramidal** (YC, ~$6M): an EEG foundation model for **clinical** use (ICU, neurology).
- **Synchron Chiral**: a foundation model built on implant data.
- **Academic models**: LaBraM, CBraMod, REVE, BrainOmni, MIRepNet (see [document 2](02-models-and-open-source.md)).

**What this means for us.** A clinical EEG foundation model is taken, and it is a medical business. A foundation model for **consumer-device control**, trained on commercially clean data and judged by calibration efficiency, is not.

## H. Platforms and operating systems

- **Apple BCI HID** (announced 13 May 2025): a Human Interface Device profile that lets BCIs drive iOS, iPadOS and visionOS through Switch Control. Synchron was first; Cognixion is also integrating.
- **Meta Wearables Device Access Toolkit**: a developer path for Ray-Ban Display and Neural Band gestures.

**What this means for us.** Our product layer does not need every app to adopt our SDK. A decoder that outputs **standard input events** (HID/switch) works with every app on a platform on day one. That is the Stage 7 "HID bridge" work package.

## Summary map

```
                     passive states ◄──────────────► active intent/control
                   ┌───────────────────────────────┬───────────────────────────────┐
 implanted         │ (Synchron Chiral – cognition) │ Neuralink, Synchron,          │
                   │                               │ Paradromics, Precision        │
                   ├───────────────────────────────┼───────────────────────────────┤
 non-invasive      │ Neurable, Arctop, Muse,       │ Emotiv/Neurosity (per-user    │
 consumer EEG      │ Emotiv, Zander Labs           │ training), Cognixion (clinic) │
                   │                               │  ► GAP: calibration-minimal,  │
                   │                               │    cross-device intent (CAP-1)│
                   ├───────────────────────────────┼───────────────────────────────┤
 non-invasive EMG  │                               │ Meta Neural Band (generic,    │
                   │                               │ no calibration)               │
                   ├───────────────────────────────┼───────────────────────────────┤
 new modalities    │ Kernel (fNIRS)                │ Merge Labs (ultrasound)       │
                   └───────────────────────────────┴───────────────────────────────┘
```
