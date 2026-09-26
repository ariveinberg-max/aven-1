# Research brief: where the real opportunity is

**Date:** 2026-09-26 · **Status:** v1, input to [CAP-1](../product/cap-1-calibration-efficient-intent.md) and the [architecture](../architecture/overview.md)

This brief answers one question before any model code is written:

> What specific, measurable technical capability can we own that existing
> companies and open-source projects do **not** already provide?

Detailed findings live in the numbered documents below. Every claim links to a
source in [sources.md](sources.md). Some primary sources (arXiv, patent
databases, dataset pages) could not be opened directly from the research
environment, so a number of figures come from abstracts or secondary coverage.
Those are marked *(reported)*. Verify them before using them in anything
external, such as a pitch deck or a patent filing.

| # | Document | What it covers |
|---|----------|----------------|
| 1 | [Companies](01-companies.md) | Who is building what: implants, consumer EEG, EMG, platforms |
| 2 | [Models and open source](02-models-and-open-source.md) | EEG foundation models, toolkits, and what 2026 evaluations show |
| 3 | [Datasets](03-datasets.md) | Public neural datasets, licenses, and which ones we can train on |
| 4 | [Patents and IP](04-patents-and-ip.md) | Patent watchlist and our IP strategy |
| 5 | [Regulation and privacy](05-regulation-and-privacy.md) | Neural-data laws, FDA, EU AI Act, and the engineering requirements they create |
| 6 | [Gap analysis](06-gap-analysis.md) | The opportunity, what we will not build, and kill criteria |

---

## Bottom line

1. **Your generalization hypothesis is right, but it is too broad to build a company on as stated.**
   Cross-person generalization for non-invasive EEG is still unsolved in 2026.
   The best open EEG foundation models beat classical methods by only a few
   points on cross-subject tasks. Several 2026 audits show these models partly
   learn *who* is wearing the headset (subject identity) instead of *what they
   intend*. "Solve generalization" is a research field. We need one slice of it
   that we can measure and ship.

2. **Meta has already shown that the generalization problem can be solved, for a different signal.**
   Their Neural Band (wrist EMG, shipping since Sept 2025) runs generic models
   that work on new users out of the box. The models were trained on data from
   thousands of people collected with one standardized device. The lesson for
   us is that the moat is data infrastructure plus evaluation discipline, not a
   clever architecture. We should not compete on wrist EMG. Meta owns it.

3. **The unowned slice: calibration-efficient intent decoding on low-channel consumer EEG, across devices.**
   - Consumer EEG companies (Neurable, Arctop, Muse, Emotiv) sell passive
     "focus/fatigue" metrics.
   - Active control products (Emotiv Mental Commands, Neurosity Kinesis) need
     per-user training.
   - Academic foundation models are trained and evaluated on 19–64-channel
     research or clinical caps. They are not tested on the 4–8 dry channels
     consumer devices actually have.
   - No standard benchmark measures what matters for adoption: how quickly a
     brand-new person on a brand-new device reaches usable control.

4. **There is a real business gap in commercially clean models.**
   Most EEG foundation models are pretrained on data with restrictive terms:
   non-commercial or no-derivatives licenses, or data-use agreements. REVE's
   weights carry a custom no-redistribution license. Hardware companies that
   license a model need clean chain-of-title. We can win that by building a
   license gate into the data pipeline from day one (already implemented in
   this repo).

5. **Be honest about the physics.** Non-invasive EEG carries little
   information. Meta's own brain-to-text research reaches a 67% character
   error rate with EEG, versus 32% with MEG. Thought-to-text and "mind
   reading" are not near-term EEG products. Realistic early products are a few
   discrete intents, reactive selection (visual-evoked), and implicit signals
   (error detection). Pair them with gaze, EMG, and AI context.

## Decision: our first measurable capability

**CAP-1: Calibration-efficient motor-intent decoding.**
A person we have never seen, wearing an electrode layout we did not train on
(down to 8 channels), gets left/right motor-intent control using as few
labeled calibration trials as possible. We measure this with a
calibration-efficiency curve:

- balanced accuracy at 0, 5, 10, 20 and 40 calibration trials per class
- on subjects and datasets the model never saw
- against a reproduced baseline suite
- with leakage controls

Full spec: [CAP-1](../product/cap-1-calibration-efficient-intent.md).

Why motor intent first:

- It is where generalization is hardest and least solved.
- It has the most public multi-dataset data with a shared label (left vs right
  hand), so we can test cross-dataset generalization with zero hardware spend.
- It maps directly to gaming, robotics and accessibility.
- Consumer devices exist that cover the motor cortex (Neurosity Crown, OpenBCI).

Next capabilities, in order:

- **CAP-2:** reactive selection for AR/VR (SSVEP/c-VEP on dry electrodes, cross-device)
- **CAP-3:** error-related potentials as implicit feedback for AI agents

## What we will not build (now)

| Idea | Why not |
|------|---------|
| Implants | Needs $200M+ and an FDA premarket-approval path. Neuralink, Synchron, Paradromics and Precision are years ahead. |
| Wrist EMG | Meta ships it, published the method, and holds a large patent portfolio. |
| Focus/stress/emotion metrics as the core product | Crowded (Neurable, Arctop, Muse, Emotiv). Emotion inference is banned in EU workplaces and schools (AI Act Art. 5(1)(f)). Scientifically soft. |
| EEG thought-to-text | Physics-limited today (see bullet 5 above). |
| Clinical EEG foundation model | Piramidal (YC) and academia are focused there. It is a medical-device business. |
| Our own EEG hardware | Not until CAP-1 results show which sensor layout matters (roadmap stage 8). |

## Business model check against the landscape

| Revenue line (your plan) | Existing competitors | Our angle |
|--------------------------|----------------------|-----------|
| Consumer subscription + devices | Muse, Neurable, Emotiv, Neurosity | Later. Consumers only care once calibration is near zero. |
| Developer API/SDK | Emotiv Cortex, Neurosity SDK (+MCP), BrainFlow (free), Arctop | One API across devices, with calibration-minimal intent decoding |
| Licensing to companies | Neurable (OEM licensing since Apr 2026), Arctop | Commercially clean models plus a published calibration-efficiency benchmark |
| Hardware integration | Neurable, Arctop, IDUN | Device-agnostic model, output through standard input profiles (Apple BCI HID) |

**Recommended lead:** developer and OEM licensing of the decoding layer. Build
the consumer app on top of it once CAP-1 and CAP-2 are proven on real devices.

## Top risks

1. **Data scale.** Meta used thousands of participants. We start with a few
   hundred public-data subjects. Mitigation: the harmonized multi-dataset
   pipeline, then our own consented consumer-device data collection (stage 8).
2. **BCI inefficiency.** 15–30% of people (up to ~50% in some studies) cannot
   reach 70% motor-imagery accuracy. Mitigation: report the usable-user rate
   honestly, and go hybrid (CAP-2, gaze, EMG).
3. **Public repository.** This repo is currently public. Public code and
   results count as public disclosure. That can destroy patent rights outside
   the US and leaks trade secrets. **Make the repository private before
   proprietary model work begins.** See [IP](04-patents-and-ip.md).
4. **Regulatory drift.** Four US states already regulate neural data, and more
   bills are pending. Treat neural data as sensitive from day one (see
   [privacy requirements](05-regulation-and-privacy.md)).
