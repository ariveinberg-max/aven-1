# Roadmap

Development follows the sequence **research → architecture → dataset → baseline model → proprietary model → API → product → real-world testing**. Each stage ends in a gate. Nothing starts until the previous gate is passed. Work packages (the unit of work handed to engineers and AI coding agents) are listed in [docs/plan/work-packages.md](../plan/work-packages.md).

| Stage | Name | Output | Exit gate | Status |
|-------|------|--------|-----------|--------|
| 0 | **Research and foundations** | Research brief, CAP-1 spec, architecture, ADRs, repo tooling, core types, dataset catalog and license gate, evaluation harness, synthetic data | Harness runs end-to-end on synthetic data in CI | **Done** |
| 1 | Data ingestion | Adapters for motor-imagery datasets → canonical `Recording`; BIDS storage; DVC-tracked raw data; license verification; dataset QA reports | All development datasets load, validate and pass QA; licenses verified by a human | Tooling built; **gate waits on license verification (WP-1.1) and downloads** |
| 2 | Signal processing | Versioned, hashed, pure transforms (resample, filter, reference, channel harmonization, epoching, artifacts); processed cache | Deterministic outputs; golden-file tests; **Gate 0** (reproduce MOABB) | Tooling built; Gate 0 needs real data |
| 3 | Neural representation | Encoder interface; Riemannian and CSP encoders; alignment methods; PyTorch training scaffold; license-gated third-party foundation-model adapters | Encoders pass contract tests | **Done** |
| 4 | Baseline models | B0–B6 through the CAP-1 harness on R0–R3; leakage controls | **Gate 1:** baseline table and ADR fixing CAP-1 thresholds | Tooling built; Gate 1 needs real data (ADR-0011 procedure fixed) |
| 5 | Proprietary model | Montage-agnostic encoder pretrained on commercially clean data, plus calibration-time adaptation. A research program run as hypotheses and experiment cards. | **Gate 2** on the locked holdout; **kill/pivot review** | v0 model built; synthetic experiments H1–H4 done; Gate 2 needs real data |
| 6 | API | Model registry and packaging; FastAPI inference (batch and streaming); auth; tenant isolation; latency SLOs | Load and latency tests; security review | Built; latency measured; security review pending |
| 7 | Product | Dashboard (upload/stream → visualize → decode); calibration game; device bridge (BrainFlow/LSL); OS input bridge (HID) prototype | Internal dogfooding | Built (dashboard, game, bridge, explainer); dogfooding needs a device |
| 8 | Real-world testing | Device selection; consent and data-rights protocol; pilot collection (≥ 20 people); proprietary dataset v1 | **Gate 3** | Tooling built (consent, pilot recorder, protocol); needs hardware, participants and counsel |

## After CAP-1

| Capability | Summary | Reuses |
|------------|---------|--------|
| CAP-2 | Reactive selection (SSVEP/c-VEP) on dry electrodes, cross-device, for AR/VR menus and accessibility | Stages 1–3 unchanged; new task head and stimulus module |
| CAP-3 | Error-related potentials as implicit feedback for AI agents (the "everyday AI" wedge) | Same pipeline; new dataset collection (public ErrP data is small) |
| CAP-1 v2 | Four classes, asynchronous control, cross-session drift (longitudinal Stieger2021) | Harness gains a session-disjoint regime |

## Business milestones mapped to gates

| Gate | What we can credibly say | Business action |
|------|--------------------------|-----------------|
| Gate 1 | "We have a rigorous calibration-efficiency benchmark and reproduced every major baseline" | Advisor and angel conversations; hiring |
| Gate 2 | "Our model beats open-source state of the art on new users and new montages, with leakage controls" | Provisional patent(s) before disclosure; developer preview conversations; OEM discovery (vs Neurable/Arctop) |
| Gate 3 | "New users reach usable control on a consumer headset in about 2 minutes" | Partner pilots (headset OEMs, XR platforms, accessibility), SDK beta, seed round |
