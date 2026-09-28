# ADR-0008: First capability is calibration-efficient motor intent (CAP-1)

- **Status:** Accepted
- **Date:** 2026-09-26

## Context
The research (`docs/research/06-gap-analysis.md`) found that cross-device, calibration-efficient intent decoding on low-channel consumer EEG is unowned by companies and unbenchmarked in academia. Motor imagery has the largest multi-dataset public data with a shared label.

## Decision
The first measurable capability is CAP-1 (`docs/product/cap-1-calibration-efficient-intent.md`): left vs right motor-intent decoding, measured with the calibration-efficiency protocol across R1–R3, with Gates 0–3. CAP-2 (reactive SSVEP/c-VEP) and CAP-3 (ErrP implicit feedback) follow on the same architecture.

## Consequences
- The pipeline must be paradigm-agnostic from the start: labels and epoching windows come from config.
- Kill and pivot criteria are pre-registered in the gap analysis §6.5.

## Alternatives considered
- SSVEP first: faster demo, thinner differentiation.
- Passive-state metrics: crowded and regulated.
- EEG text: not physically viable yet.
