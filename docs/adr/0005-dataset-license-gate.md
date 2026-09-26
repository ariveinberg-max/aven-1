# ADR-0005: Dataset license gate

- **Status:** Accepted
- **Date:** 2026-09-26

## Context
The company's value is proprietary models and data that partners can license. Many public EEG datasets and pretrained models carry restrictive terms: non-commercial, no-derivatives, share-alike, data-use agreements, or custom revocable licenses. Training shippable weights on them creates legal risk that surfaces in due diligence.

## Decision
- Every dataset has a catalog card with a license block. The block records the SPDX id, evidence, commercial-use and derivatives flags, share-alike, a **human verification** (`verified_by`, `verified_on`), and optional legal-review approvals.
- `evaluate_usage(card, purpose)` decides:
  - `exploration`: allowed unless commercial use is known to be prohibited.
  - `benchmark`: requires human verification and commercial use not prohibited.
  - `training` (shippable weights): requires human verification, commercial use = yes, derivatives = yes, and no share-alike.
  - A recorded legal review can explicitly approve any purpose.
  - Internal or synthetic data is always allowed.
- Experiment configs declare a purpose. `neurolayer run` checks the gate before loading data; `--official` runs refuse to proceed when it fails.

## Consequences
Until a founder verifies each license (WP-1.1), only synthetic data passes `benchmark`/`training`. That is deliberate friction. The catalog doubles as due-diligence documentation.

## Alternatives considered
- A spreadsheet of licenses: not enforced; agents and humans would bypass it.
- Trusting dataset loaders' metadata: often missing or wrong.
