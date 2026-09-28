# ADR-0003: Canonical data model and on-disk storage

- **Status:** Accepted
- **Date:** 2026-09-26

## Context
Public datasets differ in:

- file formats (EDF, GDF, MAT, BrainVision)
- units
- channel naming (`Fc5.`, `EEG-C3`, `EEG FP1-REF`, legacy `T3`)
- label codes
- sampling rates

Consumer devices add further variation. Without one canonical representation, every model re-solves these problems inconsistently.

## Decision
- In-memory: `Recording` (continuous) and `EpochSet` (trials) in `neurolayer.core`.
  - Validated on construction.
  - Arrays are read-only.
  - Units are volts.
  - Channel names are canonical 10-05.
  - Labels use a canonical vocabulary.
  - `order` records chronology.
- On disk:
  - `data/raw` is immutable and DVC-tracked.
  - `data/interim/bids` holds BIDS-EEG written via mne-bids.
  - `data/processed/<pipeline_hash>` holds `.npz` arrays plus JSON provenance sidecars.
- Montages are data (`CONSUMER_MONTAGES`), never hard-coded channel lists inside models.

## Consequences
Adapters (Stage 1) carry the whole burden of source quirks; everything downstream is uniform. `.npz` limits us to datasets that fit in memory per subject, which is fine for motor-imagery datasets. Revisit (Zarr) for pretraining corpora (TUEG, HBN).

## Alternatives considered
- Passing MNE `Raw`/`Epochs` objects everywhere: heavy, mutable, and couples every stage to MNE.
- HDF5 from day one: more complex, and concurrent-access pitfalls; unnecessary at current scale.
