# Data model

The canonical in-memory types live in `src/neurolayer/core/`. Every stage speaks these types. No stage passes raw MNE objects, dicts or bare arrays across a stage boundary.

## 1. `Recording`: continuous signal from one run

| Field | Type | Notes |
|-------|------|-------|
| `data` | float64 array `(n_channels, n_samples)` | **Volts.** Adapters convert units at ingestion. |
| `sfreq` | float | Hz, > 0 |
| `ch_names` | tuple[str] | Canonical names (see §4); unique |
| `dataset_id` | str | Must exist in the catalog |
| `subject_id` | str | **Pseudonymous** (PRIV-8), e.g. `sub-007` |
| `session_id`, `run_id` | str | BIDS-style identifiers |
| `events` | tuple[`Event`] | `Event(onset_sample, label, duration_samples)`; must lie within the recording |
| `device` | str or None | For example `neurosity_crown` for device data |

## 2. `EpochSet`: trials ready for modeling

| Field | Type | Notes |
|-------|------|-------|
| `X` | float64 `(n_epochs, n_channels, n_times)` | Volts |
| `y` | int64 `(n_epochs,)` | Index into `label_names`; **−1 means "label hidden"** |
| `label_names` | tuple[str] | Canonical label vocabulary (for motor imagery: `left_hand`, `right_hand`, `feet`, `tongue`, `rest`, `both_hands`) |
| `ch_names` | tuple[str] | Canonical names |
| `sfreq` | float | Hz |
| `subject`, `session`, `dataset` | str arrays `(n_epochs,)` | Provenance per epoch, so mixed-dataset sets are safe |
| `order` | int64 `(n_epochs,)` | **Chronological rank within (dataset, subject).** Sorting by it must reproduce recording order across sessions. The CAP-1 calibration split depends on it. |

Invariants, validated on construction:

- shapes agree
- no NaN or Inf values
- `y` values are in range, or all −1 when unlabeled
- channel names are unique
- arrays become **read-only**, so no stage can mutate shared data in place

Key operations:

- `subset(idx)`
- `select_channels(names)`, which raises `MissingChannelsError`
- `concat([...])`
- `without_labels()`, used by the harness to hide labels

## 3. Label harmonization

Datasets name classes differently: `left_hand`, `T1`, `769`, `1`. Adapters map native labels to the canonical vocabulary. Anything unmapped is dropped **explicitly**, with a count in the QA report. It is never silently kept.

## 4. Channel naming and montages

- **Canonical names:** 10-05 system spellings (`Fp1`, `FCz`, `C3`, `CP4`, `POz`, `TP9`, …).
- `normalize_channel_name` handles the messy real world:

  | Real-world form | Example | Normalized |
  |-----------------|---------|------------|
  | Trailing dots (PhysioNet) | `Fc5.` | `FC5` |
  | `EEG` prefix (BCI Competition) | `EEG-C3` | `C3` |
  | TUH reference suffix | `EEG FP1-REF` | `Fp1` |
  | Legacy 10-20 names | `T3`, `T4`, `T5`, `T6` | `T7`, `T8`, `P7`, `P8` |

  Unknown names are returned unchanged, and `is_standard()` reports them.
- **Montages are data.** `CONSUMER_MONTAGES` defines the channel sets of real consumer devices:

  | Montage | Channels |
  |---------|----------|
  | Neurosity Crown | 8 |
  | Muse S | 4 |
  | Emotiv EPOC X | 14 |
  | Emotiv Insight | 5 |
  | OpenBCI Cyton sensorimotor layout (our recommendation) | 8 |
  | BCI-IV-2b | 3 |

  CAP-1 regime R3 uses these to simulate consumer devices by channel subsetting.
- A montage that needs channels a dataset lacks (for example Muse `TP9`/`TP10` on a 10-10 cap) **fails loudly**. Proxy-channel substitution, if ever added, must be explicit and recorded in the run manifest.

## 5. On-disk layout

```
data/                                   # git-ignored; tracked by DVC where noted
├── raw/<dataset_id>/<version>/          # Exactly as downloaded. Immutable. Checksums in a manifest. DVC-tracked.
├── interim/bids/<dataset_id>/           # Converted to BIDS-EEG via mne-bids (canonical names, events, sidecars)
└── processed/<pipeline_hash>/<dataset_id>/
    ├── sub-XXX_epochs.npz               # EpochSet arrays
    └── sub-XXX_epochs.json              # Provenance: pipeline config, transform versions, source checksums
```

- `pipeline_hash` = SHA-256 of (canonical JSON of the pipeline config + transform versions). Change anything and you get a new directory. Processed data is never overwritten in place.
- **Why BIDS for interim:** it is the community standard (supported by MNE, OpenNeuro and NEMAR), so our converted data is inspectable with standard tools, and our own collected data (Stage 8) uses the same layout.
- **Why `.npz` for processed:** simple and fast at current scale. Revisit (Zarr or HDF5) when an EpochSet no longer fits in memory (ADR-0003).

## 6. Dataset cards (the catalog)

Every dataset, public or ours, has a YAML card in `catalog/datasets/` that is validated by `neurolayer.data.catalog.DatasetCard`. It contains:

- identity and version
- modality, paradigms, subjects, channels, sampling rate
- source URL and citation
- access type
- **license block:**
  - SPDX id
  - evidence
  - commercial-use and derivatives flags
  - share-alike flag
  - `verified_by` / `verified_on`
  - optional legal-review approvals
- roles: `pretraining`, `development`, `locked_holdout`, `benchmark_only`, `excluded`

The license gate (`evaluate_usage(card, purpose)`) decides whether a dataset may be used for `exploration`, `benchmark` or `training`. Experiment runs call it before touching data. See ADR-0005.
