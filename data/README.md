# data/ (git-ignored)

Everything in this directory except this README is ignored by git. Datasets are versioned with DVC (see [data model](../docs/architecture/data-model.md#5-on-disk-layout) and ADR-0003 and ADR-0006).

```
data/
├── raw/<dataset_id>/<version>/          immutable downloads + checksums.sha256 (DVC-tracked)
├── interim/bids/<dataset_id>/           BIDS-EEG conversions (mne-bids)
└── processed/<pipeline_hash>/<dataset>/ epoch arrays (.npz) + provenance (.json)
```

Rules:

- Only datasets with a catalog card may be placed here, and the card's license gate decides what they may be used for.
- Never copy data from people we recorded (C3) onto a machine or into a bucket that is not encrypted.
