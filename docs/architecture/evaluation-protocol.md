# Evaluation protocol (CAP-1 harness specification)

This document specifies precisely what `src/neurolayer/evaluation/` computes. The product-level definition and gates are in [CAP-1](../product/cap-1-calibration-efficient-intent.md).

> **Protected module.** Changes to splitting, metrics or the protocol loop need an ADR and owner review (see `CODEOWNERS`). Never change the measuring stick to improve a number.

## 1. Inputs

- `epochs: EpochSet`
  - labeled
  - may contain multiple datasets and subjects
  - `order` gives chronological rank within each (dataset, subject)
- `factory: () -> Decoder`, which builds a fresh, unfitted decoder
- `ProtocolConfig`

## 2. The `Decoder` contract

Defined in `neurolayer.core.interfaces`, so models and the harness never import each other.

```python
class Decoder(Protocol):
    def fit(self, source: EpochSet) -> None: ...
    def adapt(self, calibration: EpochSet | None, unlabeled: EpochSet | None) -> Decoder: ...
    def predict(self, epochs: EpochSet) -> np.ndarray: ...  # int labels, shape (n_epochs,)
```

`fit`
- Sees only **source** subjects (other subjects, or other datasets in R2/R3).

`adapt`
- Receives the target subject's labeled calibration trials (or `None` when k=0) and unlabeled calibration-window trials (or `None`).
- Returns a decoder ready to predict for **that subject only**.
- The harness calls `adapt` on a **deep copy** of the fitted decoder for every (subject, k). Adaptation to one subject can therefore never leak into another.

`predict`
- Receives test epochs with **labels stripped** (`y = -1`).
- Must return predictions in the index space of `epochs.label_names`.

The decoder is responsible for handling channel mismatches between source and target. It gets `ch_names` on every `EpochSet`. That is the montage-agnostic problem, and it is deliberately part of the decoder, not the harness.

## 3. Folds (who is source, who is target)

| Regime | Construction | Assertion |
|--------|--------------|-----------|
| `within_dataset` (R1) | Subjects are keyed as `(dataset, subject)` and split into `n_folds` groups (or leave-one-subject-out when `n_folds` is `None`) with a seeded shuffle. Source = every other subject. | Source ∩ target subjects = ∅ |
| `leave_dataset_out` (R2) | Each dataset in turn is the target; source = all subjects of all other datasets. Needs ≥ 2 datasets. | Also source datasets ∩ target datasets = ∅ |
| R3 | R2 (or R1) plus `target_montage`: target epochs are restricted to the montage's channels. Source is unrestricted unless `source_montage` is set. | Missing montage channels in a target → subject skipped, with a reason |

R0 (within-subject reference) is the special case where source is empty. It is implemented by baseline B1/B2 decoders that ignore `source` and train on calibration only.

## 4. Calibration split for one target subject

Given the subject's epochs sorted by `order`, the budget grid `ks` (for example `(0, 5, 10, 20, 40)`), `n_unlabeled = u` and `min_test_per_class = m`:

1. `kmax = max(ks)`.
2. **Calibration window length W:** the smallest prefix that contains at least `kmax` trials of **every** class and at least `u` trials in total.
3. **Test window** = everything after W. It must contain at least `m` trials of every class, otherwise the subject is **skipped with a recorded reason**, never silently dropped.
4. For each k, the **labeled calibration** set is the first k trials of each class within the window, in chronological order.
5. The **unlabeled** set is the first u trials of the window, with labels stripped.
6. The **test window is identical for every k**, so curve points are comparable within a subject.

Consequences:

- Calibration always precedes test in time, which mirrors deployment and avoids optimistic leakage from temporally adjacent trials.
- Adding a larger k to the grid shrinks the test window for all k. Changing the grid is therefore a protocol change (new config hash).

## 5. Per-(subject, k) record

`SubjectResult`:

- regime, fold, dataset, subject
- k, n_unlabeled
- n_calibration, n_test
- balanced_accuracy, accuracy
- Cohen's κ
- ITR (bits/min, if `trial_seconds` is set)

## 6. Aggregates (per regime, per k)

| Aggregate | Computation |
|-----------|-------------|
| Mean and median BA | Across subjects, with a percentile **bootstrap 95% CI over subjects** (default 2,000 resamples, seeded) |
| UUR@k | Fraction of subjects with BA ≥ `threshold` (default 0.70) |
| TTC per subject | Smallest k with BA ≥ threshold; `None` if never reached. Median TTC treats `None` as +∞. |
| AUCEC | Trapezoidal area of mean BA over x = log2(1 + k), divided by the x-range. Lies in [0, 1] when BA is in [0, 1]. |
| Paired comparison of two decoders | Wilcoxon signed-rank on per-subject BA at each k; Holm–Bonferroni across the k grid |

## 7. Leakage controls (Gate reports)

Implemented in WP-4.3:

1. **Label-shuffle** (ADR-0010): permute source and calibration labels within each subject, then run the protocol with **≥ 5 shuffle seeds** (`neurolayer control`). Pass if, at every k, |across-seed mean BA − 0.5| ≤ max(2 × SE, 0.02). A single seed is not enough: its bootstrap CI ignores the classifier randomness shared by all subjects.
2. **Identity probe:** logistic regression predicting `subject` from the frozen representation of source epochs (cross-validated). Report its accuracy. It is not a pass/fail, but it must be reported next to any foundation-model result.
3. **Dataset-ID probe:** same for `dataset` in R2.
4. **Test-window normalization audit:** static review. Decoders must not compute statistics from `predict` inputs across trials, *unless* it is declared as online test-time adaptation. In that case the causal (past-only) variant is used and reported separately.

## 8. Reproducibility of a protocol run

A run is defined by:

- the config (hash)
- the dataset card versions
- the pipeline hash
- the decoder spec
- the seeds

`neurolayer run` writes these into `manifest.json` next to `results.csv` and `summary.json` (see [reproducibility.md](reproducibility.md)).
