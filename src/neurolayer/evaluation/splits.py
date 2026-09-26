"""Leakage-safe folds and chronological calibration splits (CAP-1 protocol §3–§4).

**Protected module:** changes require an ADR (ADR-0004).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from neurolayer.core.types import EpochSet, IntArray

SubjectKey = tuple[str, str]
"""``(dataset_id, subject_id)``: subjects are only unique within a dataset."""


class LeakageError(RuntimeError):
    """Raised when a split would let target information reach the training side."""


class InsufficientTrialsError(ValueError):
    """Raised when a subject lacks the trials a calibration plan requires."""


@dataclass(frozen=True, slots=True)
class Fold:
    """One source/target partition of subjects.

    Construction fails with :class:`LeakageError` if source and target overlap, or,
    when ``dataset_disjoint`` is set, if they share a dataset.
    """

    name: str
    source: tuple[SubjectKey, ...]
    target: tuple[SubjectKey, ...]
    dataset_disjoint: bool = False

    def __post_init__(self) -> None:
        if not self.target:
            raise ValueError(f"fold {self.name!r} has no target subjects")
        if not self.source:
            raise ValueError(f"fold {self.name!r} has no source subjects")
        overlap = set(self.source) & set(self.target)
        if overlap:
            raise LeakageError(f"fold {self.name!r}: subjects in both source and target: {overlap}")
        if self.dataset_disjoint:
            shared = {d for d, _ in self.source} & {d for d, _ in self.target}
            if shared:
                raise LeakageError(f"fold {self.name!r}: datasets in source and target: {shared}")


def within_dataset_folds(keys: Sequence[SubjectKey], n_folds: int | None, seed: int) -> list[Fold]:
    """Subject-disjoint folds (regime R1).

    Parameters
    ----------
    keys
        All ``(dataset, subject)`` pairs.
    n_folds
        Number of subject groups; ``None`` (or ``>= len(keys)``) means leave-one-subject-out.
    seed
        Seed for the subject shuffle.
    """
    unique = sorted(set(keys))
    if len(unique) < 2:
        raise ValueError("need at least two subjects for subject-disjoint folds")
    if n_folds is not None and n_folds < 2:
        raise ValueError("n_folds must be >= 2 or None")
    if n_folds is None or n_folds >= len(unique):
        groups = [[key] for key in unique]
    else:
        perm = np.random.default_rng(seed).permutation(len(unique))
        groups = [[unique[i] for i in chunk] for chunk in np.array_split(perm, n_folds)]
    folds = []
    for i, group in enumerate(groups):
        target = tuple(sorted(group))
        target_set = set(target)
        source = tuple(k for k in unique if k not in target_set)
        folds.append(Fold(name=f"fold-{i:02d}", source=source, target=target))
    return folds


def leave_dataset_out_folds(keys: Sequence[SubjectKey]) -> list[Fold]:
    """Dataset-disjoint folds (regime R2): each dataset is the target once."""
    unique = sorted(set(keys))
    datasets = sorted({d for d, _ in unique})
    if len(datasets) < 2:
        raise ValueError("leave-dataset-out needs at least two datasets")
    return [
        Fold(
            name=f"lodo-{held_out}",
            source=tuple(k for k in unique if k[0] != held_out),
            target=tuple(k for k in unique if k[0] == held_out),
            dataset_disjoint=True,
        )
        for held_out in datasets
    ]


@dataclass(frozen=True, slots=True)
class CalibrationPlan:
    """Index sets (into one subject's :class:`EpochSet`) for every calibration budget.

    Attributes
    ----------
    window_end
        Number of chronologically first trials reserved as the calibration window.
    test_idx
        The fixed test window (identical for every budget), in chronological order.
    calibration_idx
        ``k -> indices`` of the first ``k`` trials of each class within the window.
    unlabeled_idx
        The first ``n_unlabeled`` trials of the window (labels to be hidden).
    """

    window_end: int
    test_idx: IntArray
    calibration_idx: dict[int, IntArray]
    unlabeled_idx: IntArray


def plan_calibration(
    subject_epochs: EpochSet,
    ks: Sequence[int],
    n_unlabeled: int = 0,
    min_test_per_class: int = 10,
) -> CalibrationPlan:
    """Build the chronological calibration plan for one subject (protocol §4).

    Raises
    ------
    InsufficientTrialsError
        If the subject cannot provide ``max(ks)`` trials per class inside the window
        and ``min_test_per_class`` trials per class after it.
    ValueError
        On invalid arguments, multiple subjects, unlabeled input or duplicate ``order``.
    """
    if len(subject_epochs.subject_keys()) != 1:
        raise ValueError("plan_calibration expects the epochs of exactly one subject")
    if not subject_epochs.is_labeled:
        raise ValueError("plan_calibration needs labeled epochs")
    if not ks or any(k < 0 for k in ks) or len(set(ks)) != len(ks):
        raise ValueError(f"ks must be unique non-negative integers, got {list(ks)}")
    if n_unlabeled < 0 or min_test_per_class < 1:
        raise ValueError("n_unlabeled must be >= 0 and min_test_per_class >= 1")
    if len(np.unique(subject_epochs.order)) != len(subject_epochs):
        raise ValueError("order must be unique within a subject")

    chronological = np.argsort(subject_epochs.order, kind="stable").astype(np.int64)
    y = subject_epochs.y[chronological]
    k_max = max(ks)
    window_end = n_unlabeled
    positions: dict[int, IntArray] = {}
    for c, label in enumerate(subject_epochs.label_names):
        positions[c] = np.flatnonzero(y == c).astype(np.int64)
        if k_max > 0:
            if len(positions[c]) < k_max:
                raise InsufficientTrialsError(
                    f"class {label!r} has {len(positions[c])} trials, budget needs {k_max}"
                )
            window_end = max(window_end, int(positions[c][k_max - 1]) + 1)
    if window_end > len(y):
        raise InsufficientTrialsError(f"calibration window ({window_end}) exceeds {len(y)} trials")

    test_idx = chronological[window_end:]
    for c, label in enumerate(subject_epochs.label_names):
        n_test = int((y[window_end:] == c).sum())
        if n_test < min_test_per_class:
            raise InsufficientTrialsError(
                f"class {label!r} has {n_test} test trials after the calibration window, "
                f"need {min_test_per_class}"
            )

    calibration_idx: dict[int, IntArray] = {}
    for k in ks:
        chosen = np.sort(np.concatenate([positions[c][:k] for c in positions]))
        calibration_idx[k] = chronological[chosen.astype(np.int64)]
    return CalibrationPlan(
        window_end=window_end,
        test_idx=test_idx,
        calibration_idx=calibration_idx,
        unlabeled_idx=chronological[:n_unlabeled],
    )
