"""Canonical in-memory data types: :class:`Recording` and :class:`EpochSet`.

Both types validate their invariants on construction and make their arrays read-only,
so no pipeline stage can mutate shared data in place. Signals are in **volts** and
channel names are canonical 10-05 names (see :mod:`neurolayer.core.channels`).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from neurolayer.core.channels import channel_indices

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
StrArray = npt.NDArray[np.str_]

UNLABELED = -1
"""Value of ``EpochSet.y`` entries whose label is hidden from the consumer."""


def _readonly(array: npt.NDArray[Any]) -> None:
    array.flags.writeable = False


def _check_unique(names: Sequence[str], what: str) -> None:
    if len(set(names)) != len(names):
        dupes = sorted({n for n in names if list(names).count(n) > 1})
        raise ValueError(f"duplicate {what}: {dupes}")


@dataclass(frozen=True, slots=True)
class Event:
    """A labelled event in a continuous recording.

    Attributes
    ----------
    onset_sample
        Sample index of the event onset (0-based).
    label
        Canonical label, for example ``"left_hand"``.
    duration_samples
        Event duration in samples (0 for instantaneous cues).
    """

    onset_sample: int
    label: str
    duration_samples: int = 0

    def __post_init__(self) -> None:
        if self.onset_sample < 0:
            raise ValueError(f"event onset must be >= 0, got {self.onset_sample}")
        if self.duration_samples < 0:
            raise ValueError(f"event duration must be >= 0, got {self.duration_samples}")
        if not self.label:
            raise ValueError("event label must be non-empty")


@dataclass(frozen=True, slots=True)
class Recording:
    """Continuous multichannel signal from one run of one session of one subject.

    Attributes
    ----------
    data
        Array of shape ``(n_channels, n_samples)`` in volts. Becomes read-only.
    sfreq
        Sampling frequency in Hz.
    ch_names
        Canonical channel names, one per row of ``data``.
    dataset_id, subject_id, session_id, run_id
        Provenance. ``subject_id`` must be a pseudonym (privacy requirement PRIV-8).
    events
        Labelled events, all within the recording.
    device
        Device identifier for data recorded with a known device, else ``None``.
    """

    data: FloatArray
    sfreq: float
    ch_names: tuple[str, ...]
    dataset_id: str
    subject_id: str
    session_id: str = "0"
    run_id: str = "0"
    events: tuple[Event, ...] = field(default_factory=tuple)
    device: str | None = None

    def __post_init__(self) -> None:
        if self.data.ndim != 2:
            raise ValueError(f"data must be 2-D (channels, samples), got shape {self.data.shape}")
        if self.data.dtype != np.float64:
            raise TypeError(f"data must be float64, got {self.data.dtype}")
        if self.data.shape[0] != len(self.ch_names):
            raise ValueError(
                f"{self.data.shape[0]} data rows but {len(self.ch_names)} channel names"
            )
        if not self.sfreq > 0:
            raise ValueError(f"sfreq must be > 0, got {self.sfreq}")
        if not np.isfinite(self.data).all():
            raise ValueError("data contains NaN or infinite values")
        _check_unique(self.ch_names, "channel names")
        for ident, value in (("dataset_id", self.dataset_id), ("subject_id", self.subject_id)):
            if not value:
                raise ValueError(f"{ident} must be non-empty")
        n_samples = self.data.shape[1]
        for event in self.events:
            if event.onset_sample + event.duration_samples > n_samples:
                raise ValueError(
                    f"event {event} extends beyond the recording ({n_samples} samples)"
                )
        _readonly(self.data)

    @property
    def n_channels(self) -> int:
        """Number of channels."""
        return int(self.data.shape[0])

    @property
    def n_samples(self) -> int:
        """Number of samples per channel."""
        return int(self.data.shape[1])

    @property
    def duration_s(self) -> float:
        """Duration in seconds."""
        return self.n_samples / self.sfreq


@dataclass(frozen=True, slots=True, eq=False)
class EpochSet:
    """A set of equal-length trials, possibly spanning several subjects and datasets.

    Attributes
    ----------
    X
        Signals, shape ``(n_epochs, n_channels, n_times)``, float64 volts.
    y
        Integer labels indexing ``label_names``, or :data:`UNLABELED` (-1) for every
        epoch when labels are hidden.
    label_names
        Canonical label vocabulary for this set.
    ch_names
        Canonical channel names.
    sfreq
        Sampling frequency in Hz.
    subject, session, dataset
        Per-epoch provenance (string arrays of length ``n_epochs``).
    order
        Per-epoch chronological rank within its (dataset, subject). Sorting a subject's
        epochs by ``order`` must reproduce recording order across sessions; the CAP-1
        calibration split relies on it.

    Notes
    -----
    Construction validates all invariants and makes every array read-only.
    """

    X: FloatArray
    y: IntArray
    label_names: tuple[str, ...]
    ch_names: tuple[str, ...]
    sfreq: float
    subject: StrArray
    session: StrArray
    dataset: StrArray
    order: IntArray

    def __post_init__(self) -> None:
        if self.X.ndim != 3:
            raise ValueError(f"X must be 3-D (epochs, channels, times), got {self.X.shape}")
        if self.X.dtype != np.float64:
            raise TypeError(f"X must be float64, got {self.X.dtype}")
        n = self.X.shape[0]
        for name in ("y", "subject", "session", "dataset", "order"):
            arr = getattr(self, name)
            if arr.shape != (n,):
                raise ValueError(f"{name} must have shape ({n},), got {arr.shape}")
        if self.y.dtype != np.int64 or self.order.dtype != np.int64:
            raise TypeError("y and order must be int64")
        if self.X.shape[1] != len(self.ch_names):
            raise ValueError(f"{self.X.shape[1]} channels in X but {len(self.ch_names)} names")
        if not self.sfreq > 0:
            raise ValueError(f"sfreq must be > 0, got {self.sfreq}")
        if not self.label_names:
            raise ValueError("label_names must be non-empty")
        _check_unique(self.ch_names, "channel names")
        _check_unique(self.label_names, "label names")
        if n and not self.is_labeled and not (self.y == UNLABELED).all():
            raise ValueError(
                f"y must be in [0, {len(self.label_names)}) or entirely {UNLABELED} (unlabeled)"
            )
        if not np.isfinite(self.X).all():
            raise ValueError("X contains NaN or infinite values")
        for arr in (self.X, self.y, self.subject, self.session, self.dataset, self.order):
            _readonly(arr)

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_arrays(
        cls,
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        *,
        label_names: Sequence[str],
        ch_names: Sequence[str],
        sfreq: float,
        subject: str | Sequence[str],
        dataset: str | Sequence[str],
        session: str | Sequence[str] = "0",
        order: npt.ArrayLike | None = None,
    ) -> EpochSet:
        """Build an :class:`EpochSet`, broadcasting scalar provenance to every epoch.

        ``order`` defaults to ``arange(n_epochs)``: the given epoch order is taken as
        chronological.
        """
        X_arr = np.array(X, dtype=np.float64)
        n = X_arr.shape[0]

        def _strings(value: str | Sequence[str]) -> StrArray:
            # np.full(..., dtype=np.str_) would truncate to one character; np.array
            # sizes the unicode dtype to the longest string.
            values = [value] * n if isinstance(value, str) else list(value)
            return np.array(values, dtype=np.str_).reshape(n)

        return cls(
            X=X_arr,
            y=np.array(y, dtype=np.int64),
            label_names=tuple(label_names),
            ch_names=tuple(ch_names),
            sfreq=float(sfreq),
            subject=_strings(subject),
            session=_strings(session),
            dataset=_strings(dataset),
            order=np.arange(n, dtype=np.int64) if order is None else np.array(order, np.int64),
        )

    @classmethod
    def concat(cls, parts: Sequence[EpochSet]) -> EpochSet:
        """Concatenate epoch sets with identical channels, labels, rate and trial length."""
        if not parts:
            raise ValueError("cannot concatenate an empty sequence of EpochSets")
        first = parts[0]
        for other in parts[1:]:
            if other.ch_names != first.ch_names:
                raise ValueError("cannot concatenate EpochSets with different ch_names")
            if other.label_names != first.label_names:
                raise ValueError("cannot concatenate EpochSets with different label_names")
            if other.sfreq != first.sfreq or other.n_times != first.n_times:
                raise ValueError("cannot concatenate EpochSets with different sfreq or n_times")
            if other.is_labeled != first.is_labeled:
                raise ValueError("cannot concatenate labeled with unlabeled EpochSets")
        return cls(
            X=np.concatenate([p.X for p in parts]),
            y=np.concatenate([p.y for p in parts]),
            label_names=first.label_names,
            ch_names=first.ch_names,
            sfreq=first.sfreq,
            subject=np.concatenate([p.subject for p in parts]),
            session=np.concatenate([p.session for p in parts]),
            dataset=np.concatenate([p.dataset for p in parts]),
            order=np.concatenate([p.order for p in parts]),
        )

    # ------------------------------------------------------------------ properties
    def __len__(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_channels(self) -> int:
        """Number of channels."""
        return int(self.X.shape[1])

    @property
    def n_times(self) -> int:
        """Samples per epoch."""
        return int(self.X.shape[2])

    @property
    def n_classes(self) -> int:
        """Size of the label vocabulary."""
        return len(self.label_names)

    @property
    def is_labeled(self) -> bool:
        """``True`` if every label indexes ``label_names``."""
        return bool(((self.y >= 0) & (self.y < self.n_classes)).all())

    def subject_keys(self) -> list[tuple[str, str]]:
        """Sorted unique ``(dataset, subject)`` pairs present in the set."""
        pairs = {(str(d), str(s)) for d, s in zip(self.dataset, self.subject, strict=True)}
        return sorted(pairs)

    def datasets(self) -> list[str]:
        """Sorted unique dataset identifiers."""
        return sorted({str(d) for d in self.dataset})

    # ------------------------------------------------------------------ selection
    def subset(self, index: npt.ArrayLike) -> EpochSet:
        """Return the epochs selected by an integer index array or boolean mask."""
        idx = np.asarray(index)
        if idx.dtype == np.bool_:
            if idx.shape != (len(self),):
                raise ValueError(f"boolean mask must have shape ({len(self)},)")
            idx = np.flatnonzero(idx)
        idx = idx.astype(np.int64)
        return EpochSet(
            X=self.X[idx],
            y=self.y[idx],
            label_names=self.label_names,
            ch_names=self.ch_names,
            sfreq=self.sfreq,
            subject=self.subject[idx],
            session=self.session[idx],
            dataset=self.dataset[idx],
            order=self.order[idx],
        )

    def for_subject(self, dataset: str, subject: str) -> EpochSet:
        """Return all epochs of one ``(dataset, subject)``."""
        return self.subset((self.dataset == dataset) & (self.subject == subject))

    def select_channels(self, names: Sequence[str]) -> EpochSet:
        """Return a copy restricted to ``names``, in that order.

        Raises
        ------
        MissingChannelsError
            If any requested channel is absent.
        """
        idx = channel_indices(self.ch_names, names, context="EpochSet.select_channels")
        return EpochSet(
            X=self.X[:, idx, :],
            y=self.y,
            label_names=self.label_names,
            ch_names=tuple(names),
            sfreq=self.sfreq,
            subject=self.subject,
            session=self.session,
            dataset=self.dataset,
            order=self.order,
        )

    def without_labels(self) -> EpochSet:
        """Return a copy whose labels are hidden (all :data:`UNLABELED`)."""
        return EpochSet(
            X=self.X,
            y=np.full(len(self), UNLABELED, dtype=np.int64),
            label_names=self.label_names,
            ch_names=self.ch_names,
            sfreq=self.sfreq,
            subject=self.subject,
            session=self.session,
            dataset=self.dataset,
            order=self.order,
        )
