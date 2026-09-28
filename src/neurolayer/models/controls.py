"""Leakage controls implemented as decoder wrappers (WP-4.3).

``ShuffleControlDecoder`` permutes labels within each subject before training and
calibration. A sound pipeline must then score chance at **every** budget; anything
clearly above chance means information leaks around the labels (for example through
subject or trial-order artifacts).
"""

from __future__ import annotations

from typing import Self

import numpy as np

from neurolayer.core.interfaces import Decoder
from neurolayer.core.types import EpochSet, IntArray


def _shuffled(epochs: EpochSet, rng: np.random.Generator) -> EpochSet:
    y = np.array(epochs.y)
    for dataset, subject in epochs.subject_keys():
        idx = np.flatnonzero((epochs.dataset == dataset) & (epochs.subject == subject))
        y[idx] = y[rng.permutation(idx)]
    return EpochSet(
        X=epochs.X,
        y=y,
        label_names=epochs.label_names,
        ch_names=epochs.ch_names,
        sfreq=epochs.sfreq,
        subject=epochs.subject,
        session=epochs.session,
        dataset=epochs.dataset,
        order=epochs.order,
    )


class ShuffleControlDecoder:
    """Wrap any decoder and train it on within-subject-shuffled labels."""

    def __init__(self, inner: Decoder, seed: int = 0) -> None:
        self.inner = inner
        self.seed = seed

    def fit(self, source: EpochSet) -> None:
        """Fit the inner decoder on shuffled source labels."""
        self.inner.fit(_shuffled(source, np.random.default_rng(self.seed)))

    def adapt(self, calibration: EpochSet | None, unlabeled: EpochSet | None) -> Self:
        """Adapt the inner decoder with shuffled calibration labels."""
        if calibration is not None and len(calibration):
            calibration = _shuffled(calibration, np.random.default_rng(self.seed + 1))
        self.inner = self.inner.adapt(calibration, unlabeled)
        return self

    def predict(self, epochs: EpochSet) -> IntArray:
        """Delegate to the inner decoder."""
        return self.inner.predict(epochs)
