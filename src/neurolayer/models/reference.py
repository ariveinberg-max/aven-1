"""Reference decoders used to validate the CAP-1 harness end to end.

These are **not** the CAP-1 baseline suite (B0–B6, work package WP-4.1). They are
small, dependency-light decoders whose expected behavior is known:

* :class:`ChanceDecoder` must score ≈ chance, which checks that the harness does not
  leak labels.
* :class:`LogVarianceDecoder` must score above chance on separable synthetic data and
  improve with calibration, which checks that calibration data reaches the decoder.
"""

from __future__ import annotations

from typing import Self

import numpy as np
from sklearn.linear_model import LogisticRegression

from neurolayer.core.channels import channel_indices
from neurolayer.core.types import EpochSet, FloatArray, IntArray

_STD_FLOOR = 1e-12


def log_variance(X: FloatArray) -> FloatArray:
    """Per-channel log-variance features, shape ``(n_epochs, n_channels)``."""
    variance = np.var(X, axis=2)
    features: FloatArray = np.log(np.maximum(variance, np.finfo(np.float64).tiny))
    return features


def _standardize_per_subject(features: FloatArray, epochs: EpochSet) -> FloatArray:
    out = np.empty_like(features)
    for dataset, subject in epochs.subject_keys():
        mask = (epochs.dataset == dataset) & (epochs.subject == subject)
        block = features[mask]
        out[mask] = (block - block.mean(axis=0)) / np.maximum(block.std(axis=0), _STD_FLOOR)
    return out


class ChanceDecoder:
    """Predicts uniformly random labels; expected balanced accuracy is 1 / n_classes."""

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)
        self._n_classes: int | None = None

    def fit(self, source: EpochSet) -> None:
        """Record the number of classes."""
        self._n_classes = source.n_classes

    def adapt(self, calibration: EpochSet | None, unlabeled: EpochSet | None) -> Self:
        """Ignore target data."""
        return self

    def predict(self, epochs: EpochSet) -> IntArray:
        """Return random labels."""
        if self._n_classes is None:
            raise RuntimeError("ChanceDecoder.predict called before fit")
        return self._rng.integers(0, self._n_classes, size=len(epochs), dtype=np.int64)


class LogVarianceDecoder:
    """Log-variance features, per-subject standardization, logistic regression.

    * **fit:** standardize each source subject's features with its own statistics, then
      fit a pooled classifier.
    * **adapt:** estimate the target's standardization from its calibration and
      unlabeled trials (never from test trials). With ``k > 0``, refit on source plus
      calibration, weighting calibration to ``calibration_share`` of the total weight.
    * **predict:** handles target montages that are a subset of the source channels by
      restricting the source features to the target's channels. With no target data
      at all (strict zero-shot), it falls back to pooled source statistics.

    Parameters
    ----------
    calibration_share
        Fraction of total sample weight given to the target's calibration trials.
    C
        Inverse regularization strength of the logistic regression.
    """

    def __init__(self, calibration_share: float = 0.5, C: float = 1.0) -> None:
        if not 0.0 < calibration_share < 1.0:
            raise ValueError("calibration_share must lie in (0, 1)")
        self.calibration_share = calibration_share
        self.C = C
        self._src_channels: tuple[str, ...] | None = None
        self._src_features: FloatArray | None = None
        self._src_y: IntArray | None = None
        self._src_mean: FloatArray | None = None
        self._src_std: FloatArray | None = None
        self._tgt_channels: tuple[str, ...] | None = None
        self._tgt_mean: FloatArray | None = None
        self._tgt_std: FloatArray | None = None
        self._cal_features: FloatArray | None = None
        self._cal_y: IntArray | None = None

    def fit(self, source: EpochSet) -> None:
        """Fit the pooled source model."""
        if not source.is_labeled:
            raise ValueError("source epochs must be labeled")
        raw = log_variance(source.X)
        self._src_channels = source.ch_names
        self._src_features = _standardize_per_subject(raw, source)
        self._src_y = source.y.copy()
        self._src_mean = raw.mean(axis=0)
        self._src_std = np.maximum(raw.std(axis=0), _STD_FLOOR)

    def adapt(self, calibration: EpochSet | None, unlabeled: EpochSet | None) -> Self:
        """Store target statistics and calibration features (no test data involved)."""
        pieces = [e for e in (calibration, unlabeled) if e is not None and len(e) > 0]
        if pieces:
            channels = pieces[0].ch_names
            if any(p.ch_names != channels for p in pieces):
                raise ValueError("calibration and unlabeled epochs must share channels")
            raw = log_variance(np.concatenate([p.X for p in pieces]))
            self._tgt_channels = channels
            self._tgt_mean = raw.mean(axis=0)
            self._tgt_std = np.maximum(raw.std(axis=0), _STD_FLOOR)
        if calibration is not None and len(calibration) > 0:
            self._cal_features = log_variance(calibration.X)
            self._cal_y = calibration.y.copy()
        return self

    def predict(self, epochs: EpochSet) -> IntArray:
        """Predict labels for one target subject's epochs."""
        if (
            self._src_channels is None
            or self._src_features is None
            or self._src_y is None
            or self._src_mean is None
            or self._src_std is None
        ):
            raise RuntimeError("LogVarianceDecoder.predict called before fit")
        channels = epochs.ch_names
        cols = channel_indices(self._src_channels, channels, context="target vs source montage")
        if self._tgt_channels is not None and self._tgt_channels != channels:
            raise ValueError("test epochs must use the same channels as the calibration data")

        if self._tgt_mean is not None and self._tgt_std is not None:
            mean, std = self._tgt_mean, self._tgt_std
        else:
            mean, std = self._src_mean[cols], self._src_std[cols]

        features = [self._src_features[:, cols]]
        labels = [self._src_y]
        weights = [np.ones(len(self._src_y))]
        if self._cal_features is not None and self._cal_y is not None:
            n_src, n_cal = len(self._src_y), len(self._cal_y)
            share = self.calibration_share
            features.append((self._cal_features - mean) / std)
            labels.append(self._cal_y)
            weights.append(np.full(n_cal, share / (1.0 - share) * n_src / n_cal))

        clf = LogisticRegression(C=self.C, max_iter=1000)
        clf.fit(
            np.concatenate(features), np.concatenate(labels), sample_weight=np.concatenate(weights)
        )
        predictions: IntArray = clf.predict((log_variance(epochs.X) - mean) / std).astype(np.int64)
        return predictions
