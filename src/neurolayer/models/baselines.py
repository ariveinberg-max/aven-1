"""Classical CAP-1 baselines B1-B3 (WP-4.1). See CAP-1 spec §4.

* **B1** ``PerSubjectCSPLDA``: CSP + shrinkage LDA on the target's calibration only.
* **B2** ``PerSubjectTangentSpaceLR``: Riemannian tangent space + logistic regression on
  the target's calibration only.
* **B3** ``PooledRiemannianDecoder``: tangent space pooled over source subjects, each
  re-centered at its own Riemannian mean; the target is re-centered with statistics from
  its calibration/unlabeled window. This is the strongest classical transfer baseline.

Per-subject decoders (B1, B2) have no model at ``k = 0``; they then predict class 0,
which scores exactly chance in balanced accuracy, as the protocol intends.
"""

from __future__ import annotations

from typing import Any, Self

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

from neurolayer.core.channels import channel_indices
from neurolayer.core.types import EpochSet, FloatArray, IntArray
from neurolayer.representation.alignment import mean_covariance, recenter
from neurolayer.representation.covariance import (
    CSPEncoder,
    TangentSpaceEncoder,
    covariances,
    tangent_vectors,
)


def _enough_calibration(calibration: EpochSet | None, minimum: int = 2) -> bool:
    if calibration is None or len(calibration) == 0:
        return False
    counts = np.bincount(calibration.y, minlength=calibration.n_classes)
    return bool((counts >= minimum).all())


class _PerSubject:
    """Shared logic of calibration-only decoders (the R0-style reference)."""

    def __init__(self) -> None:
        self._encoder: Any = None
        self._classifier: Any = None
        self._channels: tuple[str, ...] | None = None

    def _build(self) -> tuple[Any, Any]:
        raise NotImplementedError

    def fit(self, source: EpochSet) -> None:
        """Ignore source subjects by design: these are calibration-only baselines."""

    def adapt(self, calibration: EpochSet | None, unlabeled: EpochSet | None) -> Self:
        """Train on the calibration trials when at least 2 per class are available."""
        self._encoder = self._classifier = None
        if calibration is not None and _enough_calibration(calibration):
            encoder, classifier = self._build()
            encoder.fit(calibration)
            classifier.fit(encoder.transform(calibration), calibration.y)
            self._encoder, self._classifier = encoder, classifier
            self._channels = calibration.ch_names
        return self

    def predict(self, epochs: EpochSet) -> IntArray:
        """Predict with the calibration model, or class 0 when there is none."""
        if self._classifier is None:
            return np.zeros(len(epochs), dtype=np.int64)
        if epochs.ch_names != self._channels:
            raise ValueError("test epochs must use the calibration channels")
        predictions: IntArray = self._classifier.predict(self._encoder.transform(epochs))
        return predictions.astype(np.int64)


class PerSubjectCSPLDA(_PerSubject):
    """B1: CSP (log-variance) + shrinkage LDA, trained per subject on calibration only."""

    def __init__(self, n_components: int = 6, estimator: str = "oas") -> None:
        super().__init__()
        self.n_components = n_components
        self.estimator = estimator

    def _build(self) -> tuple[Any, Any]:
        return (
            CSPEncoder(self.n_components, self.estimator),
            LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        )


class PerSubjectTangentSpaceLR(_PerSubject):
    """B2: tangent space + logistic regression, trained per subject on calibration only."""

    def __init__(self, C: float = 1.0, estimator: str = "oas") -> None:
        super().__init__()
        self.C = C
        self.estimator = estimator

    def _build(self) -> tuple[Any, Any]:
        return TangentSpaceEncoder(self.estimator), LogisticRegression(C=self.C, max_iter=2000)


class PooledRiemannianDecoder:
    """B3: pooled tangent-space logistic regression with per-subject re-centering.

    Parameters
    ----------
    C
        Inverse regularization strength of the logistic regression.
    calibration_share
        Fraction of total sample weight given to target calibration trials when k > 0.
    recenter_subjects
        Re-center each subject at its own mean (unsupervised domain alignment).
    estimator
        Covariance estimator.
    """

    def __init__(
        self,
        C: float = 1.0,
        calibration_share: float = 0.5,
        recenter_subjects: bool = True,
        estimator: str = "oas",
    ) -> None:
        if not 0.0 < calibration_share < 1.0:
            raise ValueError("calibration_share must lie in (0, 1)")
        self.C = C
        self.calibration_share = calibration_share
        self.recenter_subjects = recenter_subjects
        self.estimator = estimator
        self._channels: tuple[str, ...] | None = None
        self._covs: FloatArray | None = None
        self._y: IntArray | None = None
        self._groups: list[IntArray] = []
        self._target_covs: FloatArray | None = None
        self._target_channels: tuple[str, ...] | None = None
        self._cal_covs: FloatArray | None = None
        self._cal_y: IntArray | None = None

    def fit(self, source: EpochSet) -> None:
        """Estimate source covariances once; features are built per target montage."""
        self._channels = source.ch_names
        self._covs = covariances(source.X, self.estimator)
        self._y = source.y.copy()
        self._groups = [
            np.flatnonzero((source.dataset == d) & (source.subject == s)).astype(np.int64)
            for d, s in source.subject_keys()
        ]

    def adapt(self, calibration: EpochSet | None, unlabeled: EpochSet | None) -> Self:
        """Collect target covariances from the calibration window (never the test window)."""
        pieces = [p for p in (calibration, unlabeled) if p is not None and len(p)]
        if pieces:
            if any(p.ch_names != pieces[0].ch_names for p in pieces):
                raise ValueError("calibration and unlabeled epochs must share channels")
            self._target_channels = pieces[0].ch_names
            self._target_covs = covariances(np.concatenate([p.X for p in pieces]), self.estimator)
        if calibration is not None and len(calibration):
            self._cal_covs = covariances(calibration.X, self.estimator)
            self._cal_y = calibration.y.copy()
        return self

    def _features(self, covs: FloatArray, reference: FloatArray) -> FloatArray:
        if self.recenter_subjects:
            return tangent_vectors(recenter(covs, reference), np.eye(covs.shape[1]))
        return tangent_vectors(covs, reference)

    def predict(self, epochs: EpochSet) -> IntArray:
        """Build montage-specific features, fit the classifier, predict the target."""
        if self._covs is None or self._y is None or self._channels is None:
            raise RuntimeError("PooledRiemannianDecoder.predict called before fit")
        idx = channel_indices(self._channels, epochs.ch_names, "target vs source montage")
        source_covs = self._covs[:, idx][:, :, idx]
        global_ref = mean_covariance(source_covs)
        source_feats = np.empty((len(source_covs), len(idx) * (len(idx) + 1) // 2))
        for group in self._groups:
            ref = mean_covariance(source_covs[group]) if self.recenter_subjects else global_ref
            source_feats[group] = self._features(source_covs[group], ref)

        if self._target_covs is not None:
            if self._target_channels != epochs.ch_names:
                raise ValueError("test epochs must use the calibration channels")
            target_ref = mean_covariance(self._target_covs)
        else:
            target_ref = global_ref  # strict zero-shot: no target statistics exist
        features = [source_feats]
        labels = [self._y]
        weights = [np.ones(len(self._y))]
        if self._cal_covs is not None and self._cal_y is not None:
            n_src, n_cal = len(self._y), len(self._cal_y)
            share = self.calibration_share
            features.append(self._features(self._cal_covs, target_ref))
            labels.append(self._cal_y)
            weights.append(np.full(n_cal, share / (1.0 - share) * n_src / n_cal))
        classifier = LogisticRegression(C=self.C, max_iter=2000)
        classifier.fit(
            np.concatenate(features), np.concatenate(labels), sample_weight=np.concatenate(weights)
        )
        test_feats = self._features(covariances(epochs.X, self.estimator), target_ref)
        predictions: IntArray = classifier.predict(test_feats).astype(np.int64)
        return predictions
