"""Covariance-based representations (WP-3.1): the strongest classical family for MI.

Signals are scaled to microvolts before covariance estimation for numerical comfort;
a constant scale does not change any downstream decision.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from neurolayer.core.types import EpochSet, FloatArray

_TO_MICROVOLTS = 1e6


def covariances(X: FloatArray, estimator: str = "oas") -> FloatArray:
    """Regularized spatial covariance per epoch, shape ``(n_epochs, n_ch, n_ch)``.

    ``estimator`` is any pyRiemann estimator name (``"oas"``, ``"lwf"``, ``"scm"``...).
    """
    from pyriemann.estimation import Covariances

    covs: FloatArray = Covariances(estimator=estimator).fit_transform(X * _TO_MICROVOLTS)
    return covs


def tangent_vectors(covs: FloatArray, reference: FloatArray, metric: str = "riemann") -> FloatArray:
    """Project SPD matrices to the tangent space at ``reference`` (upper-triangle vectors)."""
    from pyriemann.tangentspace import tangent_space

    vectors: FloatArray = tangent_space(covs, reference, metric=metric)
    return vectors


class TangentSpaceEncoder:
    """Covariance → Riemannian tangent space at the training mean (Barachant et al.)."""

    def __init__(self, estimator: str = "oas", metric: str = "riemann") -> None:
        self.estimator = estimator
        self.metric = metric
        self._ts: Any = None

    def fit(self, epochs: EpochSet) -> None:
        """Estimate the reference point (Riemannian mean of training covariances)."""
        from pyriemann.tangentspace import TangentSpace

        self._ts = TangentSpace(metric=self.metric).fit(covariances(epochs.X, self.estimator))

    def transform(self, epochs: EpochSet) -> FloatArray:
        """Return tangent vectors, shape ``(n_epochs, n_ch * (n_ch + 1) / 2)``."""
        if self._ts is None:
            raise RuntimeError("TangentSpaceEncoder.transform called before fit")
        vectors: FloatArray = self._ts.transform(covariances(epochs.X, self.estimator))
        return vectors


class CSPEncoder:
    """Common Spatial Patterns log-variance features (needs labels to fit)."""

    def __init__(self, n_components: int = 6, estimator: str = "oas") -> None:
        self.n_components = n_components
        self.estimator = estimator
        self._csp: Any = None

    def fit(self, epochs: EpochSet) -> None:
        """Fit spatial filters on labeled epochs."""
        from pyriemann.spatialfilters import CSP

        if not epochs.is_labeled:
            raise ValueError("CSPEncoder needs labeled epochs")
        n_filters = min(self.n_components, epochs.n_channels)
        self._csp = CSP(nfilter=n_filters, log=True).fit(
            covariances(epochs.X, self.estimator), epochs.y
        )

    def transform(self, epochs: EpochSet) -> FloatArray:
        """Return log-variance of CSP-filtered signals, shape ``(n_epochs, n_components)``."""
        if self._csp is None:
            raise RuntimeError("CSPEncoder.transform called before fit")
        features: FloatArray = self._csp.transform(covariances(epochs.X, self.estimator))
        return features


class LogVarianceEncoder:
    """Per-channel log-variance (the simplest band-power representation)."""

    def fit(self, epochs: EpochSet) -> None:
        """Stateless."""

    def transform(self, epochs: EpochSet) -> FloatArray:
        """Return ``log(var)`` per channel, shape ``(n_epochs, n_channels)``."""
        variance = np.var(epochs.X * _TO_MICROVOLTS, axis=2)
        features: FloatArray = np.log(np.maximum(variance, np.finfo(np.float64).tiny))
        return features
