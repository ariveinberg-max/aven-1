"""Calibration-time alignment components (WP-3.2).

All functions take the statistics they need as **explicit inputs**, computed by the
caller from a subject's calibration or unlabeled data. Nothing here can see test data
unless a caller passes it, which keeps the protocol's leakage rules auditable.

* **Euclidean alignment** (He & Wu, 2019): whiten each subject's trials by the inverse
  square root of their mean covariance, so every subject's mean covariance is identity.
* **Riemannian re-centering** (Zanini et al., 2018): the same idea on the SPD manifold
  using the Riemannian mean.
* **Stretching**: normalize each subject's dispersion around the identity (the "stretch"
  step of Riemannian Procrustes Analysis, Rodrigues et al., 2019). The supervised
  "rotation" step is a documented follow-up.
"""

from __future__ import annotations

import numpy as np

from neurolayer.core.types import FloatArray


def _invsqrtm(matrix: FloatArray) -> FloatArray:
    from pyriemann.geometry.base import invsqrtm

    result: FloatArray = invsqrtm(matrix)
    return result


def mean_covariance(covs: FloatArray, metric: str = "riemann") -> FloatArray:
    """Mean of SPD matrices: ``"riemann"`` (geometric) or ``"euclid"`` (arithmetic)."""
    if covs.ndim != 3 or covs.shape[0] == 0:
        raise ValueError("expected a non-empty stack of covariance matrices")
    if metric == "euclid":
        mean: FloatArray = covs.mean(axis=0)
        return mean
    if metric == "riemann":
        from pyriemann.geometry.mean import mean_riemann

        riemann: FloatArray = mean_riemann(covs)
        return riemann
    raise ValueError(f"unknown metric {metric!r}")


def recenter(covs: FloatArray, reference: FloatArray) -> FloatArray:
    """Congruence transform ``R^{-1/2} C R^{-1/2}``: moves ``reference`` to identity."""
    whitening = _invsqrtm(reference)
    recentered: FloatArray = whitening @ covs @ whitening
    return recentered


def euclidean_alignment_matrix(X: FloatArray) -> FloatArray:
    """Whitening matrix ``R^{-1/2}`` from the arithmetic mean trial covariance of ``X``."""
    if X.ndim != 3 or X.shape[0] == 0:
        raise ValueError("expected epochs of shape (n_epochs, n_channels, n_times)")
    centered = X - X.mean(axis=2, keepdims=True)
    mean_cov = np.einsum("nct,ndt->cd", centered, centered) / (X.shape[0] * X.shape[2])
    ridge = 1e-6 * np.trace(mean_cov) / mean_cov.shape[0]
    return _invsqrtm(mean_cov + ridge * np.eye(mean_cov.shape[0]))


def apply_alignment(X: FloatArray, matrix: FloatArray) -> FloatArray:
    """Apply a spatial alignment matrix to every epoch: ``W @ X_i``."""
    aligned: FloatArray = np.einsum("cd,ndt->nct", matrix, X)
    return aligned


def stretch(covs: FloatArray, target_dispersion: float = 1.0) -> FloatArray:
    """Rescale dispersion around the identity: ``C^s`` so mean ``||log C||_F^2`` = target.

    Apply to matrices already re-centered at the identity.
    """
    from pyriemann.geometry.base import logm, powm

    dispersion = float(np.mean([np.linalg.norm(logm(c), "fro") ** 2 for c in covs]))
    if dispersion <= 0:
        return covs
    exponent = float(np.sqrt(target_dispersion / dispersion))
    stretched: FloatArray = np.stack([powm(c, exponent) for c in covs])
    return stretched
