from __future__ import annotations

import numpy as np
import pytest

from neurolayer.core.types import EpochSet
from neurolayer.representation.alignment import (
    apply_alignment,
    euclidean_alignment_matrix,
    mean_covariance,
    recenter,
    stretch,
)
from neurolayer.representation.covariance import (
    CSPEncoder,
    LogVarianceEncoder,
    TangentSpaceEncoder,
    covariances,
    tangent_vectors,
)
from neurolayer.representation.registry import make_encoder

pytest.importorskip("pyriemann")


def test_covariances_are_spd(epochs: EpochSet) -> None:
    covs = covariances(epochs.X[:20])
    assert covs.shape == (20, epochs.n_channels, epochs.n_channels)
    np.testing.assert_allclose(covs, np.transpose(covs, (0, 2, 1)))
    assert (np.linalg.eigvalsh(covs) > 0).all()


def test_tangent_space_encoder(epochs: EpochSet) -> None:
    encoder = TangentSpaceEncoder()
    with pytest.raises(RuntimeError, match="before fit"):
        encoder.transform(epochs)
    encoder.fit(epochs)
    c = epochs.n_channels
    assert encoder.transform(epochs).shape == (len(epochs), c * (c + 1) // 2)
    reference = mean_covariance(covariances(epochs.X))
    assert tangent_vectors(covariances(epochs.X[:3]), reference).shape == (3, c * (c + 1) // 2)


def test_csp_and_logvar_encoders(epochs: EpochSet) -> None:
    csp = CSPEncoder(n_components=4)
    with pytest.raises(ValueError, match="labeled"):
        csp.fit(epochs.without_labels())
    csp.fit(epochs)
    assert csp.transform(epochs).shape == (len(epochs), 4)
    logvar = make_encoder("log_variance")
    logvar.fit(epochs)
    assert logvar.transform(epochs).shape == (len(epochs), epochs.n_channels)
    assert isinstance(logvar, LogVarianceEncoder)
    with pytest.raises(KeyError, match="unknown encoder"):
        make_encoder("wavelets")


def test_recentering_moves_mean_to_identity(epochs: EpochSet) -> None:
    covs = covariances(epochs.X[:40])
    centered = recenter(covs, mean_covariance(covs, "riemann"))
    np.testing.assert_allclose(
        mean_covariance(centered, "riemann"), np.eye(covs.shape[1]), atol=1e-6
    )
    with pytest.raises(ValueError, match="unknown metric"):
        mean_covariance(covs, "manhattan")


def test_euclidean_alignment_whitens_mean_covariance(epochs: EpochSet) -> None:
    X = epochs.X[:40] * 1e6
    aligned = apply_alignment(X, euclidean_alignment_matrix(X))
    centered = aligned - aligned.mean(axis=2, keepdims=True)
    mean_cov = np.einsum("nct,ndt->cd", centered, centered) / (X.shape[0] * X.shape[2])
    np.testing.assert_allclose(mean_cov, np.eye(X.shape[1]), atol=1e-3)
    with pytest.raises(ValueError, match="n_epochs"):
        euclidean_alignment_matrix(X[0])


def test_stretch_normalizes_dispersion(epochs: EpochSet) -> None:
    from pyriemann.geometry.base import logm

    covs = recenter(covariances(epochs.X[:30]), mean_covariance(covariances(epochs.X[:30])))
    stretched = stretch(covs, target_dispersion=2.0)
    dispersion = np.mean([np.linalg.norm(logm(c), "fro") ** 2 for c in stretched])
    assert dispersion == pytest.approx(2.0, rel=1e-6)
