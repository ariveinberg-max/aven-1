from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from neurolayer.data.synthetic import SyntheticMIConfig, generate_synthetic_mi
from neurolayer.models.reference import log_variance


def test_shapes_order_and_provenance() -> None:
    cfg = SyntheticMIConfig(n_subjects=3, n_trials_per_class=10, n_sessions=2, sfreq=100.0)
    result = generate_synthetic_mi(cfg)
    ep = result.epochs
    assert len(ep) == 3 * 2 * 2 * 10
    assert ep.X.shape[1:] == (len(cfg.channels), 200)
    assert ep.label_names == ("left_hand", "right_hand")
    assert set(result.efficiency) == {"sub-000", "sub-001", "sub-002"}
    one = ep.for_subject("synthetic_mi", "sub-001")
    assert sorted(one.order.tolist()) == list(range(40))  # chronological across sessions
    assert set(one.session.tolist()) == {"ses-00", "ses-01"}
    assert np.abs(ep.X).max() < 1e-3  # volts, EEG-like magnitude


def test_deterministic_given_seed() -> None:
    cfg = SyntheticMIConfig(n_subjects=2, n_trials_per_class=5, seed=42)
    a, b = generate_synthetic_mi(cfg).epochs, generate_synthetic_mi(cfg).epochs
    np.testing.assert_array_equal(a.X, b.X)
    np.testing.assert_array_equal(a.y, b.y)
    c = generate_synthetic_mi(SyntheticMIConfig(n_subjects=2, n_trials_per_class=5, seed=43)).epochs
    assert not np.allclose(a.X, c.X)


def _within_subject_accuracy(efficiency: float) -> float:
    cfg = SyntheticMIConfig(n_subjects=1, n_trials_per_class=60, efficiencies=(efficiency,), seed=5)
    ep = generate_synthetic_mi(cfg).epochs
    scores = cross_val_score(LogisticRegression(max_iter=1000), log_variance(ep.X), ep.y, cv=5)
    return float(scores.mean())


def test_efficiency_controls_separability() -> None:
    assert _within_subject_accuracy(1.0) > 0.85
    assert _within_subject_accuracy(0.0) < 0.65  # BCI-inefficient subject ≈ chance


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_subjects": 0},
        {"efficiencies": (0.5,), "n_subjects": 2},
        {"efficiencies": (1.5,), "n_subjects": 1},
        {"erd_depth": 1.0},
        {"sfreq": 20.0},
    ],
)
def test_invalid_config(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        SyntheticMIConfig(**kwargs)  # type: ignore[arg-type]
