from __future__ import annotations

import math

import pytest

from neurolayer.evaluation import metrics


def test_wolpaw_known_values() -> None:
    assert metrics.wolpaw_bits_per_trial(1.0, 2) == pytest.approx(1.0)
    assert metrics.wolpaw_bits_per_trial(0.5, 2) == 0.0
    assert metrics.wolpaw_bits_per_trial(0.3, 2) == 0.0  # below chance → 0 by convention
    # p = 0.8, N = 2: 1 + 0.8 log2 0.8 + 0.2 log2 0.2 ≈ 0.2781
    assert metrics.wolpaw_bits_per_trial(0.8, 2) == pytest.approx(0.2781, abs=1e-4)
    assert metrics.itr_bits_per_minute(1.0, 4, 2.0) == pytest.approx(60.0)
    with pytest.raises(ValueError, match="n_classes"):
        metrics.wolpaw_bits_per_trial(0.9, 1)
    with pytest.raises(ValueError, match="seconds"):
        metrics.itr_bits_per_minute(0.9, 2, 0)


def test_balanced_accuracy_and_kappa() -> None:
    assert metrics.balanced_accuracy([0, 0, 0, 1], [0, 0, 0, 0]) == pytest.approx(0.5)
    assert metrics.accuracy([0, 1, 1], [0, 1, 0]) == pytest.approx(2 / 3)
    assert metrics.cohen_kappa([0, 1, 0, 1], [0, 1, 0, 1]) == pytest.approx(1.0)
    assert metrics.cohen_kappa([0, 0], [0, 0]) == 0.0


def test_bootstrap_ci_brackets_mean() -> None:
    values = [0.5, 0.6, 0.7, 0.8, 0.9]
    low, high = metrics.bootstrap_ci(values, seed=1)
    assert low < 0.7 < high
    assert metrics.bootstrap_ci([0.6]) == (0.6, 0.6)
    assert metrics.bootstrap_ci(values, seed=1) == metrics.bootstrap_ci(values, seed=1)
    with pytest.raises(ValueError, match="empty"):
        metrics.bootstrap_ci([])


def test_usability_metrics() -> None:
    assert metrics.usable_user_rate([0.5, 0.7, 0.9]) == pytest.approx(2 / 3)
    assert metrics.trials_to_criterion({0: 0.55, 5: 0.68, 10: 0.72, 20: 0.8}) == 10
    assert metrics.trials_to_criterion({0: 0.5, 5: 0.6}) is None
    assert metrics.median_trials_to_criterion([5, 10, None]) == 10.0
    assert math.isinf(metrics.median_trials_to_criterion([None, None, 5]))


def test_aucec() -> None:
    assert metrics.aucec([0, 5, 10], [0.7, 0.7, 0.7]) == pytest.approx(0.7)
    assert metrics.aucec([10], [0.8]) == pytest.approx(0.8)
    # Order-independent, and weighted toward small budgets (log2(1 + k) axis).
    early = metrics.aucec([0, 5, 40], [0.5, 0.8, 0.8])
    late = metrics.aucec([40, 0, 5], [0.8, 0.5, 0.5])
    assert early > late
    with pytest.raises(ValueError, match="equal length"):
        metrics.aucec([0, 1], [0.5])


def test_paired_tests() -> None:
    a = [0.60, 0.62, 0.65, 0.70, 0.71, 0.72, 0.75, 0.78, 0.80, 0.82]
    b = [x + 0.08 for x in a]
    assert metrics.paired_wilcoxon(a, b) < 0.01
    assert metrics.paired_wilcoxon(a, a) == 1.0
    assert metrics.holm_bonferroni([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
