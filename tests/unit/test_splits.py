from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from neurolayer.core.types import EpochSet
from neurolayer.evaluation.splits import (
    Fold,
    InsufficientTrialsError,
    LeakageError,
    leave_dataset_out_folds,
    plan_calibration,
    within_dataset_folds,
)

KEYS = [("a", f"s{i}") for i in range(5)] + [("b", f"s{i}") for i in range(3)]


def _subject(labels: list[int], order: list[int] | None = None) -> EpochSet:
    n = len(labels)
    return EpochSet.from_arrays(
        np.zeros((n, 1, 4)),
        labels,
        label_names=("left_hand", "right_hand"),
        ch_names=("C3",),
        sfreq=100.0,
        subject="s1",
        dataset="a",
        order=order,
    )


class TestFolds:
    def test_loso_covers_every_subject_once(self) -> None:
        folds = within_dataset_folds(KEYS, n_folds=None, seed=0)
        assert len(folds) == len(KEYS)
        assert sorted(k for f in folds for k in f.target) == sorted(KEYS)

    def test_grouped_folds_are_disjoint_and_complete(self) -> None:
        folds = within_dataset_folds(KEYS, n_folds=3, seed=1)
        targets = [k for f in folds for k in f.target]
        assert sorted(targets) == sorted(KEYS)
        for fold in folds:
            assert not set(fold.source) & set(fold.target)
            assert set(fold.source) | set(fold.target) == set(KEYS)

    def test_grouped_folds_depend_on_seed_only(self) -> None:
        assert within_dataset_folds(KEYS, 3, seed=1) == within_dataset_folds(KEYS, 3, seed=1)

    def test_leave_dataset_out(self) -> None:
        folds = leave_dataset_out_folds(KEYS)
        assert [f.name for f in folds] == ["lodo-a", "lodo-b"]
        for fold in folds:
            assert not {d for d, _ in fold.source} & {d for d, _ in fold.target}

    def test_leakage_is_rejected(self) -> None:
        with pytest.raises(LeakageError):
            Fold(name="bad", source=(("a", "s1"),), target=(("a", "s1"),))
        with pytest.raises(LeakageError):
            Fold(name="bad", source=(("a", "s1"),), target=(("a", "s2"),), dataset_disjoint=True)

    def test_invalid_inputs(self) -> None:
        with pytest.raises(ValueError, match="two subjects"):
            within_dataset_folds([("a", "s0")], None, 0)
        with pytest.raises(ValueError, match="two datasets"):
            leave_dataset_out_folds([("a", "s0"), ("a", "s1")])


class TestCalibrationPlan:
    def test_chronological_and_fixed_test_window(self) -> None:
        labels = [0, 1] * 20  # 40 trials, alternating
        plan = plan_calibration(_subject(labels), ks=[0, 2, 5], n_unlabeled=4, min_test_per_class=5)
        assert plan.window_end == 10  # 5th trial of class 1 is at position 9
        np.testing.assert_array_equal(plan.test_idx, np.arange(10, 40))
        np.testing.assert_array_equal(plan.calibration_idx[2], [0, 1, 2, 3])
        assert len(plan.calibration_idx[0]) == 0
        np.testing.assert_array_equal(plan.unlabeled_idx, [0, 1, 2, 3])

    def test_uses_order_not_storage_position(self) -> None:
        labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        order = list(reversed(range(12)))  # stored newest-first
        plan = plan_calibration(_subject(labels, order), ks=[1], min_test_per_class=2)
        # Chronologically first trials are the last stored ones.
        assert set(plan.calibration_idx[1].tolist()) == {11, 10}
        assert 11 not in plan.test_idx
        assert 10 not in plan.test_idx

    def test_insufficient_trials(self) -> None:
        with pytest.raises(InsufficientTrialsError, match="budget"):
            plan_calibration(_subject([0, 1, 0, 1]), ks=[5])
        with pytest.raises(InsufficientTrialsError, match="test trials"):
            plan_calibration(_subject([0, 1] * 6), ks=[5], min_test_per_class=2)

    def test_invalid_arguments(self) -> None:
        ep = _subject([0, 1] * 10)
        with pytest.raises(ValueError, match="unique non-negative"):
            plan_calibration(ep, ks=[2, 2])
        with pytest.raises(ValueError, match="labeled"):
            plan_calibration(ep.without_labels(), ks=[1])
        with pytest.raises(ValueError, match="unique within"):
            plan_calibration(
                _subject([0, 1, 0, 1], order=[0, 0, 1, 2]), ks=[0], min_test_per_class=1
            )

    @settings(max_examples=60, deadline=None)
    @given(
        labels=st.lists(st.integers(0, 1), min_size=30, max_size=80),
        k=st.integers(0, 6),
        n_unlabeled=st.integers(0, 12),
        seed=st.integers(0, 10_000),
    )
    def test_invariants(self, labels: list[int], k: int, n_unlabeled: int, seed: int) -> None:
        order = np.random.default_rng(seed).permutation(len(labels)).tolist()
        ep = _subject(labels, order)
        try:
            plan = plan_calibration(
                ep, ks=[0, k] if k else [0], n_unlabeled=n_unlabeled, min_test_per_class=3
            )
        except InsufficientTrialsError:
            return
        test = set(plan.test_idx.tolist())
        window_orders = ep.order[plan.test_idx]
        for idx_set in [*plan.calibration_idx.values(), plan.unlabeled_idx]:
            # Calibration and unlabeled data never overlap the test window ...
            assert not set(idx_set.tolist()) & test
            # ... and always precede it in time.
            if len(idx_set) and len(window_orders):
                assert ep.order[idx_set].max() < window_orders.min()
        for budget, idx in plan.calibration_idx.items():
            assert np.bincount(ep.y[idx], minlength=2).tolist() == [budget, budget]
