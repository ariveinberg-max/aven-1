from __future__ import annotations

from typing import Self

import numpy as np
import pytest

from neurolayer.core.channels import CONSUMER_MONTAGES
from neurolayer.core.interfaces import Decoder
from neurolayer.core.types import EpochSet, IntArray
from neurolayer.data.synthetic import SyntheticMIConfig, generate_synthetic_mi
from neurolayer.evaluation.protocol import ProtocolConfig, run_protocol
from neurolayer.models.reference import ChanceDecoder, LogVarianceDecoder

FAST = {"n_bootstrap": 200}


class SharedLog:
    """Survives the harness's deep copies so the test can observe every call."""

    def __init__(self) -> None:
        self.adapt_history_lengths: list[int] = []
        self.unlabeled_labels: list[int] = []
        self.predict_labels: list[int] = []
        self.fit_target_overlap: list[bool] = []

    def __deepcopy__(self, memo: dict[int, object]) -> SharedLog:
        return self


class SpyDecoder:
    """Records what the harness shows it; always predicts class 0."""

    def __init__(self, log: SharedLog) -> None:
        self.log = log
        self.fitted_on: set[tuple[str, str]] = set()
        self.calibration_sizes: list[int] = []

    def fit(self, source: EpochSet) -> None:
        self.fitted_on = set(source.subject_keys())

    def adapt(self, calibration: EpochSet | None, unlabeled: EpochSet | None) -> Self:
        # Mutating self must not leak to other subjects/budgets (the harness deep-copies).
        self.calibration_sizes.append(0 if calibration is None else len(calibration))
        self.log.adapt_history_lengths.append(len(self.calibration_sizes))
        if unlabeled is not None:
            self.log.unlabeled_labels.extend(unlabeled.y.tolist())
        return self

    def predict(self, epochs: EpochSet) -> IntArray:
        self.log.predict_labels.extend(epochs.y.tolist())
        self.log.fit_target_overlap.append(bool(set(epochs.subject_keys()) & self.fitted_on))
        return np.zeros(len(epochs), dtype=np.int64)


def test_spy_satisfies_protocol() -> None:
    assert isinstance(SpyDecoder(SharedLog()), Decoder)
    assert isinstance(LogVarianceDecoder(), Decoder)


def test_harness_hides_labels_and_isolates_state(epochs: EpochSet) -> None:
    log = SharedLog()
    config = ProtocolConfig(ks=(0, 5, 10), n_unlabeled=6, n_folds=3, **FAST)
    result = run_protocol(epochs, lambda: SpyDecoder(log), config)

    n_calls = len(epochs.subject_keys()) * 3
    assert len(result.records) == n_calls
    # The decoder never sees a test or unlabeled label.
    assert set(log.predict_labels) == {-1}
    assert set(log.unlabeled_labels) == {-1}
    # Every adapt starts from the freshly fitted state (deep copy per subject and budget).
    assert log.adapt_history_lengths == [1] * n_calls
    # The subject being predicted was never part of that fold's fit data.
    assert log.fit_target_overlap == [False] * n_calls
    # A constant predictor scores exactly chance in balanced accuracy.
    assert {r.balanced_accuracy for r in result.records} == {0.5}


def test_chance_decoder_scores_chance() -> None:
    ep = generate_synthetic_mi(
        SyntheticMIConfig(n_subjects=8, n_trials_per_class=60, seed=1)
    ).epochs
    result = run_protocol(ep, lambda: ChanceDecoder(seed=0), ProtocolConfig(ks=(0, 10), **FAST))
    for summary in result.per_budget():
        assert 0.4 < summary.mean_ba < 0.6


def test_logvar_decoder_learns_and_benefits_from_calibration(epochs: EpochSet) -> None:
    config = ProtocolConfig(ks=(0, 5, 20), n_unlabeled=10, n_folds=3, **FAST)
    result = run_protocol(epochs, LogVarianceDecoder, config)
    by_k = {s.k: s for s in result.per_budget()}
    assert by_k[20].mean_ba > 0.65
    assert by_k[20].mean_ba >= by_k[0].mean_ba - 0.02
    summary = result.summary_dict()
    assert summary["n_subjects_scored"] == 6
    assert 0.0 <= summary["aucec"] <= 1.0
    assert len(result.trials_to_criterion()) == 6


def test_leave_dataset_out_with_consumer_montage() -> None:
    parts = [
        generate_synthetic_mi(
            SyntheticMIConfig(
                n_subjects=3, n_trials_per_class=40, seed=s, dataset_id=name, site_shift=shift
            )
        ).epochs
        for s, name, shift in [(1, "site_a", 0.0), (2, "site_b", 0.3)]
    ]
    ep = EpochSet.concat(parts)
    config = ProtocolConfig(
        regime="leave_dataset_out",
        ks=(0, 10),
        target_montage=CONSUMER_MONTAGES["bci_iv_2b"].channels,
        **FAST,
    )
    result = run_protocol(ep, LogVarianceDecoder, config)
    assert {r.fold for r in result.records} == {"lodo-site_a", "lodo-site_b"}
    assert len(result.records) == 6 * 2


def test_missing_montage_channels_are_skipped_with_reason(epochs: EpochSet) -> None:
    config = ProtocolConfig(ks=(0, 5), target_montage=CONSUMER_MONTAGES["muse_s"].channels, **FAST)
    result = run_protocol(epochs, LogVarianceDecoder, config)
    assert not result.records
    assert len(result.skipped) == 6
    assert all("TP9" in s.reason for s in result.skipped)


def test_insufficient_trials_are_skipped(epochs: EpochSet) -> None:
    result = run_protocol(epochs, LogVarianceDecoder, ProtocolConfig(ks=(0, 39), **FAST))
    assert not result.records
    assert all("test trials" in s.reason for s in result.skipped)


class BadDecoder(ChanceDecoder):
    def predict(self, epochs: EpochSet) -> IntArray:
        return np.full(len(epochs), 7, dtype=np.int64)


def test_invalid_predictions_raise(epochs: EpochSet) -> None:
    with pytest.raises(ValueError, match="invalid predictions"):
        run_protocol(epochs, BadDecoder, ProtocolConfig(ks=(0,), **FAST))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ks": (5, 0)},
        {"ks": (0, 0)},
        {"ks": ()},
        {"threshold": 1.5},
        {"n_unlabeled": -1},
        {"trial_seconds": 0.0},
    ],
)
def test_config_validation(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        ProtocolConfig(**kwargs)  # type: ignore[arg-type]


def test_unlabeled_input_rejected(epochs: EpochSet) -> None:
    with pytest.raises(ValueError, match="labeled"):
        run_protocol(epochs.without_labels(), LogVarianceDecoder, ProtocolConfig(**FAST))
