from __future__ import annotations

import numpy as np
import pytest

from neurolayer.core.channels import MissingChannelsError
from neurolayer.core.types import UNLABELED, EpochSet, Event, Recording


def _epochs(n: int = 6, labels: list[int] | None = None) -> EpochSet:
    rng = np.random.default_rng(0)
    return EpochSet.from_arrays(
        rng.standard_normal((n, 3, 16)),
        labels if labels is not None else [0, 1] * (n // 2),
        label_names=("left_hand", "right_hand"),
        ch_names=("C3", "Cz", "C4"),
        sfreq=128.0,
        subject="sub-001",
        dataset="ds",
    )


class TestRecording:
    def test_valid_recording_is_read_only(self) -> None:
        data = np.zeros((2, 100))
        rec = Recording(
            data=data,
            sfreq=100.0,
            ch_names=("C3", "C4"),
            dataset_id="d",
            subject_id="s",
            events=(Event(10, "left_hand"),),
        )
        assert rec.duration_s == pytest.approx(1.0)
        assert rec.n_channels == 2
        with pytest.raises(ValueError, match="read-only"):
            rec.data[0, 0] = 1.0

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"data": np.zeros(10)}, "2-D"),
            ({"ch_names": ("C3",)}, "channel names"),
            ({"sfreq": 0.0}, "sfreq"),
            ({"data": np.array([[np.nan, 0.0], [0.0, 0.0]])}, "NaN"),
            ({"ch_names": ("C3", "C3")}, "duplicate"),
            ({"events": (Event(1, "x", duration_samples=5),)}, "beyond"),
            ({"subject_id": ""}, "subject_id"),
        ],
    )
    def test_invalid_recording(self, kwargs: dict[str, object], message: str) -> None:
        base: dict[str, object] = {
            "data": np.zeros((2, 2)),
            "sfreq": 100.0,
            "ch_names": ("C3", "C4"),
            "dataset_id": "d",
            "subject_id": "s",
        }
        with pytest.raises(ValueError, match=message):
            Recording(**(base | kwargs))  # type: ignore[arg-type]

    def test_rejects_float32(self) -> None:
        with pytest.raises(TypeError, match="float64"):
            Recording(
                data=np.zeros((1, 4), np.float32),
                sfreq=1.0,
                ch_names=("C3",),  # type: ignore[arg-type]
                dataset_id="d",
                subject_id="s",
            )

    def test_event_validation(self) -> None:
        with pytest.raises(ValueError, match="onset"):
            Event(-1, "x")
        with pytest.raises(ValueError, match="label"):
            Event(0, "")


class TestEpochSet:
    def test_invariants_and_read_only(self) -> None:
        ep = _epochs()
        assert len(ep) == 6
        assert ep.is_labeled
        assert ep.subject_keys() == [("ds", "sub-001")]
        with pytest.raises(ValueError, match="read-only"):
            ep.X[0, 0, 0] = 0.0

    def test_label_range_checked(self) -> None:
        with pytest.raises(ValueError, match="unlabeled"):
            _epochs(labels=[0, 1, 2, 0, 1, 0])

    def test_subset_and_mask(self) -> None:
        ep = _epochs()
        assert len(ep.subset([0, 2])) == 2
        assert len(ep.subset(ep.y == 1)) == 3
        with pytest.raises(ValueError, match="mask"):
            ep.subset(np.array([True, False]))

    def test_select_channels(self) -> None:
        ep = _epochs()
        sub = ep.select_channels(["C4", "C3"])
        assert sub.ch_names == ("C4", "C3")
        np.testing.assert_array_equal(sub.X[:, 0], ep.X[:, 2])
        with pytest.raises(MissingChannelsError):
            ep.select_channels(["Pz"])

    def test_without_labels(self) -> None:
        hidden = _epochs().without_labels()
        assert not hidden.is_labeled
        assert (hidden.y == UNLABELED).all()

    def test_concat_checks_compatibility(self) -> None:
        a, b = _epochs(), _epochs()
        assert len(EpochSet.concat([a, b])) == 12
        with pytest.raises(ValueError, match="ch_names"):
            EpochSet.concat([a, a.select_channels(["C3", "C4"])])
        with pytest.raises(ValueError, match="unlabeled"):
            EpochSet.concat([a, b.without_labels()])
        with pytest.raises(ValueError, match="empty"):
            EpochSet.concat([])

    def test_shape_mismatch_rejected(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            EpochSet.from_arrays(
                np.zeros((3, 1, 4)),
                [0, 1],
                label_names=("a", "b"),
                ch_names=("C3",),
                sfreq=1.0,
                subject="s",
                dataset="d",
            )
