from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from neurolayer.data.synthetic import SyntheticMIConfig, generate_synthetic_recordings
from neurolayer.signal.epoching import EpochingSpec, epoch_recordings
from neurolayer.signal.pipeline import (
    PipelineSpec,
    TransformConfig,
    load_epochs_file,
    process_subject,
    save_epochs,
)


def _subject_recordings(n_sessions: int = 2, sfreq: float = 128.0):  # type: ignore[no-untyped-def]
    cfg = SyntheticMIConfig(
        n_subjects=1, n_trials_per_class=6, n_sessions=n_sessions, sfreq=sfreq, trial_seconds=3.0
    )
    return generate_synthetic_recordings(cfg, rest_seconds=1.0)


def test_epochs_are_exact_windows_after_the_cue() -> None:
    recordings = _subject_recordings()
    epochs, report = epoch_recordings(recordings, EpochingSpec(tmin=0.5, tmax=1.5))
    assert len(epochs) == 24
    assert epochs.n_times == 128
    first_event = recordings[0].events[0]
    start = first_event.onset_sample + 64
    np.testing.assert_array_equal(epochs.X[0], recordings[0].data[:, start : start + 128])
    assert epochs.order.tolist() == list(range(24))  # chronological across sessions
    assert epochs.session.tolist()[:12] == ["ses-00"] * 12
    assert report.kept == {"left_hand": 12, "right_hand": 12}


def test_label_filter_bounds_baseline_and_rejection() -> None:
    recordings = _subject_recordings(n_sessions=1)
    only_left, report = epoch_recordings(recordings, EpochingSpec(labels=("left_hand",)))
    assert len(only_left) == 6
    assert report.dropped_label == {"right_hand": 6}
    # A window longer than the trial runs past the recording end for the last trial.
    _, bounded = epoch_recordings(recordings, EpochingSpec(tmin=0.0, tmax=3.5))
    assert bounded.dropped_out_of_bounds >= 1
    baselined, _ = epoch_recordings(recordings, EpochingSpec(baseline=(0.5, 2.5)))
    np.testing.assert_allclose(baselined.X.mean(axis=2), 0.0, atol=1e-18)
    with pytest.raises(ValueError, match="no epochs kept"):
        epoch_recordings(recordings, EpochingSpec(reject_peak_to_peak_v=1e-9))
    _, flat = epoch_recordings(recordings, EpochingSpec(reject_flat_v=1e-12))
    assert flat.dropped_flat == 0


def test_epoching_spec_validation() -> None:
    with pytest.raises(ValidationError, match="tmax"):
        EpochingSpec(tmin=2.0, tmax=1.0)
    two_subjects = generate_synthetic_recordings(
        SyntheticMIConfig(n_subjects=2, n_trials_per_class=2)
    )
    with pytest.raises(ValueError, match="single subject"):
        epoch_recordings(two_subjects, EpochingSpec())


def test_pipeline_hash_is_stable_and_sensitive() -> None:
    default = PipelineSpec()
    assert default.pipeline_hash() == PipelineSpec().pipeline_hash()
    changed = PipelineSpec(transforms=(TransformConfig(name="bandpass", params={"l_freq": 8.0}),))
    assert changed.pipeline_hash() != default.pipeline_hash()
    window = PipelineSpec(epoching=EpochingSpec(tmin=1.0, tmax=3.0))
    assert window.pipeline_hash() != default.pipeline_hash()
    assert len(default.pipeline_hash()) == 16


def test_process_subject_and_cache_round_trip(tmp_path: Path) -> None:
    recordings = _subject_recordings(sfreq=256.0)
    epochs, report = process_subject(recordings, PipelineSpec())
    assert epochs.sfreq == 128.0  # default pipeline resamples
    assert epochs.n_times == 256  # 0.5-2.5 s at 128 Hz
    assert sum(report.kept.values()) == 24
    path = tmp_path / "sub-000_epochs"
    save_epochs(epochs, path, {"pipeline_hash": "x"})
    back = load_epochs_file(path)
    np.testing.assert_array_equal(back.X, epochs.X)
    assert back.ch_names == epochs.ch_names
    assert back.subject.tolist() == epochs.subject.tolist()
    assert back.label_names == epochs.label_names
