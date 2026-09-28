from __future__ import annotations

import numpy as np
import pytest

from neurolayer.data.mne_bridge import from_mne_raw, to_mne_raw
from neurolayer.data.synthetic import SyntheticMIConfig, generate_synthetic_recordings
from tests.fakes import make_moabb_raw

pytest.importorskip("mne")

LABELS = {"left_hand": "left_hand", "right_hand": "right_hand", "hands": "both_hands"}
CODES = {"left_hand": 2, "right_hand": 3, "hands": 4, "weird": 7}


def test_stim_channel_events_are_mapped_shifted_and_counted() -> None:
    raw = make_moabb_raw(n_trials=8, sfreq=160.0)
    recording, report = from_mne_raw(
        raw,
        dataset_id="ds",
        subject_id="sub-001",
        event_codes=CODES,
        label_map=LABELS,
        cue_offset_s=0.5,
    )
    assert recording.ch_names[:3] == ("FC3", "FCz", "FC4")  # "Fc3." normalized
    assert report.dropped_channels == ("EOG1", "STI")
    assert report.label_counts == {"left_hand": 2, "right_hand": 2, "both_hands": 2}
    assert report.dropped_labels == {"weird": 2}
    # Cue alignment: native marker at sample 10, cue 0.5 s (80 samples) later.
    assert recording.events[0].onset_sample == 10 + 80
    assert recording.data.dtype == np.float64


def test_annotation_events_round_trip() -> None:
    original = generate_synthetic_recordings(SyntheticMIConfig(n_subjects=1, n_trials_per_class=3))[
        0
    ]
    back, report = from_mne_raw(
        to_mne_raw(original), dataset_id=original.dataset_id, subject_id=original.subject_id
    )
    np.testing.assert_allclose(back.data, original.data)
    assert [(e.onset_sample, e.label) for e in back.events] == [
        (e.onset_sample, e.label) for e in original.events
    ]
    assert report.dropped_labels == {}


def test_events_outside_recording_are_dropped() -> None:
    raw = make_moabb_raw(n_trials=2, sfreq=160.0)
    recording, report = from_mne_raw(
        raw,
        dataset_id="ds",
        subject_id="s",
        event_codes=CODES,
        label_map=LABELS,
        cue_offset_s=1000.0,
    )
    assert recording.events == ()
    assert sum(report.dropped_labels.values()) == 2
