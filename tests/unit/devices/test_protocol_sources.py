from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from neurolayer.core.labels import CAP1_LABELS
from neurolayer.data.synthetic import SyntheticMIConfig, generate_synthetic_recordings
from neurolayer.devices.protocol import CueSchedule
from neurolayer.devices.sources import ReplaySource, StreamSource, simulated_source


def test_schedule_is_balanced_timed_and_seeded() -> None:
    schedule = CueSchedule(n_trials_per_class=5, rest_s=1.0, imagery_s=3.0, seed=4)
    cues = schedule.cues()
    assert Counter(c.label for c in cues) == dict.fromkeys(CAP1_LABELS, 5)
    assert [c.onset_s for c in cues[:3]] == [1.0, 5.0, 9.0]
    assert schedule.duration_s == 40.0
    assert [c.label for c in cues] == [c.label for c in CueSchedule(5, seed=4).cues()]
    assert [c.label for c in cues] != [c.label for c in CueSchedule(5, seed=5).cues()]


def test_replay_source_streams_microvolts_in_chunks() -> None:
    (recording,) = generate_synthetic_recordings(
        SyntheticMIConfig(n_subjects=1, n_trials_per_class=2)
    )
    source = ReplaySource(recording, chunk=100)
    assert isinstance(source, StreamSource)
    source.start()
    chunks = []
    while not source.exhausted:
        chunks.append(source.read())
    assert all(c.shape[1] <= 100 for c in chunks)
    np.testing.assert_allclose(np.concatenate(chunks, axis=1) * 1e-6, recording.data)
    assert source.read().shape == (recording.n_channels, 0)
    source.rewind()
    assert source.read().shape[1] == 100


def test_simulated_source_follows_the_schedule() -> None:
    schedule = CueSchedule(n_trials_per_class=3, rest_s=1.0, imagery_s=2.0, seed=1)
    source = simulated_source(schedule, live_trials=4, realtime=False)
    recording_events = source.recording.events
    assert [e.label for e in recording_events[:6]] == [c.label for c in schedule.cues()]
    assert len(recording_events) == 10
    assert recording_events[0].onset_sample == round(1.0 * source.sfreq)


def test_brainflow_synthetic_board_streams() -> None:
    pytest.importorskip("brainflow")
    from neurolayer.devices.sources import BrainFlowSource

    source = BrainFlowSource("synthetic")
    assert source.sfreq > 0
    assert "Cz" in source.ch_names
    source.start()
    try:
        import time

        time.sleep(0.3)
        data = source.read()
    finally:
        source.stop()
    assert data.shape[0] == len(source.ch_names)
    assert data.shape[1] > 0


def test_brainflow_rejects_wrong_channel_count() -> None:
    pytest.importorskip("brainflow")
    from neurolayer.devices.sources import BrainFlowSource

    with pytest.raises(ValueError, match="channel names"):
        BrainFlowSource("synthetic", channel_names=["C3", "C4"])
