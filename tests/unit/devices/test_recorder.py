from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from neurolayer.data.consent import ConsentPurpose, ConsentRegistry
from neurolayer.data.storage import verify_checksums
from neurolayer.devices.protocol import Cue, CueSchedule
from neurolayer.devices.recorder import cue_windows, record_session, save_pilot_recording
from neurolayer.devices.sources import simulated_source

SCHEDULE = CueSchedule(n_trials_per_class=3, rest_s=1.0, imagery_s=2.5, seed=2)


def _record(chunk: int = 8) -> tuple[object, object]:
    source = simulated_source(SCHEDULE, live_trials=1, realtime=False, chunk=chunk)
    shown: list[Cue] = []
    recording = record_session(
        source, SCHEDULE, participant="P001", realtime=False, on_cue=shown.append
    )
    return recording, (source, shown)


def test_records_schedule_in_volts_with_cue_events() -> None:
    recording, (source, shown) = _record()
    assert recording.n_samples == round((SCHEDULE.duration_s + 1.0) * source.sfreq)
    assert [c.label for c in shown] == [c.label for c in SCHEDULE.cues()]
    assert [e.label for e in recording.events] == [c.label for c in SCHEDULE.cues()]
    truth = source.recording.events
    for recorded, true in zip(recording.events, truth, strict=False):
        assert 0 <= recorded.onset_sample - true.onset_sample < 8  # at most one chunk late
    np.testing.assert_allclose(recording.data, source.recording.data[:, : recording.n_samples])
    assert recording.subject_id == "P001"
    assert recording.dataset_id == "neurolayer_pilot"


def test_cue_windows_are_microvolts_and_in_bounds() -> None:
    recording, _ = _record()
    windows, labels = cue_windows(recording, offset_s=0.5, length_s=2.0)
    assert len(windows) == len(labels) == 6
    assert windows[0].shape == (recording.n_channels, round(2.0 * recording.sfreq))
    start = recording.events[0].onset_sample + round(0.5 * recording.sfreq)
    np.testing.assert_allclose(windows[0] * 1e-6, recording.data[:, start : start + 256])
    too_long, _ = cue_windows(recording, offset_s=2.0, length_s=2.0)
    assert len(too_long) == 5  # the last cue's window would run past the end (2.5 + 1 s left)


def test_source_that_ends_early_is_an_error() -> None:
    short = CueSchedule(n_trials_per_class=1, rest_s=1.0, imagery_s=1.0)
    source = simulated_source(short, live_trials=0, realtime=False)
    long = CueSchedule(n_trials_per_class=5, rest_s=1.0, imagery_s=1.0)
    with pytest.raises(RuntimeError, match="source ended"):
        record_session(source, long, participant="P001", realtime=False)


def test_saving_requires_research_consent(tmp_path: Path) -> None:
    recording, _ = _record()
    ledger = ConsentRegistry(tmp_path / "consent.jsonl")
    with pytest.raises(PermissionError, match="research"):
        save_pilot_recording(recording, SCHEDULE, tmp_path / "bids", ledger)
    assert not (tmp_path / "bids").exists()


def test_saves_bids_metadata_and_checksums(tmp_path: Path) -> None:
    pytest.importorskip("mne_bids")
    recording, _ = _record()
    form = tmp_path / "form.md"
    form.write_text("consent v0.1", encoding="utf-8")
    ledger = ConsentRegistry(tmp_path / "consent.jsonl")
    ledger.grant("P001", [ConsentPurpose.RESEARCH], form, "v0.1", "tester")
    root = tmp_path / "bids"
    written = save_pilot_recording(recording, SCHEDULE, root, ledger)
    assert any(p.suffix == ".edf" for p in written)
    meta = json.loads(written[-1].read_text(encoding="utf-8"))
    assert meta["participant"] == "P001"
    assert meta["schedule"]["seed"] == 2
    assert meta["n_cues"] == 6
    assert len(meta["consent_document_sha256"]) == 64
    assert verify_checksums(root) == []
