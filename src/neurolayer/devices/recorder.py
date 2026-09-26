"""Record a cued session from a stream into a :class:`Recording` (WP-7.2, WP-8.3).

Cues are scheduled on the **sample clock**: a cue fires when the number of samples
received reaches its onset, and its event is stamped at exactly that sample. The cue is
therefore displayed slightly *after* the brain data it is aligned to, by the device's
transport latency (tens of ms for BrainFlow/LSL over Bluetooth). This is small relative
to the CAP-1 window (0.5-2.5 s after the cue) but must be measured per device with a
photodiode before Gate 3 (see ``docs/guides/pilot-protocol.md``).

Saving requires ``research`` consent from the participant's ledger (PRIV-2); nothing is
written for a participant who has not consented or has revoked.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from neurolayer.core.types import Event, FloatArray, Recording
from neurolayer.data.consent import ConsentPurpose, ConsentRegistry
from neurolayer.data.storage import write_bids, write_checksums
from neurolayer.devices.protocol import Cue, CueSchedule
from neurolayer.devices.sources import StreamSource

MICROVOLTS = 1e-6
PILOT_DATASET_ID = "neurolayer_pilot"


@dataclass(frozen=True, slots=True)
class SessionMeta:
    """Non-identifying metadata saved next to a pilot recording."""

    participant: str
    session_id: str
    device: str
    ch_names: tuple[str, ...]
    sfreq: float
    schedule: dict[str, object]
    consent_document_sha256: str
    n_cues: int


def record_session(
    source: StreamSource,
    schedule: CueSchedule,
    *,
    participant: str,
    session_id: str = "1",
    device: str | None = None,
    on_cue: Callable[[Cue], None] | None = None,
    poll_s: float = 0.02,
    realtime: bool = True,
    max_idle_s: float = 5.0,
    tail_s: float = 1.0,
) -> Recording:
    """Stream the schedule (plus ``tail_s``), firing cues on the sample clock.

    Parameters
    ----------
    source
        Any :class:`~neurolayer.devices.sources.StreamSource` (microvolts).
    schedule
        Cue timing and order.
    participant
        Pseudonym (``"P001"``); becomes ``subject_id``.
    on_cue
        Called once per cue, e.g. to display the arrow or print the instruction.
    poll_s
        Sleep between reads in real-time mode.
    realtime
        ``False`` reads as fast as the source allows (replay, tests).
    max_idle_s
        Abort if a real-time source delivers nothing for this long (device dropped).
    tail_s
        Extra seconds recorded after the last trial, so the last cue's window fits even
        when its event is stamped a few samples late.

    Returns
    -------
    Recording
        Continuous data in volts with one event per cue, ``dataset_id`` =
        ``neurolayer_pilot``.
    """
    cues = schedule.cues()
    total = round((schedule.duration_s + tail_s) * source.sfreq)
    chunks: list[FloatArray] = []
    received, next_cue = 0, 0
    events: list[Event] = []
    last_data = time.monotonic()
    source.start()
    try:
        while received < total:
            while next_cue < len(cues) and received >= round(cues[next_cue].onset_s * source.sfreq):
                cue = cues[next_cue]
                events.append(Event(onset_sample=received, label=cue.label))
                if on_cue is not None:
                    on_cue(cue)
                next_cue += 1
            chunk = source.read()
            if chunk.shape[1]:
                chunks.append(chunk)
                received += chunk.shape[1]
                last_data = time.monotonic()
            elif not realtime:
                raise RuntimeError(f"source ended after {received} of {total} samples")
            elif time.monotonic() - last_data > max_idle_s:
                raise RuntimeError(f"no data for {max_idle_s:g} s; is the device connected?")
            if realtime:
                time.sleep(poll_s)
    finally:
        source.stop()
    data = np.concatenate(chunks, axis=1)[:, :total] * MICROVOLTS
    return Recording(
        data=np.ascontiguousarray(data, dtype=np.float64),
        sfreq=source.sfreq,
        ch_names=source.ch_names,
        dataset_id=PILOT_DATASET_ID,
        subject_id=participant,
        session_id=session_id,
        events=tuple(e for e in events if e.onset_sample < total),
        device=device,
    )


def cue_windows(
    recording: Recording, offset_s: float, length_s: float
) -> tuple[list[FloatArray], list[str]]:
    """Cut one window per cue event, in **microvolts** (the API's input unit).

    Windows that would run past the end of the recording are skipped.
    """
    start = round(offset_s * recording.sfreq)
    length = round(length_s * recording.sfreq)
    windows, labels = [], []
    for event in recording.events:
        begin = event.onset_sample + start
        if begin + length <= recording.n_samples:
            windows.append(recording.data[:, begin : begin + length] / MICROVOLTS)
            labels.append(event.label)
    return windows, labels


def save_pilot_recording(
    recording: Recording,
    schedule: CueSchedule,
    root: Path,
    consent: ConsentRegistry,
) -> list[Path]:
    """Write BIDS-EEG + session metadata + checksums, after checking research consent.

    ``root`` should be on encrypted storage (C3 data). Returns the written data files.
    """
    consent.require(recording.subject_id, [ConsentPurpose.RESEARCH])
    grants = [
        e
        for e in consent.events()
        if e.participant == recording.subject_id
        and e.purpose == ConsentPurpose.RESEARCH
        and e.action == "grant"
    ]
    meta = SessionMeta(
        participant=recording.subject_id,
        session_id=recording.session_id,
        device=recording.device or "unknown",
        ch_names=recording.ch_names,
        sfreq=recording.sfreq,
        schedule=asdict(schedule),
        consent_document_sha256=grants[-1].document_sha256,
        n_cues=len(recording.events),
    )
    written = write_bids([recording], root)
    sidecar = root / "sourcedata" / "sessions" / f"{meta.participant}_{meta.session_id}.json"
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps(asdict(meta), indent=2) + "\n", encoding="utf-8")
    write_checksums(root)
    return [*written, sidecar]
