"""Conversion between :class:`~neurolayer.core.types.Recording` and MNE ``Raw`` objects.

MNE objects never cross a stage boundary (ADR-0003); this module is the only place
where they are converted. MNE is imported lazily (``neuro`` extra).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from neurolayer.core.channels import normalize_channel_names
from neurolayer.core.types import Event, Recording


@dataclass(frozen=True, slots=True)
class ConversionReport:
    """What a conversion kept and dropped (feeds the dataset QA report)."""

    dropped_channels: tuple[str, ...] = ()
    label_counts: dict[str, int] = field(default_factory=dict)
    dropped_labels: dict[str, int] = field(default_factory=dict)


def to_mne_raw(recording: Recording) -> Any:
    """Return an ``mne.io.RawArray`` (EEG channels, volts) with events as annotations."""
    import mne

    info = mne.create_info(list(recording.ch_names), recording.sfreq, ch_types="eeg")
    raw = mne.io.RawArray(np.array(recording.data), info, verbose="error")
    if recording.events:
        raw.set_annotations(
            mne.Annotations(
                onset=[e.onset_sample / recording.sfreq for e in recording.events],
                duration=[e.duration_samples / recording.sfreq for e in recording.events],
                description=[e.label for e in recording.events],
            )
        )
    return raw


def from_mne_raw(
    raw: Any,
    *,
    dataset_id: str,
    subject_id: str,
    session_id: str = "0",
    run_id: str = "0",
    event_codes: Mapping[str, int] | None = None,
    label_map: Mapping[str, str] | None = None,
    cue_offset_s: float = 0.0,
    device: str | None = None,
) -> tuple[Recording, ConversionReport]:
    """Convert an MNE ``Raw`` into a canonical :class:`Recording`.

    Parameters
    ----------
    raw
        Any ``mne.io.BaseRaw``. Only EEG channels are kept; others are reported as
        dropped. MNE stores EEG in volts, so no unit conversion is needed.
    event_codes
        Native label name → integer code, used when events live in a stim channel
        (MOABB convention). Without a stim channel, annotations are used and their
        descriptions are the native label names.
    label_map
        Native label name → canonical label. Labels absent from the map are dropped and
        counted. ``None`` keeps native names unchanged.
    cue_offset_s
        Seconds from the native event to the cue. Events are shifted so every
        canonical event marks the **cue onset** (MOABB's ``interval[0]``).
    """
    import mne

    eeg_picks = mne.pick_types(raw.info, eeg=True, stim=False, exclude=[])
    all_names = list(raw.ch_names)
    kept = [all_names[i] for i in eeg_picks]
    dropped = tuple(n for n in all_names if n not in set(kept))
    data = np.asarray(raw.get_data(picks=eeg_picks), dtype=np.float64)
    sfreq = float(raw.info["sfreq"])

    native: list[tuple[int, str]] = []
    stim_picks = mne.pick_types(raw.info, eeg=False, stim=True)
    if len(stim_picks) and event_codes:
        code_to_name = {code: name for name, code in event_codes.items()}
        found = mne.find_events(raw, shortest_event=0, verbose="error")
        native = [
            (int(s) - raw.first_samp, code_to_name[int(c)])
            for s, _, c in found
            if int(c) in code_to_name
        ]
    elif len(raw.annotations):
        # events_from_annotations accounts for first_samp and meas_date offsets.
        found, description_to_code = mne.events_from_annotations(raw, verbose="error")
        code_to_name = {code: name for name, code in description_to_code.items()}
        native = [(int(s) - raw.first_samp, code_to_name[int(c)]) for s, _, c in found]

    offset = round(cue_offset_s * sfreq)
    events: list[Event] = []
    kept_counts: Counter[str] = Counter()
    dropped_counts: Counter[str] = Counter()
    for onset, name in native:
        label = name if label_map is None else label_map.get(name)
        if label is None:
            dropped_counts[name] += 1
            continue
        onset_sample = onset + offset
        if 0 <= onset_sample < data.shape[1]:
            events.append(Event(onset_sample=onset_sample, label=label))
            kept_counts[label] += 1
        else:
            dropped_counts[f"{name} (outside recording)"] += 1

    recording = Recording(
        data=data,
        sfreq=sfreq,
        ch_names=normalize_channel_names(kept),
        dataset_id=dataset_id,
        subject_id=subject_id,
        session_id=session_id,
        run_id=run_id,
        events=tuple(sorted(events, key=lambda e: e.onset_sample)),
        device=device,
    )
    return recording, ConversionReport(
        dropped_channels=dropped,
        label_counts=dict(kept_counts),
        dropped_labels=dict(dropped_counts),
    )
