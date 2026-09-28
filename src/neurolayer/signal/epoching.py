"""Epoching and artifact rejection (WP-2.4, WP-2.5).

Events in canonical recordings mark **cue onset** (adapters align them), so one
epoching window works for every dataset: ``[tmin, tmax)`` seconds after the cue.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from neurolayer.core.labels import CAP1_LABELS
from neurolayer.core.types import EpochSet, FloatArray, Recording


class EpochingSpec(BaseModel):
    """Epoch window relative to cue onset, label filter and rejection thresholds."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tmin: float = Field(default=0.5, ge=0.0, description="Seconds after cue onset.")
    tmax: float = Field(default=2.5, gt=0.0, description="Seconds after cue onset (exclusive).")
    labels: tuple[str, ...] = CAP1_LABELS
    baseline: tuple[float, float] | None = Field(
        default=None, description="Window (s, relative to cue) whose mean is subtracted."
    )
    reject_peak_to_peak_v: float | None = Field(
        default=None, gt=0.0, description="Drop epochs with any channel exceeding this range."
    )
    reject_flat_v: float | None = Field(
        default=None, gt=0.0, description="Drop epochs with any channel below this range."
    )

    @model_validator(mode="after")
    def _window(self) -> EpochingSpec:
        if self.tmax <= self.tmin:
            raise ValueError("tmax must exceed tmin")
        if not self.labels:
            raise ValueError("labels must be non-empty")
        return self


@dataclass(slots=True)
class EpochingReport:
    """Counts of kept and dropped epochs, for QA and run manifests."""

    kept: Counter[str] = field(default_factory=Counter)
    dropped_label: Counter[str] = field(default_factory=Counter)
    dropped_out_of_bounds: int = 0
    dropped_peak_to_peak: int = 0
    dropped_flat: int = 0

    def as_dict(self) -> dict[str, object]:
        """JSON-serializable counts."""
        return {
            "kept": dict(self.kept),
            "dropped_label": dict(self.dropped_label),
            "dropped_out_of_bounds": self.dropped_out_of_bounds,
            "dropped_peak_to_peak": self.dropped_peak_to_peak,
            "dropped_flat": self.dropped_flat,
        }


def epoch_recordings(
    recordings: Sequence[Recording], spec: EpochingSpec
) -> tuple[EpochSet, EpochingReport]:
    """Cut labeled epochs from recordings of **one subject** of one dataset.

    Recordings must be given in chronological order (adapters yield them that way);
    ``order`` numbers epochs consecutively across runs and sessions.

    Raises
    ------
    ValueError
        If recordings mix subjects, datasets, channel sets or sampling rates, or if no
        epoch survives.
    """
    if not recordings:
        raise ValueError("no recordings to epoch")
    first = recordings[0]
    for rec in recordings[1:]:
        if (rec.dataset_id, rec.subject_id) != (first.dataset_id, first.subject_id):
            raise ValueError("epoch_recordings expects recordings of a single subject")
        if rec.ch_names != first.ch_names or rec.sfreq != first.sfreq:
            raise ValueError("recordings of one subject must share channels and sampling rate")

    label_index = {label: i for i, label in enumerate(spec.labels)}
    start = round(spec.tmin * first.sfreq)
    length = round((spec.tmax - spec.tmin) * first.sfreq)
    report = EpochingReport()
    windows: list[FloatArray] = []
    labels: list[int] = []
    sessions: list[str] = []
    for rec in recordings:
        for event in rec.events:
            if event.label not in label_index:
                report.dropped_label[event.label] += 1
                continue
            begin = event.onset_sample + start
            if begin < 0 or begin + length > rec.n_samples:
                report.dropped_out_of_bounds += 1
                continue
            window = rec.data[:, begin : begin + length]
            if spec.baseline is not None:
                b0 = event.onset_sample + round(spec.baseline[0] * rec.sfreq)
                b1 = event.onset_sample + round(spec.baseline[1] * rec.sfreq)
                if b0 < 0 or b1 > rec.n_samples or b1 <= b0:
                    report.dropped_out_of_bounds += 1
                    continue
                window = window - rec.data[:, b0:b1].mean(axis=1, keepdims=True)
            ptp = np.ptp(window, axis=1)
            if spec.reject_peak_to_peak_v is not None and ptp.max() > spec.reject_peak_to_peak_v:
                report.dropped_peak_to_peak += 1
                continue
            if spec.reject_flat_v is not None and ptp.min() < spec.reject_flat_v:
                report.dropped_flat += 1
                continue
            windows.append(np.array(window))
            labels.append(label_index[event.label])
            sessions.append(rec.session_id)
            report.kept[event.label] += 1
    if not windows:
        raise ValueError(f"no epochs kept for {first.dataset_id}/{first.subject_id}")
    epochs = EpochSet.from_arrays(
        np.stack(windows),
        labels,
        label_names=spec.labels,
        ch_names=first.ch_names,
        sfreq=first.sfreq,
        subject=first.subject_id,
        session=sessions,
        dataset=first.dataset_id,
    )
    return epochs, report
