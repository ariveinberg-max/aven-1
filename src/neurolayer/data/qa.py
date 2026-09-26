"""Per-recording data quality report (WP-1.5).

Flags problems that silently corrupt results if unnoticed:

* flat channels (disconnected electrodes);
* noisy channels (robust outliers in log-variance across channels);
* strong power-line noise (50 or 60 Hz);
* implausible amplitudes (for example microvolts mislabeled as volts);
* labels and channels dropped during ingestion.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from scipy.signal import welch

from neurolayer.core.types import Recording
from neurolayer.data.mne_bridge import ConversionReport

FLAT_STD_V = 1e-7
"""Channels with a standard deviation below 0.1 µV are considered flat."""

NOISY_Z = 5.0
"""Robust z-score of per-channel log-variance above which a channel is flagged noisy."""

LINE_NOISE_RATIO = 10.0
"""Line-frequency power this many times above its neighborhood is flagged."""

PLAUSIBLE_MEDIAN_ABS_V = (1e-8, 1e-3)
"""Median absolute EEG amplitude outside this range suggests a unit error."""


@dataclass(frozen=True, slots=True)
class RecordingQA:
    """Quality metrics of one recording."""

    dataset_id: str
    subject_id: str
    session_id: str
    run_id: str
    duration_s: float
    sfreq: float
    n_channels: int
    flat_channels: tuple[str, ...]
    noisy_channels: tuple[str, ...]
    line_noise_ratio: dict[str, float | None]
    median_abs_amplitude_v: float
    event_counts: dict[str, int]
    dropped_labels: dict[str, int] = field(default_factory=dict)
    dropped_channels: tuple[str, ...] = ()

    @property
    def issues(self) -> list[str]:
        """Human-readable list of problems (empty when the recording looks clean)."""
        problems = []
        if self.flat_channels:
            problems.append(f"flat: {', '.join(self.flat_channels)}")
        if self.noisy_channels:
            problems.append(f"noisy: {', '.join(self.noisy_channels)}")
        for freq, ratio in self.line_noise_ratio.items():
            if ratio is not None and ratio > LINE_NOISE_RATIO:
                problems.append(f"line noise at {freq} Hz ({ratio:.0f}x)")
        low, high = PLAUSIBLE_MEDIAN_ABS_V
        if not low <= self.median_abs_amplitude_v <= high:
            problems.append(f"implausible amplitude ({self.median_abs_amplitude_v:.2e} V)")
        if not self.event_counts:
            problems.append("no events")
        return problems

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable form."""
        return asdict(self) | {"issues": self.issues}


def _line_ratio(data: np.ndarray[Any, Any], sfreq: float, line_hz: float) -> float | None:
    if line_hz + 6.0 >= sfreq / 2.0:
        return None
    nperseg = int(min(data.shape[1], 4 * sfreq))
    freqs, power = welch(data, fs=sfreq, nperseg=nperseg, axis=1)
    spectrum = np.median(power, axis=0)
    at_line = spectrum[np.abs(freqs - line_hz) <= 0.5].max()
    neighborhood = (np.abs(freqs - line_hz) >= 2.0) & (np.abs(freqs - line_hz) <= 6.0)
    baseline = float(np.median(spectrum[neighborhood]))
    return float(at_line / baseline) if baseline > 0 else None


def qa_recording(recording: Recording, report: ConversionReport | None = None) -> RecordingQA:
    """Compute quality metrics for one recording."""
    data = recording.data
    std = data.std(axis=1)
    flat = tuple(ch for ch, s in zip(recording.ch_names, std, strict=True) if s < FLAT_STD_V)
    noisy: tuple[str, ...] = ()
    live = std >= FLAT_STD_V
    if live.sum() >= 4:
        log_var = np.log(std[live] ** 2)
        median = np.median(log_var)
        mad = np.median(np.abs(log_var - median)) * 1.4826
        if mad > 0:
            z = (log_var - median) / mad
            names = [ch for ch, keep in zip(recording.ch_names, live, strict=True) if keep]
            noisy = tuple(ch for ch, score in zip(names, z, strict=True) if score > NOISY_Z)
    return RecordingQA(
        dataset_id=recording.dataset_id,
        subject_id=recording.subject_id,
        session_id=recording.session_id,
        run_id=recording.run_id,
        duration_s=recording.duration_s,
        sfreq=recording.sfreq,
        n_channels=recording.n_channels,
        flat_channels=flat,
        noisy_channels=noisy,
        line_noise_ratio={
            "50": _line_ratio(data, recording.sfreq, 50.0),
            "60": _line_ratio(data, recording.sfreq, 60.0),
        },
        median_abs_amplitude_v=float(np.median(np.abs(data))),
        event_counts=dict(Counter(e.label for e in recording.events)),
        dropped_labels=dict(report.dropped_labels) if report else {},
        dropped_channels=report.dropped_channels if report else (),
    )


def qa_markdown(results: Sequence[RecordingQA]) -> str:
    """Summarize QA results as markdown (one row per recording)."""
    lines = [
        "| dataset | subject | session | run | duration (s) | channels | events | issues |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for qa in results:
        events = ", ".join(f"{k}:{v}" for k, v in sorted(qa.event_counts.items()))
        lines.append(
            f"| {qa.dataset_id} | {qa.subject_id} | {qa.session_id} | {qa.run_id} | "
            f"{qa.duration_s:.1f} | {qa.n_channels} | {events} | "
            f"{'; '.join(qa.issues) or 'ok'} |"
        )
    flagged = sum(1 for qa in results if qa.issues)
    lines += ["", f"{flagged} of {len(results)} recordings flagged."]
    return "\n".join(lines) + "\n"
