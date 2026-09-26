"""Pure, versioned signal transforms: ``Recording -> Recording`` (WP-2.1-2.3).

Every transform:

* never mutates its input (recording arrays are read-only anyway);
* declares ``name`` and ``version``; bump ``version`` whenever outputs could change,
  which changes the pipeline hash and therefore the processed-data cache key;
* exposes ``config()`` (JSON-serializable) so pipelines are reproducible from YAML.

Filters are zero-phase (forward-backward), so they do not shift event timing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from fractions import Fraction
from typing import Any, ClassVar, Literal, Protocol, runtime_checkable

import numpy as np
from scipy import signal as sps

from neurolayer.core.channels import CONSUMER_MONTAGES, channel_indices
from neurolayer.core.types import Event, FloatArray, Recording


@runtime_checkable
class Transform(Protocol):
    """A pure, versioned recording transform."""

    name: ClassVar[str]
    version: ClassVar[int]

    def config(self) -> dict[str, Any]:
        """JSON-serializable parameters."""
        ...

    def __call__(self, recording: Recording) -> Recording:
        """Return a new, transformed recording."""
        ...


def _with_data(
    recording: Recording,
    data: FloatArray,
    *,
    sfreq: float | None = None,
    ch_names: tuple[str, ...] | None = None,
    events: tuple[Event, ...] | None = None,
) -> Recording:
    return Recording(
        data=np.ascontiguousarray(data, dtype=np.float64),
        sfreq=recording.sfreq if sfreq is None else sfreq,
        ch_names=recording.ch_names if ch_names is None else ch_names,
        dataset_id=recording.dataset_id,
        subject_id=recording.subject_id,
        session_id=recording.session_id,
        run_id=recording.run_id,
        events=recording.events if events is None else events,
        device=recording.device,
    )


class _Base:
    """Dataclass-based config and ``name``/``version`` plumbing."""

    name: ClassVar[str]
    version: ClassVar[int]

    def config(self) -> dict[str, Any]:
        params: dict[str, Any] = asdict(self)  # type: ignore[call-overload]
        return {k: list(v) if isinstance(v, tuple) else v for k, v in params.items()}


@dataclass(frozen=True, slots=True)
class Bandpass(_Base):
    """Zero-phase Butterworth band-pass (or high-/low-pass when one edge is ``None``)."""

    name: ClassVar[str] = "bandpass"
    version: ClassVar[int] = 1
    l_freq: float | None = 1.0
    h_freq: float | None = 40.0
    order: int = 4

    def __post_init__(self) -> None:
        if self.l_freq is None and self.h_freq is None:
            raise ValueError("bandpass needs l_freq and/or h_freq")
        if self.l_freq is not None and self.h_freq is not None and self.l_freq >= self.h_freq:
            raise ValueError("l_freq must be below h_freq")

    def __call__(self, recording: Recording) -> Recording:
        """Return the transformed recording (the input is not modified)."""
        nyquist = recording.sfreq / 2.0
        if self.h_freq is not None and self.h_freq >= nyquist:
            raise ValueError(f"h_freq {self.h_freq} Hz must be below Nyquist ({nyquist} Hz)")
        if self.l_freq is not None and self.h_freq is not None:
            sos = sps.butter(
                self.order, [self.l_freq, self.h_freq], "bandpass", fs=recording.sfreq, output="sos"
            )
        elif self.l_freq is not None:
            sos = sps.butter(self.order, self.l_freq, "highpass", fs=recording.sfreq, output="sos")
        else:
            sos = sps.butter(self.order, self.h_freq, "lowpass", fs=recording.sfreq, output="sos")
        return _with_data(recording, sps.sosfiltfilt(sos, recording.data, axis=1))


@dataclass(frozen=True, slots=True)
class Notch(_Base):
    """Zero-phase IIR notch at each frequency below Nyquist (power-line removal)."""

    name: ClassVar[str] = "notch"
    version: ClassVar[int] = 1
    freqs: tuple[float, ...] = (50.0,)
    quality: float = 30.0

    def __call__(self, recording: Recording) -> Recording:
        """Return the transformed recording (the input is not modified)."""
        data = recording.data
        for freq in self.freqs:
            if freq < recording.sfreq / 2.0:
                b, a = sps.iirnotch(freq, self.quality, fs=recording.sfreq)
                data = sps.filtfilt(b, a, data, axis=1)
        return _with_data(recording, np.asarray(data))


@dataclass(frozen=True, slots=True)
class Resample(_Base):
    """Polyphase resampling to ``sfreq``; event onsets are rescaled."""

    name: ClassVar[str] = "resample"
    version: ClassVar[int] = 1
    sfreq: float = 128.0

    def __call__(self, recording: Recording) -> Recording:
        """Return the transformed recording (the input is not modified)."""
        if recording.sfreq == self.sfreq:
            return recording
        ratio = Fraction(self.sfreq / recording.sfreq).limit_denominator(1000)
        data = sps.resample_poly(recording.data, ratio.numerator, ratio.denominator, axis=1)
        scale = self.sfreq / recording.sfreq
        n_samples = data.shape[1]
        events = tuple(
            Event(onset_sample=min(round(e.onset_sample * scale), n_samples - 1), label=e.label,
                  duration_samples=round(e.duration_samples * scale))
            for e in recording.events
        )  # fmt: skip
        return _with_data(recording, data, sfreq=self.sfreq, events=events)


@dataclass(frozen=True, slots=True)
class Rereference(_Base):
    """Re-reference to the common average or to the mean of given channels."""

    name: ClassVar[str] = "rereference"
    version: ClassVar[int] = 1
    kind: Literal["average", "channels"] = "average"
    channels: tuple[str, ...] = ()

    def __call__(self, recording: Recording) -> Recording:
        """Return the transformed recording (the input is not modified)."""
        if self.kind == "average":
            reference = recording.data.mean(axis=0, keepdims=True)
        else:
            if not self.channels:
                raise ValueError("kind='channels' needs at least one reference channel")
            idx = channel_indices(recording.ch_names, self.channels, "rereference")
            reference = recording.data[idx].mean(axis=0, keepdims=True)
        return _with_data(recording, recording.data - reference)


@dataclass(frozen=True, slots=True)
class SelectChannels(_Base):
    """Keep exactly ``channels`` (or a named consumer ``montage``), in that order.

    Missing channels raise :class:`~neurolayer.core.channels.MissingChannelsError`;
    nothing is interpolated or substituted silently (WP-2.3).
    """

    name: ClassVar[str] = "select_channels"
    version: ClassVar[int] = 1
    channels: tuple[str, ...] = ()
    montage: str | None = None

    def __post_init__(self) -> None:
        if bool(self.channels) == (self.montage is not None):
            raise ValueError("give exactly one of channels or montage")
        if self.montage is not None and self.montage not in CONSUMER_MONTAGES:
            raise ValueError(f"unknown montage {self.montage!r}")

    def target(self) -> tuple[str, ...]:
        """Return the channel list this transform selects."""
        return CONSUMER_MONTAGES[self.montage].channels if self.montage else self.channels

    def __call__(self, recording: Recording) -> Recording:
        """Return the transformed recording (the input is not modified)."""
        names = self.target()
        idx = channel_indices(
            recording.ch_names, names, f"{recording.dataset_id}/{recording.subject_id}"
        )
        return _with_data(recording, recording.data[idx], ch_names=names)


@dataclass(frozen=True, slots=True)
class UnitCheck(_Base):
    """Fail fast when amplitudes are implausible for EEG in volts (e.g. µV stored as V)."""

    name: ClassVar[str] = "unit_check"
    version: ClassVar[int] = 1
    min_median_abs_v: float = 1e-8
    max_median_abs_v: float = 1e-3

    def __call__(self, recording: Recording) -> Recording:
        """Return the transformed recording (the input is not modified)."""
        median = float(np.median(np.abs(recording.data)))
        if not self.min_median_abs_v <= median <= self.max_median_abs_v:
            raise ValueError(
                f"{recording.dataset_id}/{recording.subject_id}: median |x| = {median:.2e} V is "
                "implausible for EEG in volts; check the adapter's unit conversion"
            )
        return recording


TRANSFORMS: dict[str, type[_Base]] = {
    cls.name: cls for cls in (Bandpass, Notch, Resample, Rereference, SelectChannels, UnitCheck)
}
"""Registry used by pipeline specs in YAML (``{name: ..., params: {...}}``)."""


@dataclass(frozen=True, slots=True)
class TransformSpec:
    """A transform reference as written in experiment configs."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)

    def build(self) -> Transform:
        """Instantiate the registered transform."""
        if self.name not in TRANSFORMS:
            raise KeyError(f"unknown transform {self.name!r}; known: {sorted(TRANSFORMS)}")
        params = {k: tuple(v) if isinstance(v, list) else v for k, v in self.params.items()}
        transform: Transform = TRANSFORMS[self.name](**params)  # type: ignore[assignment]
        return transform
