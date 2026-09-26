"""Synthetic motor-imagery EEG with realistic between-subject variability.

Used for tests, smoke runs and CI. It lets the whole pipeline and the CAP-1 harness run
end-to-end without downloading any data. It is **not** a substitute for real data and
is never used to support a capability claim.

Generative model (per subject)
------------------------------
* Two narrow-band "mu rhythm" sources over the left (near C3) and right (near C4)
  sensorimotor cortex, with a subject-specific peak frequency (9–13 Hz).
* Imagining a hand movement suppresses the *contralateral* source's power
  (event-related desynchronization, ERD). Its depth scales with a per-subject
  **efficiency** in [0, 1] drawn from a Beta prior. Efficiency 0 means the subject
  produces no class information at all, mimicking BCI inefficiency.
* 1/f background sources with a subject-specific random mixing, plus white sensor
  noise.
* Subject-specific perturbation of the motor source patterns (``subject_variability``)
  and a per-site perturbation shared by all subjects of a synthetic dataset
  (``site_shift``), mimicking lab and amplifier differences for regime R2.
* Small per-session channel gain drift.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from neurolayer.core.labels import CAP1_LABELS
from neurolayer.core.types import EpochSet, Event, FloatArray, IntArray, Recording

_VOLTS_SCALE = 1e-5
_DEFAULT_CHANNELS = ("FC3", "FCz", "FC4", "C3", "Cz", "C4", "CP3", "CPz", "CP4")
_SITE_RE = re.compile(r"^(?P<row>[A-Za-z]+?)(?P<pos>z|\d+)$")


@dataclass(frozen=True, slots=True)
class SyntheticMIConfig:
    """Parameters of the synthetic motor-imagery generator.

    Attributes
    ----------
    n_subjects
        Number of subjects.
    n_trials_per_class
        Trials per class *per session*.
    n_sessions
        Sessions per subject.
    sfreq
        Sampling rate in Hz.
    trial_seconds
        Trial length in seconds.
    channels
        Canonical channel names; lateral position is inferred from the 10-10 name.
    subject_variability
        Scale of per-subject perturbation of source patterns (0 = identical subjects).
    site_shift
        Scale of a perturbation shared by all subjects of this dataset (0 = none).
    efficiency_alpha, efficiency_beta
        Beta prior over per-subject ERD efficiency.
    efficiencies
        Optional explicit per-subject efficiencies (overrides the prior).
    erd_depth
        Relative amplitude drop of the contralateral source at efficiency 1.
    signal_to_noise
        Motor-source amplitude relative to the background sources.
    seed
        Master seed; each subject gets an independent, reproducible stream.
    dataset_id
        Dataset identifier written into the epochs.
    """

    n_subjects: int = 8
    n_trials_per_class: int = 60
    n_sessions: int = 1
    sfreq: float = 128.0
    trial_seconds: float = 2.0
    channels: tuple[str, ...] = _DEFAULT_CHANNELS
    subject_variability: float = 0.5
    site_shift: float = 0.0
    efficiency_alpha: float = 4.0
    efficiency_beta: float = 2.0
    efficiencies: tuple[float, ...] | None = None
    erd_depth: float = 0.6
    signal_to_noise: float = 1.0
    seed: int = 0
    dataset_id: str = "synthetic_mi"

    def __post_init__(self) -> None:
        if self.n_subjects < 1 or self.n_trials_per_class < 1 or self.n_sessions < 1:
            raise ValueError("n_subjects, n_trials_per_class and n_sessions must be >= 1")
        if self.efficiencies is not None:
            if len(self.efficiencies) != self.n_subjects:
                raise ValueError("efficiencies must have one entry per subject")
            if not all(0.0 <= e <= 1.0 for e in self.efficiencies):
                raise ValueError("efficiencies must lie in [0, 1]")
        if not 0.0 <= self.erd_depth < 1.0:
            raise ValueError("erd_depth must lie in [0, 1)")
        if self.sfreq < 40.0:
            raise ValueError("sfreq must be >= 40 Hz to represent mu rhythms")


@dataclass(frozen=True, slots=True)
class SyntheticMIResult:
    """Generated epochs plus the ground-truth efficiency of each subject."""

    epochs: EpochSet
    efficiency: dict[str, float]


def _site_coordinates(name: str) -> tuple[float, float]:
    """Return (lateral position in [-1, 1], sensorimotor weight) for a 10-10 name."""
    match = _SITE_RE.match(name)
    if match is None:
        return 0.0, 0.2
    row, pos = match.group("row").upper(), match.group("pos")
    if pos == "z":
        lateral = 0.0
    else:
        digit = int(pos)
        step = (digit + 1) // 2 if digit % 2 else digit // 2
        lateral = (-1.0 if digit % 2 else 1.0) * min(step * 0.25, 1.0)
    weight = {"C": 1.0, "FC": 0.7, "CP": 0.7}.get(row, 0.25)
    return lateral, weight


def _motor_patterns(channels: tuple[str, ...]) -> FloatArray:
    """Return base spatial patterns (n_channels, 2) of the left and right motor sources."""
    coords = np.array([_site_coordinates(c) for c in channels])
    lateral, weight = coords[:, 0], coords[:, 1]
    left = weight * np.exp(-(((lateral + 0.5) / 0.35) ** 2))
    right = weight * np.exp(-(((lateral - 0.5) / 0.35) ** 2))
    patterns: FloatArray = np.stack([left, right], axis=1) + 0.02
    return patterns


def _shaped_noise(
    rng: np.random.Generator, shape: tuple[int, ...], sfreq: float, band: tuple[float, float] | None
) -> FloatArray:
    """Unit-variance noise: band-limited if ``band`` is given, else 1/f (pink)."""
    n_times = shape[-1]
    spectrum = np.fft.rfft(rng.standard_normal(shape), axis=-1)
    freqs = np.fft.rfftfreq(n_times, d=1.0 / sfreq)
    if band is None:
        gain = np.zeros_like(freqs)
        gain[1:] = 1.0 / np.sqrt(freqs[1:])
    else:
        gain = ((freqs >= band[0]) & (freqs <= band[1])).astype(np.float64)
    signal: FloatArray = np.fft.irfft(spectrum * gain, n=n_times, axis=-1)
    std = signal.std(axis=-1, keepdims=True)
    return signal / np.maximum(std, 1e-12)


@dataclass(frozen=True, slots=True)
class _SubjectModel:
    patterns: FloatArray
    mixing: FloatArray
    band: tuple[float, float]
    efficiency: float


def _subject_model(
    cfg: SyntheticMIConfig, rng: np.random.Generator, site_patterns: FloatArray, efficiency: float
) -> _SubjectModel:
    shape = site_patterns.shape
    patterns = site_patterns * (1.0 + cfg.subject_variability * rng.normal(0.0, 0.3, shape))
    patterns += cfg.subject_variability * rng.normal(0.0, 0.1, shape)
    n_ch = shape[0]
    mixing = rng.normal(0.0, 1.0, (n_ch, n_ch)) / np.sqrt(n_ch)
    mu_freq = rng.uniform(9.0, 13.0)
    return _SubjectModel(patterns, mixing, (mu_freq - 1.5, mu_freq + 1.5), efficiency)


def _amplitudes(
    cfg: SyntheticMIConfig, rng: np.random.Generator, labels: IntArray, efficiency: float
) -> FloatArray:
    """Per-trial motor-source amplitudes; label -1 means rest (no desynchronization)."""
    amplitude = np.full((len(labels), 2), cfg.signal_to_noise)
    amplitude *= np.exp(0.2 * rng.standard_normal((len(labels), 2)))
    suppression = 1.0 - cfg.erd_depth * efficiency
    # Left-hand imagery (0) suppresses the right-hemisphere source (index 1) and
    # right-hand imagery (1) suppresses the left-hemisphere source (index 0).
    amplitude[labels == 0, 1] *= suppression
    amplitude[labels == 1, 0] *= suppression
    return amplitude


def _render(
    cfg: SyntheticMIConfig,
    rng: np.random.Generator,
    model: _SubjectModel,
    amplitude: FloatArray,
    n_times: int,
    gain: FloatArray | None = None,
) -> FloatArray:
    """Mix sources into sensor signals, shape (n_trials, n_channels, n_times), in volts."""
    n, n_ch = amplitude.shape[0], model.mixing.shape[0]
    motor = _shaped_noise(rng, (n, 2, n_times), cfg.sfreq, model.band)
    motor *= amplitude[:, :, None]
    background = _shaped_noise(rng, (n, n_ch, n_times), cfg.sfreq, None)
    sensor = 0.3 * rng.standard_normal((n, n_ch, n_times))
    channel_gain: FloatArray = (
        gain if gain is not None else np.asarray(1.0 + 0.05 * rng.standard_normal(n_ch))
    )
    X: FloatArray = np.einsum("cs,nst->nct", model.patterns, motor)
    X += np.einsum("cs,nst->nct", model.mixing, background)
    X += sensor
    X *= channel_gain[None, :, None] * _VOLTS_SCALE
    return X


def _prepare(cfg: SyntheticMIConfig) -> tuple[FloatArray, FloatArray]:
    """Site-level patterns and per-subject efficiencies."""
    base = _motor_patterns(tuple(cfg.channels))
    site_rng = np.random.default_rng([cfg.seed, 1_000_003])
    site_patterns = base * (1.0 + cfg.site_shift * site_rng.normal(0.0, 0.4, base.shape))
    if cfg.efficiencies is not None:
        efficiencies = np.asarray(cfg.efficiencies, dtype=np.float64)
    else:
        prior_rng = np.random.default_rng([cfg.seed, 2_000_003])
        efficiencies = prior_rng.beta(cfg.efficiency_alpha, cfg.efficiency_beta, cfg.n_subjects)
    return site_patterns, efficiencies


def generate_synthetic_mi(config: SyntheticMIConfig | None = None) -> SyntheticMIResult:
    """Generate a labeled two-class (left/right hand) motor-imagery :class:`EpochSet`.

    Epochs are ordered chronologically per subject (``order`` increases across
    sessions), with class order shuffled within each session.
    """
    cfg = config or SyntheticMIConfig()
    n_times = round(cfg.sfreq * cfg.trial_seconds)
    n_per_session = 2 * cfg.n_trials_per_class
    site_patterns, efficiencies = _prepare(cfg)

    parts: list[EpochSet] = []
    truth: dict[str, float] = {}
    for s in range(cfg.n_subjects):
        rng = np.random.default_rng([cfg.seed, s])
        subject = f"sub-{s:03d}"
        truth[subject] = float(efficiencies[s])
        model = _subject_model(cfg, rng, site_patterns, truth[subject])
        for session in range(cfg.n_sessions):
            labels = rng.permutation(
                np.repeat(np.arange(2, dtype=np.int64), cfg.n_trials_per_class)
            )
            amplitude = _amplitudes(cfg, rng, labels, model.efficiency)
            X = _render(cfg, rng, model, amplitude, n_times)
            parts.append(
                EpochSet.from_arrays(
                    X,
                    labels,
                    label_names=CAP1_LABELS,
                    ch_names=tuple(cfg.channels),
                    sfreq=cfg.sfreq,
                    subject=subject,
                    session=f"ses-{session:02d}",
                    dataset=cfg.dataset_id,
                    order=session * n_per_session + np.arange(n_per_session),
                )
            )
    return SyntheticMIResult(epochs=EpochSet.concat(parts), efficiency=truth)


def generate_synthetic_recordings(
    config: SyntheticMIConfig | None = None, rest_seconds: float = 1.5
) -> list[Recording]:
    """Generate continuous recordings (one per subject and session) with cue events.

    Each trial is ``rest_seconds`` of rest (no desynchronization) followed by
    ``trial_seconds`` of imagery; the event marks the cue at the start of the imagery
    segment. Used to test ingestion, QA and epoching without downloads.
    """
    cfg = config or SyntheticMIConfig()
    if rest_seconds <= 0:
        raise ValueError("rest_seconds must be > 0")
    n_trial = round(cfg.sfreq * cfg.trial_seconds)
    n_rest = round(cfg.sfreq * rest_seconds)
    site_patterns, efficiencies = _prepare(cfg)
    recordings: list[Recording] = []
    for s in range(cfg.n_subjects):
        rng = np.random.default_rng([cfg.seed, s])
        model = _subject_model(cfg, rng, site_patterns, float(efficiencies[s]))
        for session in range(cfg.n_sessions):
            labels = rng.permutation(
                np.repeat(np.arange(2, dtype=np.int64), cfg.n_trials_per_class)
            )
            gain = 1.0 + 0.05 * rng.standard_normal(len(cfg.channels))
            rest_labels = np.full(len(labels), -1, dtype=np.int64)
            rest = _render(cfg, rng, model, _amplitudes(cfg, rng, rest_labels, 0.0), n_rest, gain)
            trials = _render(
                cfg, rng, model, _amplitudes(cfg, rng, labels, model.efficiency), n_trial, gain
            )
            segments = np.concatenate([rest, trials], axis=2)  # (n, ch, rest + trial)
            data = np.ascontiguousarray(np.moveaxis(segments, 0, 1).reshape(len(cfg.channels), -1))
            period = n_rest + n_trial
            events = tuple(
                Event(onset_sample=i * period + n_rest, label=CAP1_LABELS[int(label)])
                for i, label in enumerate(labels)
            )
            recordings.append(
                Recording(
                    data=data,
                    sfreq=cfg.sfreq,
                    ch_names=tuple(cfg.channels),
                    dataset_id=cfg.dataset_id,
                    subject_id=f"sub-{s:03d}",
                    session_id=f"ses-{session:02d}",
                    run_id="0",
                    events=events,
                )
            )
    return recordings
