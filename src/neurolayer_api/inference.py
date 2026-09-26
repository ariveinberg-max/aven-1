"""Turn API trial payloads into model-ready epochs, mirroring training preprocessing.

Clients send **microvolts**, one window per trial (channels × samples) starting at the
same cue-relative offset as training. The service converts to volts, normalizes channel
names, resamples to the model rate and applies the bundle's filters, then checks the
window length.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from neurolayer.core.channels import normalize_channel_names
from neurolayer.core.types import UNLABELED, EpochSet, FloatArray, Recording
from neurolayer.models.bundle import BundleInfo
from neurolayer.representation.spatial_field import electrode_positions
from neurolayer.signal.pipeline import PipelineSpec
from neurolayer.signal.transforms import Resample, Transform

MICROVOLTS = 1e-6


class PayloadError(ValueError):
    """The request is well-formed JSON but not usable by the model."""


def canonical_channels(ch_names: Sequence[str]) -> tuple[str, ...]:
    """Normalize names and require a known scalp position for each."""
    try:
        names = normalize_channel_names(ch_names)
        electrode_positions(names)
    except (KeyError, ValueError) as exc:
        raise PayloadError(str(exc)) from exc
    return names


def _transforms(info: BundleInfo) -> list[Transform]:
    pipeline = info.preprocessing.get("pipeline")
    if not pipeline:
        return []
    spec = PipelineSpec.model_validate(pipeline)
    return [t for t in spec.build() if t.name not in {"resample", "unit_check", "select_channels"}]


def to_epochs(
    trials: Sequence[FloatArray],
    labels: Sequence[int] | None,
    ch_names: tuple[str, ...],
    sfreq: float,
    info: BundleInfo,
) -> EpochSet:
    """Validate, preprocess and stack trials into an :class:`EpochSet`."""
    if not trials:
        raise PayloadError("no trials")
    transforms = _transforms(info)
    windows = []
    for data in trials:
        if data.ndim != 2 or data.shape[0] != len(ch_names):
            raise PayloadError(f"each trial must be {len(ch_names)} channels x samples")
        recording = Recording(
            data=np.ascontiguousarray(data, dtype=np.float64) * MICROVOLTS,
            sfreq=sfreq,
            ch_names=ch_names,
            dataset_id="api",
            subject_id="session",
        )
        recording = Resample(info.sfreq)(recording)
        for transform in transforms:
            recording = transform(recording)
        if recording.n_samples != info.n_times:
            seconds = info.n_times / info.sfreq
            raise PayloadError(
                f"each trial must be {seconds:g} s long (got {recording.duration_s:g} s)"
            )
        windows.append(recording.data)
    y = list(labels) if labels is not None else [UNLABELED] * len(windows)
    return EpochSet.from_arrays(
        np.stack(windows),
        y,
        label_names=info.label_names,
        ch_names=ch_names,
        sfreq=info.sfreq,
        subject="session",
        dataset="api",
    )
