"""Preprocessing pipelines, their content hash, and the processed-epoch cache (WP-2.1, 2.6).

``pipeline_hash`` covers every transform's name, version and config plus the epoching
spec and the channel-harmonization rule. It addresses
``data/processed/<pipeline_hash>/<dataset>/`` so changing anything produces a new cache
entry and nothing is ever overwritten in place (ADR-0003).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from neurolayer.core.types import EpochSet, Recording
from neurolayer.signal.epoching import EpochingReport, EpochingSpec, epoch_recordings
from neurolayer.signal.transforms import Transform, TransformSpec

PIPELINE_FORMAT_VERSION = 1


class TransformConfig(BaseModel):
    """YAML form of one transform: ``{name: bandpass, params: {l_freq: 1, h_freq: 40}}``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    params: dict[str, Any] = Field(default_factory=dict)

    def build(self) -> Transform:
        """Instantiate the transform."""
        return TransformSpec(self.name, dict(self.params)).build()


def _default_transforms() -> tuple[TransformConfig, ...]:
    return (
        TransformConfig(name="unit_check"),
        TransformConfig(name="bandpass", params={"l_freq": 1.0, "h_freq": 40.0, "order": 4}),
        TransformConfig(name="resample", params={"sfreq": 128.0}),
    )


class PipelineSpec(BaseModel):
    """Full preprocessing definition for real datasets (synthetic data skips it)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    transforms: tuple[TransformConfig, ...] = Field(default_factory=_default_transforms)
    epoching: EpochingSpec = Field(default_factory=EpochingSpec)
    channels: Literal["common"] | tuple[str, ...] = Field(
        default="common",
        description="'common' keeps the channels shared by all datasets in a run; a list "
        "selects exactly those channels.",
    )

    def build(self) -> list[Transform]:
        """Instantiate the transforms in order."""
        return [t.build() for t in self.transforms]

    def pipeline_hash(self) -> str:
        """Content hash of the full pipeline definition."""
        payload = {
            "format": PIPELINE_FORMAT_VERSION,
            "transforms": [
                {"name": t.name, "version": t.version, "config": t.config()} for t in self.build()
            ],
            "epoching": self.epoching.model_dump(mode="json"),
            "channels": self.channels if self.channels == "common" else list(self.channels),
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def preprocess(recording: Recording, transforms: Sequence[Transform]) -> Recording:
    """Apply transforms in order."""
    for transform in transforms:
        recording = transform(recording)
    return recording


def process_subject(
    recordings: Sequence[Recording], spec: PipelineSpec
) -> tuple[EpochSet, EpochingReport]:
    """Preprocess and epoch all recordings of one subject."""
    transforms = spec.build()
    processed = [preprocess(r, transforms) for r in recordings]
    return epoch_recordings(processed, spec.epoching)


# ----------------------------------------------------------------------------- cache


def save_epochs(epochs: EpochSet, path: Path, provenance: dict[str, Any]) -> None:
    """Write ``<path>.npz`` (arrays) and ``<path>.json`` (metadata + provenance)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path.with_suffix(".npz"),
        X=epochs.X,
        y=epochs.y,
        subject=epochs.subject,
        session=epochs.session,
        dataset=epochs.dataset,
        order=epochs.order,
    )
    meta = {
        "label_names": list(epochs.label_names),
        "ch_names": list(epochs.ch_names),
        "sfreq": epochs.sfreq,
        "provenance": provenance,
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def load_epochs_file(path: Path) -> EpochSet:
    """Read an EpochSet written by :func:`save_epochs` (no pickle involved)."""
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    with np.load(path.with_suffix(".npz"), allow_pickle=False) as arrays:
        return EpochSet(
            X=arrays["X"].astype(np.float64),
            y=arrays["y"].astype(np.int64),
            label_names=tuple(meta["label_names"]),
            ch_names=tuple(meta["ch_names"]),
            sfreq=float(meta["sfreq"]),
            subject=arrays["subject"],
            session=arrays["session"],
            dataset=arrays["dataset"],
            order=arrays["order"].astype(np.int64),
        )
