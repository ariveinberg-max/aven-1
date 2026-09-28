"""Typed experiment configuration (YAML → validated pydantic models).

Every run starts from a file in ``configs/experiments/``; see
``docs/architecture/reproducibility.md``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from neurolayer.core.channels import CONSUMER_MONTAGES
from neurolayer.data.catalog import Purpose
from neurolayer.data.synthetic import SyntheticMIConfig
from neurolayer.evaluation.protocol import ProtocolConfig
from neurolayer.signal.pipeline import PipelineSpec


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SyntheticSpec(_Strict):
    """Parameters for synthetic sources (see :class:`SyntheticMIConfig`)."""

    n_subjects: int = Field(default=8, ge=2)
    n_trials_per_class: int = Field(default=60, ge=1)
    n_sessions: int = Field(default=1, ge=1)
    sfreq: float = Field(default=128.0, ge=40.0)
    trial_seconds: float = Field(default=2.0, gt=0)
    subject_variability: float = Field(default=0.5, ge=0)
    signal_to_noise: float = Field(default=1.0, gt=0)
    seed: int = 0

    def to_generator_config(self) -> SyntheticMIConfig:
        """Convert to the generator's config."""
        return SyntheticMIConfig(**self.model_dump())


class DecoderSpec(_Strict):
    """Registered decoder name plus constructor parameters."""

    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class ProtocolSpec(_Strict):
    """CAP-1 protocol parameters; montages are referenced by name."""

    regime: Literal["within_dataset", "leave_dataset_out"] = "within_dataset"
    ks: tuple[int, ...] = (0, 5, 10, 20, 40)
    n_unlabeled: int = Field(default=0, ge=0)
    n_folds: int | None = Field(default=5, ge=2)
    min_test_per_class: int = Field(default=10, ge=1)
    source_montage: str | None = None
    target_montage: str | None = None
    threshold: float = Field(default=0.70, gt=0, lt=1)
    trial_seconds: float | None = Field(default=None, gt=0)
    n_bootstrap: int = Field(default=2000, ge=100)
    seed: int = 0

    @field_validator("source_montage", "target_montage")
    @classmethod
    def _known_montage(cls, value: str | None) -> str | None:
        if value is not None and value not in CONSUMER_MONTAGES:
            raise ValueError(f"unknown montage {value!r}; known: {sorted(CONSUMER_MONTAGES)}")
        return value

    def to_protocol_config(self) -> ProtocolConfig:
        """Resolve montage names and build the harness config."""
        data = self.model_dump()
        for key in ("source_montage", "target_montage"):
            name = data[key]
            data[key] = None if name is None else CONSUMER_MONTAGES[name].channels
        return ProtocolConfig(**data)


class ExperimentConfig(_Strict):
    """A complete, reproducible experiment definition."""

    name: str = Field(pattern=r"^[a-z0-9][a-z0-9_\-]*$")
    description: str = ""
    purpose: Purpose
    datasets: tuple[str, ...] = Field(min_length=1)
    synthetic: SyntheticSpec = Field(default_factory=SyntheticSpec)
    decoder: DecoderSpec
    protocol: ProtocolSpec = Field(default_factory=ProtocolSpec)
    pipeline: PipelineSpec = Field(
        default_factory=PipelineSpec, description="Preprocessing for real datasets (Stage 2)."
    )
    max_subjects: int | None = Field(
        default=None, ge=2, description="Use only the first N subjects of each real dataset."
    )


def load_experiment_config(path: Path) -> ExperimentConfig:
    """Load and validate an experiment YAML file."""
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return ExperimentConfig.model_validate(raw)
