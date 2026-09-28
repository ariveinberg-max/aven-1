"""Multi-seed label-shuffle control (WP-4.3, ADR-0010).

One shuffled classifier is one random direction shared by every target subject, so a
single run's bootstrap CI (over subjects) understates the control's variability and
can exclude 0.5 by chance. The control is therefore repeated over several shuffle
seeds; it **passes** when, at every budget, the across-seed mean balanced accuracy is
within ``max(2 x standard error, 0.02)`` of 0.5.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from neurolayer.core.types import EpochSet
from neurolayer.evaluation.protocol import ProtocolConfig, run_protocol
from neurolayer.experiments.config import DecoderSpec
from neurolayer.models.registry import make_factory

MIN_TOLERANCE = 0.02


@dataclass(frozen=True, slots=True)
class ShuffleControlResult:
    """Across-seed statistics of the shuffle control at each budget."""

    ks: tuple[int, ...]
    per_seed: tuple[tuple[float, ...], ...]  # seed -> mean BA per budget

    @property
    def mean(self) -> tuple[float, ...]:
        """Across-seed mean BA per budget."""
        return tuple(float(v) for v in np.mean(self.per_seed, axis=0))

    @property
    def standard_error(self) -> tuple[float, ...]:
        """Across-seed standard error per budget."""
        values = np.asarray(self.per_seed)
        return tuple(float(v) for v in values.std(axis=0, ddof=1) / np.sqrt(len(values)))

    @property
    def passed(self) -> bool:
        """``True`` when every budget's mean is compatible with chance."""
        return all(
            abs(m - 0.5) <= max(2.0 * se, MIN_TOLERANCE)
            for m, se in zip(self.mean, self.standard_error, strict=True)
        )


def run_shuffle_control(
    epochs: EpochSet, decoder: DecoderSpec, config: ProtocolConfig, seeds: Sequence[int]
) -> ShuffleControlResult:
    """Run ``decoder`` wrapped in the label-shuffle control once per seed."""
    if len(seeds) < 2:
        raise ValueError("the shuffle control needs at least two seeds (ADR-0010 uses >= 5)")
    per_seed = []
    for seed in seeds:
        factory = make_factory(
            "shuffle_control",
            {"decoder": {"name": decoder.name, "params": dict(decoder.params)}, "seed": seed},
        )
        per_seed.append(
            tuple(s.mean_ba for s in run_protocol(epochs, factory, config).per_budget())
        )
    return ShuffleControlResult(ks=config.ks, per_seed=tuple(per_seed))
