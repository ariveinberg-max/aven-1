"""The cue protocol shared by the calibration game, pilot recordings and CAP-1.

Using the **same** trial structure (rest → cue → imagery window) in data collection,
the product's calibration flow and the evaluation windows keeps the lab-to-product gap
small (CAP-1 spec §7).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neurolayer.core.labels import CAP1_LABELS


@dataclass(frozen=True, slots=True)
class Cue:
    """One trial: at ``onset_s`` (from session start) the user sees ``label``."""

    index: int
    onset_s: float
    label: str


@dataclass(frozen=True, slots=True)
class CueSchedule:
    """Balanced, shuffled cue sequence with fixed timing.

    Attributes
    ----------
    n_trials_per_class
        Trials per label.
    labels
        Canonical labels to cue.
    rest_s, imagery_s
        Rest before each cue, and imagery duration after it. The CAP-1 epoch
        (0.5-2.5 s after the cue) must fit inside ``imagery_s``.
    seed
        Shuffle seed (recorded with the data).
    """

    n_trials_per_class: int = 20
    labels: tuple[str, ...] = CAP1_LABELS
    rest_s: float = 2.0
    imagery_s: float = 4.0
    seed: int = 0

    def cues(self) -> list[Cue]:
        """Return the trial sequence."""
        rng = np.random.default_rng(self.seed)
        order = rng.permutation(np.repeat(np.arange(len(self.labels)), self.n_trials_per_class))
        period = self.rest_s + self.imagery_s
        return [
            Cue(index=i, onset_s=i * period + self.rest_s, label=self.labels[int(c)])
            for i, c in enumerate(order)
        ]

    @property
    def duration_s(self) -> float:
        """Total session length."""
        return len(self.labels) * self.n_trials_per_class * (self.rest_s + self.imagery_s)
