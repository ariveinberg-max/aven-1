"""Identity and dataset-ID probes (WP-4.3, protocol §7).

A *probe* asks how well a simple classifier can recover **who** produced an epoch
(subject) or **which dataset** it came from, using only a representation. High values
flag the "identity trap": a representation may be solving the task through identity or
dataset shortcuts instead of transferable intent structure. Probe accuracy is reported
next to every foundation-model and proprietary-model result. It doubles as a privacy
audit (PRIV-7): lower identity decodability is better.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from neurolayer.core.interfaces import Encoder
from neurolayer.core.types import EpochSet


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """Cross-validated probe accuracy against its chance level."""

    target: str
    n_classes: int
    accuracy: float
    chance: float

    @property
    def above_chance(self) -> float:
        """Accuracy minus chance (0 = no information about the target)."""
        return self.accuracy - self.chance


def probe(
    epochs: EpochSet,
    encoder: Encoder,
    target: Literal["subject", "dataset"] = "subject",
    n_splits: int = 5,
    seed: int = 0,
) -> ProbeResult:
    """Fit ``encoder`` on all epochs, then cross-validate a linear probe for ``target``.

    The encoder is fitted without the probe target, and chance is the majority-class
    rate. Use with source (training) data only; probes never touch target subjects.
    """
    if target == "subject":
        groups = np.array([f"{d}/{s}" for d, s in zip(epochs.dataset, epochs.subject, strict=True)])
    else:
        groups = np.asarray(epochs.dataset)
    labels, y = np.unique(groups, return_inverse=True)
    if len(labels) < 2:
        raise ValueError(f"probe needs at least two distinct {target} values")
    encoder.fit(epochs)
    features = encoder.transform(epochs)
    folds = min(n_splits, int(np.bincount(y).min()))
    if folds < 2:
        raise ValueError(f"every {target} needs at least 2 epochs for cross-validation")
    classifier = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = cross_val_score(classifier, features, y, cv=cv)
    chance = float(np.bincount(y).max() / len(y))
    return ProbeResult(
        target=target, n_classes=len(labels), accuracy=float(np.mean(scores)), chance=chance
    )
