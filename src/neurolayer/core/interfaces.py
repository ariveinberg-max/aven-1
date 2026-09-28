"""Stage interfaces shared by models and the evaluation harness.

They are defined here, in the lowest layer, so that ``neurolayer.models`` (which
implements them) and ``neurolayer.evaluation`` (which measures them) never import each
other (see the import-linter contracts in ``pyproject.toml``).
"""

from __future__ import annotations

from typing import Protocol, Self, runtime_checkable

import numpy as np
import numpy.typing as npt

from neurolayer.core.types import EpochSet


@runtime_checkable
class Decoder(Protocol):
    """A task model that can be fitted on source subjects and adapted to a new one.

    The CAP-1 harness (``docs/architecture/evaluation-protocol.md``) guarantees:

    * :meth:`fit` receives only source subjects, never the target.
    * :meth:`adapt` is called on a deep copy of the fitted decoder for every
      (target subject, calibration budget), so adaptation can never leak across
      subjects or budgets.
    * :meth:`predict` receives test epochs with labels hidden.

    Decoders must handle channel mismatches between source and target themselves,
    using ``EpochSet.ch_names``; that is the montage-agnostic problem.
    """

    def fit(self, source: EpochSet) -> None:
        """Train on labeled source-subject epochs."""
        ...

    def adapt(self, calibration: EpochSet | None, unlabeled: EpochSet | None) -> Self:
        """Adapt to one target subject and return a decoder ready for that subject.

        Parameters
        ----------
        calibration
            The target subject's first ``k`` labeled trials per class, or ``None`` when
            ``k == 0``.
        unlabeled
            Unlabeled trials from the target's calibration window, or ``None``.
        """
        ...

    def predict(self, epochs: EpochSet) -> npt.NDArray[np.int64]:
        """Predict label indices (into ``epochs.label_names``) for unlabeled epochs."""
        ...


@runtime_checkable
class Encoder(Protocol):
    """Maps epochs to fixed-size representations (Stage 3)."""

    def fit(self, epochs: EpochSet) -> None:
        """Learn the encoder's parameters (may ignore labels)."""
        ...

    def transform(self, epochs: EpochSet) -> npt.NDArray[np.float64]:
        """Return an array of shape ``(n_epochs, n_features)``."""
        ...
