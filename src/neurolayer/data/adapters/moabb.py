"""MOABB dataset adapter (WP-1.2).

Wraps any MOABB dataset class named by a catalog card's ``loader`` field
(``moabb:<ClassName>``). MOABB handles downloading and file parsing; this adapter
converts its MNE ``Raw`` objects into canonical recordings with:

* EEG channels only, canonical 10-05 names;
* volts (MNE's native unit);
* events aligned to **cue onset** (MOABB ``interval[0]``);
* labels mapped to the canonical vocabulary, with unmapped labels counted, not kept;
* pseudonymous ``sub-XXX`` subject ids.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from neurolayer.core.types import Recording
from neurolayer.data.adapters.base import (
    AdapterNotAvailableError,
    IngestionLog,
    pseudonymize,
    require_exploration,
)
from neurolayer.data.catalog import DatasetCard
from neurolayer.data.mne_bridge import from_mne_raw

MOABB_LABEL_MAP: dict[str, str] = {
    "left_hand": "left_hand",
    "right_hand": "right_hand",
    "hands": "both_hands",
    "both_hands": "both_hands",
    "feet": "feet",
    "tongue": "tongue",
    "rest": "rest",
}
"""MOABB event names → canonical labels. Anything else is dropped and counted."""


class MoabbAdapter:
    """Canonical-recording adapter around a MOABB dataset instance.

    Parameters
    ----------
    card
        The dataset's catalog card; its license must permit exploration.
    dataset
        A MOABB dataset object (or any object with ``subject_list``, ``event_id``,
        ``interval`` and ``get_data(subjects=[...])``). Injected so tests need no
        downloads.
    data_dir
        Where MOABB/MNE store the files (``data/raw/<id>/<version>``). ``None`` keeps
        MNE's default (``~/mne_data``).
    """

    def __init__(self, card: DatasetCard, dataset: Any, data_dir: Path | None = None) -> None:
        require_exploration(card)
        self.card = card
        self.log = IngestionLog()
        self._dataset = dataset
        self._data_dir = data_dir

    @contextmanager
    def _mne_data(self) -> Iterator[None]:
        """Point MNE's dataset cache at ``data_dir`` for the duration of a call."""
        if self._data_dir is None:
            yield
            return
        previous = os.environ.get("MNE_DATA")
        os.environ["MNE_DATA"] = str(self._data_dir)
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop("MNE_DATA", None)
            else:
                os.environ["MNE_DATA"] = previous

    @classmethod
    def from_card(cls, card: DatasetCard, data_dir: Path | None = None) -> MoabbAdapter:
        """Instantiate the MOABB class named in ``card.loader``."""
        if card.loader is None or not card.loader.startswith("moabb:"):
            raise AdapterNotAvailableError(f"{card.id} is not a MOABB dataset")
        try:
            import moabb.datasets
        except ImportError as exc:
            raise AdapterNotAvailableError("install the 'neuro' extra to use MOABB") from exc
        name = card.loader.split(":", 1)[1]
        dataset_cls = getattr(moabb.datasets, name, None)
        if dataset_cls is None:
            raise AdapterNotAvailableError(f"MOABB has no dataset class {name!r}")
        return cls(card, dataset_cls(), data_dir)

    @property
    def cue_offset_s(self) -> float:
        """Seconds from MOABB's event marker to the task cue (``interval[0]``)."""
        return float(self._dataset.interval[0])

    @property
    def max_trial_s(self) -> float:
        """Length of the task window after the cue, in seconds."""
        return float(self._dataset.interval[1] - self._dataset.interval[0])

    def subjects(self) -> list[str]:
        """Native subject ids."""
        return [str(s) for s in self._dataset.subject_list]

    def download(self, subjects: list[str]) -> None:
        """Download ``subjects`` into ``data_dir`` (network; ``neurolayer data fetch``)."""
        if self._data_dir is not None:
            self._data_dir.mkdir(parents=True, exist_ok=True)
        with self._mne_data():
            self._dataset.download(
                subject_list=[int(s) for s in subjects],
                path=None if self._data_dir is None else str(self._data_dir),
                update_path=False,
            )

    def recordings(self, subject: str) -> Iterator[Recording]:
        """Yield canonical recordings for every session and run of ``subject``."""
        native = int(subject)
        with self._mne_data():
            data = self._dataset.get_data(subjects=[native])
        sessions = data[native]
        for session_name in sorted(sessions):
            runs = sessions[session_name]
            for run_name in sorted(runs):
                recording, report = from_mne_raw(
                    runs[run_name],
                    dataset_id=self.card.id,
                    subject_id=pseudonymize(native),
                    session_id=str(session_name),
                    run_id=str(run_name),
                    event_codes=self._dataset.event_id,
                    label_map=MOABB_LABEL_MAP,
                    cue_offset_s=self.cue_offset_s,
                )
                self.log.add(recording, report)
                yield recording
