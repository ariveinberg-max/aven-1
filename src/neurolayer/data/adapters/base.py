"""Dataset adapter contract (WP-1.2).

An adapter turns one catalogued dataset into canonical
:class:`~neurolayer.core.types.Recording` objects. No library objects (MNE, MOABB)
cross this boundary.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from neurolayer.core.types import Recording
from neurolayer.data.catalog import DatasetCard, Purpose, evaluate_usage
from neurolayer.data.mne_bridge import ConversionReport


class AdapterNotAvailableError(RuntimeError):
    """Raised when no adapter exists for a card's ``loader`` or its extra is missing."""


@dataclass(slots=True)
class IngestionLog:
    """Accumulates per-recording conversion reports for the QA report (WP-1.5)."""

    reports: dict[tuple[str, str, str], ConversionReport] = field(default_factory=dict)

    def add(self, recording: Recording, report: ConversionReport) -> None:
        """Record the report of one converted recording."""
        key = (recording.subject_id, recording.session_id, recording.run_id)
        self.reports[key] = report


@runtime_checkable
class DatasetAdapter(Protocol):
    """Loads one dataset as canonical recordings."""

    card: DatasetCard
    log: IngestionLog

    def subjects(self) -> list[str]:
        """Native subject identifiers, as strings."""
        ...

    def recordings(self, subject: str) -> Iterator[Recording]:
        """Yield every run of every session of ``subject``, in chronological order."""
        ...

    def download(self, subjects: list[str]) -> None:
        """Fetch the raw files of ``subjects`` into the adapter's data directory."""
        ...


def pseudonymize(subject: str | int) -> str:
    """Map a native subject id to a pseudonymous canonical id (privacy requirement PRIV-8).

    Public datasets already use numeric pseudonyms, which are kept stable as
    ``sub-XXX``. Non-numeric ids are rejected rather than guessed.
    """
    text = str(subject).strip()
    if not text.isdigit():
        raise ValueError(f"expected a numeric native subject id, got {subject!r}")
    return f"sub-{int(text):03d}"


def require_exploration(card: DatasetCard) -> None:
    """Refuse to touch data whose license does not even permit local exploration."""
    decision = evaluate_usage(card, Purpose.EXPLORATION)
    if not decision.allowed:
        raise PermissionError(f"dataset {card.id} refused: {'; '.join(decision.reasons)}")


def get_adapter(card: DatasetCard, data_dir: Path | None = None) -> DatasetAdapter:
    """Build the adapter named by ``card.loader`` (for example ``moabb:PhysionetMI``)."""
    if card.loader is None:
        raise AdapterNotAvailableError(f"dataset {card.id} has no loader configured")
    kind, _, _ = card.loader.partition(":")
    if kind == "moabb":
        from neurolayer.data.adapters.moabb import MoabbAdapter

        return MoabbAdapter.from_card(card, data_dir)
    raise AdapterNotAvailableError(f"no adapter for loader {card.loader!r} ({card.id})")
