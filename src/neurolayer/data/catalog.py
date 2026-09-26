"""Dataset catalog and license gate (ADR-0005).

Every dataset, public or ours, has a YAML card in ``catalog/datasets/``. The card's
license block decides what the dataset may be used for:

``exploration``
    Local inspection and pipeline development; nothing trained on it ships.
``benchmark``
    Reporting evaluation numbers.
``training``
    Training weights that may ship in a product or be licensed to a partner.

The gate is deliberately conservative: until a human records ``verified_by`` and
``verified_on`` for a license, only ``exploration`` is permitted. A recorded legal
review can explicitly approve any purpose.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class Purpose(StrEnum):
    """What a dataset is being used for, from least to most demanding."""

    EXPLORATION = "exploration"
    BENCHMARK = "benchmark"
    TRAINING = "training"


Tri = Literal["yes", "no", "unclear"]
Role = Literal[
    "pretraining", "development", "locked_holdout", "benchmark_only", "unassigned", "excluded"
]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LegalReview(_Strict):
    """A recorded legal decision that explicitly approves purposes."""

    approved_purposes: tuple[Purpose, ...]
    reviewer: str
    reference: str = Field(description="Ticket, memo or agreement identifier.")
    reviewed_on: date


class LicenseInfo(_Strict):
    """License facts for a dataset, with evidence and human verification."""

    spdx: str = Field(description="SPDX id or LicenseRef-* (e.g. LicenseRef-DUA).")
    url: HttpUrl | None = None
    evidence: str = Field(description="Where the license facts were read from.")
    commercial_use: Tri
    derivatives: Tri
    share_alike: bool = False
    verified_by: str | None = None
    verified_on: date | None = None
    legal_review: LegalReview | None = None

    @model_validator(mode="after")
    def _verification_complete(self) -> LicenseInfo:
        if (self.verified_by is None) != (self.verified_on is None):
            raise ValueError("verified_by and verified_on must be set together")
        return self

    @property
    def is_verified(self) -> bool:
        """``True`` once a human has read the license and recorded it."""
        return self.verified_by is not None


class DatasetCard(_Strict):
    """Provenance and licensing record for one dataset."""

    id: str = Field(pattern=r"^[a-z0-9_]+$")
    name: str
    version: str
    modality: Literal["eeg", "emg", "meg", "ecog", "spikes", "fnirs", "synthetic"]
    paradigms: tuple[str, ...]
    n_subjects: int | None = Field(default=None, ge=1)
    n_channels: int | None = Field(default=None, ge=1)
    sfreq_hz: float | None = Field(default=None, gt=0)
    sessions_per_subject: str | None = None
    labels: tuple[str, ...] = ()
    source_url: HttpUrl | None = None
    citation: str
    access: Literal["open", "registration", "dua", "internal"]
    loader: str | None = Field(default=None, description="Adapter id, e.g. 'moabb:PhysionetMI'.")
    roles: tuple[Role, ...]
    license: LicenseInfo
    notes: str = ""

    @property
    def is_internal(self) -> bool:
        """Internal data (synthetic or our own, consented) with an internal license."""
        return self.access == "internal" and self.license.spdx.startswith("LicenseRef-Internal")


class UsageDecision(_Strict):
    """Outcome of :func:`evaluate_usage`."""

    dataset_id: str
    purpose: Purpose
    allowed: bool
    reasons: tuple[str, ...]


class LicenseGateError(PermissionError):
    """Raised when one or more datasets are not permitted for the requested purpose."""

    def __init__(self, decisions: Iterable[UsageDecision]) -> None:
        self.decisions = tuple(decisions)
        lines = [
            f"  - {d.dataset_id} ({d.purpose}): {'; '.join(d.reasons)}" for d in self.decisions
        ]
        super().__init__("license gate refused dataset usage:\n" + "\n".join(lines))


def evaluate_usage(card: DatasetCard, purpose: Purpose) -> UsageDecision:
    """Decide whether ``card`` may be used for ``purpose``.

    Rules (ADR-0005), checked in order:

    1. ``excluded`` role → refused.
    2. Internal data → allowed.
    3. A legal review approving ``purpose`` → allowed.
    4. ``exploration`` → allowed unless commercial use is known to be prohibited.
    5. ``benchmark`` → requires human verification and commercial use not prohibited.
    6. ``training`` → requires human verification, commercial use ``yes``,
       derivatives ``yes`` and no share-alike clause.
    """
    lic = card.license

    def decide(allowed: bool, *reasons: str) -> UsageDecision:
        return UsageDecision(
            dataset_id=card.id, purpose=purpose, allowed=allowed, reasons=tuple(reasons)
        )

    if "excluded" in card.roles:
        return decide(False, "dataset is marked 'excluded' in the catalog")
    if card.is_internal:
        return decide(True, "internal data")
    if lic.legal_review is not None and purpose in lic.legal_review.approved_purposes:
        return decide(True, f"approved by legal review {lic.legal_review.reference}")

    problems: list[str] = []
    if lic.commercial_use == "no":
        problems.append(f"license {lic.spdx} prohibits commercial use")
    if purpose in (Purpose.BENCHMARK, Purpose.TRAINING) and not lic.is_verified:
        problems.append("license not yet verified by a human (set verified_by/verified_on)")
    if purpose is Purpose.TRAINING:
        if lic.commercial_use != "yes":
            problems.append(f"commercial use is '{lic.commercial_use}', must be 'yes'")
        if lic.derivatives != "yes":
            problems.append(f"derivatives is '{lic.derivatives}', must be 'yes'")
        if lic.share_alike:
            problems.append("share-alike license needs a legal review before training")
    # A prohibition may be listed twice (rule 4 and rule 6); keep reasons unique.
    unique = tuple(dict.fromkeys(problems))
    if unique:
        return decide(False, *unique)
    return decide(True, f"license {lic.spdx} permits {purpose.value}")


def require_usage(cards: Iterable[DatasetCard], purpose: Purpose) -> list[UsageDecision]:
    """Evaluate every card and raise :class:`LicenseGateError` if any is refused."""
    decisions = [evaluate_usage(card, purpose) for card in cards]
    refused = [d for d in decisions if not d.allowed]
    if refused:
        raise LicenseGateError(refused)
    return decisions


def load_card(path: Path) -> DatasetCard:
    """Load and validate one dataset card; its ``id`` must match the file name."""
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    card = DatasetCard.model_validate(raw)
    if card.id != path.stem:
        raise ValueError(f"card id {card.id!r} does not match file name {path.name!r}")
    return card


def load_catalog(directory: Path) -> dict[str, DatasetCard]:
    """Load every ``*.yaml`` card in ``directory``, keyed by dataset id."""
    cards: dict[str, DatasetCard] = {}
    for path in sorted(directory.glob("*.yaml")):
        card = load_card(path)
        if card.id in cards:
            raise ValueError(f"duplicate dataset id {card.id!r}")
        cards[card.id] = card
    if not cards:
        raise FileNotFoundError(f"no dataset cards found in {directory}")
    return cards


def select_cards(catalog: Mapping[str, DatasetCard], ids: Iterable[str]) -> list[DatasetCard]:
    """Return the cards for ``ids``, raising ``KeyError`` for unknown datasets."""
    missing = [i for i in ids if i not in catalog]
    if missing:
        raise KeyError(f"datasets not in catalog: {missing}")
    return [catalog[i] for i in ids]
