"""Third-party pretrained model governance (WP-3.4, ADR-0007).

Every pretrained model has a card in ``catalog/models/``. A card records where the
weights come from, their pinned SHA-256, the code and weight licenses, and what data
they were pretrained on. The model gate mirrors the dataset license gate:

* ``benchmark`` (using the model as a baseline): permissive code license and pinned
  weight hash.
* ``training`` (using the weights to initialize a shippable model): additionally
  requires a recorded legal review, because weights inherit their pretraining-data
  terms.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from neurolayer.data.catalog import LegalReview, Purpose

PERMISSIVE_CODE_LICENSES = frozenset({"MIT", "BSD-3-Clause", "BSD-2-Clause", "Apache-2.0"})


class WeightsInfo(BaseModel):
    """Where the weights come from and what constrains them."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    url: str | None = None
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    license: str = Field(description="SPDX id or LicenseRef-* of the weights themselves.")
    pretraining_data: tuple[str, ...] = Field(
        default=(), description="Catalog ids or names of pretraining corpora."
    )
    evidence: str = ""


class ModelCard(BaseModel):
    """Provenance and licensing record for one third-party pretrained model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^[a-z0-9_]+$")
    name: str
    architecture: str = Field(description="e.g. 'braindecode:Labram'.")
    source_url: str
    citation: str
    code_license: str
    weights: WeightsInfo
    roles: tuple[str, ...] = ("baseline",)
    legal_review: LegalReview | None = None
    notes: str = ""


class ModelUsageDecision(BaseModel):
    """Outcome of :func:`evaluate_model_usage`."""

    model_config = ConfigDict(frozen=True)

    model_id: str
    purpose: Purpose
    allowed: bool
    reasons: tuple[str, ...]


def evaluate_model_usage(card: ModelCard, purpose: Purpose) -> ModelUsageDecision:
    """Decide whether a pretrained model may be used for ``purpose`` (ADR-0007)."""
    reasons: list[str] = []
    if card.legal_review is not None and purpose in card.legal_review.approved_purposes:
        return ModelUsageDecision(
            model_id=card.id, purpose=purpose, allowed=True,
            reasons=(f"approved by legal review {card.legal_review.reference}",),
        )  # fmt: skip
    if card.code_license not in PERMISSIVE_CODE_LICENSES:
        reasons.append(f"code license {card.code_license} is not on the permissive list")
    if purpose in (Purpose.BENCHMARK, Purpose.TRAINING) and card.weights.sha256 is None:
        reasons.append("weights SHA-256 not pinned (download, review, then record the hash)")
    if purpose is Purpose.TRAINING:
        reasons.append("training on third-party weights requires a recorded legal review")
    return ModelUsageDecision(
        model_id=card.id,
        purpose=purpose,
        allowed=not reasons,
        reasons=tuple(reasons) or (f"{card.code_license} code, pinned weights",),
    )


def load_model_card(path: Path) -> ModelCard:
    """Load and validate one model card; its ``id`` must match the file name."""
    with path.open(encoding="utf-8") as fh:
        card = ModelCard.model_validate(yaml.safe_load(fh))
    if card.id != path.stem:
        raise ValueError(f"model card id {card.id!r} does not match file name {path.name!r}")
    return card


def load_model_catalog(directory: Path) -> dict[str, ModelCard]:
    """Load every model card in ``directory``."""
    return {card.id: card for card in map(load_model_card, sorted(directory.glob("*.yaml")))}


def load_pretrained(model: Any, card: ModelCard, weights_path: Path, purpose: Purpose) -> Any:
    """Load third-party weights into ``model`` after the gate and a hash check."""
    from neurolayer.representation.torch_utils import load_checkpoint

    decision = evaluate_model_usage(card, purpose)
    if not decision.allowed:
        raise PermissionError(f"model {card.id} refused: {'; '.join(decision.reasons)}")
    load_checkpoint(model, weights_path, expected_sha256=card.weights.sha256, strict=False)
    return model
