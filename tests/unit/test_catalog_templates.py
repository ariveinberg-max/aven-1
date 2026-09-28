from __future__ import annotations

from pathlib import Path

import yaml

from neurolayer.data.catalog import DatasetCard, Purpose, evaluate_usage


def test_pilot_card_template_is_valid_and_not_internal(repo_root: Path) -> None:
    raw = yaml.safe_load((repo_root / "catalog/templates/neurolayer_pilot.yaml").read_text("utf-8"))
    card = DatasetCard.model_validate(raw)
    assert card.access == "internal"
    assert not card.is_internal  # consent, not a blanket internal license (ADR-0012)
    assert not evaluate_usage(card, Purpose.TRAINING).allowed
