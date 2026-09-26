from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from neurolayer.data.catalog import (
    DatasetCard,
    LicenseGateError,
    Purpose,
    evaluate_usage,
    load_card,
    load_catalog,
    require_usage,
    select_cards,
)


def _card(**license_overrides: Any) -> DatasetCard:
    license_block: dict[str, Any] = {
        "spdx": "ODC-By-1.0",
        "evidence": "test",
        "commercial_use": "yes",
        "derivatives": "yes",
    } | license_overrides
    return DatasetCard.model_validate(
        {
            "id": "demo",
            "name": "Demo",
            "version": "1",
            "modality": "eeg",
            "paradigms": ["motor_imagery"],
            "citation": "x",
            "access": "open",
            "roles": ["development"],
            "license": license_block,
        }
    )


VERIFIED = {"verified_by": "founder", "verified_on": date(2026, 9, 26)}


class TestGateRules:
    def test_unverified_allows_only_exploration(self) -> None:
        card = _card()
        assert evaluate_usage(card, Purpose.EXPLORATION).allowed
        assert not evaluate_usage(card, Purpose.BENCHMARK).allowed
        assert not evaluate_usage(card, Purpose.TRAINING).allowed

    def test_verified_permissive_license_allows_everything(self) -> None:
        card = _card(**VERIFIED)
        assert all(evaluate_usage(card, p).allowed for p in Purpose)

    def test_non_commercial_blocks_all_purposes(self) -> None:
        card = _card(spdx="CC-BY-NC-4.0", commercial_use="no", **VERIFIED)
        for purpose in Purpose:
            decision = evaluate_usage(card, purpose)
            assert not decision.allowed
            assert any("prohibits commercial use" in r for r in decision.reasons)

    def test_no_derivatives_allows_benchmark_not_training(self) -> None:
        card = _card(spdx="CC-BY-ND-4.0", derivatives="no", **VERIFIED)
        assert evaluate_usage(card, Purpose.BENCHMARK).allowed
        assert not evaluate_usage(card, Purpose.TRAINING).allowed

    def test_share_alike_needs_legal_review_for_training(self) -> None:
        card = _card(spdx="CC-BY-SA-4.0", share_alike=True, **VERIFIED)
        assert evaluate_usage(card, Purpose.BENCHMARK).allowed
        assert not evaluate_usage(card, Purpose.TRAINING).allowed

    def test_legal_review_can_approve(self) -> None:
        review = {
            "approved_purposes": ["training"],
            "reviewer": "counsel",
            "reference": "LEGAL-1",
            "reviewed_on": "2026-10-01",
        }
        card = _card(spdx="CC-BY-SA-4.0", share_alike=True, legal_review=review, **VERIFIED)
        decision = evaluate_usage(card, Purpose.TRAINING)
        assert decision.allowed
        assert "LEGAL-1" in decision.reasons[0]

    def test_verification_fields_must_be_set_together(self) -> None:
        with pytest.raises(ValidationError, match="together"):
            _card(verified_by="founder")

    def test_require_usage_raises_with_all_refusals(self) -> None:
        with pytest.raises(LicenseGateError) as info:
            require_usage([_card(), _card(**VERIFIED)], Purpose.TRAINING)
        assert len(info.value.decisions) == 1


class TestRepositoryCatalog:
    def test_all_cards_validate(self, catalog_dir: Path) -> None:
        catalog = load_catalog(catalog_dir)
        assert {"synthetic_mi", "physionet_mi", "bnci2014_001", "meta_emg_generic"} <= set(catalog)

    def test_expected_gate_decisions(self, catalog_dir: Path) -> None:
        catalog = load_catalog(catalog_dir)
        assert evaluate_usage(catalog["synthetic_mi"], Purpose.TRAINING).allowed
        # Non-commercial Meta EMG data is excluded for every purpose.
        assert not evaluate_usage(catalog["meta_emg_generic"], Purpose.EXPLORATION).allowed
        # No-derivatives BCI IV 2a must never be trained on.
        assert not evaluate_usage(catalog["bnci2014_001"], Purpose.TRAINING).allowed
        # Nothing public is usable beyond exploration until a human verifies it (WP-1.1).
        for card in catalog.values():
            if not card.is_internal:
                assert not evaluate_usage(card, Purpose.TRAINING).allowed, card.id

    def test_select_cards_unknown(self, catalog_dir: Path) -> None:
        with pytest.raises(KeyError, match="nope"):
            select_cards(load_catalog(catalog_dir), ["nope"])

    def test_card_id_must_match_filename(self, tmp_path: Path, catalog_dir: Path) -> None:
        raw = yaml.safe_load((catalog_dir / "synthetic_mi.yaml").read_text())
        wrong = tmp_path / "other_name.yaml"
        wrong.write_text(yaml.safe_dump(raw))
        with pytest.raises(ValueError, match="does not match"):
            load_card(wrong)

    def test_empty_catalog_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_catalog(tmp_path)
