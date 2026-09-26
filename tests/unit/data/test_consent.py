from __future__ import annotations

import json
from pathlib import Path

import pytest

from neurolayer.data.consent import ConsentPurpose, ConsentRegistry, document_hash


@pytest.fixture
def form(tmp_path: Path) -> Path:
    path = tmp_path / "consent-v0.1.md"
    path.write_text("# Consent form v0.1\n", encoding="utf-8")
    return path


def test_purposes_are_separate(tmp_path: Path, form: Path) -> None:
    ledger = ConsentRegistry(tmp_path / "ledger.jsonl")
    ledger.grant("P001", [ConsentPurpose.RESEARCH], form, "v0.1", "tester")
    assert ledger.allowed("P001", ConsentPurpose.RESEARCH)
    assert not ledger.allowed("P001", ConsentPurpose.COMMERCIAL_TRAINING)
    assert not ledger.allowed("P002", ConsentPurpose.RESEARCH)
    with pytest.raises(PermissionError, match="commercial_training"):
        ledger.require("P001", [ConsentPurpose.RESEARCH, ConsentPurpose.COMMERCIAL_TRAINING])


def test_revocation_wins_and_history_is_kept(tmp_path: Path, form: Path) -> None:
    ledger = ConsentRegistry(tmp_path / "ledger.jsonl")
    ledger.grant("P001", [ConsentPurpose.RESEARCH, ConsentPurpose.PRODUCT], form, "v0.1", "t")
    ledger.revoke("P001", ConsentPurpose.RESEARCH, "t")
    assert not ledger.allowed("P001", ConsentPurpose.RESEARCH)
    assert ledger.allowed("P001", ConsentPurpose.PRODUCT)
    assert [e.action for e in ledger.events()] == ["grant", "grant", "revoke"]
    ledger.grant("P001", [ConsentPurpose.RESEARCH], form, "v0.2", "t")
    assert ledger.allowed("P001", ConsentPurpose.RESEARCH)


def test_grant_binds_document_hash(tmp_path: Path, form: Path) -> None:
    ledger = ConsentRegistry(tmp_path / "ledger.jsonl")
    (event,) = ledger.grant("P007", [ConsentPurpose.RESEARCH], form, "v0.1", "t")
    assert event.document_sha256 == document_hash(form)
    line = json.loads((tmp_path / "ledger.jsonl").read_text(encoding="utf-8"))
    assert line["participant"] == "P007"
    assert line["document_version"] == "v0.1"


@pytest.mark.parametrize("name", ["Alice", "P", "P01a", "sub-001"])
def test_rejects_non_pseudonyms(tmp_path: Path, form: Path, name: str) -> None:
    ledger = ConsentRegistry(tmp_path / "ledger.jsonl")
    with pytest.raises(ValueError, match="pseudonym"):
        ledger.grant(name, [ConsentPurpose.RESEARCH], form, "v0.1", "t")
    assert ledger.events() == []


def test_participants_by_purpose(tmp_path: Path, form: Path) -> None:
    ledger = ConsentRegistry(tmp_path / "ledger.jsonl")
    both = [ConsentPurpose.RESEARCH, ConsentPurpose.COMMERCIAL_TRAINING]
    ledger.grant("P002", both, form, "1", "t")
    ledger.grant("P001", [ConsentPurpose.RESEARCH], form, "1", "t")
    ledger.grant("P003", [ConsentPurpose.COMMERCIAL_TRAINING], form, "1", "t")
    ledger.revoke("P003", ConsentPurpose.COMMERCIAL_TRAINING, "t")
    assert ledger.participants(ConsentPurpose.RESEARCH) == ["P001", "P002"]
    assert ledger.participants(ConsentPurpose.COMMERCIAL_TRAINING) == ["P002"]
    assert ledger.participants(ConsentPurpose.PRODUCT) == []
