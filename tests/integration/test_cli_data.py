from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from neurolayer.cli import main
from neurolayer.data.adapters.moabb import MoabbAdapter
from neurolayer.data.catalog import DatasetCard
from tests.fakes import FakeMoabbDataset

pytest.importorskip("mne")


@pytest.fixture
def fake_adapter(monkeypatch: pytest.MonkeyPatch) -> FakeMoabbDataset:
    fake = FakeMoabbDataset(subjects=(1, 2, 3))

    def build(card: DatasetCard, data_dir: Path | None = None) -> Any:
        return MoabbAdapter(card, fake, data_dir)

    monkeypatch.setattr("neurolayer.cli.data.get_adapter", build)
    return fake


def _base(catalog_dir: Path, tmp_path: Path) -> list[str]:
    return ["--catalog", str(catalog_dir), "--data-root", str(tmp_path / "data")]


def test_fetch_writes_checksums_and_bids_then_verify(
    catalog_dir: Path,
    tmp_path: Path,
    fake_adapter: FakeMoabbDataset,
    capsys: pytest.CaptureFixture[str],
) -> None:
    pytest.importorskip("mne_bids")
    base = _base(catalog_dir, tmp_path)
    raw = tmp_path / "data/raw/physionet_mi/1.0.0"
    raw.mkdir(parents=True)
    (raw / "S001R01.edf.placeholder").write_text("pretend download")
    assert main([*base, "data", "fetch", "physionet_mi", "--subjects", "1"]) == 0
    assert fake_adapter.downloads == [([1], str(raw))]
    assert (raw / "checksums.sha256").exists()
    assert list((tmp_path / "data/interim/bids/physionet_mi").rglob("*.edf"))
    assert main([*base, "data", "verify", "physionet_mi"]) == 0
    (raw / "S001R01.edf.placeholder").write_text("tampered")
    assert main([*base, "data", "verify", "physionet_mi"]) == 1
    assert "modified" in capsys.readouterr().out


def test_audit_and_qa(
    catalog_dir: Path,
    tmp_path: Path,
    fake_adapter: FakeMoabbDataset,
    capsys: pytest.CaptureFixture[str],
) -> None:
    base = _base(catalog_dir, tmp_path)
    out = tmp_path / "audit.md"
    assert (
        main([*base, "data", "audit", "physionet_mi", "--max-subjects", "2", "--out", str(out)])
        == 0
    )
    assert "| physionet_mi | 2 |" in out.read_text()
    details = tmp_path / "qa.json"
    code = main([*base, "data", "qa", "physionet_mi", "--subjects", "1", "--out", str(details)])
    assert code == 0
    rows = json.loads(details.read_text())
    assert len(rows) == 3  # three runs for subject 1
    assert rows[0]["dropped_labels"] == {"weird": 2}  # 10 trials cycling 4 codes
    assert "recordings flagged" in capsys.readouterr().out


def test_errors_are_reported_not_raised(
    catalog_dir: Path,
    tmp_path: Path,
    fake_adapter: FakeMoabbDataset,
    capsys: pytest.CaptureFixture[str],
) -> None:
    base = _base(catalog_dir, tmp_path)
    assert main([*base, "data", "qa", "not_a_dataset"]) == 2
    assert main([*base, "data", "qa", "physionet_mi", "--subjects", "99"]) == 2
    assert "unknown subjects" in capsys.readouterr().err
