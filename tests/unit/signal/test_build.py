from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from neurolayer.data.adapters.moabb import MoabbAdapter
from neurolayer.data.catalog import DatasetCard, load_catalog
from neurolayer.signal.build import build_epochs
from neurolayer.signal.pipeline import PipelineSpec
from tests.fakes import FakeMoabbDataset

pytest.importorskip("mne")


@pytest.fixture
def fake_adapters(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    calls: dict[str, int] = {}

    def build(card: DatasetCard, data_dir: Path | None = None) -> Any:
        calls[card.id] = calls.get(card.id, 0) + 1
        return MoabbAdapter(card, FakeMoabbDataset(subjects=(1, 2)), data_dir)

    monkeypatch.setattr("neurolayer.signal.build.get_adapter", build)
    return calls


def test_real_dataset_path_with_cache(
    catalog_dir: Path, tmp_path: Path, fake_adapters: dict[str, int]
) -> None:
    catalog = load_catalog(catalog_dir)
    spec = PipelineSpec()
    epochs, report = build_epochs(
        ["physionet_mi"], catalog=catalog, data_root=tmp_path, pipeline=spec
    )
    # 2 subjects x 3 runs x 10 trials cycling 4 codes: 3 left + 3 right per run.
    assert len(epochs) == 2 * 3 * 6
    assert epochs.sfreq == 128.0
    assert set(epochs.subject.tolist()) == {"sub-001", "sub-002"}
    assert report.cache_misses == 2
    assert "physionet_mi/sub-001" in report.epoching
    cached = tmp_path / "processed" / spec.pipeline_hash() / "physionet_mi"
    assert sorted(p.name for p in cached.glob("*.npz")) == [
        "sub-001_epochs.npz",
        "sub-002_epochs.npz",
    ]

    again, report2 = build_epochs(
        ["physionet_mi"], catalog=catalog, data_root=tmp_path, pipeline=spec
    )
    assert report2.cache_hits == 2
    assert again.X.shape == epochs.X.shape


def test_mixing_real_and_synthetic_keeps_common_channels(
    catalog_dir: Path, tmp_path: Path, fake_adapters: dict[str, int]
) -> None:
    catalog = load_catalog(catalog_dir)
    epochs, _ = build_epochs(
        ["physionet_mi", "synthetic_mi"],
        catalog=catalog,
        data_root=tmp_path,
        pipeline=PipelineSpec(),
    )
    assert epochs.datasets() == ["physionet_mi", "synthetic_mi"]
    assert set(epochs.ch_names) == {"FC3", "FCz", "FC4", "C3", "Cz", "C4", "CP3", "CPz", "CP4"}


def test_sampling_rate_mismatch_is_an_error(
    catalog_dir: Path, tmp_path: Path, fake_adapters: dict[str, int]
) -> None:
    catalog = load_catalog(catalog_dir)
    no_resample = PipelineSpec(transforms=())
    with pytest.raises(ValueError, match="sampling rates"):
        build_epochs(
            ["physionet_mi", "synthetic_mi"],
            catalog=catalog,
            data_root=tmp_path,
            pipeline=no_resample,
        )
