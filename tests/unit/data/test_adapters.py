from __future__ import annotations

from pathlib import Path

import pytest

from neurolayer.data.adapters import AdapterNotAvailableError, get_adapter, pseudonymize
from neurolayer.data.adapters.base import DatasetAdapter
from neurolayer.data.adapters.moabb import MoabbAdapter
from neurolayer.data.catalog import load_catalog
from tests.fakes import FakeMoabbDataset

pytest.importorskip("mne")


def test_pseudonymize() -> None:
    assert pseudonymize(7) == "sub-007"
    assert pseudonymize("12") == "sub-012"
    with pytest.raises(ValueError, match="numeric"):
        pseudonymize("alice")


def test_moabb_adapter_yields_canonical_recordings(catalog_dir: Path, tmp_path: Path) -> None:
    card = load_catalog(catalog_dir)["physionet_mi"]
    fake = FakeMoabbDataset()
    adapter = MoabbAdapter(card, fake, data_dir=tmp_path)
    assert isinstance(adapter, DatasetAdapter)
    assert adapter.subjects() == ["1", "2"]
    assert adapter.cue_offset_s == 0.5
    assert adapter.max_trial_s == 3.0
    recordings = list(adapter.recordings("2"))
    # Sessions and runs come out in sorted (chronological) order.
    assert [(r.session_id, r.run_id) for r in recordings] == [
        ("0session", "0run"),
        ("1session", "0run"),
        ("1session", "1run"),
    ]
    assert {r.subject_id for r in recordings} == {"sub-002"}
    assert all(r.dataset_id == "physionet_mi" for r in recordings)
    assert "C3" in recordings[0].ch_names
    assert len(adapter.log.reports) == 3
    adapter.download(["1"])
    assert fake.downloads == [([1], str(tmp_path))]


def test_adapter_refuses_excluded_dataset(catalog_dir: Path) -> None:
    card = load_catalog(catalog_dir)["meta_emg_generic"]
    with pytest.raises(PermissionError, match="excluded"):
        MoabbAdapter(card, FakeMoabbDataset())


def test_get_adapter_dispatch(catalog_dir: Path) -> None:
    catalog = load_catalog(catalog_dir)
    with pytest.raises(AdapterNotAvailableError, match="no loader"):
        get_adapter(catalog["tuh_eeg"])
    with pytest.raises(AdapterNotAvailableError, match="no adapter"):
        get_adapter(catalog["synthetic_mi"])


def test_get_adapter_builds_real_moabb_class(catalog_dir: Path) -> None:
    pytest.importorskip("moabb")
    adapter = get_adapter(load_catalog(catalog_dir)["physionet_mi"])
    assert isinstance(adapter, MoabbAdapter)
    assert adapter.subjects()[:3] == ["1", "2", "3"]


@pytest.mark.network
def test_physionet_subject_1_end_to_end(catalog_dir: Path, tmp_path: Path) -> None:
    """Opt-in (`pytest -m network`): downloads PhysioNet subject 1 (~20 MB)."""
    adapter = get_adapter(load_catalog(catalog_dir)["physionet_mi"], tmp_path)
    recordings = list(adapter.recordings("1"))
    assert recordings
    assert all(r.sfreq == 160.0 for r in recordings)
    labels = {e.label for r in recordings for e in r.events}
    assert {"left_hand", "right_hand"} <= labels
