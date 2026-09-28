from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neurolayer.data.catalog import load_catalog
from neurolayer.data.storage import (
    bids_root,
    processed_dir,
    raw_dir,
    read_bids,
    verify_checksums,
    write_bids,
    write_checksums,
)
from neurolayer.data.synthetic import SyntheticMIConfig, generate_synthetic_recordings


def test_layout(catalog_dir: Path) -> None:
    card = load_catalog(catalog_dir)["physionet_mi"]
    root = Path("data")
    assert raw_dir(root, card) == Path("data/raw/physionet_mi/1.0.0")
    assert bids_root(root, card) == Path("data/interim/bids/physionet_mi")
    assert processed_dir(root, "abc", "physionet_mi") == Path("data/processed/abc/physionet_mi")


def test_checksums_detect_modified_missing_and_unexpected(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.txt").write_text("alpha")
    (tmp_path / "sub" / "b.txt").write_text("beta")
    write_checksums(tmp_path)
    assert verify_checksums(tmp_path) == []

    (tmp_path / "a.txt").write_text("tampered")
    (tmp_path / "sub" / "b.txt").unlink()
    (tmp_path / "new.txt").write_text("x")
    problems = verify_checksums(tmp_path)
    assert "modified: a.txt" in problems
    assert "missing: sub/b.txt" in problems
    assert "unexpected: new.txt" in problems


def test_checksum_manifest_required(tmp_path: Path) -> None:
    assert verify_checksums(tmp_path) == ["checksums.sha256 missing"]


def test_bids_round_trip(tmp_path: Path) -> None:
    pytest.importorskip("mne_bids")
    pytest.importorskip("edfio")
    recordings = generate_synthetic_recordings(
        SyntheticMIConfig(n_subjects=1, n_trials_per_class=4, n_sessions=2)
    )
    files = write_bids(recordings, tmp_path)
    assert len(files) == 2
    assert all(f.suffix == ".edf" for f in files)
    for original in recordings:
        back = read_bids(tmp_path, original)
        assert back.ch_names == original.ch_names
        assert back.sfreq == original.sfreq
        # EDF stores 16-bit samples; quantization error stays far below 0.01 µV here.
        assert np.abs(back.data - original.data).max() < 1e-8
        assert [(e.onset_sample, e.label) for e in back.events] == [
            (e.onset_sample, e.label) for e in original.events
        ]
