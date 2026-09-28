from __future__ import annotations

from pathlib import Path

import pytest

from neurolayer.cli import main

pytest.importorskip("torch")


def _config(tmp_path: Path, datasets: str) -> Path:
    path = tmp_path / "train.yaml"
    path.write_text(
        f"""name: tiny-model
purpose: training
datasets: {datasets}
synthetic: {{n_subjects: 3, n_trials_per_class: 20}}
decoder: {{name: nl_spatial_field, params: {{epochs: 3, device: cpu}}}}
"""
    )
    return path


def test_train_exports_bundle(
    catalog_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base = ["--catalog", str(catalog_dir)]
    models = tmp_path / "models"
    cmd = [
        *base,
        "train",
        str(_config(tmp_path, "[synthetic_mi]")),
        "--version",
        "0.0.1",
        "--models-dir",
        str(models),
    ]
    assert main(cmd) == 0
    assert (models / "tiny-model" / "0.0.1" / "bundle.json").exists()
    assert "sha256" in capsys.readouterr().out
    assert main(cmd) == 2  # versions are immutable


def test_train_requires_training_license(
    unverified_catalog: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cmd = [
        "--catalog",
        str(unverified_catalog),
        "train",
        str(_config(tmp_path, "[physionet_mi]")),
        "--version",
        "0.0.1",
        "--models-dir",
        str(tmp_path / "m"),
    ]
    assert main(cmd) == 2
    assert "license gate" in capsys.readouterr().err
