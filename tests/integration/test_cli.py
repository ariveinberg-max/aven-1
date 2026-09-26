from __future__ import annotations

from pathlib import Path

import pytest

from neurolayer.cli import main


def test_catalog_list(catalog_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--catalog", str(catalog_dir), "catalog", "list"]) == 0
    out = capsys.readouterr().out
    assert "physionet_mi" in out
    assert "blocked" in out


def test_catalog_check_exit_codes(catalog_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ok = ["--catalog", str(catalog_dir), "catalog", "check", "--purpose", "training"]
    assert main([*ok, "--datasets", "synthetic_mi"]) == 0
    assert main([*ok, "--datasets", "synthetic_mi,bnci2014_001"]) == 1
    assert main([*ok, "--datasets", "unknown_ds"]) == 1
    assert "REFUSED" in capsys.readouterr().out


def test_run_and_smoke(
    repo_root: Path,
    catalog_dir: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(repo_root)
    base = ["--catalog", str(catalog_dir), "--output", str(tmp_path)]
    assert main([*base, "smoke"]) == 0
    out = capsys.readouterr().out
    assert "AUCEC" in out
    assert "mean BA" in out
    blocked = main([*base, "run", "configs/experiments/cap1_r1_physionet_template.yaml"])
    assert blocked == 2
    assert "license gate" in capsys.readouterr().err
