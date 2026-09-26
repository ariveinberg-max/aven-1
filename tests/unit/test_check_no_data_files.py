from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_no_data_files.py"


@pytest.fixture(scope="module")
def guard() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_no_data_files", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "path",
    [
        "S001R04.edf",
        "sub-01_eeg.BDF",
        "x/y/model.safetensors",
        "weights.pt",
        "cache.npz",
        "data/raw/physionet/readme.txt",
        "artifacts/runs/r/manifest.json",
        "notebook_dump.PKL",
        "archive.tar.npy",
    ],
)
def test_blocks_data_and_weights(guard: ModuleType, path: str) -> None:
    assert guard.blocked([path]) == [path]


@pytest.mark.parametrize(
    "path",
    [
        "src/neurolayer/core/types.py",
        "catalog/datasets/physionet_mi.yaml",
        "data/README.md",
        "data/raw/physionet_mi/1.0.0.dvc",  # DVC pointer files must be committed
        "data/raw/.gitignore",
    ],
)
def test_allows_code_and_docs(guard: ModuleType, path: str) -> None:
    assert guard.blocked([path]) == []


def test_main_exit_codes(guard: ModuleType, capsys: pytest.CaptureFixture[str]) -> None:
    assert guard.main(["README.md"]) == 0
    assert guard.main(["README.md", "x.fif"]) == 1
    assert "x.fif" in capsys.readouterr().out
