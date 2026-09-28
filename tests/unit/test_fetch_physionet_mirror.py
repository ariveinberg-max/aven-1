from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "fetch_physionet_mirror.py"


@pytest.fixture(scope="module")
def mirror() -> ModuleType:
    spec = importlib.util.spec_from_file_location("fetch_physionet_mirror", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_subjects(mirror: ModuleType) -> None:
    assert mirror.parse_subjects("1-3,7,2") == [1, 2, 3, 7]
    with pytest.raises(ValueError, match="1-109"):
        mirror.parse_subjects("0-3")
    with pytest.raises(ValueError, match="1-109"):
        mirror.parse_subjects("110")


def test_refuses_urls_outside_the_mirror(mirror: ModuleType) -> None:
    with pytest.raises(ValueError, match="outside the mirror"):
        mirror._get("https://example.com/S001/S001R04.edf")


def test_fetch_one_verifies_checksums(
    mirror: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"0       fake edf bytes"
    good = hashlib.sha256(payload).hexdigest()
    calls: list[str] = []

    def fake_get(url: str) -> bytes:
        calls.append(url)
        return payload

    monkeypatch.setattr(mirror, "_get", fake_get)
    target = tmp_path / "S001" / "S001R04.edf"
    assert mirror.fetch_one("S001/S001R04.edf", target, good) == "downloaded"
    assert target.read_bytes() == payload
    assert mirror.fetch_one("S001/S001R04.edf", target, good) == "cached"
    assert len(calls) == 1

    other = tmp_path / "S002" / "S002R04.edf"
    with pytest.raises(ValueError, match="checksum mismatch"):
        mirror.fetch_one("S002/S002R04.edf", other, "0" * 64)
    assert not other.exists()
    assert not other.with_suffix(".edf.part").exists()
