from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "license_report.py"


@pytest.fixture(scope="module")
def report() -> ModuleType:
    spec = importlib.util.spec_from_file_location("license_report", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their module via sys.modules
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("expression", "declared", "classifiers", "verdict"),
    [
        ("MIT", "", [], "ok"),
        ("BSD-3-Clause", "", [], "ok"),
        ("GPL-3.0-or-later", "", [], "forbidden"),
        ("", "AGPL-3.0", [], "forbidden"),
        ("", "", ["License :: OSI Approved :: GNU General Public License v3 (GPLv3)"], "forbidden"),
        ("LGPL-2.1-only", "", [], "lgpl"),
        (
            "",
            "",
            ["License :: OSI Approved :: GNU Lesser General Public License v2 (LGPLv2)"],
            "lgpl",
        ),
        ("SSPL-1.0", "", [], "forbidden"),
        ("MPL-2.0", "", [], "ok"),
        ("", "", [], "unknown"),
    ],
)
def test_classify(
    report: ModuleType, expression: str, declared: str, classifiers: list[str], verdict: str
) -> None:
    assert report.classify("pkg", expression, declared, classifiers) == verdict


def test_parse_requirements(report: ModuleType) -> None:
    lines = [
        "# comment",
        "numpy==2.5.3",
        "torch==2.14.0 ; sys_platform == 'linux'",
        "uvicorn[standard]==0.54.0",
        "-e .",
        "",
    ]
    assert report.parse_requirements(lines) == ["numpy", "torch", "uvicorn"]


def test_main_on_installed_packages(report: ModuleType, tmp_path: Path) -> None:
    reqs = tmp_path / "reqs.txt"
    reqs.write_text("numpy==0\npydantic==0\n")
    assert report.main([str(reqs)]) == 0
    assert report.main([]) == 2
