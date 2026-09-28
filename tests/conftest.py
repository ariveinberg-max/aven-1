from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from neurolayer.core.types import EpochSet
from neurolayer.data.synthetic import SyntheticMIConfig, SyntheticMIResult, generate_synthetic_mi

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def catalog_dir() -> Path:
    return REPO_ROOT / "catalog" / "datasets"


@pytest.fixture(scope="session")
def synthetic() -> SyntheticMIResult:
    """Six subjects, 40 trials per class: small enough to keep the suite fast."""
    return generate_synthetic_mi(SyntheticMIConfig(n_subjects=6, n_trials_per_class=40, seed=3))


@pytest.fixture(scope="session")
def epochs(synthetic: SyntheticMIResult) -> EpochSet:
    return synthetic.epochs


@pytest.fixture
def unverified_catalog(tmp_path: Path) -> Path:
    """Copy of the catalog in which ``physionet_mi``'s license is not yet verified.

    License-gate tests use it so they do not depend on which real datasets a human
    has verified so far.
    """
    target = tmp_path / "catalog"
    shutil.copytree(REPO_ROOT / "catalog" / "datasets", target)
    card = target / "physionet_mi.yaml"
    raw = yaml.safe_load(card.read_text(encoding="utf-8"))
    raw["license"]["verified_by"] = None
    raw["license"]["verified_on"] = None
    card.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return target
