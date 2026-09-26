from __future__ import annotations

from pathlib import Path

import pytest

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
