from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

from neurolayer.experiments.config import load_experiment_config

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = sorted((ROOT / "configs" / "experiments").rglob("*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: str(p.relative_to(ROOT)))
def test_every_config_loads(path: Path) -> None:
    load_experiment_config(path)


def test_baseline_configs_match_their_generator() -> None:
    script = ROOT / "scripts" / "make_baseline_configs.py"
    spec = importlib.util.spec_from_file_location("make_baseline_configs", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    for pool in module.POOLS:
        for baseline in module.BASELINES:
            for regime in module.regimes_for(pool):
                path = module.OUT / f"{pool}_{baseline}_{regime}.yaml"
                committed = yaml.safe_load(path.read_text(encoding="utf-8"))
                assert committed == module.config_for(pool, baseline, regime), path.name
