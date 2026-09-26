from __future__ import annotations

import pytest

from neurolayer.data.synthetic import SyntheticMIConfig, generate_synthetic_mi
from neurolayer.evaluation.protocol import ProtocolConfig
from neurolayer.experiments.config import DecoderSpec
from neurolayer.experiments.controls import ShuffleControlResult, run_shuffle_control


def test_multi_seed_shuffle_control_passes_on_clean_pipeline() -> None:
    ep = generate_synthetic_mi(
        SyntheticMIConfig(n_subjects=5, n_trials_per_class=30, seed=4)
    ).epochs
    config = ProtocolConfig(ks=(0, 10), n_unlabeled=5, n_folds=None, n_bootstrap=100)
    result = run_shuffle_control(ep, DecoderSpec(name="logvar_logreg"), config, seeds=[0, 1, 2, 3])
    assert result.ks == (0, 10)
    assert len(result.per_seed) == 4
    assert all(0.4 < m < 0.6 for m in result.mean)
    with pytest.raises(ValueError, match="at least two seeds"):
        run_shuffle_control(ep, DecoderSpec(name="logvar_logreg"), config, seeds=[0])


def test_pass_rule() -> None:
    chance = ShuffleControlResult(ks=(0,), per_seed=((0.49,), (0.51,), (0.50,)))
    assert chance.passed
    leaky = ShuffleControlResult(ks=(0,), per_seed=((0.70,), (0.72,), (0.71,)))
    assert not leaky.passed
