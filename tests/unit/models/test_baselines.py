from __future__ import annotations

import numpy as np
import pytest

from neurolayer.core.channels import MissingChannelsError
from neurolayer.core.interfaces import Decoder
from neurolayer.data.synthetic import SyntheticMIConfig, generate_synthetic_mi
from neurolayer.evaluation.protocol import ProtocolConfig, run_protocol
from neurolayer.models.baselines import (
    PerSubjectCSPLDA,
    PerSubjectTangentSpaceLR,
    PooledRiemannianDecoder,
)
from neurolayer.models.registry import BASELINES, available_decoders, make_factory

pytest.importorskip("pyriemann")

FAST = {"n_bootstrap": 100}


@pytest.fixture(scope="module")
def easy():  # type: ignore[no-untyped-def]
    cfg = SyntheticMIConfig(n_subjects=5, n_trials_per_class=40, seed=2, efficiencies=(0.9,) * 5)
    return generate_synthetic_mi(cfg).epochs


def _curve(epochs, name: str, params: dict | None = None, **protocol):  # type: ignore[no-untyped-def]
    config = ProtocolConfig(ks=(0, 5, 20), n_unlabeled=10, n_folds=None, **(FAST | protocol))
    return {
        s.k: s.mean_ba
        for s in run_protocol(epochs, make_factory(name, params), config).per_budget()
    }


def test_per_subject_baselines_need_calibration(easy) -> None:  # type: ignore[no-untyped-def]
    for name in ("csp_lda_subject", "ts_lr_subject"):
        curve = _curve(easy, name)
        assert curve[0] == 0.5  # no model without calibration: exactly chance
        assert curve[20] > 0.8


def test_pooled_riemannian_transfers_zero_shot_and_on_montage(easy) -> None:  # type: ignore[no-untyped-def]
    curve = _curve(easy, "ts_lr_pooled")
    assert curve[0] > 0.75
    montage = _curve(easy, "ts_lr_pooled", target_montage=("C3", "Cz", "C4"))
    assert montage[20] > 0.65


def test_shuffle_control_is_at_chance(easy) -> None:  # type: ignore[no-untyped-def]
    curve = _curve(easy, "shuffle_control", {"decoder": {"name": "ts_lr_pooled"}, "seed": 1})
    assert all(0.35 < v < 0.65 for v in curve.values())


def test_contracts_and_errors(easy) -> None:  # type: ignore[no-untyped-def]
    for decoder in (PerSubjectCSPLDA(), PerSubjectTangentSpaceLR(), PooledRiemannianDecoder()):
        assert isinstance(decoder, Decoder)
    source = easy.subset(easy.subject != "sub-000")
    target = easy.for_subject("synthetic_mi", "sub-000")
    pooled = PooledRiemannianDecoder()
    with pytest.raises(RuntimeError, match="before fit"):
        pooled.predict(target)
    pooled.fit(source.select_channels(["C3", "C4"]))
    with pytest.raises(MissingChannelsError):
        pooled.predict(target.select_channels(["C3", "Cz"]).without_labels())
    with pytest.raises(ValueError, match="calibration_share"):
        PooledRiemannianDecoder(calibration_share=0.0)
    b1 = PerSubjectCSPLDA().adapt(target.subset(np.arange(20)), None)
    with pytest.raises(ValueError, match="calibration channels"):
        b1.predict(target.select_channels(["C3", "C4"]).without_labels())


def test_registry_lists_baselines_and_wrapper() -> None:
    assert {"csp_lda_subject", "ts_lr_subject", "ts_lr_pooled", "shuffle_control"} <= set(
        available_decoders()
    )
    for name, params in BASELINES.values():
        if name != "braindecode":
            assert isinstance(make_factory(name, params)(), Decoder)
    with pytest.raises(KeyError, match="shuffle_control needs"):
        make_factory("shuffle_control", {})


def test_braindecode_eegnet_runs_and_guards_montage(easy) -> None:  # type: ignore[no-untyped-def]
    pytest.importorskip("torch")
    params = {"architecture": "EEGNet", "train_epochs": 2, "finetune_epochs": 1, "device": "cpu"}
    curve = _curve(easy, "braindecode", params)
    assert set(curve) == {0, 5, 20}
    with pytest.raises(ValueError, match="source_montage"):
        _curve(easy, "braindecode", params, target_montage=("C3", "Cz", "C4"))
    same = _curve(
        easy,
        "braindecode",
        params,
        target_montage=("C3", "Cz", "C4"),
        source_montage=("C3", "Cz", "C4"),
    )
    assert set(same) == {0, 5, 20}
    with pytest.raises(ValueError, match="model_card"):
        make_factory("braindecode", params | {"weights": "w.safetensors"})().fit(easy)
