from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neurolayer.core.interfaces import Decoder
from neurolayer.core.positions import POSITIONS
from neurolayer.data.synthetic import SyntheticMIConfig, generate_synthetic_mi
from neurolayer.evaluation.protocol import ProtocolConfig, run_protocol
from neurolayer.models.registry import make_factory
from neurolayer.representation.spatial_field import electrode_positions, sinc_filter_bank

torch = pytest.importorskip("torch")

from neurolayer.models.bundle import export_bundle, load_bundle, read_bundle_info  # noqa: E402
from neurolayer.models.proprietary import SpatialFieldDecoder  # noqa: E402

FAST = {"epochs": 8, "patience": 8, "finetune_epochs": 5, "device": "cpu"}


@pytest.fixture(scope="module")
def easy():  # type: ignore[no-untyped-def]
    cfg = SyntheticMIConfig(n_subjects=5, n_trials_per_class=40, seed=9, efficiencies=(0.9,) * 5)
    return generate_synthetic_mi(cfg).epochs


def test_positions_table_is_geometrically_sane() -> None:
    assert len(POSITIONS) >= 85
    c3, cz, c4 = electrode_positions(["C3", "Cz", "C4"])
    assert c3[0] < 0 < c4[0]  # left / right hemisphere
    assert cz[2] > 0.9  # vertex on top
    np.testing.assert_allclose(
        np.linalg.norm(electrode_positions(["Fp1", "O2"]), axis=1), 1.0, atol=1e-3
    )
    with pytest.raises(KeyError, match="no scalp position"):
        electrode_positions(["C3", "EXG1"])


def test_sinc_bank_is_bandpass() -> None:
    bank = sinc_filter_bank(128.0, [(8.0, 12.0)], 33)
    freqs = np.fft.rfftfreq(512, 1 / 128.0)
    response = np.abs(np.fft.rfft(bank[0], 512))
    assert response[np.argmin(np.abs(freqs - 10))] > 5 * response[np.argmin(np.abs(freqs - 30))]


def test_decoder_learns_and_is_montage_agnostic(easy) -> None:  # type: ignore[no-untyped-def]
    assert isinstance(SpatialFieldDecoder(), Decoder)
    config = ProtocolConfig(ks=(0, 10), n_unlabeled=10, n_folds=None, n_bootstrap=100)
    curve = {
        s.k: s.mean_ba
        for s in run_protocol(easy, make_factory("nl_spatial_field", FAST), config).per_budget()
    }
    assert curve[0] > 0.7
    # A 3-channel target montage works without retraining or source_montage.
    r3 = ProtocolConfig(
        ks=(0,), n_unlabeled=10, n_folds=None, n_bootstrap=100, target_montage=("C3", "Cz", "C4")
    )
    assert (
        run_protocol(easy, make_factory("nl_spatial_field", FAST), r3).per_budget()[0].mean_ba > 0.6
    )


def test_online_alignment_is_causal_and_probabilities_normalized(easy) -> None:  # type: ignore[no-untyped-def]
    source = easy.subset(easy.subject != "sub-000")
    target = easy.for_subject("synthetic_mi", "sub-000")
    decoder = SpatialFieldDecoder(online_alignment=True, **FAST)
    decoder.fit(source)
    test = target.without_labels()
    first = decoder.predict(test)
    # Causality: predictions for the first half do not change when later trials are removed.
    half = test.subset(np.argsort(test.order)[: len(test) // 2])
    np.testing.assert_array_equal(
        decoder.predict(half), first[np.argsort(test.order)[: len(test) // 2]]
    )
    offline = SpatialFieldDecoder(**FAST)
    offline.fit(source)
    adapted = offline.adapt(None, target.subset(np.arange(10)).without_labels())
    proba = adapted.predict_proba(test)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert adapted.embed(source).shape == (len(source), 8 * 6)


def test_errors(easy) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError, match="channel_keep"):
        SpatialFieldDecoder(channel_keep=0.0)
    with pytest.raises(RuntimeError, match="before fit"):
        SpatialFieldDecoder().predict(easy)


def test_bundle_round_trip_and_tamper_detection(easy, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    decoder = SpatialFieldDecoder(**FAST)
    decoder.fit(easy)
    info = export_bundle(
        decoder,
        tmp_path / "m" / "0.1.0",
        name="m",
        version="0.1.0",
        label_names=easy.label_names,
        sfreq=easy.sfreq,
        n_times=easy.n_times,
        trained_channels=easy.ch_names,
        preprocessing={"synthetic_only": True},
        provenance={"test": True},
    )
    assert info.model_id == "m@0.1.0"
    assert read_bundle_info(tmp_path / "m" / "0.1.0").weights_sha256 == info.weights_sha256
    loaded, _ = load_bundle(tmp_path / "m" / "0.1.0")
    test = easy.without_labels()
    np.testing.assert_allclose(loaded.predict_proba(test), decoder.predict_proba(test), atol=1e-5)
    weights = tmp_path / "m" / "0.1.0" / "weights.safetensors"
    weights.write_bytes(weights.read_bytes()[:-8] + b"tampered")
    with pytest.raises(ValueError, match="do not match"):
        load_bundle(tmp_path / "m" / "0.1.0")
    with pytest.raises(FileExistsError):
        export_bundle(
            decoder,
            tmp_path / "m" / "0.1.0",
            name="m",
            version="0.1.0",
            label_names=easy.label_names,
            sfreq=easy.sfreq,
            n_times=easy.n_times,
            trained_channels=easy.ch_names,
            preprocessing={},
        )
