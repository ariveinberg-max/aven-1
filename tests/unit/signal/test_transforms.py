from __future__ import annotations

import numpy as np
import pytest

from neurolayer.core.channels import MissingChannelsError
from neurolayer.core.types import Event, Recording
from neurolayer.signal.transforms import (
    Bandpass,
    Notch,
    Rereference,
    Resample,
    SelectChannels,
    Transform,
    TransformSpec,
    UnitCheck,
)

SFREQ = 256.0


def _sines(freqs: list[float], seconds: float = 8.0, amplitude: float = 1e-5) -> Recording:
    t = np.arange(int(seconds * SFREQ)) / SFREQ
    data = np.stack([amplitude * np.sin(2 * np.pi * f * t) for f in freqs])
    names = ("C3", "Cz", "C4", "Pz")[: len(freqs)]
    return Recording(
        data=data,
        sfreq=SFREQ,
        ch_names=names,
        dataset_id="d",
        subject_id="s",
        events=(Event(512, "left_hand"), Event(1024, "right_hand")),
    )


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x**2)))


def _middle(x: np.ndarray) -> np.ndarray:
    return x[..., x.shape[-1] // 4 : -x.shape[-1] // 4]  # ignore filter edge effects


def test_bandpass_passes_band_and_attenuates_outside() -> None:
    rec = _sines([10.0, 60.0, 0.3])
    out = Bandpass(1.0, 40.0)(rec)
    assert isinstance(Bandpass(), Transform)
    passed = _rms(_middle(out.data[0])) / _rms(_middle(rec.data[0]))
    stop_hi = _rms(_middle(out.data[1])) / _rms(_middle(rec.data[1]))
    stop_lo = _rms(_middle(out.data[2])) / _rms(_middle(rec.data[2]))
    assert passed == pytest.approx(1.0, abs=0.02)
    assert 20 * np.log10(stop_hi) < -30  # ≥ 30 dB down at 60 Hz
    assert 20 * np.log10(stop_lo) < -20
    # Zero phase: the 10 Hz sine is not shifted in time.
    lag = np.argmax(np.correlate(_middle(out.data[0]), _middle(rec.data[0]), "full"))
    assert lag == len(_middle(rec.data[0])) - 1


def test_bandpass_validation_and_purity() -> None:
    rec = _sines([10.0])
    before = rec.data.copy()
    Bandpass(1.0, 40.0)(rec)
    np.testing.assert_array_equal(rec.data, before)
    with pytest.raises(ValueError, match="Nyquist"):
        Bandpass(1.0, 200.0)(rec)
    with pytest.raises(ValueError, match="below h_freq"):
        Bandpass(40.0, 1.0)
    with pytest.raises(ValueError, match="needs"):
        Bandpass(None, None)
    assert Bandpass(None, 30.0)(rec).data.shape == rec.data.shape
    assert Bandpass(1.0, None)(rec).data.shape == rec.data.shape


def test_notch_removes_line_frequency_only() -> None:
    rec = _sines([50.0, 10.0])
    out = Notch(freqs=(50.0, 1000.0))(rec)  # frequencies above Nyquist are ignored
    assert _rms(_middle(out.data[0])) < 0.05 * _rms(_middle(rec.data[0]))
    assert _rms(_middle(out.data[1])) == pytest.approx(_rms(_middle(rec.data[1])), rel=0.02)


def test_resample_rescales_samples_and_events() -> None:
    rec = _sines([5.0])
    out = Resample(128.0)(rec)
    assert out.sfreq == 128.0
    assert out.n_samples == rec.n_samples // 2
    assert [e.onset_sample for e in out.events] == [256, 512]
    assert Resample(SFREQ)(rec) is rec  # no-op returns the input


def test_rereference() -> None:
    rec = _sines([5.0, 7.0, 9.0])
    avg = Rereference()(rec)
    np.testing.assert_allclose(avg.data.mean(axis=0), 0.0, atol=1e-18)
    to_cz = Rereference(kind="channels", channels=("Cz",))(rec)
    np.testing.assert_allclose(to_cz.data[1], 0.0)
    with pytest.raises(ValueError, match="at least one"):
        Rereference(kind="channels")(rec)


def test_select_channels() -> None:
    rec = _sines([5.0, 7.0, 9.0, 11.0])
    out = SelectChannels(channels=("C4", "C3"))(rec)
    assert out.ch_names == ("C4", "C3")
    np.testing.assert_array_equal(out.data[0], rec.data[2])
    with pytest.raises(MissingChannelsError):
        SelectChannels(montage="neurosity_crown")(rec)
    with pytest.raises(ValueError, match="exactly one"):
        SelectChannels()
    with pytest.raises(ValueError, match="unknown montage"):
        SelectChannels(montage="nope")


def test_unit_check_catches_microvolts_as_volts() -> None:
    assert UnitCheck()(_sines([5.0])) is not None
    with pytest.raises(ValueError, match="implausible"):
        UnitCheck()(_sines([5.0], amplitude=10.0))


def test_transform_spec_from_yaml_like_dict() -> None:
    transform = TransformSpec("notch", {"freqs": [50, 60]}).build()
    assert transform.config() == {"freqs": [50, 60], "quality": 30.0}
    with pytest.raises(KeyError, match="unknown transform"):
        TransformSpec("wavelet_magic").build()
