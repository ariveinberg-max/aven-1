from __future__ import annotations

import numpy as np
import pytest

from neurolayer.core.channels import MissingChannelsError
from neurolayer.core.types import EpochSet
from neurolayer.models.reference import ChanceDecoder, LogVarianceDecoder, log_variance
from neurolayer.models.registry import available_decoders, make_factory


def _split(epochs: EpochSet) -> tuple[EpochSet, EpochSet]:
    target_mask = epochs.subject == "sub-000"
    return epochs.subset(~target_mask), epochs.subset(target_mask)


def test_log_variance_shape(epochs: EpochSet) -> None:
    assert log_variance(epochs.X).shape == (len(epochs), epochs.n_channels)


def test_logvar_predicts_on_target_montage_subset(epochs: EpochSet) -> None:
    source, target = _split(epochs)
    decoder = LogVarianceDecoder()
    decoder.fit(source)
    restricted = target.select_channels(["C3", "Cz", "C4"])
    calibration = restricted.subset(np.arange(10))
    adapted = decoder.adapt(calibration, None)
    predictions = adapted.predict(
        restricted.subset(np.arange(10, len(restricted))).without_labels()
    )
    assert predictions.dtype == np.int64
    assert set(np.unique(predictions)) <= {0, 1}


def test_logvar_requires_target_channels_in_source(epochs: EpochSet) -> None:
    source, target = _split(epochs)
    decoder = LogVarianceDecoder()
    decoder.fit(source.select_channels(["C3", "C4"]))
    with pytest.raises(MissingChannelsError, match="Cz"):
        decoder.predict(target.select_channels(["C3", "Cz"]).without_labels())


def test_logvar_rejects_mixed_channel_sets(epochs: EpochSet) -> None:
    source, target = _split(epochs)
    decoder = LogVarianceDecoder()
    decoder.fit(source)
    adapted = decoder.adapt(target.select_channels(["C3", "C4"]).subset(np.arange(4)), None)
    with pytest.raises(ValueError, match="same channels"):
        adapted.predict(target.without_labels())


def test_decoders_require_fit(epochs: EpochSet) -> None:
    with pytest.raises(RuntimeError, match="before fit"):
        LogVarianceDecoder().predict(epochs)
    with pytest.raises(RuntimeError, match="before fit"):
        ChanceDecoder().predict(epochs)
    with pytest.raises(ValueError, match="calibration_share"):
        LogVarianceDecoder(calibration_share=1.0)


def test_registry() -> None:
    assert {"chance", "logvar_logreg"} <= set(available_decoders())
    decoder = make_factory("logvar_logreg", {"C": 0.5})()
    assert isinstance(decoder, LogVarianceDecoder)
    assert decoder.C == 0.5
    with pytest.raises(KeyError, match="unknown decoder"):
        make_factory("does_not_exist")
