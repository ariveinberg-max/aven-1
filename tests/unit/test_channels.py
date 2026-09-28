from __future__ import annotations

import pytest

from neurolayer.core.channels import (
    CONSUMER_MONTAGES,
    MissingChannelsError,
    Montage,
    channel_indices,
    is_standard,
    normalize_channel_name,
    normalize_channel_names,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Fc5.", "FC5"),  # PhysioNet trailing dots + case
        ("Cz..", "Cz"),
        ("Fpz.", "Fpz"),
        ("EEG-C3", "C3"),  # BCI Competition prefix
        ("EEG FP1-REF", "Fp1"),  # TUH prefix + reference suffix
        ("EEG T3-REF", "T7"),  # TUH legacy name
        ("T4", "T8"),
        ("t5", "P7"),
        ("POZ", "POz"),
        ("  cp4  ", "CP4"),
        ("C3-C4", "C3-C4"),  # bipolar derivation left alone
        ("EEG-0", "0"),  # unknown kept (flagged by is_standard)
    ],
)
def test_normalize_channel_name(raw: str, expected: str) -> None:
    assert normalize_channel_name(raw) == expected


def test_is_standard() -> None:
    assert is_standard("FCz")
    assert not is_standard("FCZ")
    assert not is_standard("T3")


def test_normalize_names_rejects_collisions() -> None:
    with pytest.raises(ValueError, match="collide"):
        normalize_channel_names(["C3", "c3."])


def test_channel_indices_never_substitutes() -> None:
    assert channel_indices(["C3", "Cz", "C4"], ["C4", "C3"]) == [2, 0]
    with pytest.raises(MissingChannelsError) as info:
        channel_indices(["C3", "C4"], ["C3", "TP9"])
    assert info.value.missing == ("TP9",)


def test_consumer_montages_are_canonical() -> None:
    crown = CONSUMER_MONTAGES["neurosity_crown"]
    assert len(crown.channels) == 8
    assert {"C3", "C4", "CP3", "CP4"} <= set(crown.channels)
    assert CONSUMER_MONTAGES["muse_s"].channels == ("TP9", "AF7", "AF8", "TP10")
    for montage in CONSUMER_MONTAGES.values():
        assert all(is_standard(c) for c in montage.channels)


def test_montage_validation() -> None:
    with pytest.raises(ValueError, match="non-canonical"):
        Montage(name="bad", channels=("C3", "FCZ"), kind="research", description="")
    with pytest.raises(ValueError, match="duplicate"):
        Montage(name="dup", channels=("C3", "C3"), kind="research", description="")


def test_analysis_montages_split_the_crown() -> None:
    crown = set(CONSUMER_MONTAGES["neurosity_crown"].channels)
    motor = CONSUMER_MONTAGES["neurosity_crown_motor"]
    nonmotor = CONSUMER_MONTAGES["neurosity_crown_nonmotor"]
    assert motor.kind == nonmotor.kind == "analysis"
    assert set(motor.channels) | set(nonmotor.channels) == crown
    assert not set(motor.channels) & set(nonmotor.channels)
