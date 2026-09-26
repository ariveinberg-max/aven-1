"""Canonical types, channel naming and stage interfaces shared by every layer."""

from neurolayer.core.channels import (
    CONSUMER_MONTAGES,
    MissingChannelsError,
    Montage,
    is_standard,
    normalize_channel_name,
)
from neurolayer.core.interfaces import Decoder, Encoder
from neurolayer.core.types import EpochSet, Event, Recording

__all__ = [
    "CONSUMER_MONTAGES",
    "Decoder",
    "Encoder",
    "EpochSet",
    "Event",
    "MissingChannelsError",
    "Montage",
    "Recording",
    "is_standard",
    "normalize_channel_name",
]
