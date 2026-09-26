"""Name → encoder factory registry (used by leakage probes and configs)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from neurolayer.core.interfaces import Encoder
from neurolayer.representation.covariance import (
    CSPEncoder,
    LogVarianceEncoder,
    TangentSpaceEncoder,
)

ENCODERS: dict[str, Callable[..., Encoder]] = {
    "tangent_space": TangentSpaceEncoder,
    "csp": CSPEncoder,
    "log_variance": LogVarianceEncoder,
}


def make_encoder(name: str, params: dict[str, Any] | None = None) -> Encoder:
    """Instantiate a registered encoder."""
    if name not in ENCODERS:
        raise KeyError(f"unknown encoder {name!r}; available: {sorted(ENCODERS)}")
    return ENCODERS[name](**(params or {}))
