"""Name → decoder factory registry used by experiment configs.

Add baselines (WP-4.1) and proprietary models (Stage 5) here, so every model is
reachable from a YAML config and therefore from a reproducible, manifest-backed run.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from neurolayer.core.interfaces import Decoder
from neurolayer.models.reference import ChanceDecoder, LogVarianceDecoder

DecoderFactory = Callable[[], Decoder]

_REGISTRY: dict[str, Callable[..., Decoder]] = {
    "chance": ChanceDecoder,
    "logvar_logreg": LogVarianceDecoder,
}


def available_decoders() -> list[str]:
    """Names of all registered decoders."""
    return sorted(_REGISTRY)


def make_factory(name: str, params: Mapping[str, Any] | None = None) -> DecoderFactory:
    """Return a zero-argument factory building a fresh decoder for each fold.

    Raises
    ------
    KeyError
        If ``name`` is not registered.
    """
    if name not in _REGISTRY:
        raise KeyError(f"unknown decoder {name!r}; available: {available_decoders()}")
    constructor = _REGISTRY[name]
    kwargs = dict(params or {})

    def factory() -> Decoder:
        return constructor(**kwargs)

    return factory
