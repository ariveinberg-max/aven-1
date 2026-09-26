"""Name → decoder factory registry used by experiment configs.

Entries are import paths resolved lazily, so heavy dependencies (PyTorch) load only
when a config actually asks for a deep model. Every model reachable from a YAML config
runs through the same manifest-backed protocol.

CAP-1 baseline suite (spec §4):

====  =====================================  ===================================
B0    ``chance``                              chance level
B1    ``csp_lda_subject``                     CSP + LDA, calibration only
B2    ``ts_lr_subject``                       tangent space + LR, calibration only
B3    ``ts_lr_pooled``                        pooled tangent space, re-centering
B4    ``braindecode`` (``EEGNet``)            pooled EEGNet + fine-tuning
B5    MIRepNet                                vendoring pending (WP-3.4 follow-up)
B6    ``braindecode`` (``Labram``/``CBraMod``)  foundation model, pinned weights
====  =====================================  ===================================

``shuffle_control`` wraps any decoder: ``{name: shuffle_control, params: {decoder:
{name: ts_lr_pooled}, seed: 0}}``.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from typing import Any

from neurolayer.core.interfaces import Decoder

DecoderFactory = Callable[[], Decoder]

_REGISTRY: dict[str, str] = {
    "chance": "neurolayer.models.reference:ChanceDecoder",
    "logvar_logreg": "neurolayer.models.reference:LogVarianceDecoder",
    "csp_lda_subject": "neurolayer.models.baselines:PerSubjectCSPLDA",
    "ts_lr_subject": "neurolayer.models.baselines:PerSubjectTangentSpaceLR",
    "ts_lr_pooled": "neurolayer.models.baselines:PooledRiemannianDecoder",
    "braindecode": "neurolayer.models.deep:BraindecodeDecoder",
}

BASELINES: dict[str, tuple[str, dict[str, Any]]] = {
    "B0": ("chance", {}),
    "B1": ("csp_lda_subject", {}),
    "B2": ("ts_lr_subject", {}),
    "B3": ("ts_lr_pooled", {}),
    "B4": ("braindecode", {"architecture": "EEGNet"}),
}
"""Baselines runnable without third-party weights (B5/B6 need pinned weights)."""


def available_decoders() -> list[str]:
    """Names of all registered decoders (plus the ``shuffle_control`` wrapper)."""
    return sorted([*_REGISTRY, "shuffle_control"])


def _resolve(name: str) -> Callable[..., Decoder]:
    module_name, _, attribute = _REGISTRY[name].partition(":")
    constructor: Callable[..., Decoder] = getattr(importlib.import_module(module_name), attribute)
    return constructor


def make_factory(name: str, params: Mapping[str, Any] | None = None) -> DecoderFactory:
    """Return a zero-argument factory building a fresh decoder for each fold.

    Raises
    ------
    KeyError
        If ``name`` (or a wrapped decoder's name) is not registered.
    """
    kwargs = dict(params or {})
    if name == "shuffle_control":
        inner_spec = kwargs.pop("decoder", None)
        if not isinstance(inner_spec, Mapping) or "name" not in inner_spec:
            raise KeyError("shuffle_control needs params.decoder = {name: ..., params: {...}}")
        inner_factory = make_factory(str(inner_spec["name"]), inner_spec.get("params"))
        seed = int(kwargs.pop("seed", 0))
        from neurolayer.models.controls import ShuffleControlDecoder

        def wrapped() -> Decoder:
            return ShuffleControlDecoder(inner_factory(), seed=seed)

        return wrapped
    if name not in _REGISTRY:
        raise KeyError(f"unknown decoder {name!r}; available: {available_decoders()}")
    constructor = _resolve(name)

    def factory() -> Decoder:
        return constructor(**kwargs)

    return factory
