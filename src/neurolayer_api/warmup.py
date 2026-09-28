"""Startup warm-up: pay first-request costs before the first user does.

The first decode after startup was measured at about 3 s (weight loading, first PyTorch
forward pass, lazy SciPy imports), and the first demo-data request at 1.2 s, versus
milliseconds afterwards. Under CPU contention the first demo request once took 56 s.
Warming up runs each model once through the real inference path (microvolt window →
preprocessing → forward pass) on seeded noise. No stored or user data is involved.
"""

from __future__ import annotations

import time

import numpy as np

from neurolayer.data.synthetic import SyntheticMIConfig, generate_synthetic_mi
from neurolayer_api.inference import to_epochs
from neurolayer_api.models import ModelRegistry


def warm_up(registry: ModelRegistry, demo: bool) -> dict[str, float]:
    """Load every model and run one forward pass; returns seconds per step."""
    timings: dict[str, float] = {}
    rng = np.random.default_rng(0)
    for model_id, info in registry.available().items():
        started = time.perf_counter()
        model = registry.get(model_id)
        window = rng.standard_normal((len(info.trained_channels), info.n_times)) * 10.0  # µV
        epochs = to_epochs([window], None, info.trained_channels, info.sfreq, model.info)
        model.decoder.predict_proba(epochs)
        timings[model_id] = round(time.perf_counter() - started, 3)
    if demo:
        started = time.perf_counter()
        generate_synthetic_mi(SyntheticMIConfig(n_subjects=1, n_trials_per_class=1, seed=0))
        timings["demo_generator"] = round(time.perf_counter() - started, 3)
    return timings
