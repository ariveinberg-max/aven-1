"""Spatial-field network: montage-agnostic learned spatial filtering (proprietary v0).

Classical motor-imagery decoders (CSP, Riemannian methods) learn spatial filters as a
weight per **channel index**, so they break whenever the electrode layout changes. Here
spatial filters are a learned **function of scalp position**::

    w_q,f(electrode) = MLP(xyz(electrode))[q, f]

so the same model applies to any subset or layout of electrodes with known positions
(research caps, consumer headsets) without retraining. Pipeline per trial:

1. Learnable temporal filter bank (initialized as windowed-sinc band-passes).
2. Position-defined spatial filters, re-normalized over the channels present.
3. Log-power of each (spatial filter, band) "virtual channel" feeds a linear head.

Training uses random channel dropout (robustness to missing electrodes) and an optional
subset-consistency loss (features from two random channel subsets of the same trial
should agree), which targets montage invariance directly.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from neurolayer.core.positions import POSITIONS
from neurolayer.core.types import FloatArray

DEFAULT_BANDS: tuple[tuple[float, float], ...] = (
    (4.0, 8.0), (8.0, 12.0), (12.0, 16.0), (16.0, 24.0), (24.0, 32.0), (32.0, 40.0),
)  # fmt: skip


def electrode_positions(ch_names: Sequence[str]) -> FloatArray:
    """Unit-sphere coordinates, shape ``(n_channels, 3)``; unknown names raise ``KeyError``."""
    missing = [c for c in ch_names if c not in POSITIONS]
    if missing:
        raise KeyError(f"no scalp position for channels {missing}; they cannot be used")
    return np.array([POSITIONS[c] for c in ch_names], dtype=np.float64)


def sinc_filter_bank(sfreq: float, bands: Sequence[tuple[float, float]], kernel: int) -> FloatArray:
    """Hamming-windowed sinc band-pass kernels, shape ``(n_bands, kernel)``."""
    from scipy.signal import firwin

    nyquist = sfreq / 2.0
    kernels = []
    for low, high in bands:
        high = min(high, 0.95 * nyquist)
        low = min(low, high - 1.0)
        kernels.append(firwin(kernel, [low, high], pass_zero=False, fs=sfreq))
    return np.asarray(kernels, dtype=np.float64)


def build_spatial_field_net(
    n_classes: int,
    sfreq: float,
    n_spatial: int = 8,
    bands: Sequence[tuple[float, float]] = DEFAULT_BANDS,
    kernel_seconds: float = 0.25,
    hidden: int = 32,
    dropout: float = 0.25,
) -> Any:
    """Construct the network (PyTorch ``nn.Module``; requires the ``dl`` extra)."""
    import torch
    from torch import nn

    kernel = max(9, round(kernel_seconds * sfreq) | 1)  # odd length keeps zero phase
    n_bands = len(bands)

    class SpatialFieldNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.n_spatial, self.n_bands = n_spatial, n_bands
            self.temporal = nn.Conv1d(
                1, n_bands, kernel_size=kernel, padding=kernel // 2, bias=False
            )
            with torch.no_grad():
                bank = sinc_filter_bank(sfreq, bands, kernel)
                self.temporal.weight.copy_(torch.as_tensor(bank, dtype=torch.float32)[:, None, :])
            self.spatial = nn.Sequential(
                nn.Linear(3, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                nn.Linear(hidden, n_spatial * n_bands),
            )  # fmt: skip
            self.head = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(n_spatial * n_bands, n_classes)
            )

        def features(self, x: Any, positions: Any, mask: Any = None) -> Any:
            """Log-power of position-defined virtual channels: ``(batch, n_spatial * n_bands)``."""
            batch, channels, times = x.shape
            filtered = self.temporal(x.reshape(batch * channels, 1, times))
            filtered = filtered.reshape(batch, channels, self.n_bands, -1)
            weights = self.spatial(positions).reshape(channels, self.n_spatial, self.n_bands)
            weights = weights.unsqueeze(0).expand(batch, -1, -1, -1)
            if mask is not None:
                weights = weights * mask[:, :, None, None]
            norm = torch.sqrt((weights**2).sum(dim=1, keepdim=True) + 1e-8)
            weights = weights / norm
            virtual = torch.einsum("bcqf,bcft->bqft", weights, filtered)
            power = torch.log(virtual.pow(2).mean(dim=-1) + 1e-6)
            return power.reshape(batch, -1)

        def forward(self, x: Any, positions: Any, mask: Any = None) -> Any:
            return self.head(self.features(x, positions, mask))

    return SpatialFieldNet()
