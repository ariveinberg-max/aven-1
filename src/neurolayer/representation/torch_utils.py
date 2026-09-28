"""PyTorch training scaffold (WP-3.3): seeding, devices, training, safe checkpoints.

Requires the ``dl`` extra. Training uses early stopping on a validation split drawn
from **source subjects only** (grouped by subject when possible), so target data never
influences model selection. Checkpoints use safetensors (no pickle).
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Optimization settings for :func:`train_classifier`."""

    epochs: int = 50
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.0
    patience: int = 10
    val_fraction: float = 0.2
    seed: int = 0


@dataclass(slots=True)
class TrainHistory:
    """Per-epoch losses and the epoch whose weights were kept."""

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    best_epoch: int = -1


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch; optionally force deterministic kernels."""
    import torch

    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - also seed libraries that use the legacy global RNG
    torch.manual_seed(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def select_device(preference: str = "auto") -> Any:
    """Return a ``torch.device``: ``auto`` picks CUDA, then Apple MPS, then CPU."""
    import torch

    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def grouped_split(
    y: npt.NDArray[np.int64], groups: npt.NDArray[Any] | None, fraction: float, seed: int
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Train/validation indices; whole groups (subjects) go to validation when possible."""
    rng = np.random.default_rng(seed)
    n = len(y)
    if groups is not None and len(np.unique(groups)) >= 3:
        unique = rng.permutation(np.unique(groups))
        n_val = max(1, round(len(unique) * fraction))
        val_mask = np.isin(groups, unique[:n_val])
    else:
        val_mask = np.zeros(n, dtype=bool)
        val_mask[rng.permutation(n)[: max(1, round(n * fraction))]] = True
    return np.flatnonzero(~val_mask).astype(np.int64), np.flatnonzero(val_mask).astype(np.int64)


def _batches(n: int, batch_size: int, rng: np.random.Generator | None) -> list[npt.NDArray[Any]]:
    order = rng.permutation(n) if rng is not None else np.arange(n)
    return [order[i : i + batch_size] for i in range(0, n, batch_size)]


def train_classifier(
    model: Any,
    X: npt.NDArray[np.float32],
    y: npt.NDArray[np.int64],
    *,
    groups: npt.NDArray[Any] | None = None,
    config: TrainConfig | None = None,
    device: Any = None,
    forward_kwargs: Mapping[str, Any] | None = None,
) -> TrainHistory:
    """Train with Adam + cross-entropy; early stopping restores the best weights.

    Seed with :func:`seed_everything` **before constructing** ``model``: weights are
    initialized at construction time, so seeding here alone does not make runs repeatable.
    """
    import torch

    cfg = config or TrainConfig()
    device = device or select_device()
    kwargs = dict(forward_kwargs or {})
    seed_everything(cfg.seed)
    model.to(device)
    train_idx, val_idx = grouped_split(y, groups, cfg.val_fraction, cfg.seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()
    # np.array(...) copies: EpochSet arrays are read-only and torch refuses to share them.
    Xt = torch.as_tensor(np.array(X, dtype=np.float32))
    yt = torch.as_tensor(np.array(y, dtype=np.int64))
    rng = np.random.default_rng(cfg.seed)
    history = TrainHistory()
    best_state: dict[str, Any] | None = None
    best_loss = float("inf")
    stale = 0
    for epoch in range(cfg.epochs):
        model.train()
        losses = []
        for batch in _batches(len(train_idx), cfg.batch_size, rng):
            idx = train_idx[batch]
            optimizer.zero_grad()
            loss = loss_fn(model(Xt[idx].to(device), **kwargs), yt[idx].to(device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            val = [
                float(
                    loss_fn(model(Xt[val_idx[b]].to(device), **kwargs), yt[val_idx[b]].to(device))
                )
                for b in _batches(len(val_idx), cfg.batch_size, None)
            ]
        history.train_loss.append(float(np.mean(losses)))
        history.val_loss.append(float(np.mean(val)))
        if history.val_loss[-1] < best_loss - 1e-6:
            best_loss, stale, history.best_epoch = history.val_loss[-1], 0, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return history


def finetune(
    model: Any,
    X: npt.NDArray[np.float32],
    y: npt.NDArray[np.int64],
    *,
    epochs: int = 20,
    lr: float = 5e-4,
    batch_size: int = 32,
    seed: int = 0,
    device: Any = None,
    forward_kwargs: Mapping[str, Any] | None = None,
) -> None:
    """Few-shot fine-tuning on calibration data (no validation split: sets are tiny)."""
    import torch

    device = device or select_device()
    kwargs = dict(forward_kwargs or {})
    seed_everything(seed)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    Xt = torch.as_tensor(np.array(X, dtype=np.float32))
    yt = torch.as_tensor(np.array(y, dtype=np.int64))
    rng = np.random.default_rng(seed)
    model.train()
    for _ in range(epochs):
        for batch in _batches(len(y), batch_size, rng):
            optimizer.zero_grad()
            loss = loss_fn(model(Xt[batch].to(device), **kwargs), yt[batch].to(device))
            loss.backward()
            optimizer.step()
    model.eval()


def predict_logits(
    model: Any,
    X: npt.NDArray[np.float32],
    *,
    batch_size: int = 256,
    device: Any = None,
    forward_kwargs: Mapping[str, Any] | None = None,
) -> npt.NDArray[np.float64]:
    """Forward pass in batches without gradients; returns logits as float64."""
    import torch

    device = device or select_device()
    kwargs = dict(forward_kwargs or {})
    model.to(device).eval()
    Xt = torch.as_tensor(np.array(X, dtype=np.float32))
    outputs = []
    with torch.no_grad():
        for batch in _batches(len(X), batch_size, None):
            outputs.append(model(Xt[batch].to(device), **kwargs).cpu().numpy())
    return np.concatenate(outputs).astype(np.float64)


def file_sha256(path: Path) -> str:
    """Hex SHA-256 of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def save_checkpoint(model: Any, path: Path, metadata: Mapping[str, Any] | None = None) -> str:
    """Save weights with safetensors plus a JSON sidecar; returns the file's SHA-256."""
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    save_file(state, str(path))
    digest = file_sha256(path)
    sidecar = {"sha256": digest, "metadata": dict(metadata or {})}
    path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    return digest


def load_checkpoint(
    model: Any, path: Path, expected_sha256: str | None = None, strict: bool = True
) -> tuple[list[str], list[str]]:
    """Load weights after verifying the SHA-256; returns (missing, unexpected) keys.

    ``.safetensors`` files are loaded with safetensors; anything else only with
    ``torch.load(weights_only=True)``, which refuses arbitrary pickled objects (AGENTS.md
    rule 9).
    """
    import torch

    if expected_sha256 is not None and file_sha256(path) != expected_sha256:
        raise ValueError(f"SHA-256 mismatch for {path}; refusing to load")
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(path))
    else:
        state = torch.load(path, map_location="cpu", weights_only=True)
    result = model.load_state_dict(state, strict=strict)
    return list(result.missing_keys), list(result.unexpected_keys)
