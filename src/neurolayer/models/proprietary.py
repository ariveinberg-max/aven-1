"""Proprietary decoder v0: ``SpatialFieldDecoder`` (Stage 5, registered as ``nl_spatial_field``).

Calibration-efficient, montage-agnostic motor-intent decoding:

* **Source training:** each source subject is Euclidean-aligned with its own trials; the
  spatial-field net (:mod:`neurolayer.representation.spatial_field`) trains with channel
  dropout and an optional subset-consistency loss, early-stopped on held-out **source**
  subjects.
* **Calibration-time adaptation:** the target is aligned with statistics from its
  calibration window only; with labeled trials (k > 0) the classifier head is fine-tuned.
* **Optional causal online alignment (H3):** at prediction time, trials are processed in
  chronological order and the alignment is updated with each trial **after** it has
  been predicted, so only past data is used (a declared causal online adaptation, as the
  protocol allows).

Because spatial filters are functions of scalp position, the decoder accepts any target
montage whose channels have known positions, without ``source_montage`` tricks.
"""

from __future__ import annotations

from typing import Any, Self

import numpy as np

from neurolayer.core.types import EpochSet, FloatArray, IntArray
from neurolayer.representation.alignment import apply_alignment, euclidean_alignment_matrix
from neurolayer.representation.spatial_field import build_spatial_field_net, electrode_positions
from neurolayer.representation.torch_utils import grouped_split, seed_everything, select_device

_TO_MICROVOLTS = 1e6


class SpatialFieldDecoder:
    """Montage-agnostic spatial-field decoder with calibration-time adaptation.

    Parameters
    ----------
    epochs, batch_size, lr, patience
        Source-training settings (early stopping on held-out source subjects).
    channel_keep
        Probability of keeping each channel in a training sample (channel dropout).
    consistency_weight
        Weight of the subset-consistency loss (0 disables it).
    finetune_epochs, finetune_lr
        Head fine-tuning on calibration trials (k > 0).
    online_alignment
        Causally update the target alignment during prediction (H3).
    align_target
        Align the target with its calibration-window statistics (H2 ablation switch).
    n_spatial
        Spatial filters per frequency band.
    seed, device
        Reproducibility and hardware.
    """

    def __init__(
        self,
        epochs: int = 60,
        batch_size: int = 64,
        lr: float = 2e-3,
        patience: int = 10,
        channel_keep: float = 0.7,
        consistency_weight: float = 0.0,
        finetune_epochs: int = 30,
        finetune_lr: float = 5e-3,
        online_alignment: bool = False,
        align_target: bool = True,
        n_spatial: int = 8,
        seed: int = 0,
        device: str = "auto",
    ) -> None:
        if not 0.0 < channel_keep <= 1.0:
            raise ValueError("channel_keep must lie in (0, 1]")
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.channel_keep = channel_keep
        self.consistency_weight = consistency_weight
        self.finetune_epochs = finetune_epochs
        self.finetune_lr = finetune_lr
        self.online_alignment = online_alignment
        self.align_target = align_target
        self.n_spatial = n_spatial
        self.seed = seed
        self.device = device
        self._net: Any = None
        self._target_alignment: FloatArray | None = None
        self._target_window: FloatArray | None = None
        self._target_channels: tuple[str, ...] | None = None

    # ------------------------------------------------------------------ helpers
    def _torch(self) -> Any:
        import torch

        return torch

    def _aligned_source(self, source: EpochSet) -> FloatArray:
        X = np.array(source.X * _TO_MICROVOLTS)
        for dataset, subject in source.subject_keys():
            mask = (source.dataset == dataset) & (source.subject == subject)
            X[mask] = apply_alignment(X[mask], euclidean_alignment_matrix(X[mask]))
        return X

    def _mask(self, batch: int, channels: int, rng: np.random.Generator) -> FloatArray:
        mask = (rng.random((batch, channels)) < self.channel_keep).astype(np.float64)
        too_few = mask.sum(axis=1) < min(3, channels)
        mask[too_few] = 1.0
        return mask

    # ------------------------------------------------------------------ Decoder API
    def fit(self, source: EpochSet) -> None:
        """Train on pooled, per-subject-aligned source subjects."""
        torch = self._torch()
        device = select_device(self.device)
        seed_everything(self.seed)
        self._net = build_spatial_field_net(source.n_classes, source.sfreq, self.n_spatial).to(
            device
        )
        positions = torch.as_tensor(
            electrode_positions(source.ch_names), dtype=torch.float32, device=device
        )
        X = torch.as_tensor(self._aligned_source(source), dtype=torch.float32)
        y = torch.as_tensor(np.array(source.y), dtype=torch.long)
        groups = np.array([f"{d}/{s}" for d, s in zip(source.dataset, source.subject, strict=True)])
        train_idx, val_idx = grouped_split(np.array(source.y), groups, 0.2, self.seed)
        rng = np.random.default_rng(self.seed)
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        loss_fn = torch.nn.CrossEntropyLoss()
        best, best_state, stale = float("inf"), None, 0
        for _ in range(self.epochs):
            self._net.train()
            order = rng.permutation(train_idx)
            for start in range(0, len(order), self.batch_size):
                idx = order[start : start + self.batch_size]
                xb, yb = X[idx].to(device), y[idx].to(device)
                mask = torch.as_tensor(
                    self._mask(len(idx), X.shape[1], rng), dtype=torch.float32, device=device
                )
                features = self._net.features(xb, positions, mask)
                loss = loss_fn(self._net.head(features), yb)
                if self.consistency_weight > 0:
                    other = torch.as_tensor(
                        self._mask(len(idx), X.shape[1], rng), dtype=torch.float32, device=device
                    )
                    target = self._net.features(xb, positions, other).detach()
                    loss = loss + self.consistency_weight * torch.mean((features - target) ** 2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            self._net.eval()
            with torch.no_grad():
                val = float(
                    loss_fn(self._net(X[val_idx].to(device), positions), y[val_idx].to(device))
                )
            if val < best - 1e-6:
                best, stale = val, 0
                best_state = {k: v.detach().clone() for k, v in self._net.state_dict().items()}
            else:
                stale += 1
                if stale >= self.patience:
                    break
        if best_state is not None:
            self._net.load_state_dict(best_state)
        self._net.eval()

    def adapt(self, calibration: EpochSet | None, unlabeled: EpochSet | None) -> Self:
        """Align the target with calibration-window statistics; fine-tune the head if k > 0."""
        if self._net is None:
            raise RuntimeError("SpatialFieldDecoder.adapt called before fit")
        pieces = [p for p in (calibration, unlabeled) if p is not None and len(p)]
        if pieces:
            if any(p.ch_names != pieces[0].ch_names for p in pieces):
                raise ValueError("calibration and unlabeled epochs must share channels")
            self._target_channels = pieces[0].ch_names
            if self.align_target:
                window = np.concatenate([p.X for p in pieces]) * _TO_MICROVOLTS
                self._target_window = window
                self._target_alignment = euclidean_alignment_matrix(window)
        if calibration is not None and len(calibration) and self.finetune_epochs > 0:
            self._finetune_head(calibration)
        return self

    def _finetune_head(self, calibration: EpochSet) -> None:
        torch = self._torch()
        device = select_device(self.device)
        seed_everything(self.seed)
        positions = torch.as_tensor(
            electrode_positions(calibration.ch_names), dtype=torch.float32, device=device
        )
        X = calibration.X * _TO_MICROVOLTS
        if self._target_alignment is not None:
            X = apply_alignment(X, self._target_alignment)
        with torch.no_grad():
            features = self._net.features(
                torch.as_tensor(np.array(X), dtype=torch.float32, device=device), positions
            )
        y = torch.as_tensor(np.array(calibration.y), dtype=torch.long, device=device)
        optimizer = torch.optim.Adam(self._net.head.parameters(), lr=self.finetune_lr)
        loss_fn = torch.nn.CrossEntropyLoss()
        self._net.head.train()
        for _ in range(self.finetune_epochs):
            optimizer.zero_grad()
            loss = loss_fn(self._net.head(features), y)
            loss.backward()
            optimizer.step()
        self._net.eval()

    def _logits(self, X: FloatArray, ch_names: tuple[str, ...]) -> FloatArray:
        torch = self._torch()
        device = select_device(self.device)
        positions = torch.as_tensor(
            electrode_positions(ch_names), dtype=torch.float32, device=device
        )
        with torch.no_grad():
            logits = self._net(
                torch.as_tensor(np.array(X), dtype=torch.float32, device=device), positions
            )
        result: FloatArray = logits.cpu().numpy().astype(np.float64)
        return result

    def predict_proba(self, epochs: EpochSet) -> FloatArray:
        """Class probabilities (softmax of logits) with offline target alignment."""
        if self._net is None:
            raise RuntimeError("SpatialFieldDecoder.predict_proba called before fit")
        if self._target_channels is not None and epochs.ch_names != self._target_channels:
            raise ValueError("test epochs must use the calibration channels")
        X = epochs.X * _TO_MICROVOLTS
        if self._target_alignment is not None:
            X = apply_alignment(X, self._target_alignment)
        logits = self._logits(X, epochs.ch_names)
        shifted = np.exp(logits - logits.max(axis=1, keepdims=True))
        probabilities: FloatArray = shifted / shifted.sum(axis=1, keepdims=True)
        return probabilities

    def predict(self, epochs: EpochSet) -> IntArray:
        """Predict; with ``online_alignment`` trials are processed causally in time order."""
        if self._net is None:
            raise RuntimeError("SpatialFieldDecoder.predict called before fit")
        if self._target_channels is not None and epochs.ch_names != self._target_channels:
            raise ValueError("test epochs must use the calibration channels")
        X = epochs.X * _TO_MICROVOLTS
        if not self.online_alignment:
            if self._target_alignment is not None:
                X = apply_alignment(X, self._target_alignment)
            predictions: IntArray = self._logits(X, epochs.ch_names).argmax(axis=1).astype(np.int64)
            return predictions
        order = np.argsort(epochs.order, kind="stable")
        history = [] if self._target_window is None else list(self._target_window)
        out = np.zeros(len(epochs), dtype=np.int64)
        for i in order:
            trial = X[i : i + 1]
            if len(history) >= 2:
                trial = apply_alignment(trial, euclidean_alignment_matrix(np.stack(history)))
            out[i] = int(self._logits(trial, epochs.ch_names).argmax(axis=1)[0])
            history.append(X[i])  # only now does this trial become "past" data
        return out

    def embed(self, epochs: EpochSet, align_per_subject: bool = True) -> FloatArray:
        """Learned features (for identity probes, H4); aligns each subject with its own trials."""
        if self._net is None:
            raise RuntimeError("SpatialFieldDecoder.embed called before fit")
        torch = self._torch()
        device = select_device(self.device)
        X = self._aligned_source(epochs) if align_per_subject else epochs.X * _TO_MICROVOLTS
        positions = torch.as_tensor(
            electrode_positions(epochs.ch_names), dtype=torch.float32, device=device
        )
        with torch.no_grad():
            features = self._net.features(
                torch.as_tensor(np.array(X), dtype=torch.float32, device=device), positions
            )
        result: FloatArray = features.cpu().numpy().astype(np.float64)
        return result
