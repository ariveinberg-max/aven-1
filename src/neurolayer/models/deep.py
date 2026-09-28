"""Deep-learning baselines via Braindecode (WP-4.1: B4 EEGNet, B6 foundation models).

``BraindecodeDecoder`` trains any Braindecode architecture on pooled source subjects
(early stopping on source-only validation), then fine-tunes a copy on the target's
calibration trials. Optional Euclidean alignment whitens each subject with statistics
from its own trials (source) or its calibration window (target).

Fixed-montage networks cannot change input channels after training: when a target
montage differs from the source, set ``protocol.source_montage`` to the same montage.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Self

import numpy as np

from neurolayer.core.types import EpochSet, FloatArray, IntArray
from neurolayer.data.catalog import Purpose
from neurolayer.representation.alignment import apply_alignment, euclidean_alignment_matrix
from neurolayer.representation.pretrained import load_model_card, load_pretrained
from neurolayer.representation.torch_utils import (
    TrainConfig,
    finetune,
    predict_logits,
    seed_everything,
    select_device,
    train_classifier,
)

_TO_MICROVOLTS = 1e6
_NEEDS_CHANNEL_NAMES = frozenset({"Labram"})
_ARCHITECTURE_DEFAULTS: dict[str, dict[str, Any]] = {
    # Max-norm constraint on the classifier, as in the original EEGNet paper (norm 0.25);
    # braindecode deprecates the unconstrained default.
    "EEGNet": {"final_layer_with_constraint": True},
}


class BraindecodeDecoder:
    """Pooled deep model + calibration fine-tuning (B4 with EEGNet, B6 with pinned weights).

    Parameters
    ----------
    architecture
        Class name in ``braindecode.models`` (``EEGNet``, ``ShallowFBCSPNet``, ``CBraMod``...).
    train_epochs, finetune_epochs, batch_size, lr, finetune_lr, patience
        Optimization settings.
    euclidean_align
        Whiten each subject with its own mean covariance (target: calibration window).
    model_card, weights
        Optional pretrained initialization: a ``catalog/models`` card path and a
        weights file. Loading goes through the model license gate and a hash check.
    seed, device
        Reproducibility and hardware (``auto`` = CUDA → MPS → CPU).
    model_kwargs
        Extra constructor arguments for the architecture.
    """

    def __init__(
        self,
        architecture: str = "EEGNet",
        train_epochs: int = 50,
        finetune_epochs: int = 20,
        batch_size: int = 64,
        lr: float = 1e-3,
        finetune_lr: float = 5e-4,
        patience: int = 10,
        euclidean_align: bool = True,
        model_card: str | None = None,
        weights: str | None = None,
        seed: int = 0,
        device: str = "auto",
        model_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.architecture = architecture
        self.train_config = TrainConfig(
            epochs=train_epochs, batch_size=batch_size, lr=lr, patience=patience, seed=seed
        )
        self.finetune_epochs = finetune_epochs
        self.finetune_lr = finetune_lr
        self.euclidean_align = euclidean_align
        self.model_card = model_card
        self.weights = weights
        self.seed = seed
        self.device = device
        self.model_kwargs = dict(model_kwargs or {})
        self._model: Any = None
        self._channels: tuple[str, ...] | None = None
        self._target_alignment: FloatArray | None = None

    def _forward_kwargs(self) -> dict[str, Any]:
        if self.architecture in _NEEDS_CHANNEL_NAMES and self._channels is not None:
            return {"ch_names": list(self._channels)}
        return {}

    def _build(self, epochs: EpochSet) -> Any:
        import braindecode.models

        cls = getattr(braindecode.models, self.architecture, None)
        if cls is None:
            raise ValueError(f"braindecode has no model {self.architecture!r}")
        kwargs: dict[str, Any] = {
            "n_chans": epochs.n_channels,
            "n_outputs": epochs.n_classes,
            "n_times": epochs.n_times,
        }
        if "sfreq" in inspect.signature(cls.__init__).parameters:
            kwargs["sfreq"] = epochs.sfreq
        defaults = _ARCHITECTURE_DEFAULTS.get(self.architecture, {})
        model = cls(**(kwargs | defaults | self.model_kwargs))
        if self.weights is not None:
            if self.model_card is None:
                raise ValueError("pretrained weights need a model_card (ADR-0007)")
            card = load_model_card(Path(self.model_card))
            load_pretrained(model, card, Path(self.weights), Purpose.BENCHMARK)
        return model

    def _prepare(self, epochs: EpochSet, alignment: FloatArray | None) -> Any:
        X = epochs.X * _TO_MICROVOLTS
        if alignment is not None:
            X = apply_alignment(X, alignment)
        return X.astype(np.float32)

    def fit(self, source: EpochSet) -> None:
        """Train on pooled source subjects (each Euclidean-aligned with its own trials)."""
        self._channels = source.ch_names
        X = source.X * _TO_MICROVOLTS
        if self.euclidean_align:
            X = np.array(X)
            for dataset, subject in source.subject_keys():
                mask = (source.dataset == dataset) & (source.subject == subject)
                X[mask] = apply_alignment(X[mask], euclidean_alignment_matrix(X[mask]))
        seed_everything(self.seed)  # weight initialization happens at construction
        self._model = self._build(source)
        groups = np.array([f"{d}/{s}" for d, s in zip(source.dataset, source.subject, strict=True)])
        train_classifier(
            self._model,
            X.astype(np.float32),
            source.y,
            groups=groups,
            config=self.train_config,
            device=select_device(self.device),
            forward_kwargs=self._forward_kwargs(),
        )

    def adapt(self, calibration: EpochSet | None, unlabeled: EpochSet | None) -> Self:
        """Estimate target alignment from the calibration window, then fine-tune."""
        if self._model is None:
            raise RuntimeError("BraindecodeDecoder.adapt called before fit")
        pieces = [p for p in (calibration, unlabeled) if p is not None and len(p)]
        for piece in pieces:
            self._check_channels(piece)
        if self.euclidean_align and pieces:
            stacked = np.concatenate([p.X for p in pieces]) * _TO_MICROVOLTS
            self._target_alignment = euclidean_alignment_matrix(stacked)
        if calibration is not None and len(calibration) and self.finetune_epochs > 0:
            finetune(
                self._model,
                self._prepare(calibration, self._target_alignment),
                calibration.y,
                epochs=self.finetune_epochs,
                lr=self.finetune_lr,
                seed=self.seed,
                device=select_device(self.device),
                forward_kwargs=self._forward_kwargs(),
            )
        return self

    def _check_channels(self, epochs: EpochSet) -> None:
        if epochs.ch_names != self._channels:
            raise ValueError(
                f"{self.architecture} was trained on {len(self._channels or ())} channels "
                f"and cannot take {epochs.n_channels} different ones; set "
                "protocol.source_montage to the target montage for fixed-montage models"
            )

    def predict(self, epochs: EpochSet) -> IntArray:
        """Argmax of the network's logits."""
        if self._model is None:
            raise RuntimeError("BraindecodeDecoder.predict called before fit")
        self._check_channels(epochs)
        logits = predict_logits(
            self._model,
            self._prepare(epochs, self._target_alignment),
            device=select_device(self.device),
            forward_kwargs=self._forward_kwargs(),
        )
        predictions: IntArray = logits.argmax(axis=1).astype(np.int64)
        return predictions
