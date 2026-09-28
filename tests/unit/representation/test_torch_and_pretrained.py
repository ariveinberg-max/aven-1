from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pytest

from neurolayer.data.catalog import LegalReview, Purpose
from neurolayer.representation.pretrained import (
    ModelCard,
    evaluate_model_usage,
    load_model_catalog,
    load_pretrained,
)

torch = pytest.importorskip("torch")

from neurolayer.representation.torch_utils import (  # noqa: E402
    TrainConfig,
    file_sha256,
    finetune,
    load_checkpoint,
    predict_logits,
    save_checkpoint,
    seed_everything,
    select_device,
    train_classifier,
)


def _toy(n: int = 120, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    X = rng.normal(0, 1, (n, 4, 16)).astype(np.float32)
    X[:, 0, :] += (y[:, None] * 2.0 - 1.0).astype(np.float32)  # linearly separable
    groups = np.repeat(np.arange(4), n // 4)
    return X, y.astype(np.int64), groups


def _net() -> torch.nn.Module:
    return torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(64, 2))


def test_training_learns_and_is_deterministic() -> None:
    X, y, groups = _toy()
    cpu = torch.device("cpu")
    seed_everything(0)
    a = _net()
    seed_everything(0)
    b = _net()
    history = train_classifier(
        a, X, y, groups=groups, config=TrainConfig(epochs=30, patience=30), device=cpu
    )
    train_classifier(b, X, y, groups=groups, config=TrainConfig(epochs=30, patience=30), device=cpu)
    assert history.best_epoch >= 0
    assert (predict_logits(a, X, device=cpu).argmax(1) == y).mean() > 0.9
    for pa, pb in zip(a.parameters(), b.parameters(), strict=True):
        torch.testing.assert_close(pa, pb)
    finetune(a, X[:10], y[:10], epochs=2, device=cpu)
    assert select_device("cpu") == cpu
    assert select_device().type in {"cpu", "cuda", "mps"}


def test_checkpoints_are_hashed_and_verified(tmp_path: Path) -> None:
    model = _net()
    path = tmp_path / "model.safetensors"
    digest = save_checkpoint(model, path, {"note": "test"})
    assert digest == file_sha256(path)
    assert (tmp_path / "model.json").exists()
    fresh = _net()
    assert load_checkpoint(fresh, path, expected_sha256=digest) == ([], [])
    torch.testing.assert_close(fresh[1].weight, model[1].weight)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_checkpoint(fresh, path, expected_sha256="0" * 64)
    legacy = tmp_path / "model.pt"
    torch.save(model.state_dict(), legacy)
    load_checkpoint(_net(), legacy)  # weights_only=True path


def test_repository_model_cards_are_gated(repo_root: Path) -> None:
    cards = load_model_catalog(repo_root / "catalog" / "models")
    assert {"labram_base", "cbramod", "reve_base", "mirepnet"} <= set(cards)
    for card in cards.values():
        # No hash is pinned yet, so nothing may be used beyond exploration.
        assert not evaluate_model_usage(card, Purpose.BENCHMARK).allowed
        assert not evaluate_model_usage(card, Purpose.TRAINING).allowed


def _card(sha: str | None, review: LegalReview | None = None) -> ModelCard:
    return ModelCard.model_validate(
        {
            "id": "toy",
            "name": "toy",
            "architecture": "torch:Sequential",
            "source_url": "https://example.org",
            "citation": "none",
            "code_license": "MIT",
            "weights": {"sha256": sha, "license": "MIT"},
            "legal_review": review,
        }
    )


def test_model_gate_and_pretrained_loading(tmp_path: Path) -> None:
    model = _net()
    path = tmp_path / "w.safetensors"
    digest = save_checkpoint(model, path)
    card = _card(digest)
    assert evaluate_model_usage(card, Purpose.BENCHMARK).allowed
    assert not evaluate_model_usage(card, Purpose.TRAINING).allowed
    reviewed = _card(
        digest,
        LegalReview(
            approved_purposes=(Purpose.TRAINING,),
            reviewer="counsel",
            reference="L-1",
            reviewed_on=date(2026, 9, 26),
        ),
    )
    assert evaluate_model_usage(reviewed, Purpose.TRAINING).allowed
    loaded = load_pretrained(_net(), card, path, Purpose.BENCHMARK)
    torch.testing.assert_close(loaded[1].weight, model[1].weight)
    with pytest.raises(PermissionError, match="refused"):
        load_pretrained(_net(), _card(None), path, Purpose.BENCHMARK)
