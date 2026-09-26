"""Model registry for the service: discovers and caches bundles (WP-6.1)."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from neurolayer.models.bundle import BUNDLE_FILE, BundleInfo, load_bundle, read_bundle_info


@dataclass(frozen=True, slots=True)
class LoadedModel:
    """A fitted base decoder and its bundle metadata."""

    decoder: Any
    info: BundleInfo


class ModelRegistry:
    """Bundles under ``<root>/<name>/<version>/``, loaded lazily and verified by hash."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._cache: dict[str, LoadedModel] = {}
        self._lock = threading.Lock()

    def available(self) -> dict[str, BundleInfo]:
        """``model_id -> info`` for every bundle on disk."""
        if not self.root.exists():
            return {}
        found = {}
        for manifest in sorted(self.root.glob(f"*/*/{BUNDLE_FILE}")):
            info = read_bundle_info(manifest.parent)
            found[info.model_id] = info
        return found

    def get(self, model_id: str) -> LoadedModel:
        """Load (once) and return a model; ``KeyError`` if unknown."""
        with self._lock:
            if model_id in self._cache:
                return self._cache[model_id]
            name, _, version = model_id.partition("@")
            directory = self.root / name / version
            if not version or not (directory / BUNDLE_FILE).exists():
                raise KeyError(model_id)
            decoder, info = load_bundle(directory)
            model = LoadedModel(decoder=decoder, info=info)
            self._cache[model_id] = model
            return model
