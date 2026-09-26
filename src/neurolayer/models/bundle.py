"""Model bundles: the deployable unit served by the API (WP-6.1).

A bundle is a directory::

    <bundle>/
    ├── bundle.json            # metadata: format, model, labels, sfreq, preprocessing, provenance
    └── weights.safetensors    # network weights (no pickle anywhere)

``bundle.json`` pins the weights' SHA-256; loading refuses any mismatch. Bundles are
produced by ``neurolayer train`` from an experiment config whose purpose is
``training`` (the license gate must allow shipping weights trained on the data).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from neurolayer.models.proprietary import SpatialFieldDecoder
from neurolayer.representation.spatial_field import build_spatial_field_net
from neurolayer.representation.torch_utils import file_sha256, load_checkpoint, save_checkpoint

BUNDLE_FORMAT = 1
BUNDLE_FILE = "bundle.json"
WEIGHTS_FILE = "weights.safetensors"


@dataclass(frozen=True, slots=True)
class BundleInfo:
    """Metadata stored in ``bundle.json``."""

    name: str
    version: str
    decoder: str
    decoder_params: dict[str, Any]
    label_names: tuple[str, ...]
    sfreq: float
    n_times: int
    trained_channels: tuple[str, ...]
    preprocessing: dict[str, Any]
    weights_sha256: str
    provenance: dict[str, Any] = field(default_factory=dict)
    format: int = BUNDLE_FORMAT

    @property
    def model_id(self) -> str:
        """``name@version``."""
        return f"{self.name}@{self.version}"


def export_bundle(
    decoder: SpatialFieldDecoder,
    directory: Path,
    *,
    name: str,
    version: str,
    label_names: tuple[str, ...],
    sfreq: float,
    n_times: int,
    trained_channels: tuple[str, ...],
    preprocessing: dict[str, Any],
    provenance: dict[str, Any] | None = None,
) -> BundleInfo:
    """Write a fitted spatial-field decoder as a bundle directory."""
    if decoder._net is None:
        raise RuntimeError("export_bundle needs a fitted decoder")
    directory.mkdir(parents=True, exist_ok=False)
    digest = save_checkpoint(decoder._net, directory / WEIGHTS_FILE, {"model": f"{name}@{version}"})
    (directory / "weights.json").unlink(missing_ok=True)  # bundle.json is the single sidecar
    info = BundleInfo(
        name=name,
        version=version,
        decoder="nl_spatial_field",
        decoder_params={
            "n_spatial": decoder.n_spatial,
            "finetune_epochs": decoder.finetune_epochs,
            "finetune_lr": decoder.finetune_lr,
            "seed": decoder.seed,
        },
        label_names=tuple(label_names),
        sfreq=float(sfreq),
        n_times=int(n_times),
        trained_channels=tuple(trained_channels),
        preprocessing=preprocessing,
        weights_sha256=digest,
        provenance=dict(provenance or {}),
    )
    (directory / BUNDLE_FILE).write_text(
        json.dumps(asdict(info), indent=2) + "\n", encoding="utf-8"
    )
    return info


def read_bundle_info(directory: Path) -> BundleInfo:
    """Parse ``bundle.json``."""
    raw = json.loads((directory / BUNDLE_FILE).read_text(encoding="utf-8"))
    if raw.get("format") != BUNDLE_FORMAT:
        raise ValueError(f"unsupported bundle format {raw.get('format')!r} in {directory}")
    raw["label_names"] = tuple(raw["label_names"])
    raw["trained_channels"] = tuple(raw["trained_channels"])
    return BundleInfo(**raw)


def load_bundle(directory: Path, device: str = "cpu") -> tuple[SpatialFieldDecoder, BundleInfo]:
    """Rebuild a fitted decoder from a bundle after verifying the weights' SHA-256."""
    info = read_bundle_info(directory)
    weights = directory / WEIGHTS_FILE
    if file_sha256(weights) != info.weights_sha256:
        raise ValueError(f"weights of {info.model_id} do not match bundle.json; refusing to load")
    params = dict(info.decoder_params)
    decoder = SpatialFieldDecoder(device=device, **params)
    net = build_spatial_field_net(len(info.label_names), info.sfreq, decoder.n_spatial)
    load_checkpoint(net, weights, expected_sha256=info.weights_sha256)
    net.eval()
    decoder._net = net
    return decoder, info
