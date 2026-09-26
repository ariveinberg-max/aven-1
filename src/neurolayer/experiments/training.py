"""Train a deployable model bundle from an experiment config (WP-6.1).

Unlike ``run_experiment`` (evaluation), training produces **shippable weights**, so the
license gate is always checked for ``training`` regardless of the config's purpose
(ADR-0005). The bundle records full provenance: git state, config hash, datasets with
their licenses, and the preprocessing pipeline hash.
"""

from __future__ import annotations

from pathlib import Path

from neurolayer.data.catalog import Purpose, load_catalog, require_usage, select_cards
from neurolayer.data.sources import SYNTHETIC_SOURCES
from neurolayer.experiments.config import ExperimentConfig
from neurolayer.models.bundle import BundleInfo, export_bundle
from neurolayer.models.proprietary import SpatialFieldDecoder
from neurolayer.signal.build import build_epochs
from neurolayer.tracking.manifest import config_hash, git_state


def train_bundle(
    config: ExperimentConfig,
    *,
    catalog_dir: Path,
    data_root: Path,
    repo_root: Path,
    out_dir: Path,
    version: str,
) -> BundleInfo:
    """Fit the configured spatial-field decoder on all subjects and export a bundle."""
    if config.decoder.name != "nl_spatial_field":
        raise ValueError("bundles are produced for the proprietary decoder 'nl_spatial_field'")
    catalog = load_catalog(catalog_dir)
    cards = select_cards(catalog, config.datasets)
    require_usage(cards, Purpose.TRAINING)
    epochs, report = build_epochs(
        config.datasets,
        catalog=catalog,
        data_root=data_root,
        pipeline=config.pipeline,
        synthetic=config.synthetic.to_generator_config(),
        max_subjects=config.max_subjects,
    )
    decoder = SpatialFieldDecoder(**config.decoder.params)
    decoder.fit(epochs)
    commit, dirty = git_state(repo_root)
    synthetic_only = all(d in SYNTHETIC_SOURCES for d in config.datasets)
    return export_bundle(
        decoder,
        out_dir / config.name / version,
        name=config.name,
        version=version,
        label_names=epochs.label_names,
        sfreq=epochs.sfreq,
        n_times=epochs.n_times,
        trained_channels=epochs.ch_names,
        preprocessing={
            "synthetic_only": synthetic_only,
            "pipeline": None if synthetic_only else config.pipeline.model_dump(mode="json"),
            "pipeline_hash": report.pipeline_hash,
        },
        provenance={
            "git_commit": commit,
            "git_dirty": dirty,
            "config_hash": config_hash(config.model_dump(mode="json")),
            "datasets": [
                {"id": c.id, "version": c.version, "license": c.license.spdx} for c in cards
            ],
            "n_subjects": len(epochs.subject_keys()),
            "n_epochs": len(epochs),
        },
    )
