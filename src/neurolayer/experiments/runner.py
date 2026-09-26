"""Orchestrates one experiment: license gate → data → protocol → manifest.

The license gate runs **before** any data is loaded (ADR-0005). ``official`` runs also
require a clean, committed git tree (ADR-0006).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from neurolayer.data.catalog import load_catalog, require_usage, select_cards
from neurolayer.data.sources import load_epochs
from neurolayer.evaluation.protocol import ProtocolResult, run_protocol
from neurolayer.experiments.config import ExperimentConfig
from neurolayer.models.registry import make_factory
from neurolayer.tracking.manifest import RunManifest, create_manifest, git_state, write_run


class DirtyTreeError(RuntimeError):
    """Raised when an official run is requested on an uncommitted working tree."""


@dataclass(frozen=True, slots=True)
class ExperimentOutcome:
    """What a finished experiment produced."""

    manifest: RunManifest
    result: ProtocolResult
    run_dir: Path


def run_experiment(
    config: ExperimentConfig,
    *,
    catalog_dir: Path,
    output_dir: Path,
    repo_root: Path,
    official: bool = False,
) -> ExperimentOutcome:
    """Run ``config`` end to end and write its run directory.

    Raises
    ------
    LicenseGateError
        If any dataset is not permitted for ``config.purpose``.
    DirtyTreeError
        If ``official`` and the git tree is dirty or not a repository.
    """
    cards = select_cards(load_catalog(catalog_dir), config.datasets)
    require_usage(cards, config.purpose)

    if official:
        commit, dirty = git_state(repo_root)
        if commit is None or dirty:
            raise DirtyTreeError(
                "official runs require a clean, committed git tree; commit or stash changes"
            )

    epochs = load_epochs(config.datasets, config.synthetic.to_generator_config())
    protocol_config = config.protocol.to_protocol_config()
    result = run_protocol(
        epochs, make_factory(config.decoder.name, config.decoder.params), protocol_config
    )

    manifest = create_manifest(
        name=config.name,
        config=config.model_dump(mode="json"),
        datasets=cards,
        purpose=config.purpose,
        seed=protocol_config.seed,
        official=official,
        repo_root=repo_root,
    )
    run_dir = write_run(output_dir, manifest, result)
    return ExperimentOutcome(manifest=manifest, result=result, run_dir=run_dir)
