"""Orchestrates one experiment: license gate → data → protocol → manifest.

The license gate runs **before** any data is loaded (ADR-0005). ``official`` runs also
require a clean, committed git tree (ADR-0006).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from neurolayer.data.catalog import load_catalog, require_usage, select_cards
from neurolayer.evaluation.protocol import ProtocolResult, run_protocol
from neurolayer.experiments.config import ExperimentConfig
from neurolayer.models.registry import make_factory
from neurolayer.signal.build import build_epochs
from neurolayer.tracking.manifest import RunManifest, create_manifest, git_state, write_run
from neurolayer.tracking.mlflow_logger import log_run


class DirtyTreeError(RuntimeError):
    """Raised when an official run is requested on an uncommitted working tree."""


@dataclass(frozen=True, slots=True)
class ExperimentOutcome:
    """What a finished experiment produced."""

    manifest: RunManifest
    result: ProtocolResult
    run_dir: Path
    mlflow_run_id: str | None = None


def run_experiment(
    config: ExperimentConfig,
    *,
    catalog_dir: Path,
    output_dir: Path,
    repo_root: Path,
    data_root: Path = Path("data"),
    official: bool = False,
    mlflow: bool = False,
    mlflow_uri: str | None = None,
) -> ExperimentOutcome:
    """Run ``config`` end to end and write its run directory.

    Raises
    ------
    LicenseGateError
        If any dataset is not permitted for ``config.purpose``.
    DirtyTreeError
        If ``official`` and the git tree is dirty or not a repository.

    Notes
    -----
    With ``mlflow=True`` the finished run is mirrored to MLflow (WP-0.7). An MLflow
    failure only logs a warning; the on-disk run directory is the source of truth.
    """
    catalog = load_catalog(catalog_dir)
    cards = select_cards(catalog, config.datasets)
    require_usage(cards, config.purpose)

    if official:
        commit, dirty = git_state(repo_root)
        if commit is None or dirty:
            raise DirtyTreeError(
                "official runs require a clean, committed git tree; commit or stash changes"
            )

    epochs, build_report = build_epochs(
        config.datasets,
        catalog=catalog,
        data_root=data_root,
        pipeline=config.pipeline,
        synthetic=config.synthetic.to_generator_config(),
        max_subjects=config.max_subjects,
    )
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
        pipeline_hash=build_report.pipeline_hash,
        data_report={
            "cache_hits": build_report.cache_hits,
            "cache_misses": build_report.cache_misses,
            "dropped_channels": list(build_report.dropped_channels),
            "epoching": build_report.epoching,
        },
    )
    run_dir = write_run(output_dir, manifest, result)
    mlflow_run_id = log_run(manifest, result, run_dir, tracking_uri=mlflow_uri) if mlflow else None
    return ExperimentOutcome(
        manifest=manifest, result=result, run_dir=run_dir, mlflow_run_id=mlflow_run_id
    )
