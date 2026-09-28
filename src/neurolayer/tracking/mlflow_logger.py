"""Mirror run manifests into MLflow for browsing and comparison (WP-0.7, ADR-0006).

The manifest in ``artifacts/runs/<run_id>/`` stays the source of truth. MLflow is a
convenience view, so any MLflow failure is logged as a warning and never fails a run.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from neurolayer.evaluation.protocol import ProtocolResult
from neurolayer.tracking.manifest import RunManifest

logger = logging.getLogger(__name__)

DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
"""MLflow's recommended local backend (the file store is deprecated). Git-ignored."""

_MAX_PARAM_CHARS = 500


def flatten_params(config: Mapping[str, Any], prefix: str = "") -> dict[str, str]:
    """Flatten a nested config into ``dotted.key -> str`` pairs suitable for MLflow params."""
    flat: dict[str, str] = {}
    for key, value in config.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(flatten_params(value, prefix=f"{name}."))
        else:
            flat[name] = str(value)[:_MAX_PARAM_CHARS]
    return flat


def log_run(
    manifest: RunManifest,
    result: ProtocolResult,
    run_dir: Path,
    *,
    tracking_uri: str | None = None,
    experiment: str = "neurolayer",
) -> str | None:
    """Log one finished run to MLflow and return the MLflow run id (``None`` on failure).

    Per-budget metrics use the calibration budget ``k`` as the MLflow *step*, so the
    MLflow UI plots the calibration-efficiency curve directly.
    """
    try:
        import mlflow
    except ImportError:
        logger.warning("MLflow not installed; install the 'tracking' extra to mirror runs")
        return None
    try:
        mlflow.set_tracking_uri(tracking_uri or DEFAULT_TRACKING_URI)
        mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name=manifest.run_id) as run:
            mlflow.set_tags(
                {
                    "neurolayer.run_id": manifest.run_id,
                    "neurolayer.official": str(manifest.official),
                    "neurolayer.purpose": manifest.purpose,
                    "neurolayer.config_hash": manifest.config_hash,
                    "neurolayer.git_commit": str(manifest.git_commit),
                    "neurolayer.git_dirty": str(manifest.git_dirty),
                    "neurolayer.datasets": ",".join(d.id for d in manifest.datasets),
                }
            )
            mlflow.log_params(flatten_params(manifest.config))
            for summary in result.per_budget():
                for name in ("mean_ba", "median_ba", "ci_low", "ci_high", "usable_user_rate"):
                    mlflow.log_metric(name, float(getattr(summary, name)), step=summary.k)
            if result.records:
                mlflow.log_metric("aucec", result.aucec())
                median_ttc = result.median_ttc()
                if not math.isinf(median_ttc):
                    mlflow.log_metric("median_ttc", median_ttc)
            mlflow.log_artifacts(str(run_dir))
            return str(run.info.run_id)
    except Exception as exc:  # MLflow must never break an experiment run.
        logger.warning("MLflow logging failed (run %s kept on disk): %s", manifest.run_id, exc)
        return None
