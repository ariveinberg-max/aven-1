from __future__ import annotations

from pathlib import Path

import pytest

from neurolayer.experiments.config import load_experiment_config
from neurolayer.experiments.runner import run_experiment
from neurolayer.tracking.mlflow_logger import flatten_params

mlflow = pytest.importorskip("mlflow")

# MLflow and SQLAlchemy emit their own deprecation warnings; they are not ours to fix.
pytestmark = pytest.mark.filterwarnings("ignore")


def test_flatten_params() -> None:
    assert flatten_params({"a": 1, "b": {"c": [1, 2], "d": {"e": "x"}}}) == {
        "a": "1",
        "b.c": "[1, 2]",
        "b.d.e": "x",
    }


def test_run_is_mirrored_with_curve_and_artifacts(
    repo_root: Path, catalog_dir: Path, tmp_path: Path
) -> None:
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    config = load_experiment_config(repo_root / "configs/experiments/smoke_synthetic.yaml")
    outcome = run_experiment(
        config,
        catalog_dir=catalog_dir,
        output_dir=tmp_path / "runs",
        repo_root=repo_root,
        mlflow=True,
        mlflow_uri=uri,
    )
    assert outcome.mlflow_run_id is not None

    client = mlflow.tracking.MlflowClient(tracking_uri=uri)
    run = client.get_run(outcome.mlflow_run_id)
    assert run.data.tags["neurolayer.run_id"] == outcome.manifest.run_id
    assert run.data.params["decoder.name"] == "logvar_logreg"
    history = client.get_metric_history(outcome.mlflow_run_id, "mean_ba")
    assert sorted(m.step for m in history) == [0, 5, 10, 20]
    assert "aucec" in run.data.metrics
    artifacts = {a.path for a in client.list_artifacts(outcome.mlflow_run_id)}
    assert {"manifest.json", "results.csv", "summary.json"} <= artifacts


def test_mlflow_failure_does_not_fail_the_run(
    repo_root: Path, catalog_dir: Path, tmp_path: Path
) -> None:
    config = load_experiment_config(repo_root / "configs/experiments/smoke_synthetic.yaml")
    outcome = run_experiment(
        config,
        catalog_dir=catalog_dir,
        output_dir=tmp_path / "runs",
        repo_root=repo_root,
        mlflow=True,
        mlflow_uri="unsupported-scheme://nowhere",
    )
    assert outcome.mlflow_run_id is None
    assert (outcome.run_dir / "manifest.json").exists()
