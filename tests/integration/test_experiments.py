from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from neurolayer.data.catalog import LicenseGateError
from neurolayer.experiments.config import ExperimentConfig, load_experiment_config
from neurolayer.experiments.runner import DirtyTreeError, run_experiment


def _configs(repo_root: Path) -> list[Path]:
    return sorted((repo_root / "configs" / "experiments").rglob("*.yaml"))


def test_every_config_in_repo_validates(repo_root: Path) -> None:
    paths = _configs(repo_root)
    assert paths
    for path in paths:
        load_experiment_config(path)


def test_unknown_montage_rejected() -> None:
    with pytest.raises(ValidationError, match="unknown montage"):
        ExperimentConfig.model_validate(
            {
                "name": "x",
                "purpose": "benchmark",
                "datasets": ["synthetic_mi"],
                "decoder": {"name": "chance"},
                "protocol": {"target_montage": "brainco_9000"},
            }
        )


def test_smoke_run_writes_manifest_results_and_summary(
    repo_root: Path, catalog_dir: Path, tmp_path: Path
) -> None:
    config = load_experiment_config(repo_root / "configs/experiments/smoke_synthetic.yaml")
    outcome = run_experiment(
        config, catalog_dir=catalog_dir, output_dir=tmp_path, repo_root=repo_root
    )

    manifest = json.loads((outcome.run_dir / "manifest.json").read_text())
    assert manifest["manifest_version"] == 2
    assert len(manifest["pipeline_hash"]) == 16
    assert manifest["purpose"] == "benchmark"
    assert manifest["datasets"][0]["id"] == "synthetic_mi"
    assert len(manifest["config_hash"]) == 64
    assert manifest["official"] is False

    with (outcome.run_dir / "results.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 6 * 4  # subjects × budgets
    summary = json.loads((outcome.run_dir / "summary.json").read_text())
    assert [b["k"] for b in summary["per_budget"]] == [0, 5, 10, 20]
    assert summary["per_budget"][-1]["mean_ba"] > 0.6


def test_same_config_same_hash_same_scores(
    repo_root: Path, catalog_dir: Path, tmp_path: Path
) -> None:
    config = load_experiment_config(repo_root / "configs/experiments/smoke_synthetic.yaml")
    a = run_experiment(config, catalog_dir=catalog_dir, output_dir=tmp_path, repo_root=repo_root)
    b = run_experiment(config, catalog_dir=catalog_dir, output_dir=tmp_path, repo_root=repo_root)
    assert a.manifest.config_hash == b.manifest.config_hash
    assert a.manifest.run_id != b.manifest.run_id
    assert a.result.records == b.result.records


def test_license_gate_blocks_unverified_public_data(
    repo_root: Path, unverified_catalog: Path, tmp_path: Path
) -> None:
    config = load_experiment_config(
        repo_root / "configs/experiments/cap1_r1_physionet_template.yaml"
    )
    with pytest.raises(LicenseGateError, match="physionet_mi"):
        run_experiment(
            config, catalog_dir=unverified_catalog, output_dir=tmp_path, repo_root=repo_root
        )


def test_official_run_requires_git_repository(
    catalog_dir: Path, repo_root: Path, tmp_path: Path
) -> None:
    config = load_experiment_config(repo_root / "configs/experiments/smoke_synthetic.yaml")
    with pytest.raises(DirtyTreeError):
        run_experiment(
            config, catalog_dir=catalog_dir, output_dir=tmp_path, repo_root=tmp_path, official=True
        )


def test_real_dataset_without_adapter_fails_loudly() -> None:
    from neurolayer.data.sources import load_epochs

    with pytest.raises(NotImplementedError, match="not synthetic"):
        load_epochs(["physionet_mi"])


def test_real_dataset_experiment_end_to_end(
    catalog_dir: Path, repo_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """License gate → adapter → preprocessing → epoching → cache → harness → manifest."""
    pytest.importorskip("mne")
    from neurolayer.data.adapters.moabb import MoabbAdapter
    from tests.fakes import FakeMoabbDataset

    monkeypatch.setattr(
        "neurolayer.signal.build.get_adapter",
        lambda card, data_dir=None: MoabbAdapter(card, FakeMoabbDataset((1, 2, 3)), data_dir),
    )
    config = ExperimentConfig.model_validate(
        {
            "name": "fake-physionet",
            "purpose": "exploration",  # unverified license: exploration only (ADR-0005)
            "datasets": ["physionet_mi"],
            "decoder": {"name": "logvar_logreg"},
            "protocol": {"ks": [0, 2], "n_folds": None, "min_test_per_class": 3},
        }
    )
    outcome = run_experiment(
        config,
        catalog_dir=catalog_dir,
        output_dir=tmp_path / "runs",
        repo_root=repo_root,
        data_root=tmp_path / "data",
    )
    assert {r.subject for r in outcome.result.records} == {"sub-001", "sub-002", "sub-003"}
    manifest = json.loads((outcome.run_dir / "manifest.json").read_text())
    assert manifest["data_report"]["cache_misses"] == 3
    assert manifest["pipeline_hash"] == config.pipeline.pipeline_hash()
