from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neurolayer.core.types import EpochSet
from neurolayer.data.synthetic import SyntheticMIConfig, generate_synthetic_mi
from neurolayer.experiments.config import ExperimentConfig
from neurolayer.experiments.gate0 import (
    compare,
    gate0_pipeline,
    sklearn_pipeline,
    within_session_auc,
)
from neurolayer.experiments.probes import probe
from neurolayer.experiments.reports import comparison_markdown, load_run, plot_curves
from neurolayer.experiments.runner import run_experiment
from neurolayer.representation.registry import make_encoder

pytest.importorskip("pyriemann")


def test_identity_probe_detects_subject_information() -> None:
    ep = generate_synthetic_mi(
        SyntheticMIConfig(n_subjects=4, n_trials_per_class=20, subject_variability=2.0)
    ).epochs
    result = probe(ep, make_encoder("tangent_space"), target="subject")
    assert result.n_classes == 4
    assert result.chance == pytest.approx(0.25)
    assert result.above_chance > 0.3


def test_dataset_probe_and_errors() -> None:
    parts = [
        generate_synthetic_mi(
            SyntheticMIConfig(
                n_subjects=2, n_trials_per_class=15, seed=s, dataset_id=name, site_shift=1.0
            )
        ).epochs
        for s, name in [(1, "a"), (2, "b")]
    ]
    result = probe(EpochSet.concat(parts), make_encoder("log_variance"), target="dataset")
    assert result.n_classes == 2
    with pytest.raises(ValueError, match="two distinct"):
        probe(parts[0], make_encoder("log_variance"), target="dataset")


def test_comparison_report_and_plot(catalog_dir: Path, repo_root: Path, tmp_path: Path) -> None:
    runs = []
    for decoder in ("chance", "ts_lr_pooled"):
        config = ExperimentConfig.model_validate(
            {
                "name": f"cmp-{decoder.replace('_', '-')}",
                "purpose": "benchmark",
                "datasets": ["synthetic_mi"],
                "synthetic": {"n_subjects": 5, "n_trials_per_class": 30},
                "decoder": {"name": decoder},
                "protocol": {"ks": [0, 10], "n_unlabeled": 5, "n_folds": None, "n_bootstrap": 100},
            }
        )
        outcome = run_experiment(
            config, catalog_dir=catalog_dir, output_dir=tmp_path, repo_root=repo_root
        )
        runs.append(load_run(outcome.run_dir))
    report = comparison_markdown(runs, reference=runs[0])
    assert "ts_lr_pooled" in report
    assert "Paired vs reference: chance" in report
    assert "exploratory" in report  # non-official runs are labeled as such
    svg = plot_curves(runs, tmp_path / "curves.svg")
    assert svg.read_text().startswith("<?xml")


def test_within_session_auc_and_compare() -> None:
    cfg = SyntheticMIConfig(
        n_subjects=2, n_trials_per_class=25, n_sessions=2, efficiencies=(1.0, 1.0)
    )
    ep = generate_synthetic_mi(cfg).epochs
    scores = within_session_auc(ep, "ts_lr")
    assert len(scores) == 4  # 2 subjects x 2 sessions
    assert np.mean(list(scores.values())) > 0.8
    assert within_session_auc(ep, "csp_lda")
    reference = {(s, e): v - 0.01 for (_, s, e), v in scores.items()}
    assert compare(scores, reference, "ts_lr").passed
    assert not compare(scores, {(s, e): v - 0.2 for (s, e), v in reference.items()}, "ts_lr").passed
    assert not compare(scores, {}, "ts_lr").passed
    with pytest.raises(ValueError, match="unknown Gate 0 pipeline"):
        sklearn_pipeline("svm")
    spec = gate0_pipeline(3.0)
    assert spec.epoching.tmax == 3.0
    assert spec.transforms[1].params == {"l_freq": 8.0, "h_freq": 32.0, "order": 4}


@pytest.mark.network
def test_moabb_reference_physionet() -> None:
    """Opt-in: runs MOABB's own evaluation on 2 PhysioNet subjects (downloads data)."""
    from neurolayer.experiments.gate0 import moabb_reference

    table = moabb_reference("PhysionetMI", [1, 2])
    assert set(table) == {"csp_lda", "ts_lr"}
