"""``neurolayer run CONFIG`` and ``neurolayer smoke``."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

from neurolayer.data.catalog import LicenseGateError
from neurolayer.experiments.config import load_experiment_config
from neurolayer.experiments.runner import DirtyTreeError, ExperimentOutcome, run_experiment
from neurolayer.experiments.training import train_bundle

SMOKE_CONFIG = Path("configs/experiments/smoke_synthetic.yaml")


def print_outcome(outcome: ExperimentOutcome) -> None:
    """Print the calibration-efficiency table of a finished run."""
    summary = outcome.result.summary_dict()
    print(f"run:        {outcome.manifest.run_id}")
    print(f"regime:     {summary['regime']}  (n_unlabeled={summary['n_unlabeled']})")
    scored, skipped = summary["n_subjects_scored"], summary["n_subjects_skipped"]
    print(f"subjects:   {scored} scored, {skipped} skipped")
    print(f"{'k/class':>8} {'mean BA':>8} {'95% CI':>17} {'median':>7} {'UUR@70%':>8}")
    for row in summary["per_budget"]:
        ci = f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}]"
        print(
            f"{row['k']:>8} {row['mean_ba']:>8.3f} {ci:>17} {row['median_ba']:>7.3f} "
            f"{row['usable_user_rate']:>8.2f}"
        )
    ttc = summary["median_ttc"]
    if summary["aucec"] is not None:
        print(f"AUCEC:      {summary['aucec']:.3f}")
    print(f"median TTC: {'never' if ttc is None or math.isinf(ttc) else ttc} trials/class")
    print(f"written to: {outcome.run_dir}")
    if outcome.mlflow_run_id:
        print(f"mlflow run: {outcome.mlflow_run_id}")


def _run_config(args: argparse.Namespace, config_path: Path) -> int:
    config = load_experiment_config(config_path)
    try:
        outcome = run_experiment(
            config,
            catalog_dir=args.catalog,
            output_dir=args.output,
            repo_root=Path.cwd(),
            data_root=args.data_root,
            official=getattr(args, "official", False),
            mlflow=getattr(args, "mlflow", False),
            mlflow_uri=getattr(args, "mlflow_uri", None),
        )
    except (LicenseGateError, DirtyTreeError, NotImplementedError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print_outcome(outcome)
    return 0


def _run(args: argparse.Namespace) -> int:
    return _run_config(args, args.config)


def _smoke(args: argparse.Namespace) -> int:
    return _run_config(args, SMOKE_CONFIG)


def _train(args: argparse.Namespace) -> int:
    config = load_experiment_config(args.config)
    try:
        info = train_bundle(
            config,
            catalog_dir=args.catalog,
            data_root=args.data_root,
            repo_root=Path.cwd(),
            out_dir=args.models_dir,
            version=args.version,
        )
    except (LicenseGateError, ValueError, FileExistsError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"bundle:  {info.model_id}")
    print(f"weights: sha256 {info.weights_sha256}")
    print(f"written: {args.models_dir / info.name / info.version}")
    return 0


def register(commands: Any) -> None:
    """Register ``run``, ``smoke`` and ``train``."""
    run = commands.add_parser("run", help="run an experiment config")
    run.add_argument("config", type=Path)
    run.add_argument("--official", action="store_true", help="require a clean git tree")
    run.add_argument("--mlflow", action="store_true", help="also mirror the run to MLflow")
    run.add_argument("--mlflow-uri", help="MLflow tracking URI (default: sqlite:///mlflow.db)")
    run.set_defaults(handler=_run)
    smoke = commands.add_parser("smoke", help="run the synthetic end-to-end smoke experiment")
    smoke.set_defaults(handler=_smoke)
    train = commands.add_parser("train", help="fit the proprietary model and export a bundle")
    train.add_argument("config", type=Path)
    train.add_argument("--version", required=True, help="bundle version, e.g. 0.1.0")
    train.add_argument("--models-dir", type=Path, default=Path("artifacts/models"))
    train.set_defaults(handler=_train)
