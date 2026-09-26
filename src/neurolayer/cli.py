"""Command-line entry point: ``neurolayer <command>``.

Commands
--------
``catalog list``
    Show every catalogued dataset with its roles and license status.
``catalog check --purpose P [--datasets a,b]``
    Run the license gate; exits non-zero if any dataset is refused.
``run CONFIG [--official]``
    Run an experiment config end to end and write a run directory.
``smoke``
    Run the synthetic end-to-end smoke experiment (used in CI).
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sequence
from pathlib import Path

from neurolayer import __version__
from neurolayer.data.catalog import LicenseGateError, Purpose, evaluate_usage, load_catalog
from neurolayer.experiments.config import load_experiment_config
from neurolayer.experiments.runner import DirtyTreeError, ExperimentOutcome, run_experiment

DEFAULT_CATALOG = Path("catalog/datasets")
DEFAULT_OUTPUT = Path("artifacts/runs")
SMOKE_CONFIG = Path("configs/experiments/smoke_synthetic.yaml")


def _print_outcome(outcome: ExperimentOutcome) -> None:
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
    print(f"AUCEC:      {summary['aucec']:.3f}")
    print(f"median TTC: {'never' if ttc is None or math.isinf(ttc) else ttc} trials/class")
    print(f"written to: {outcome.run_dir}")
    if outcome.mlflow_run_id:
        print(f"mlflow run: {outcome.mlflow_run_id}")


def _cmd_catalog_list(args: argparse.Namespace) -> int:
    catalog = load_catalog(args.catalog)
    print(f"{'id':<22} {'roles':<32} {'license':<18} {'verified':<9} {'training?'}")
    for card in catalog.values():
        verified = "yes" if card.license.is_verified else "NO"
        training = "allowed" if evaluate_usage(card, Purpose.TRAINING).allowed else "blocked"
        roles = ",".join(card.roles)
        print(f"{card.id:<22} {roles:<32} {card.license.spdx:<18} {verified:<9} {training}")
    return 0


def _cmd_catalog_check(args: argparse.Namespace) -> int:
    catalog = load_catalog(args.catalog)
    ids = args.datasets.split(",") if args.datasets else list(catalog)
    purpose = Purpose(args.purpose)
    refused = 0
    for dataset_id in ids:
        if dataset_id not in catalog:
            print(f"UNKNOWN  {dataset_id}")
            refused += 1
            continue
        decision = evaluate_usage(catalog[dataset_id], purpose)
        refused += not decision.allowed
        status = "ALLOWED" if decision.allowed else "REFUSED"
        print(f"{status:<8} {dataset_id:<22} {'; '.join(decision.reasons)}")
    return 1 if refused else 0


def _cmd_run(args: argparse.Namespace, config_path: Path) -> int:
    config = load_experiment_config(config_path)
    try:
        outcome = run_experiment(
            config,
            catalog_dir=args.catalog,
            output_dir=args.output,
            repo_root=Path.cwd(),
            official=getattr(args, "official", False),
            mlflow=getattr(args, "mlflow", False),
            mlflow_uri=getattr(args, "mlflow_uri", None),
        )
    except (LicenseGateError, DirtyTreeError, NotImplementedError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    _print_outcome(outcome)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(prog="neurolayer", description=__doc__.splitlines()[0])
    parser.add_argument("--version", action="version", version=f"neurolayer {__version__}")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG, help="dataset card dir")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="run output dir")
    commands = parser.add_subparsers(dest="command", required=True)

    catalog = commands.add_parser("catalog", help="dataset catalog and license gate")
    catalog_commands = catalog.add_subparsers(dest="catalog_command", required=True)
    catalog_commands.add_parser("list", help="list catalogued datasets")
    check = catalog_commands.add_parser("check", help="run the license gate")
    check.add_argument("--purpose", choices=[p.value for p in Purpose], required=True)
    check.add_argument("--datasets", help="comma-separated dataset ids (default: all)")

    run = commands.add_parser("run", help="run an experiment config")
    run.add_argument("config", type=Path)
    run.add_argument("--official", action="store_true", help="require a clean git tree")
    run.add_argument("--mlflow", action="store_true", help="also mirror the run to MLflow")
    run.add_argument("--mlflow-uri", help="MLflow tracking URI (default: sqlite:///mlflow.db)")

    commands.add_parser("smoke", help="run the synthetic end-to-end smoke experiment")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``neurolayer`` console script."""
    args = build_parser().parse_args(argv)
    if args.command == "catalog":
        return (
            _cmd_catalog_list(args) if args.catalog_command == "list" else _cmd_catalog_check(args)
        )
    if args.command == "run":
        return _cmd_run(args, args.config)
    return _cmd_run(args, SMOKE_CONFIG)


if __name__ == "__main__":
    raise SystemExit(main())
