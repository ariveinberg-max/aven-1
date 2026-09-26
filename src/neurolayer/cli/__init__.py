"""Command-line entry point: ``neurolayer <command>``.

Command groups live in submodules; each registers its subcommands and a handler:

``catalog``  dataset catalog and license gate (:mod:`neurolayer.cli.catalog`)
``run``      run an experiment config; ``smoke`` runs the synthetic smoke test
             (:mod:`neurolayer.cli.experiments`)
``data``     fetch, verify, audit and QA real datasets (:mod:`neurolayer.cli.data`)
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from neurolayer import __version__
from neurolayer.cli import catalog, data, experiments

DEFAULT_CATALOG = Path("catalog/datasets")
DEFAULT_OUTPUT = Path("artifacts/runs")
DEFAULT_DATA_ROOT = Path("data")


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with every command group registered."""
    parser = argparse.ArgumentParser(
        prog="neurolayer", description="neurolayer: neural intelligence layer tooling"
    )
    parser.add_argument("--version", action="version", version=f"neurolayer {__version__}")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG, help="dataset card dir")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="run output dir")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="data dir")
    commands = parser.add_subparsers(dest="command", required=True)
    for module in (catalog, experiments, data):
        module.register(commands)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``neurolayer`` console script."""
    args = build_parser().parse_args(argv)
    code: int = args.handler(args)
    return code


__all__ = ["build_parser", "main"]
