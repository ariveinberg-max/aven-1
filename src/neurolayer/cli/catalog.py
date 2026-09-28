"""``neurolayer catalog list|check``: dataset catalog and license gate."""

from __future__ import annotations

import argparse
from typing import Any

from neurolayer.data.catalog import Purpose, evaluate_usage, load_catalog


def _list(args: argparse.Namespace) -> int:
    catalog = load_catalog(args.catalog)
    print(f"{'id':<22} {'roles':<32} {'license':<18} {'verified':<9} {'training?'}")
    for card in catalog.values():
        verified = "yes" if card.license.is_verified else "NO"
        training = "allowed" if evaluate_usage(card, Purpose.TRAINING).allowed else "blocked"
        roles = ",".join(card.roles)
        print(f"{card.id:<22} {roles:<32} {card.license.spdx:<18} {verified:<9} {training}")
    return 0


def _check(args: argparse.Namespace) -> int:
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


def register(commands: Any) -> None:
    """Register the ``catalog`` command group."""
    catalog = commands.add_parser("catalog", help="dataset catalog and license gate")
    sub = catalog.add_subparsers(dest="catalog_command", required=True)
    sub.add_parser("list", help="list catalogued datasets").set_defaults(handler=_list)
    check = sub.add_parser("check", help="run the license gate")
    check.add_argument("--purpose", choices=[p.value for p in Purpose], required=True)
    check.add_argument("--datasets", help="comma-separated dataset ids (default: all)")
    check.set_defaults(handler=_check)
