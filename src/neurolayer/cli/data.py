"""``neurolayer data fetch|verify|audit|qa``: real-dataset ingestion (Stage 1).

All commands go through the dataset's catalog card: an adapter refuses datasets whose
license does not permit at least local exploration (ADR-0005).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from neurolayer.data.adapters import get_adapter
from neurolayer.data.audit import audit_dataset, audit_markdown
from neurolayer.data.catalog import DatasetCard, load_catalog
from neurolayer.data.qa import qa_markdown, qa_recording
from neurolayer.data.storage import (
    bids_root,
    raw_dir,
    verify_checksums,
    write_bids,
    write_checksums,
)


def _card(args: argparse.Namespace, dataset_id: str) -> DatasetCard:
    catalog = load_catalog(args.catalog)
    if dataset_id not in catalog:
        raise KeyError(f"dataset {dataset_id!r} is not in the catalog ({args.catalog})")
    return catalog[dataset_id]


def _subjects(requested: str | None, available: list[str], limit: int | None = None) -> list[str]:
    if requested:
        chosen = [s.strip() for s in requested.split(",") if s.strip()]
        unknown = [s for s in chosen if s not in available]
        if unknown:
            raise ValueError(f"unknown subjects {unknown}; available: {available[:10]}...")
        return chosen
    return available[:limit] if limit else available


def _fetch(args: argparse.Namespace) -> int:
    card = _card(args, args.dataset)
    target = raw_dir(args.data_root, card)
    adapter = get_adapter(card, target)
    subjects = _subjects(args.subjects, adapter.subjects())
    print(f"downloading {card.id} subjects {subjects} into {target} ...")
    adapter.download(subjects)
    manifest = write_checksums(target)
    print(f"checksums: {manifest}")
    if not args.no_bids:
        root = bids_root(args.data_root, card)
        files = [f for s in subjects for f in write_bids(adapter.recordings(s), root)]
        print(f"BIDS: {len(files)} recordings written under {root}")
    print(f"next: version the download with `uvx dvc add {target}` and commit the .dvc file")
    return 0


def _verify(args: argparse.Namespace) -> int:
    card = _card(args, args.dataset)
    problems = verify_checksums(raw_dir(args.data_root, card))
    for problem in problems:
        print(problem)
    print("intact" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


def _audit(args: argparse.Namespace) -> int:
    audits = []
    for dataset_id in args.datasets:
        card = _card(args, dataset_id)
        adapter = get_adapter(card, raw_dir(args.data_root, card))
        subjects = _subjects(None, adapter.subjects(), args.max_subjects)
        recordings = (r for s in subjects for r in adapter.recordings(s))
        audits.append(audit_dataset(card.id, recordings))
    report = audit_markdown(audits, args.min_test_per_class)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        print(f"written to {args.out}")
    else:
        print(report)
    return 0


def _qa(args: argparse.Namespace) -> int:
    card = _card(args, args.dataset)
    adapter = get_adapter(card, raw_dir(args.data_root, card))
    subjects = _subjects(args.subjects, adapter.subjects(), args.max_subjects)
    results = []
    for subject in subjects:
        for recording in adapter.recordings(subject):
            key = (recording.subject_id, recording.session_id, recording.run_id)
            results.append(qa_recording(recording, adapter.log.reports.get(key)))
    print(qa_markdown(results))
    out = args.out or Path("artifacts/qa") / f"{card.id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([r.to_dict() for r in results], indent=2) + "\n", encoding="utf-8")
    print(f"details: {out}")
    return 1 if args.strict and any(r.issues for r in results) else 0


def _guarded(handler: Any) -> Any:
    def run(args: argparse.Namespace) -> int:
        try:
            code: int = handler(args)
        except (KeyError, ValueError, PermissionError, RuntimeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        return code

    return run


def register(commands: Any) -> None:
    """Register the ``data`` command group."""
    data = commands.add_parser("data", help="fetch, verify, audit and QA datasets")
    sub = data.add_subparsers(dest="data_command", required=True)

    fetch = sub.add_parser("fetch", help="download a dataset, write checksums and BIDS")
    fetch.add_argument("dataset")
    fetch.add_argument("--subjects", help="comma-separated native subject ids (default: all)")
    fetch.add_argument("--no-bids", action="store_true", help="skip the BIDS conversion")
    fetch.set_defaults(handler=_guarded(_fetch))

    verify = sub.add_parser("verify", help="verify raw-data checksums")
    verify.add_argument("dataset")
    verify.set_defaults(handler=_guarded(_verify))

    audit = sub.add_parser("audit", help="channel/label audit across datasets (WP-1.4)")
    audit.add_argument("datasets", nargs="+")
    audit.add_argument("--max-subjects", type=int, default=None)
    audit.add_argument("--min-test-per-class", type=int, default=10)
    audit.add_argument("--out", type=Path, help="write markdown here instead of stdout")
    audit.set_defaults(handler=_guarded(_audit))

    qa = sub.add_parser("qa", help="per-recording quality report (WP-1.5)")
    qa.add_argument("dataset")
    qa.add_argument("--subjects", help="comma-separated native subject ids")
    qa.add_argument("--max-subjects", type=int, default=None)
    qa.add_argument("--out", type=Path, help="JSON details path (default artifacts/qa/<id>.json)")
    qa.add_argument("--strict", action="store_true", help="exit 1 if any recording is flagged")
    qa.set_defaults(handler=_guarded(_qa))
