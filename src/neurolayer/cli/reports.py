"""``neurolayer report compare``, ``neurolayer probe`` and ``neurolayer gate0`` (Stage 4)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from neurolayer.data.catalog import LicenseGateError, Purpose, load_catalog, require_usage
from neurolayer.experiments.config import load_experiment_config
from neurolayer.experiments.controls import run_shuffle_control
from neurolayer.experiments.gate0 import (
    PIPELINES,
    compare,
    gate0_pipeline,
    moabb_reference,
    within_session_auc,
)
from neurolayer.experiments.probes import probe
from neurolayer.experiments.reports import comparison_markdown, load_run, plot_curves
from neurolayer.representation.registry import ENCODERS, make_encoder
from neurolayer.signal.build import build_epochs


def _compare(args: argparse.Namespace) -> int:
    runs = [load_run(Path(d)) for d in args.runs]
    reference = load_run(Path(args.reference)) if args.reference else None
    if reference is not None and all(r.run_id != reference.run_id for r in runs):
        runs.insert(0, reference)
    if reference is not None:
        reference = next(r for r in runs if r.run_id == reference.run_id)
    report = comparison_markdown(runs, reference)
    if args.plot:
        plot_curves(runs, args.plot)
        report += f"\n![Calibration-efficiency curves]({args.plot.name})\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        print(f"written to {args.out}")
    else:
        print(report)
    return 0


def _probe(args: argparse.Namespace) -> int:
    config = load_experiment_config(args.config)
    catalog = load_catalog(args.catalog)
    require_usage([catalog[d] for d in config.datasets], config.purpose)
    epochs, _ = build_epochs(
        config.datasets,
        catalog=catalog,
        data_root=args.data_root,
        pipeline=config.pipeline,
        synthetic=config.synthetic.to_generator_config(),
        max_subjects=config.max_subjects,
    )
    targets = ["subject"] + (["dataset"] if len(epochs.datasets()) > 1 else [])
    print("| encoder | target | classes | probe accuracy | chance | above chance |")
    print("|---|---|---|---|---|---|")
    for name in args.encoders:
        for target in targets:
            result = probe(epochs, make_encoder(name), target=target)  # type: ignore[arg-type]
            print(
                f"| {name} | {target} | {result.n_classes} | {result.accuracy:.3f} | "
                f"{result.chance:.3f} | {result.above_chance:+.3f} |"
            )
    return 0


def _control(args: argparse.Namespace) -> int:
    config = load_experiment_config(args.config)
    catalog = load_catalog(args.catalog)
    require_usage([catalog[d] for d in config.datasets], config.purpose)
    epochs, _ = build_epochs(
        config.datasets,
        catalog=catalog,
        data_root=args.data_root,
        pipeline=config.pipeline,
        synthetic=config.synthetic.to_generator_config(),
        max_subjects=config.max_subjects,
    )
    protocol = config.protocol.to_protocol_config()
    result = run_shuffle_control(epochs, config.decoder, protocol, list(range(args.seeds)))
    print(f"label-shuffle control for {config.decoder.name}, {args.seeds} seeds (ADR-0010)")
    print("| k | mean BA across seeds | SE | compatible with chance |")
    print("|---|---|---|---|")
    for k, mean, se in zip(result.ks, result.mean, result.standard_error, strict=True):
        ok = abs(mean - 0.5) <= max(2 * se, 0.02)
        print(f"| {k} | {mean:.3f} | {se:.3f} | {'yes' if ok else 'NO'} |")
    print("PASS" if result.passed else "FAIL: possible leakage, investigate before any claim")
    return 0 if result.passed else 1


def _gate0(args: argparse.Namespace) -> int:
    catalog = load_catalog(args.catalog)
    card = catalog[args.dataset]
    require_usage([card], Purpose.BENCHMARK)
    if card.loader is None or not card.loader.startswith("moabb:"):
        raise ValueError(f"{card.id} has no MOABB loader; Gate 0 compares against MOABB")
    import moabb.datasets

    moabb_class = card.loader.split(":", 1)[1]
    reference_dataset = getattr(moabb.datasets, moabb_class)()
    task_seconds = float(reference_dataset.interval[1] - reference_dataset.interval[0])
    subjects = [int(s) for s in reference_dataset.subject_list[: args.max_subjects]]
    epochs, _ = build_epochs(
        [card.id],
        catalog=catalog,
        data_root=args.data_root,
        pipeline=gate0_pipeline(task_seconds),
        max_subjects=args.max_subjects,
    )
    reference = moabb_reference(moabb_class, subjects)
    failed = False
    print("| pipeline | sessions | ours (AUC) | MOABB (AUC) | Δ | pass |")
    print("|---|---|---|---|---|---|")
    for name in PIPELINES:
        result = compare(within_session_auc(epochs, name), reference[name], name, args.tolerance)
        failed |= not result.passed
        print(
            f"| {name} | {result.n_sessions} | {result.mean_ours:.3f} | "
            f"{result.mean_reference:.3f} | {result.mean_difference:+.3f} | "
            f"{'yes' if result.passed else 'NO'} |"
        )
    return 1 if failed else 0


def _guarded(handler: Any) -> Any:
    def run(args: argparse.Namespace) -> int:
        try:
            code: int = handler(args)
        except (LicenseGateError, KeyError, ValueError, FileNotFoundError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        return code

    return run


def register(commands: Any) -> None:
    """Register ``report``, ``probe`` and ``gate0``."""
    report = commands.add_parser("report", help="reports across runs")
    sub = report.add_subparsers(dest="report_command", required=True)
    comp = sub.add_parser("compare", help="compare runs (Gate 1 baseline table)")
    comp.add_argument("runs", nargs="+", help="run directories")
    comp.add_argument("--reference", help="run directory used for paired comparisons")
    comp.add_argument("--plot", type=Path, help="write calibration-efficiency curves (SVG)")
    comp.add_argument("--out", type=Path, help="write markdown here instead of stdout")
    comp.set_defaults(handler=_guarded(_compare))

    prb = commands.add_parser("probe", help="identity / dataset-ID leakage probes (WP-4.3)")
    prb.add_argument("config", type=Path, help="experiment config (datasets + pipeline)")
    prb.add_argument("--encoders", nargs="+", default=["tangent_space"], choices=sorted(ENCODERS))
    prb.set_defaults(handler=_guarded(_probe))

    ctl = commands.add_parser("control", help="multi-seed label-shuffle control (ADR-0010)")
    ctl.add_argument("config", type=Path, help="experiment config of the decoder to control")
    ctl.add_argument("--seeds", type=int, default=5)
    ctl.set_defaults(handler=_guarded(_control))

    gate = commands.add_parser("gate0", help="reproduce MOABB within-session results (network)")
    gate.add_argument("dataset")
    gate.add_argument("--max-subjects", type=int, default=5)
    gate.add_argument("--tolerance", type=float, default=0.03)
    gate.set_defaults(handler=_guarded(_gate0))
