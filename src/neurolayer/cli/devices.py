"""``neurolayer consent|pilot|bridge``: consent ledger, pilot recording, live bridge.

Pilot data and the consent ledger live under ``<data-root>/pilot/`` (git-ignored; keep
the data root on encrypted storage because this is C3 personal data). Access tokens are
read from the ``NEUROLAYER_TOKEN`` environment variable, never from the command line.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from neurolayer.data.consent import ConsentPurpose, ConsentRegistry
from neurolayer.data.qa import qa_recording
from neurolayer.devices.bridge import run_bridge
from neurolayer.devices.client import ApiError, NeurolayerClient
from neurolayer.devices.output import DebouncedSink, IntentSink, KeyboardSink, PrintSink
from neurolayer.devices.protocol import Cue, CueSchedule
from neurolayer.devices.recorder import record_session, save_pilot_recording
from neurolayer.devices.sources import (
    BrainFlowSource,
    LSLSource,
    StreamSource,
    simulated_source,
)

ARROWS = {"left_hand": "<-- LEFT hand", "right_hand": "RIGHT hand -->"}


def _ledger(args: argparse.Namespace) -> ConsentRegistry:
    path = args.ledger or args.data_root / "pilot" / "consent" / "ledger.jsonl"
    return ConsentRegistry(path)


def _purposes(text: str) -> list[ConsentPurpose]:
    return [ConsentPurpose(p.strip()) for p in text.split(",") if p.strip()]


def _source(args: argparse.Namespace) -> StreamSource:
    if args.lsl:
        return LSLSource(stream_type=args.lsl)
    if args.board == "simulated":
        return simulated_source(_schedule(args), seed=args.seed)
    channels = [c.strip() for c in args.channels.split(",")] if args.channels else None
    return BrainFlowSource(
        args.board,
        serial_port=args.serial_port,
        mac_address=args.mac_address,
        channel_names=channels,
    )


def _schedule(args: argparse.Namespace) -> CueSchedule:
    return CueSchedule(
        n_trials_per_class=args.trials_per_class,
        rest_s=args.rest_s,
        imagery_s=args.imagery_s,
        seed=args.seed,
    )


def _show_cue(total: int) -> Any:
    def show(cue: Cue) -> None:
        print(f"[{cue.index + 1:>3}/{total}] {ARROWS.get(cue.label, cue.label)}", flush=True)

    return show


def _grant(args: argparse.Namespace) -> int:
    events = _ledger(args).grant(
        args.participant,
        _purposes(args.purposes),
        document=args.document,
        document_version=args.document_version,
        recorded_by=args.recorded_by,
    )
    for event in events:
        print(f"granted {event.purpose.value} for {event.participant} ({event.timestamp})")
    return 0


def _revoke(args: argparse.Namespace) -> int:
    ledger = _ledger(args)
    for purpose in _purposes(args.purposes):
        ledger.revoke(args.participant, purpose, args.recorded_by)
        print(f"revoked {purpose.value} for {args.participant}")
    if ConsentPurpose.RESEARCH in _purposes(args.purposes):
        print("next: delete this participant's recordings (PRIV-4) and log the deletion")
    return 0


def _check(args: argparse.Namespace) -> int:
    ledger = _ledger(args)
    for purpose in ConsentPurpose:
        state = "granted" if ledger.allowed(args.participant, purpose) else "no"
        print(f"{purpose.value:<20} {state}")
    return 0


def _record(args: argparse.Namespace) -> int:
    ledger = _ledger(args)
    ledger.require(args.participant, [ConsentPurpose.RESEARCH])
    schedule = _schedule(args)
    source = _source(args)
    total = len(schedule.labels) * schedule.n_trials_per_class
    print(f"recording {schedule.duration_s:.0f} s, {total} cues; relax, imagine on each arrow")
    recording = record_session(
        source,
        schedule,
        participant=args.participant,
        session_id=args.session,
        device=f"lsl:{args.lsl}" if args.lsl else args.board,
        on_cue=_show_cue(total),
    )
    qa = qa_recording(recording)
    print("QA: " + ("clean" if not qa.issues else "; ".join(qa.issues)))
    written = save_pilot_recording(recording, schedule, args.data_root / "pilot" / "bids", ledger)
    for path in written:
        print(f"written: {path}")
    return 0


def _bridge(args: argparse.Namespace) -> int:
    client = NeurolayerClient.connect(args.api, os.environ.get("NEUROLAYER_TOKEN"))
    base: IntentSink = KeyboardSink() if args.sink == "keyboard" else PrintSink()
    sink = DebouncedSink(base, threshold=args.threshold, min_consecutive=args.consecutive)
    schedule = _schedule(args)
    total = len(schedule.labels) * schedule.n_trials_per_class
    print(f"calibration: {schedule.duration_s:.0f} s, {total} cues")
    result = run_bridge(
        _source(args),
        client,
        args.model,
        schedule,
        sink,
        on_cue=_show_cue(total),
        hop_s=args.hop_s,
        duration_s=args.duration,
    )
    print(f"done: {result.calibration_trials} calibration trials, {result.decoded} decodes")
    return 0


def _guarded(handler: Any) -> Any:
    def run(args: argparse.Namespace) -> int:
        try:
            code: int = handler(args)
        except (KeyError, ValueError, PermissionError, RuntimeError, ImportError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        except ApiError as exc:
            print(f"api error: {exc}", file=sys.stderr)
            return 2
        return code

    return run


def _source_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--board",
        default="synthetic",
        help="simulated (cue-aware synthetic), synthetic (BrainFlow), CYTON_BOARD, ...",
    )
    parser.add_argument("--serial-port", default="", help="e.g. COM3 or /dev/ttyUSB0")
    parser.add_argument("--mac-address", default="")
    parser.add_argument("--channels", help="comma-separated 10-05 names (configurable boards)")
    parser.add_argument("--lsl", metavar="TYPE", help="use an LSL stream of this type instead")


def _schedule_args(parser: argparse.ArgumentParser, trials: int) -> None:
    parser.add_argument("--trials-per-class", type=int, default=trials)
    parser.add_argument("--rest-s", type=float, default=2.0)
    parser.add_argument("--imagery-s", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=0)


def register(commands: Any) -> None:
    """Register the ``consent``, ``pilot`` and ``bridge`` commands."""
    consent = commands.add_parser("consent", help="participant consent ledger (WP-8.2)")
    consent.add_argument("--ledger", type=Path, help="default <data-root>/pilot/consent/...")
    sub = consent.add_subparsers(dest="consent_command", required=True)
    grant = sub.add_parser("grant", help="record consent for one or more purposes")
    grant.add_argument("participant", help="pseudonym, e.g. P001")
    grant.add_argument("--purposes", required=True, help="product,research,commercial_training")
    grant.add_argument("--document", type=Path, required=True, help="consent form they signed")
    grant.add_argument("--document-version", required=True)
    grant.add_argument("--recorded-by", required=True)
    grant.set_defaults(handler=_guarded(_grant))
    revoke = sub.add_parser("revoke", help="record a withdrawal")
    revoke.add_argument("participant")
    revoke.add_argument("--purposes", required=True)
    revoke.add_argument("--recorded-by", required=True)
    revoke.set_defaults(handler=_guarded(_revoke))
    check = sub.add_parser("check", help="show a participant's current consent")
    check.add_argument("participant")
    check.set_defaults(handler=_guarded(_check))

    pilot = commands.add_parser("pilot", help="pilot data collection (WP-8.3)")
    pilot_sub = pilot.add_subparsers(dest="pilot_command", required=True)
    record = pilot_sub.add_parser("record", help="record one cued session (needs consent)")
    record.add_argument("participant", help="pseudonym, e.g. P001")
    record.add_argument("--session", default="1")
    record.add_argument("--ledger", type=Path)
    _source_args(record)
    _schedule_args(record, trials=20)
    record.set_defaults(handler=_guarded(_record))

    bridge = commands.add_parser("bridge", help="calibrate and decode a device live (WP-7.3)")
    bridge.add_argument("--api", default="http://localhost:8000")
    bridge.add_argument("--model", required=True, help="model id, e.g. name@version")
    bridge.add_argument("--sink", choices=("print", "keyboard"), default="print")
    bridge.add_argument("--threshold", type=float, default=0.7)
    bridge.add_argument("--consecutive", type=int, default=2)
    bridge.add_argument("--hop-s", type=float, default=0.5)
    bridge.add_argument("--duration", type=float, default=60.0, help="seconds of live decoding")
    _source_args(bridge)
    _schedule_args(bridge, trials=10)
    bridge.set_defaults(handler=_guarded(_bridge))
