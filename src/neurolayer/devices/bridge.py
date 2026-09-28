"""Device → API bridge: calibrate with cues, then decode live into a sink (WP-7.3).

``neurolayer bridge`` wires a stream source (BrainFlow board or LSL stream) to a running
API: it opens a session, runs a short cued calibration (the same protocol as the pilot
and the dashboard game), sends the cue-locked windows as labeled calibration trials,
then decodes the live stream and forwards intents to a sink. The session, and with it
the calibration data, is deleted on the server when the bridge exits.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from neurolayer.devices.client import LiveDecoder, NeurolayerClient
from neurolayer.devices.output import Intent, IntentSink
from neurolayer.devices.protocol import Cue, CueSchedule
from neurolayer.devices.recorder import cue_windows, record_session
from neurolayer.devices.sources import StreamSource


@dataclass(frozen=True, slots=True)
class BridgeResult:
    """Summary of one bridge run (no neural data)."""

    session_id: str
    calibration_trials: int
    decoded: int
    labels: tuple[str, ...]


def run_bridge(
    source: StreamSource,
    client: NeurolayerClient,
    model_id: str,
    schedule: CueSchedule,
    sink: IntentSink,
    *,
    on_cue: Callable[[Cue], None] | None = None,
    hop_s: float = 0.5,
    max_decodes: int | None = None,
    duration_s: float | None = None,
    realtime: bool = True,
) -> BridgeResult:
    """Calibrate on ``schedule`` then decode until ``max_decodes`` or ``duration_s``.

    Raises
    ------
    KeyError
        If ``model_id`` is not served to this caller.
    """
    model = client.models()[model_id]
    session_id = client.create_session(model_id, source.ch_names, source.sfreq)
    decoded: list[Intent] = []
    try:
        calibration = record_session(
            source, schedule, participant="live", on_cue=on_cue, realtime=realtime
        )
        windows, labels = cue_windows(calibration, model.window_offset_s, model.trial_seconds)
        client.calibrate(session_id, windows, labels)
        live = LiveDecoder(source, client, session_id, model.trial_seconds, hop_s)
        source.start()
        started = time.monotonic()
        try:
            while max_decodes is None or len(decoded) < max_decodes:
                if duration_s is not None and time.monotonic() - started > duration_s:
                    break
                intent = live.step()
                if intent is not None:
                    decoded.append(intent)
                    sink(intent)
                elif realtime:
                    time.sleep(0.02)
                elif _exhausted(source):
                    break
        finally:
            source.stop()
    finally:
        client.delete_session(session_id)
    return BridgeResult(session_id, len(windows), len(decoded), model.labels)


def _exhausted(source: StreamSource) -> bool:
    return bool(getattr(source, "exhausted", False))
