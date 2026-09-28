"""Streaming sources: anything that yields multichannel samples in microvolts.

``StreamSource`` is the contract; implementations wrap BrainFlow boards, LSL streams, or
replay synthetic data (tests, demos). Channel names are canonical 10-05 names, so every
downstream component (recorder, decoder) is device-agnostic.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np

from neurolayer.core.channels import normalize_channel_names
from neurolayer.core.types import FloatArray, Recording
from neurolayer.data.synthetic import SyntheticMIConfig, generate_synthetic_recordings
from neurolayer.devices.protocol import CueSchedule


@runtime_checkable
class StreamSource(Protocol):
    """A live (or replayed) multichannel stream."""

    ch_names: tuple[str, ...]
    sfreq: float

    def start(self) -> None:
        """Begin streaming."""
        ...

    def read(self) -> FloatArray:
        """Return all samples since the last read, shape ``(n_channels, n_new)``, in µV."""
        ...

    def stop(self) -> None:
        """Stop streaming and release the device."""
        ...


class ReplaySource:
    """Replays a recording as if it were live (optionally in real time).

    Used for demos and tests; ``realtime=False`` returns ``chunk`` samples per read.
    """

    def __init__(self, recording: Recording, chunk: int = 32, realtime: bool = False) -> None:
        self.recording = recording  # ground truth (events) for tests and demos
        self.ch_names = recording.ch_names
        self.sfreq = recording.sfreq
        self._data = recording.data / 1e-6  # volts → microvolts
        self._chunk = chunk
        self._realtime = realtime
        self._cursor = 0
        self._started_at = 0.0

    def start(self) -> None:
        """Resume from the current position (a fresh source starts at the beginning)."""
        self._started_at = time.monotonic() - self._cursor / self.sfreq

    def rewind(self) -> None:
        """Go back to the first sample."""
        self._cursor = 0

    def read(self) -> FloatArray:
        """Next chunk (or everything that 'arrived' since start in real-time mode)."""
        if self._realtime:
            elapsed = time.monotonic() - self._started_at
            target = min(self._data.shape[1], int(elapsed * self.sfreq))
        else:
            target = min(self._data.shape[1], self._cursor + self._chunk)
        out: FloatArray = self._data[:, self._cursor : target]
        self._cursor = target
        return out

    def stop(self) -> None:
        """Nothing to release."""

    @property
    def exhausted(self) -> bool:
        """``True`` once every sample has been read."""
        return bool(self._cursor >= self._data.shape[1])


class BrainFlowSource:
    """A BrainFlow board (``devices`` extra). ``board="synthetic"`` needs no hardware.

    Parameters
    ----------
    board
        BrainFlow ``BoardIds`` name (``SYNTHETIC_BOARD``, ``CYTON_BOARD``, ``MUSE_S_BOARD``,
        ``NOTION_2_BOARD``...) or the shortcut ``"synthetic"``.
    serial_port, mac_address
        Connection parameters for real hardware.
    channel_names
        Override the board's channel labels (required for configurable boards such as the
        OpenBCI Cyton, whose electrodes are placed by the user).
    """

    def __init__(
        self,
        board: str = "synthetic",
        serial_port: str = "",
        mac_address: str = "",
        channel_names: Sequence[str] | None = None,
    ) -> None:
        from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams

        BoardShim.disable_board_logger()
        name = "SYNTHETIC_BOARD" if board == "synthetic" else board.upper()
        self._board_id = int(getattr(BoardIds, name).value)
        params = BrainFlowInputParams()
        params.serial_port = serial_port
        params.mac_address = mac_address
        self._shim: Any = BoardShim(self._board_id, params)
        self._eeg_rows = list(BoardShim.get_eeg_channels(self._board_id))
        labels = channel_names or BoardShim.get_eeg_names(self._board_id)
        if len(labels) != len(self._eeg_rows):
            raise ValueError(f"{len(labels)} channel names for {len(self._eeg_rows)} EEG channels")
        self.ch_names = normalize_channel_names(labels)
        self.sfreq = float(BoardShim.get_sampling_rate(self._board_id))

    def start(self) -> None:
        """Open the device and start streaming."""
        self._shim.prepare_session()
        self._shim.start_stream()

    def read(self) -> FloatArray:
        """All EEG samples since the last read (BrainFlow reports microvolts)."""
        data = self._shim.get_board_data()
        out: FloatArray = np.asarray(data[self._eeg_rows], dtype=np.float64)
        return out

    def stop(self) -> None:
        """Stop streaming and release the device."""
        self._shim.stop_stream()
        self._shim.release_session()


class LSLSource:
    """Any Lab Streaming Layer EEG stream (``devices`` extra), e.g. from the Neurosity SDK."""

    def __init__(self, stream_type: str = "EEG", timeout_s: float = 5.0) -> None:
        from pylsl import StreamInlet, resolve_byprop

        streams = resolve_byprop("type", stream_type, timeout=timeout_s)
        if not streams:
            raise RuntimeError(f"no LSL stream of type {stream_type!r} found")
        self._inlet: Any = StreamInlet(streams[0])
        info = self._inlet.info()
        labels, channel = [], info.desc().child("channels").child("channel")
        for _ in range(info.channel_count()):
            labels.append(channel.child_value("label"))
            channel = channel.next_sibling()
        self.ch_names = normalize_channel_names(labels)
        self.sfreq = float(info.nominal_srate())

    def start(self) -> None:
        """Open the inlet."""
        self._inlet.open_stream()

    def read(self) -> FloatArray:
        """All pending samples (assumes the stream is in microvolts, as is conventional)."""
        chunk, _ = self._inlet.pull_chunk()
        out: FloatArray = np.asarray(chunk, dtype=np.float64).T.reshape(len(self.ch_names), -1)
        return out

    def stop(self) -> None:
        """Close the inlet."""
        self._inlet.close_stream()


def simulated_source(
    schedule: CueSchedule,
    *,
    live_trials: int = 40,
    efficiency: float = 0.8,
    seed: int = 0,
    realtime: bool = True,
    chunk: int = 8,
) -> ReplaySource:
    """Build a synthetic "headset" that imagines exactly what ``schedule`` cues.

    The stream follows the schedule's timing and labels, then continues with
    ``live_trials`` randomly chosen trials for live decoding. Lets the pilot recorder,
    the bridge and the dashboard run end to end without hardware. Synthetic data only.
    """
    rng = np.random.default_rng([seed, 7])
    cued = [cue.label for cue in schedule.cues()]
    live = [str(label) for label in rng.choice(schedule.labels, size=live_trials)]
    config = SyntheticMIConfig(
        n_subjects=1,
        n_trials_per_class=schedule.n_trials_per_class,
        trial_seconds=schedule.imagery_s,
        signal_to_noise=0.5,
        efficiencies=(efficiency,),
        seed=seed,
        dataset_id="simulated",
    )
    (recording,) = generate_synthetic_recordings(config, schedule.rest_s, cue_labels=cued + live)
    return ReplaySource(recording, chunk=chunk, realtime=realtime)
