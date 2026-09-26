"""Sandboxed parser for uploaded recordings (WP-7.1; ``python -m`` entry point).

Uploaded files are untrusted. The API never parses them in its own process: it runs
this module in a subprocess with a wall-clock timeout, and the worker caps its own
memory, CPU time and open files before touching the file. Output is one JSON object on
stdout. Exit codes: ``0`` ok, ``3`` unreadable file, ``4`` parser not installed.

The report contains channel names, rates, annotation counts, band power, QA flags and a
short downsampled preview for plotting. It is returned only to the uploader and never
logged.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

MAX_MEMORY_BYTES = 2 * 1024**3
MAX_CPU_SECONDS = 20
MAX_OPEN_FILES = 64
PREVIEW_SECONDS = 10.0
PREVIEW_POINTS = 500
PREVIEW_CHANNELS = 8
BANDS = {"mu": (8.0, 13.0), "beta": (13.0, 30.0)}

EXIT_UNREADABLE = 3
EXIT_NO_PARSER = 4


def _limit_resources() -> None:
    import resource  # POSIX only; main() refuses to parse without it

    for limit, value in (
        (resource.RLIMIT_AS, MAX_MEMORY_BYTES),
        (resource.RLIMIT_CPU, MAX_CPU_SECONDS),
        (resource.RLIMIT_NOFILE, MAX_OPEN_FILES),
    ):
        _, hard = resource.getrlimit(limit)
        cap = value if hard == resource.RLIM_INFINITY else min(value, hard)
        resource.setrlimit(limit, (cap, hard))


def _band_power_db(data: np.ndarray[Any, Any], sfreq: float) -> dict[str, list[float | None]]:
    """Mean band power in dB re 1 µV²/Hz (``None`` for flat channels or bands above Nyquist)."""
    from scipy.signal import welch

    nperseg = int(min(data.shape[1], 2 * sfreq))
    freqs, power = welch(data / 1e-6, fs=sfreq, nperseg=nperseg, axis=1)
    out: dict[str, list[float | None]] = {}
    for band, (low, high) in BANDS.items():
        mask = (freqs >= low) & (freqs < high)
        if not mask.any() or high >= sfreq / 2:
            out[band] = [None] * data.shape[0]
            continue
        values = power[:, mask].mean(axis=1)
        out[band] = [round(float(10 * np.log10(v)), 2) if v > 1e-6 else None for v in values]
    return out


def _preview(data: np.ndarray[Any, Any], sfreq: float) -> tuple[float, list[list[float]]]:
    n = min(data.shape[1], int(PREVIEW_SECONDS * sfreq))
    step = max(1, int(np.ceil(n / PREVIEW_POINTS)))
    window = data[:PREVIEW_CHANNELS, :n:step] / 1e-6
    return sfreq / step, np.round(window, 2).tolist()


def inspect(path: Path, file_format: str) -> dict[str, Any]:
    """Parse ``path`` (``edf`` or ``bdf``) and summarize it."""
    import mne

    from neurolayer.core.channels import is_standard, normalize_channel_name
    from neurolayer.core.types import Recording
    from neurolayer.data.qa import qa_recording

    reader = mne.io.read_raw_bdf if file_format == "bdf" else mne.io.read_raw_edf
    raw = reader(path, preload=True, verbose="error")
    if "eeg" not in raw.get_channel_types():
        raise ValueError("no EEG channels")
    eeg = raw.copy().pick("eeg", exclude=[])
    data = np.ascontiguousarray(eeg.get_data(), dtype=np.float64)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    sfreq = float(eeg.info["sfreq"])
    raw_names = list(eeg.ch_names)
    canonical = [normalize_channel_name(n) for n in raw_names]
    unique = len(set(canonical)) == len(canonical)
    names = tuple(canonical) if unique else tuple(raw_names)
    annotations: dict[str, int] = {}
    for description in raw.annotations.description:
        annotations[str(description)] = annotations.get(str(description), 0) + 1
    qa = qa_recording(
        Recording(data=data, sfreq=sfreq, ch_names=names, dataset_id="upload", subject_id="upload")
    )
    power = _band_power_db(data, sfreq)
    preview_rate, preview = _preview(data, sfreq)
    return {
        "format": file_format,
        "sfreq": sfreq,
        "duration_s": round(data.shape[1] / sfreq, 3),
        "n_channels": len(names),
        "channels": [
            {
                "name": raw_name,
                "canonical": name if is_standard(name) else None,
                "mu_db": power["mu"][i],
                "beta_db": power["beta"][i],
                "flat": name in qa.flat_channels,
                "noisy": name in qa.noisy_channels,
            }
            for i, (raw_name, name) in enumerate(zip(raw_names, names, strict=True))
        ],
        "annotations": annotations,
        "line_noise_ratio": qa.line_noise_ratio,
        "median_abs_amplitude_uv": round(qa.median_abs_amplitude_v / 1e-6, 3),
        "issues": [i for i in qa.issues if i != "no events"],
        "preview": {
            "sfreq": preview_rate,
            "channels": list(raw_names[:PREVIEW_CHANNELS]),
            "data_uv": preview,
        },
    }


def main(argv: list[str] | None = None) -> int:
    """``python -m neurolayer_api.inspect_worker PATH FORMAT``."""
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2 or args[1] not in {"edf", "bdf"}:
        print(json.dumps({"error": "usage: inspect_worker PATH edf|bdf"}))
        return 2
    try:
        _limit_resources()
    except ImportError:
        print(json.dumps({"error": "file inspection needs a POSIX host (resource limits)"}))
        return EXIT_NO_PARSER
    try:
        import mne  # noqa: F401
    except ImportError:
        print(json.dumps({"error": "file inspection needs the 'neuro' extra (mne)"}))
        return EXIT_NO_PARSER
    try:
        report = inspect(Path(args[0]), args[1])
    except Exception as exc:  # any parse failure means an unreadable upload
        print(json.dumps({"error": f"could not read the file: {type(exc).__name__}"}))
        return EXIT_UNREADABLE
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
    raise SystemExit(main())
