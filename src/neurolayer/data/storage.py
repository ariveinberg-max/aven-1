"""On-disk layout, checksums and BIDS conversion for datasets (WP-1.3, ADR-0003).

::

    data/raw/<dataset_id>/<version>/          immutable downloads + checksums.sha256
    data/interim/bids/<dataset_id>/           BIDS-EEG (EDF) written via mne-bids
    data/processed/<pipeline_hash>/<dataset>/ epoch arrays + provenance (Stage 2)
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from neurolayer.core.types import Recording
from neurolayer.data.catalog import DatasetCard
from neurolayer.data.mne_bridge import from_mne_raw, to_mne_raw

CHECKSUM_FILE = "checksums.sha256"
_CHUNK = 1 << 20


def raw_dir(data_root: Path, card: DatasetCard) -> Path:
    """Directory holding the immutable raw download of ``card``."""
    return data_root / "raw" / card.id / card.version


def bids_root(data_root: Path, card: DatasetCard) -> Path:
    """BIDS root of ``card``'s interim conversion."""
    return data_root / "interim" / "bids" / card.id


def processed_dir(data_root: Path, pipeline_hash: str, dataset_id: str) -> Path:
    """Cache directory for epochs produced by one pipeline (content-addressed)."""
    return data_root / "processed" / pipeline_hash / dataset_id


def sha256_file(path: Path) -> str:
    """Hex SHA-256 of a file, streamed in 1 MiB chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _data_files(directory: Path) -> list[Path]:
    return sorted(p for p in directory.rglob("*") if p.is_file() and p.name != CHECKSUM_FILE)


def write_checksums(directory: Path) -> Path:
    """Write ``checksums.sha256`` (``sha256  relative/path`` lines) for every file."""
    lines = [
        f"{sha256_file(p)}  {p.relative_to(directory).as_posix()}" for p in _data_files(directory)
    ]
    target = directory / CHECKSUM_FILE
    target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return target


def verify_checksums(directory: Path) -> list[str]:
    """Return problems (modified, missing or unexpected files); empty means intact."""
    manifest = directory / CHECKSUM_FILE
    if not manifest.exists():
        return [f"{CHECKSUM_FILE} missing"]
    expected: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, _, relative = line.partition("  ")
        expected[relative] = digest
    problems = []
    actual = {p.relative_to(directory).as_posix(): p for p in _data_files(directory)}
    for relative, digest in expected.items():
        if relative not in actual:
            problems.append(f"missing: {relative}")
        elif sha256_file(actual[relative]) != digest:
            problems.append(f"modified: {relative}")
    problems.extend(f"unexpected: {r}" for r in sorted(set(actual) - set(expected)))
    return problems


def _bids_path(root: Path, recording: Recording) -> Any:
    from mne_bids import BIDSPath

    def clean(value: str, prefix: str) -> str:
        text = value.removeprefix(prefix)
        return "".join(ch for ch in text if ch.isalnum()) or "0"

    return BIDSPath(
        subject=clean(recording.subject_id, "sub-"),
        session=clean(recording.session_id, "ses-"),
        run=None,
        acquisition=clean(recording.run_id, "run-"),
        task="motor",
        datatype="eeg",
        root=root,
    )


def write_bids(recordings: Iterable[Recording], root: Path) -> list[Path]:
    """Write recordings as BIDS-EEG (EDF + sidecars) under ``root``; returns file paths.

    Requires the ``neuro`` extra (mne, mne-bids, edfio).
    """
    from mne_bids import write_raw_bids

    written = []
    for recording in recordings:
        raw = to_mne_raw(recording)
        raw.set_meas_date(None)
        path = _bids_path(root, recording)
        write_raw_bids(
            raw,
            path,
            allow_preload=True,
            format="EDF",
            overwrite=True,
            verbose="error",
        )
        written.append(Path(str(path.fpath)))
    return written


def read_bids(root: Path, template: Recording) -> Recording:
    """Read back the BIDS recording matching ``template``'s identifiers."""
    from mne_bids import read_raw_bids

    raw = read_raw_bids(_bids_path(root, template), verbose="error")
    raw.load_data()
    recording, _ = from_mne_raw(
        raw,
        dataset_id=template.dataset_id,
        subject_id=template.subject_id,
        session_id=template.session_id,
        run_id=template.run_id,
    )
    return recording
