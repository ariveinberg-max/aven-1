"""Fetch PhysioNet EEGMMIDB from PhysioNet's official AWS Open Data mirror (WP-1.2).

Some environments can reach ``*.amazonaws.com`` but not ``physionet.org``. PhysioNet
publishes its open-access databases, byte-identical, in the ``physionet-open`` bucket
(https://registry.opendata.aws/physionet/). This script downloads the runs MOABB's
``PhysionetMI`` needs into the cache layout MOABB expects, so ``neurolayer data fetch
physionet_mi`` then runs without contacting physionet.org.

Every file is verified against PhysioNet's published ``SHA256SUMS.txt`` before it is
kept. The catalog license gate is checked first (``exploration`` purpose): this script
never grants benchmark or training use.

Usage::

    uv run python scripts/fetch_physionet_mirror.py --subjects 1-109 [--data-root data]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from neurolayer.data.adapters.base import require_exploration
from neurolayer.data.catalog import load_catalog
from neurolayer.data.storage import raw_dir

MIRROR = "https://physionet-open.s3.amazonaws.com/eegmmidb/1.0.0/"
# MOABB PhysionetMI.data_path: baselines 1-2, imagined hands 4/8/12, imagined feet 6/10/14.
RUNS = (1, 2, 4, 6, 8, 10, 12, 14)
# Where MOABB's data_dl() looks for https://physionet.org/files/eegmmidb/1.0.0/<file>.
CACHE_SUBDIR = Path("MNE-eegbci-data/files/eegmmidb/1.0.0")


def parse_subjects(spec: str) -> list[int]:
    """``"1-3,7"`` → ``[1, 2, 3, 7]``."""
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            low, high = (int(x) for x in part.split("-", 1))
            out.extend(range(low, high + 1))
        elif part.strip():
            out.append(int(part))
    if not out or min(out) < 1 or max(out) > 109:
        raise ValueError(f"subjects must be within 1-109, got {spec!r}")
    return sorted(set(out))


def _get(url: str) -> bytes:
    if not url.startswith(MIRROR):
        raise ValueError(f"refusing to fetch outside the mirror: {url}")
    with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310  (fixed https prefix)
        data: bytes = response.read()
    return data


def checksums() -> dict[str, str]:
    """PhysioNet's published SHA-256 per relative path."""
    lines = _get(MIRROR + "SHA256SUMS.txt").decode().splitlines()
    return {rel: digest for digest, _, rel in (line.partition(" ") for line in lines) if rel}


def fetch_one(rel: str, target: Path, expected: str) -> str:
    """Download ``rel`` unless a verified copy exists; returns a status word."""
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == expected:
        return "cached"
    data = _get(MIRROR + rel)
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"checksum mismatch for {rel}; not saved")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    partial.write_bytes(data)
    partial.replace(target)
    return "downloaded"


def main(argv: Sequence[str] | None = None) -> int:
    """Mirror the requested subjects; exit 1 if any file fails verification."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--subjects", default="1-109")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--catalog", type=Path, default=Path("catalog/datasets"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)

    card = load_catalog(args.catalog)["physionet_mi"]
    require_exploration(card)
    subjects = parse_subjects(args.subjects)
    sums = checksums()
    root = raw_dir(args.data_root, card) / CACHE_SUBDIR
    jobs = []
    for subject in subjects:
        for run in RUNS:
            rel = f"S{subject:03d}/S{subject:03d}R{run:02d}.edf"
            if rel not in sums:
                print(f"error: {rel} is not in PhysioNet's SHA256SUMS", file=sys.stderr)
                return 1
            jobs.append((rel, root / rel, sums[rel]))
    failures = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_one, *job): job[0] for job in jobs}
        counts = {"downloaded": 0, "cached": 0}
        for future, rel in futures.items():
            try:
                counts[future.result()] += 1
            except (OSError, ValueError) as exc:
                failures += 1
                print(f"error: {rel}: {exc}", file=sys.stderr)
    print(
        f"{len(subjects)} subjects, {len(jobs)} files: {counts['downloaded']} downloaded, "
        f"{counts['cached']} already present, {failures} failed (all SHA-256 verified)"
    )
    print(f"cache: {root}")
    print("next: uv run neurolayer data fetch physionet_mi --subjects ... (offline from here)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
