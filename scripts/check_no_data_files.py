"""Refuse to commit neural data, arrays or model weights (security control, C1-C3).

Used as a pre-commit hook (paths passed as arguments) and in CI with ``--all`` (checks
every tracked file). Exit code 1 lists the offending paths.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from collections.abc import Iterable, Sequence
from pathlib import PurePath

BLOCKED_SUFFIXES: frozenset[str] = frozenset(
    {
        # neural data formats
        ".edf", ".bdf", ".gdf", ".fif", ".set", ".fdt", ".vhdr", ".vmrk", ".eeg",
        ".xdf", ".cnt", ".mat", ".nwb", ".mef", ".snirf",
        # arrays and tabular dumps
        ".npy", ".npz", ".h5", ".hdf5", ".zarr", ".parquet", ".feather",
        # model weights and pickles
        ".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".pkl", ".pickle", ".joblib", ".bin",
    }
)  # fmt: skip

BLOCKED_DIRS: tuple[str, ...] = (
    "data/raw/",
    "data/interim/",
    "data/processed/",
    "artifacts/",
    "mlruns/",
)


# DVC pointer files and the .gitignore files DVC writes are small text files that must be
# committed so data versions are tracked in git (ADR-0006).
ALLOWED_SUFFIXES: tuple[str, ...] = (".dvc", "/.gitignore")


def blocked(paths: Iterable[str]) -> list[str]:
    """Return the subset of ``paths`` that must not be committed."""
    offenders = []
    for path in paths:
        normalized = path.replace("\\", "/")
        if normalized.endswith(ALLOWED_SUFFIXES):
            continue
        suffixes = {s.lower() for s in PurePath(normalized).suffixes}
        if suffixes & BLOCKED_SUFFIXES or normalized.startswith(BLOCKED_DIRS):
            offenders.append(path)
    return offenders


def tracked_files() -> list[str]:
    """All files tracked by git in the current repository."""
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git not found")
    # Fixed argument list, no shell, no user input.
    out = subprocess.run([git, "ls-files"], check=True, capture_output=True, text=True)  # noqa: S603
    return [line for line in out.stdout.splitlines() if line]


def main(argv: Sequence[str] | None = None) -> int:
    """Check the given paths (or all tracked files with ``--all``)."""
    args = list(sys.argv[1:] if argv is None else argv)
    paths = tracked_files() if args == ["--all"] else args
    offenders = blocked(paths)
    if offenders:
        print(
            "Refusing to commit data/weight files (see docs/architecture/security-and-privacy.md):"
        )
        for path in offenders:
            print(f"  {path}")
        print("Store data under data/ (git-ignored, DVC-tracked) and weights under artifacts/.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
