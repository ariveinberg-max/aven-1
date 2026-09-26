"""Run manifests: the ground truth for reproducibility (ADR-0006).

Every run writes ``artifacts/runs/<run_id>/`` with:

* ``manifest.json``: git state, config and its hash, dataset card versions and
  licenses, license-gate purpose, seeds, environment.
* ``results.csv``: one row per (subject, calibration budget).
* ``summary.json``: CAP-1 aggregates.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import re
import secrets
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from neurolayer.data.catalog import DatasetCard, Purpose
from neurolayer.evaluation.protocol import ProtocolResult

MANIFEST_VERSION = 1
TRACKED_PACKAGES: tuple[str, ...] = (
    "neurolayer", "numpy", "scipy", "scikit-learn", "pydantic",
    "mne", "moabb", "pyriemann", "braindecode", "torch",
)  # fmt: skip


@dataclass(frozen=True, slots=True)
class DatasetRef:
    """Dataset provenance recorded in a manifest."""

    id: str
    version: str
    license_spdx: str
    license_verified: bool


@dataclass(frozen=True, slots=True)
class RunManifest:
    """Everything needed to reproduce and audit a run."""

    manifest_version: int
    run_id: str
    name: str
    created_utc: str
    official: bool
    purpose: str
    git_commit: str | None
    git_dirty: bool | None
    config_hash: str
    config: dict[str, Any]
    datasets: tuple[DatasetRef, ...]
    seed: int
    python: str
    platform: str
    packages: dict[str, str]

    def to_json(self) -> str:
        """Serialize to pretty-printed JSON."""
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def config_hash(config: Mapping[str, Any]) -> str:
    """SHA-256 of the canonical JSON form of ``config``."""
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def git_state(repo_root: Path) -> tuple[str | None, bool | None]:
    """Return ``(commit, dirty)`` for ``repo_root``, or ``(None, None)`` outside git."""
    git = shutil.which("git")
    if git is None:
        return None, None
    try:
        # Fixed argument lists, no shell, no user input: safe subprocess use.
        commit = subprocess.run(  # noqa: S603
            [git, "rev-parse", "HEAD"], cwd=repo_root, check=True, capture_output=True, text=True
        ).stdout.strip()
        status = subprocess.run(  # noqa: S603
            [git, "status", "--porcelain"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (subprocess.CalledProcessError, OSError):
        return None, None
    return commit, bool(status.strip())


def package_versions(names: Sequence[str] = TRACKED_PACKAGES) -> dict[str, str]:
    """Installed versions of ``names`` (``"not installed"`` if absent)."""
    versions = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def new_run_id(name: str, now: datetime | None = None) -> str:
    """Sortable, unique run id: ``<UTC timestamp>-<slug>-<random>``."""
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40] or "run"
    return f"{stamp}-{slug}-{secrets.token_hex(3)}"


def create_manifest(
    *,
    name: str,
    config: Mapping[str, Any],
    datasets: Sequence[DatasetCard],
    purpose: Purpose,
    seed: int,
    official: bool,
    repo_root: Path,
) -> RunManifest:
    """Assemble a manifest for a run that is about to be (or has been) executed."""
    commit, dirty = git_state(repo_root)
    config_dict = json.loads(json.dumps(config, default=str))
    return RunManifest(
        manifest_version=MANIFEST_VERSION,
        run_id=new_run_id(name),
        name=name,
        created_utc=datetime.now(UTC).isoformat(timespec="seconds"),
        official=official,
        purpose=purpose.value,
        git_commit=commit,
        git_dirty=dirty,
        config_hash=config_hash(config_dict),
        config=config_dict,
        datasets=tuple(
            DatasetRef(
                id=card.id,
                version=card.version,
                license_spdx=card.license.spdx,
                license_verified=card.license.is_verified,
            )
            for card in datasets
        ),
        seed=seed,
        python=sys.version.split()[0],
        platform=platform.platform(),
        packages=package_versions(),
    )


def write_run(output_dir: Path, manifest: RunManifest, result: ProtocolResult) -> Path:
    """Write manifest, per-subject results and summary; returns the run directory."""
    run_dir = output_dir / manifest.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "manifest.json").write_text(manifest.to_json() + "\n", encoding="utf-8")
    rows = result.to_rows()
    with (run_dir / "results.csv").open("w", newline="", encoding="utf-8") as fh:
        if rows:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = result.summary_dict() | {
        "run_id": manifest.run_id,
        "config_hash": manifest.config_hash,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return run_dir
