"""Report runtime dependency licenses and fail on strong copyleft (WP-0.9).

Usage (CI)::

    uv export --locked --no-hashes --no-dev --extra api --extra neuro --no-emit-project > reqs.txt
    uv run python scripts/license_report.py reqs.txt

Reads pinned requirements, looks up each installed distribution's license metadata
(License-Expression, License, and trove classifiers) and exits 1 if any runtime
dependency is GPL, AGPL or SSPL, since those could force disclosure of proprietary
code. LGPL is reported but allowed (dynamic linking from Python). Unknown licenses are
reported for human review.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import metadata

# Strong copyleft: incompatible with a proprietary, distributed product.
_FORBIDDEN = re.compile(r"(?<![L])\bA?GPL|Affero|SSPL|Server Side Public", re.IGNORECASE)
_LGPL = re.compile(r"LGPL|Lesser General Public", re.IGNORECASE)
_GPL_CLASSIFIER = re.compile(r"GNU (Affero )?General Public License", re.IGNORECASE)

# Packages whose metadata trips the pattern but whose actual terms are acceptable.
# Every entry needs a reason a reviewer can check.
ALLOWLIST: dict[str, str] = {}


@dataclass(frozen=True, slots=True)
class LicenseRow:
    """License facts for one distribution."""

    name: str
    version: str
    license: str
    verdict: str  # "ok" | "lgpl" | "unknown" | "forbidden" | "allowlisted"


def parse_requirements(lines: Sequence[str]) -> list[str]:
    """Return distribution names from a pinned requirements file."""
    names = []
    for raw in lines:
        line = raw.split("#", 1)[0].split(";", 1)[0].strip()
        if not line or line.startswith(("-", "--")):
            continue
        name = re.split(r"[=<>!~\[ ]", line, maxsplit=1)[0]
        if name:
            names.append(name)
    return names


def classify(name: str, expression: str, declared: str, classifiers: Sequence[str]) -> str:
    """Return the verdict for one distribution's license metadata."""
    text = " | ".join(filter(None, [expression, declared, *classifiers]))
    forbidden = bool(_FORBIDDEN.search(_LGPL.sub("", text))) or any(
        _GPL_CLASSIFIER.search(c) and not _LGPL.search(c) for c in classifiers
    )
    if forbidden:
        return "allowlisted" if name.lower() in ALLOWLIST else "forbidden"
    if _LGPL.search(text):
        return "lgpl"
    return "ok" if text else "unknown"


def describe(name: str) -> LicenseRow:
    """Look up and classify the license of an installed distribution."""
    try:
        meta = metadata.metadata(name)
    except metadata.PackageNotFoundError:
        return LicenseRow(name, "not installed", "?", "unknown")
    expression = meta.get("License-Expression") or ""
    declared = (meta.get("License") or "").splitlines()[0][:80] if meta.get("License") else ""
    classifiers = [c for c in meta.get_all("Classifier") or [] if c.startswith("License ::")]
    text = " | ".join(filter(None, [expression, declared, *classifiers])) or "?"
    verdict = classify(name, expression, declared, classifiers)
    return LicenseRow(name, meta.get("Version") or "?", text, verdict)


def main(argv: Sequence[str] | None = None) -> int:
    """Print a license table for the requirements file; exit 1 on forbidden licenses."""
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: license_report.py <pinned-requirements.txt>")
        return 2
    with open(args[0], encoding="utf-8") as fh:
        rows = [describe(name) for name in parse_requirements(fh.read().splitlines())]
    print("| package | version | verdict | license |")
    print("|---------|---------|---------|---------|")
    for row in sorted(rows, key=lambda r: (r.verdict != "forbidden", r.name.lower())):
        print(f"| {row.name} | {row.version} | {row.verdict} | {row.license[:120]} |")
    forbidden = [r.name for r in rows if r.verdict == "forbidden"]
    unknown = [r.name for r in rows if r.verdict == "unknown"]
    if unknown:
        print(f"\nReview manually (license metadata missing): {', '.join(unknown)}")
    if forbidden:
        print(f"\nFORBIDDEN strong-copyleft runtime dependencies: {', '.join(forbidden)}")
        return 1
    print(f"\n{len(rows)} runtime dependencies checked; no strong copyleft found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
