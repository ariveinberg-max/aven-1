"""Comparison reports across runs: the Gate 1 baseline table (WP-4.2).

Reads run directories written by ``neurolayer run`` and produces a markdown report:
per-budget BA with CIs, usable-user rate, AUCEC and median TTC per run, plus paired
per-subject comparisons against a reference run (Wilcoxon signed-rank, Holm-corrected
across budgets). Optionally renders calibration-efficiency curves to SVG.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from neurolayer.evaluation import metrics


@dataclass(frozen=True, slots=True)
class LoadedRun:
    """A finished run read back from disk."""

    run_id: str
    label: str
    official: bool
    summary: dict[str, Any]
    scores: dict[int, dict[tuple[str, str], float]]  # k -> (dataset, subject) -> BA


def load_run(run_dir: Path) -> LoadedRun:
    """Read ``manifest.json``, ``summary.json`` and ``results.csv`` of a run."""
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    scores: dict[int, dict[tuple[str, str], float]] = defaultdict(dict)
    with (run_dir / "results.csv").open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            scores[int(row["k"])][(row["dataset"], row["subject"])] = float(
                row["balanced_accuracy"]
            )
    decoder = manifest["config"].get("decoder", {})
    params = decoder.get("params") or {}
    label = decoder.get("name", "?") + (
        f" ({params.get('architecture')})" if "architecture" in params else ""
    )
    return LoadedRun(
        run_id=manifest["run_id"],
        label=label,
        official=bool(manifest["official"]),
        summary=summary,
        scores=dict(scores),
    )


def paired_comparison(run: LoadedRun, reference: LoadedRun) -> dict[int, tuple[float, float, int]]:
    """Per budget: (mean BA difference, Holm-adjusted p, n paired subjects)."""
    budgets = sorted(set(run.scores) & set(reference.scores))
    raw: list[tuple[int, float, float, int]] = []
    for k in budgets:
        common = sorted(set(run.scores[k]) & set(reference.scores[k]))
        if len(common) < 2:
            continue
        a = [run.scores[k][s] for s in common]
        b = [reference.scores[k][s] for s in common]
        diff = sum(x - y for x, y in zip(a, b, strict=True)) / len(common)
        raw.append((k, diff, metrics.paired_wilcoxon(a, b), len(common)))
    adjusted = metrics.holm_bonferroni([p for _, _, p, _ in raw]) if raw else []
    return {k: (diff, p_adj, n) for (k, diff, _, n), p_adj in zip(raw, adjusted, strict=True)}


def comparison_markdown(runs: list[LoadedRun], reference: LoadedRun | None = None) -> str:
    """Render the comparison report."""
    lines = ["# Run comparison", ""]
    if any(not r.official for r in runs):
        lines += ["> Contains exploratory (non-official) runs: not citable as results.", ""]
    budgets = sorted({k for r in runs for k in r.scores})
    header = (
        "| run | decoder | "
        + " | ".join(f"BA@{k}" for k in budgets)
        + " | UUR@max | AUCEC | median TTC |"
    )
    lines += [header, "|" + "---|" * (len(budgets) + 5)]
    for run in runs:
        per_k = {row["k"]: row for row in run.summary["per_budget"]}
        cells = []
        for k in budgets:
            row = per_k.get(k)
            cells.append(
                f"{row['mean_ba']:.3f} [{row['ci_low']:.2f}, {row['ci_high']:.2f}]" if row else "–"
            )
        last = per_k.get(budgets[-1]) if budgets else None
        uur = f"{last['usable_user_rate']:.2f}" if last else "–"
        aucec = run.summary.get("aucec")
        aucec_text = "–" if aucec is None else f"{aucec:.3f}"
        ttc = run.summary.get("median_ttc")
        ttc_text = "never" if ttc is None or math.isinf(ttc) else f"{ttc:g}"
        lines.append(
            f"| `{run.run_id}` | {run.label} | {' | '.join(cells)} | {uur} | {aucec_text} | "
            f"{ttc_text} |"
        )
    if reference is not None:
        lines += ["", f"## Paired vs reference: {reference.label} (`{reference.run_id}`)", ""]
        lines += ["| decoder | k | mean ΔBA | Holm p | n subjects |", "|---|---|---|---|---|"]
        for run in runs:
            if run is reference:
                continue
            for k, (diff, p_adj, n) in paired_comparison(run, reference).items():
                lines.append(f"| {run.label} | {k} | {diff:+.3f} | {p_adj:.3g} | {n} |")
    return "\n".join(lines) + "\n"


def plot_curves(runs: list[LoadedRun], path: Path) -> Path:
    """Save calibration-efficiency curves (mean BA vs k with 95% CI) as SVG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for run in runs:
        rows = sorted(run.summary["per_budget"], key=lambda r: r["k"])
        ks = [r["k"] for r in rows]
        ax.plot(ks, [r["mean_ba"] for r in rows], marker="o", label=run.label)
        ax.fill_between(ks, [r["ci_low"] for r in rows], [r["ci_high"] for r in rows], alpha=0.15)
    ax.axhline(0.7, linestyle="--", linewidth=1, color="grey")
    ax.axhline(0.5, linestyle=":", linewidth=1, color="grey")
    ax.set_xlabel("calibration trials per class (k)")
    ax.set_ylabel("balanced accuracy (held-out subjects)")
    ax.set_title("Calibration-efficiency curves")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg")
    plt.close(fig)
    return path
