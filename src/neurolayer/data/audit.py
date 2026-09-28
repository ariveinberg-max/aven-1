"""Channel and label audit across datasets (WP-1.4).

Answers, per dataset:

* which channels exist (canonical and non-canonical)?
* which consumer montages (CAP-1 regime R3) can be simulated?
* how many trials per class does each subject have for the CAP-1 budget grid?

The output feeds the choice of the development pool and the locked holdout (ADR-0009).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from neurolayer.core.channels import CONSUMER_MONTAGES, is_standard
from neurolayer.core.labels import CAP1_LABELS
from neurolayer.core.types import Recording


@dataclass(frozen=True, slots=True)
class SubjectSummary:
    """Trial counts of one subject."""

    subject_id: str
    n_sessions: int
    n_runs: int
    trials_per_label: dict[str, int]


@dataclass(frozen=True, slots=True)
class DatasetAudit:
    """Audit of one dataset's channels, sampling rates and trial counts."""

    dataset_id: str
    sampling_rates: tuple[float, ...]
    common_channels: tuple[str, ...]
    non_standard_channels: tuple[str, ...]
    montage_missing: dict[str, tuple[str, ...]]
    subjects: tuple[SubjectSummary, ...]
    labels: tuple[str, ...]

    def min_trials_per_class(self) -> int:
        """Smallest per-class trial count over subjects (for the audited labels)."""
        counts = [s.trials_per_label.get(label, 0) for s in self.subjects for label in self.labels]
        return min(counts) if counts else 0

    def supported_montages(self) -> list[str]:
        """Consumer montages whose channels all exist in every recording."""
        return sorted(name for name, missing in self.montage_missing.items() if not missing)

    def max_budget(self, min_test_per_class: int = 10) -> int:
        """Largest per-class calibration budget every subject can supply (protocol §4)."""
        return max(0, self.min_trials_per_class() - min_test_per_class)


def audit_dataset(
    dataset_id: str, recordings: Iterable[Recording], labels: Sequence[str] = CAP1_LABELS
) -> DatasetAudit:
    """Audit all recordings of one dataset."""
    rates: set[float] = set()
    channel_sets: list[set[str]] = []
    sessions: dict[str, set[str]] = defaultdict(set)
    runs: Counter[str] = Counter()
    trials: dict[str, Counter[str]] = defaultdict(Counter)
    for recording in recordings:
        if recording.dataset_id != dataset_id:
            raise ValueError(f"recording from {recording.dataset_id} in audit of {dataset_id}")
        rates.add(recording.sfreq)
        channel_sets.append(set(recording.ch_names))
        sessions[recording.subject_id].add(recording.session_id)
        runs[recording.subject_id] += 1
        trials[recording.subject_id].update(e.label for e in recording.events)
    if not channel_sets:
        raise ValueError(f"no recordings to audit for {dataset_id}")
    common = set.intersection(*channel_sets)
    every = set.union(*channel_sets)
    return DatasetAudit(
        dataset_id=dataset_id,
        sampling_rates=tuple(sorted(rates)),
        common_channels=tuple(sorted(common)),
        non_standard_channels=tuple(sorted(c for c in every if not is_standard(c))),
        montage_missing={
            name: tuple(c for c in montage.channels if c not in common)
            for name, montage in CONSUMER_MONTAGES.items()
            if montage.kind != "analysis"
        },
        subjects=tuple(
            SubjectSummary(
                subject_id=subject,
                n_sessions=len(sessions[subject]),
                n_runs=runs[subject],
                trials_per_label={label: trials[subject][label] for label in labels},
            )
            for subject in sorted(trials.keys() | sessions.keys())
        ),
        labels=tuple(labels),
    )


def audit_markdown(audits: Sequence[DatasetAudit], min_test_per_class: int = 10) -> str:
    """Render audits as the markdown report committed to ``docs/results/dataset-audit.md``."""
    lines = [
        "# Dataset audit (WP-1.4)",
        "",
        "| dataset | subjects | rates (Hz) | common channels | min trials/class | "
        f"max budget k (test ≥ {min_test_per_class}/class) | simulable consumer montages |",
        "|---|---|---|---|---|---|---|",
    ]
    for audit in audits:
        lines.append(
            f"| {audit.dataset_id} | {len(audit.subjects)} | "
            f"{', '.join(f'{r:g}' for r in audit.sampling_rates)} | {len(audit.common_channels)} | "
            f"{audit.min_trials_per_class()} | {audit.max_budget(min_test_per_class)} | "
            f"{', '.join(audit.supported_montages()) or 'none'} |"
        )
    for audit in audits:
        lines += ["", f"## {audit.dataset_id}", ""]
        if audit.non_standard_channels:
            lines.append(f"- Non-canonical channels: {', '.join(audit.non_standard_channels)}")
        for name, missing in sorted(audit.montage_missing.items()):
            if missing:
                lines.append(f"- `{name}` missing: {', '.join(missing)}")
        lines += ["", "| subject | sessions | runs | " + " | ".join(audit.labels) + " |"]
        lines.append("|---|---|---|" + "---|" * len(audit.labels))
        for s in audit.subjects:
            counts = " | ".join(str(s.trials_per_label[label]) for label in audit.labels)
            lines.append(f"| {s.subject_id} | {s.n_sessions} | {s.n_runs} | {counts} |")
    return "\n".join(lines) + "\n"
