"""The CAP-1 calibration-efficiency protocol runner.

Specification: ``docs/architecture/evaluation-protocol.md``. **Protected module:**
changes require an ADR (ADR-0004).
"""

from __future__ import annotations

import copy
import math
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

from neurolayer.core.channels import MissingChannelsError
from neurolayer.core.interfaces import Decoder
from neurolayer.core.types import EpochSet
from neurolayer.evaluation import metrics
from neurolayer.evaluation.splits import (
    Fold,
    InsufficientTrialsError,
    LeakageError,
    SubjectKey,
    leave_dataset_out_folds,
    plan_calibration,
    within_dataset_folds,
)

Regime = Literal["within_dataset", "leave_dataset_out"]
DecoderFactory = Callable[[], Decoder]


@dataclass(frozen=True, slots=True)
class ProtocolConfig:
    """Parameters of one protocol run (see protocol spec §3–§6)."""

    regime: Regime = "within_dataset"
    ks: tuple[int, ...] = (0, 5, 10, 20, 40)
    n_unlabeled: int = 0
    n_folds: int | None = 5
    min_test_per_class: int = 10
    source_montage: tuple[str, ...] | None = None
    target_montage: tuple[str, ...] | None = None
    threshold: float = metrics.USABILITY_THRESHOLD
    trial_seconds: float | None = None
    n_bootstrap: int = 2000
    seed: int = 0

    def __post_init__(self) -> None:
        if not self.ks or list(self.ks) != sorted(set(self.ks)) or self.ks[0] < 0:
            raise ValueError(f"ks must be sorted, unique and non-negative, got {self.ks}")
        if self.n_unlabeled < 0:
            raise ValueError("n_unlabeled must be >= 0")
        if not 0.0 < self.threshold < 1.0:
            raise ValueError("threshold must lie in (0, 1)")
        if self.trial_seconds is not None and self.trial_seconds <= 0:
            raise ValueError("trial_seconds must be > 0")


@dataclass(frozen=True, slots=True)
class SubjectResult:
    """Score of one target subject at one calibration budget."""

    regime: str
    fold: str
    dataset: str
    subject: str
    k: int
    n_unlabeled: int
    n_calibration: int
    n_test: int
    balanced_accuracy: float
    accuracy: float
    kappa: float
    itr_bits_per_min: float | None


@dataclass(frozen=True, slots=True)
class SkippedSubject:
    """A target subject excluded from scoring, with the reason (never silent)."""

    fold: str
    dataset: str
    subject: str
    reason: str


@dataclass(frozen=True, slots=True)
class BudgetSummary:
    """Aggregate over subjects at one calibration budget."""

    k: int
    n_subjects: int
    mean_ba: float
    median_ba: float
    ci_low: float
    ci_high: float
    usable_user_rate: float


@dataclass(frozen=True, slots=True)
class ProtocolResult:
    """All per-subject records of a protocol run plus derived CAP-1 aggregates."""

    config: ProtocolConfig
    records: tuple[SubjectResult, ...]
    skipped: tuple[SkippedSubject, ...]

    def _scores_by_k(self) -> dict[int, list[float]]:
        grouped: dict[int, list[float]] = defaultdict(list)
        for record in self.records:
            grouped[record.k].append(record.balanced_accuracy)
        return dict(sorted(grouped.items()))

    def per_budget(self) -> list[BudgetSummary]:
        """Mean/median BA with bootstrap CI and usable-user rate for each budget."""
        summaries = []
        for k, scores in self._scores_by_k().items():
            low, high = metrics.bootstrap_ci(
                scores, n_boot=self.config.n_bootstrap, seed=self.config.seed
            )
            summaries.append(
                BudgetSummary(
                    k=k,
                    n_subjects=len(scores),
                    mean_ba=float(np.mean(scores)),
                    median_ba=float(np.median(scores)),
                    ci_low=low,
                    ci_high=high,
                    usable_user_rate=metrics.usable_user_rate(scores, self.config.threshold),
                )
            )
        return summaries

    def trials_to_criterion(self) -> dict[SubjectKey, int | None]:
        """Per-subject TTC: smallest budget reaching the threshold, else ``None``."""
        by_subject: dict[SubjectKey, dict[int, float]] = defaultdict(dict)
        for record in self.records:
            by_subject[(record.dataset, record.subject)][record.k] = record.balanced_accuracy
        return {
            key: metrics.trials_to_criterion(scores, self.config.threshold)
            for key, scores in sorted(by_subject.items())
        }

    def median_ttc(self) -> float:
        """Median trials-to-criterion; ``inf`` if most subjects never reach it."""
        return metrics.median_trials_to_criterion(list(self.trials_to_criterion().values()))

    def aucec(self) -> float:
        """Return the normalized area under the mean calibration-efficiency curve."""
        summaries = self.per_budget()
        return metrics.aucec([s.k for s in summaries], [s.mean_ba for s in summaries])

    def to_rows(self) -> list[dict[str, Any]]:
        """Per-subject records as plain dicts (for CSV export)."""
        return [asdict(record) for record in self.records]

    def summary_dict(self) -> dict[str, Any]:
        """JSON-serializable summary used in ``summary.json`` and CLI output."""
        median_ttc = self.median_ttc() if self.records else math.inf
        return {
            "regime": self.config.regime,
            "ks": list(self.config.ks),
            "n_unlabeled": self.config.n_unlabeled,
            "threshold": self.config.threshold,
            "n_subjects_scored": len({(r.dataset, r.subject) for r in self.records}),
            "n_subjects_skipped": len(self.skipped),
            "aucec": self.aucec() if self.records else None,
            "median_ttc": None if math.isinf(median_ttc) else median_ttc,
            "per_budget": [asdict(s) for s in self.per_budget()],
            "skipped": [asdict(s) for s in self.skipped],
        }


def _folds(epochs: EpochSet, config: ProtocolConfig) -> list[Fold]:
    keys = epochs.subject_keys()
    if config.regime == "within_dataset":
        return within_dataset_folds(keys, config.n_folds, config.seed)
    return leave_dataset_out_folds(keys)


def _mask_for(
    epochs: EpochSet, keys: tuple[SubjectKey, ...]
) -> np.ndarray[Any, np.dtype[np.bool_]]:
    wanted = set(keys)
    return np.array(
        [(str(d), str(s)) in wanted for d, s in zip(epochs.dataset, epochs.subject, strict=True)],
        dtype=np.bool_,
    )


def run_protocol(
    epochs: EpochSet, factory: DecoderFactory, config: ProtocolConfig
) -> ProtocolResult:
    """Run the CAP-1 protocol for one decoder over all folds and target subjects.

    For each fold, a fresh decoder is fitted on the source subjects. For every target
    subject a chronological :class:`~neurolayer.evaluation.splits.CalibrationPlan` is
    built. Then, for every budget ``k``, a **deep copy** of the fitted decoder is
    adapted and scored on the subject's fixed test window, with labels hidden.
    """
    if not epochs.is_labeled:
        raise ValueError("protocol input must be labeled")
    records: list[SubjectResult] = []
    skipped: list[SkippedSubject] = []

    for fold in _folds(epochs, config):
        source = epochs.subset(_mask_for(epochs, fold.source))
        if set(source.subject_keys()) & set(fold.target):
            raise LeakageError(f"fold {fold.name}: target subject present in source data")
        if config.source_montage is not None:
            source = source.select_channels(config.source_montage)
        fitted = factory()
        fitted.fit(source)

        for dataset, subject in fold.target:
            target = epochs.for_subject(dataset, subject)
            if config.target_montage is not None:
                try:
                    target = target.select_channels(config.target_montage)
                except MissingChannelsError as exc:
                    skipped.append(SkippedSubject(fold.name, dataset, subject, str(exc)))
                    continue
            try:
                plan = plan_calibration(
                    target, config.ks, config.n_unlabeled, config.min_test_per_class
                )
            except InsufficientTrialsError as exc:
                skipped.append(SkippedSubject(fold.name, dataset, subject, str(exc)))
                continue

            test = target.subset(plan.test_idx)
            hidden_test = test.without_labels()
            unlabeled = (
                target.subset(plan.unlabeled_idx).without_labels() if config.n_unlabeled else None
            )
            for k in config.ks:
                calibration = target.subset(plan.calibration_idx[k]) if k > 0 else None
                decoder = copy.deepcopy(fitted).adapt(calibration, unlabeled)
                predictions = np.asarray(decoder.predict(hidden_test))
                if (
                    predictions.shape != (len(test),)
                    or not np.isin(predictions, np.arange(test.n_classes)).all()
                ):
                    raise ValueError(
                        f"decoder returned invalid predictions for {dataset}/{subject} at k={k}"
                    )
                acc = metrics.accuracy(test.y, predictions)
                records.append(
                    SubjectResult(
                        regime=config.regime,
                        fold=fold.name,
                        dataset=dataset,
                        subject=subject,
                        k=k,
                        n_unlabeled=config.n_unlabeled,
                        n_calibration=0 if calibration is None else len(calibration),
                        n_test=len(test),
                        balanced_accuracy=metrics.balanced_accuracy(test.y, predictions),
                        accuracy=acc,
                        kappa=metrics.cohen_kappa(test.y, predictions),
                        itr_bits_per_min=(
                            None
                            if config.trial_seconds is None
                            else metrics.itr_bits_per_minute(
                                acc, test.n_classes, config.trial_seconds
                            )
                        ),
                    )
                )
    return ProtocolResult(config=config, records=tuple(records), skipped=tuple(skipped))
