"""Gate 0: our ingestion + preprocessing reproduces MOABB's own results (WP-2.6).

Runs the same two classic pipelines MOABB benchmarks, CSP+LDA and TS+LR, with 5-fold
within-session cross-validation (ROC AUC, MOABB's binary metric):

* on **our** epochs (adapter → our transforms → our epoching), and
* through **MOABB's** own paradigm and ``WithinSessionEvaluation`` (network).

Gate 0 passes when the mean per-session difference is within ``tolerance`` (default
0.03 AUC, the CAP-1 spec's ±3 pp). A failure means our pipeline differs from the
reference implementation somewhere (units, events, filtering, windows) and nothing
downstream can be trusted until it is fixed.

MOABB ``LeftRightImagery`` filters 8-32 Hz with MNE's default IIR (4th-order
Butterworth, zero-phase) and epochs the dataset's full task interval; our Gate 0
pipeline mirrors that exactly.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score

from neurolayer.core.types import EpochSet
from neurolayer.signal.epoching import EpochingSpec
from neurolayer.signal.pipeline import PipelineSpec, TransformConfig

SessionKey = tuple[str, str, str]  # (dataset, subject, session)
PIPELINES = ("csp_lda", "ts_lr")


def gate0_pipeline(task_seconds: float) -> PipelineSpec:
    """Preprocessing that mirrors MOABB's LeftRightImagery defaults."""
    return PipelineSpec(
        transforms=(
            TransformConfig(name="unit_check"),
            TransformConfig(name="bandpass", params={"l_freq": 8.0, "h_freq": 32.0, "order": 4}),
        ),
        epoching=EpochingSpec(tmin=0.0, tmax=task_seconds),
    )


def sklearn_pipeline(name: str) -> Any:
    """Return a MOABB benchmark pipeline operating on raw epochs (scale-free)."""
    from pyriemann.estimation import Covariances
    from pyriemann.spatialfilters import CSP
    from pyriemann.tangentspace import TangentSpace
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    if name == "csp_lda":
        return make_pipeline(
            Covariances("oas"), CSP(nfilter=8), LinearDiscriminantAnalysis(solver="svd")
        )
    if name == "ts_lr":
        return make_pipeline(
            Covariances("oas"), TangentSpace(metric="riemann"), LogisticRegression(C=1.0)
        )
    raise ValueError(f"unknown Gate 0 pipeline {name!r}; use one of {PIPELINES}")


def within_session_auc(
    epochs: EpochSet, pipeline: str, n_splits: int = 5, seed: int = 42
) -> dict[SessionKey, float]:
    """Mean ROC AUC of 5-fold within-session CV for every (dataset, subject, session)."""
    scores: dict[SessionKey, float] = {}
    keys = sorted(
        {
            (str(d), str(s), str(e))
            for d, s, e in zip(epochs.dataset, epochs.subject, epochs.session, strict=True)
        }
    )
    for dataset, subject, session in keys:
        mask = (
            (epochs.dataset == dataset) & (epochs.subject == subject) & (epochs.session == session)
        )
        X, y = epochs.X[mask] * 1e6, epochs.y[mask]
        if len(np.unique(y)) < 2 or np.bincount(y).min() < n_splits:
            continue
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        scores[(dataset, subject, session)] = float(
            np.mean(cross_val_score(sklearn_pipeline(pipeline), X, y, cv=cv, scoring="roc_auc"))
        )
    return scores


def moabb_reference(
    moabb_class: str, subjects: list[int], pipelines: tuple[str, ...] = PIPELINES
) -> dict[str, dict[tuple[str, str], float]]:
    """MOABB's own within-session scores (downloads data): pipeline -> (subject, session) -> AUC."""
    import tempfile

    import moabb.datasets
    from moabb.evaluations import WithinSessionEvaluation
    from moabb.paradigms import LeftRightImagery

    dataset = getattr(moabb.datasets, moabb_class)()
    dataset.subject_list = subjects
    with tempfile.TemporaryDirectory() as tmp:
        evaluation = WithinSessionEvaluation(
            paradigm=LeftRightImagery(), datasets=[dataset], overwrite=True, hdf5_path=tmp
        )
        results = evaluation.process({name: sklearn_pipeline(name) for name in pipelines})
    table: dict[str, dict[tuple[str, str], float]] = {name: {} for name in pipelines}
    for row in results.itertuples():
        table[str(row.pipeline)][(f"sub-{int(row.subject):03d}", str(row.session))] = float(
            row.score
        )
    return table


@dataclass(frozen=True, slots=True)
class Gate0Result:
    """Per-pipeline comparison of our scores with MOABB's."""

    pipeline: str
    n_sessions: int
    mean_ours: float
    mean_reference: float
    tolerance: float

    @property
    def mean_difference(self) -> float:
        """Ours minus reference (AUC)."""
        return self.mean_ours - self.mean_reference

    @property
    def passed(self) -> bool:
        """Within tolerance and computed on at least one session."""
        return self.n_sessions > 0 and abs(self.mean_difference) <= self.tolerance


def compare(
    ours: Mapping[SessionKey, float],
    reference: Mapping[tuple[str, str], float],
    pipeline: str,
    tolerance: float = 0.03,
) -> Gate0Result:
    """Match sessions by (subject, session) and compare mean AUC."""
    pairs = [
        (score, reference[(subject, session)])
        for (_, subject, session), score in ours.items()
        if (subject, session) in reference
    ]
    return Gate0Result(
        pipeline=pipeline,
        n_sessions=len(pairs),
        mean_ours=float(np.mean([a for a, _ in pairs])) if pairs else float("nan"),
        mean_reference=float(np.mean([b for _, b in pairs])) if pairs else float("nan"),
        tolerance=tolerance,
    )
