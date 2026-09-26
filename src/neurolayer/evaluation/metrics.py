"""Metrics for the CAP-1 calibration-efficiency protocol (protocol §5–§6).

**Protected module:** changes require an ADR (ADR-0004).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence

import numpy as np
import numpy.typing as npt
from scipy import stats
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score

USABILITY_THRESHOLD = 0.70
"""Conventional minimum accuracy for usable binary BCI communication."""


def balanced_accuracy(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> float:
    """Mean per-class recall; robust to class imbalance."""
    return float(balanced_accuracy_score(y_true, y_pred))


def accuracy(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> float:
    """Fraction of correct predictions."""
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def cohen_kappa(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> float:
    """Chance-corrected agreement. Returns 0.0 when undefined (a single class present)."""
    true, pred = np.asarray(y_true), np.asarray(y_pred)
    if len(np.unique(np.concatenate([true, pred]))) < 2:
        return 0.0
    return float(cohen_kappa_score(true, pred))


def wolpaw_bits_per_trial(p: float, n_classes: int) -> float:
    """Wolpaw information transfer per selection, in bits.

    Returns 0 at or below chance (by convention) and ``log2(n_classes)`` at ``p == 1``.
    """
    if n_classes < 2:
        raise ValueError("n_classes must be >= 2")
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must lie in [0, 1]")
    if p <= 1.0 / n_classes:
        return 0.0
    bits = math.log2(n_classes) + p * math.log2(p)
    if p < 1.0:
        bits += (1.0 - p) * math.log2((1.0 - p) / (n_classes - 1))
    return bits


def itr_bits_per_minute(p: float, n_classes: int, seconds_per_selection: float) -> float:
    """Wolpaw information transfer rate in bits per minute."""
    if seconds_per_selection <= 0:
        raise ValueError("seconds_per_selection must be > 0")
    return wolpaw_bits_per_trial(p, n_classes) * 60.0 / seconds_per_selection


def bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[npt.NDArray[np.float64]], float] = lambda a: float(np.mean(a)),
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval of ``statistic`` over ``values``."""
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0:
        raise ValueError("cannot bootstrap an empty sample")
    if data.size == 1:
        return float(data[0]), float(data[0])
    rng = np.random.default_rng(seed)
    resamples = rng.integers(0, data.size, size=(n_boot, data.size))
    boot = np.array([statistic(data[row]) for row in resamples])
    low, high = np.quantile(boot, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(low), float(high)


def usable_user_rate(scores: Sequence[float], threshold: float = USABILITY_THRESHOLD) -> float:
    """Fraction of subjects whose score reaches ``threshold`` (UUR)."""
    data = np.asarray(scores, dtype=np.float64)
    if data.size == 0:
        raise ValueError("no scores")
    return float(np.mean(data >= threshold))


def trials_to_criterion(
    scores_by_k: Mapping[int, float], threshold: float = USABILITY_THRESHOLD
) -> int | None:
    """Smallest budget ``k`` whose score reaches ``threshold``; ``None`` if never (TTC)."""
    for k in sorted(scores_by_k):
        if scores_by_k[k] >= threshold:
            return k
    return None


def median_trials_to_criterion(ttc: Sequence[int | None]) -> float:
    """Median TTC over subjects, treating "never reached" as +infinity."""
    if not ttc:
        raise ValueError("no subjects")
    return float(np.median([math.inf if t is None else float(t) for t in ttc]))


def aucec(ks: Sequence[int], scores: Sequence[float]) -> float:
    """Compute the normalized area under the calibration-efficiency curve (AUCEC).

    Uses x = log2(1 + k), so small budgets carry more weight, and divides by the
    x-range, so the result lies in [0, 1] when scores do. A single budget returns
    its score.
    """
    if len(ks) != len(scores) or not ks:
        raise ValueError("ks and scores must be non-empty and of equal length")
    order = np.argsort(ks)
    x = np.log2(1.0 + np.asarray(ks, dtype=np.float64)[order])
    y = np.asarray(scores, dtype=np.float64)[order]
    if len(x) == 1 or x[-1] == x[0]:
        return float(y.mean())
    return float(np.trapezoid(y, x) / (x[-1] - x[0]))


def paired_wilcoxon(a: Sequence[float], b: Sequence[float]) -> float:
    """Two-sided Wilcoxon signed-rank p-value for paired per-subject scores.

    Returns 1.0 when all differences are zero (no evidence of a difference).
    """
    x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if x.shape != y.shape or x.size == 0:
        raise ValueError("a and b must be non-empty and of equal length")
    if np.allclose(x, y):
        return 1.0
    return float(stats.wilcoxon(x, y, zero_method="wilcox").pvalue)


def holm_bonferroni(p_values: Sequence[float]) -> list[float]:
    """Holm–Bonferroni adjusted p-values (same order as the input)."""
    p = np.asarray(p_values, dtype=np.float64)
    m = p.size
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, float((m - rank) * p[idx])))
        adjusted[idx] = running
    return [float(v) for v in adjusted]
