"""Resolve catalog dataset ids to :class:`~neurolayer.core.types.EpochSet` objects.

Only synthetic sources exist today. Real-dataset adapters (MOABB/MNE → ``Recording``
→ Stage 2 epoching) are work package WP-1.2; until then, requesting a real dataset
fails loudly instead of silently substituting data.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from neurolayer.core.types import EpochSet
from neurolayer.data.synthetic import SyntheticMIConfig, generate_synthetic_mi

SYNTHETIC_SOURCES: dict[str, dict[str, float | int]] = {
    # A second synthetic "lab": different seed stream and a shared site perturbation,
    # so leave-dataset-out (regime R2) can be exercised without real data.
    "synthetic_mi": {"seed_offset": 0, "site_shift": 0.0},
    "synthetic_mi_shifted": {"seed_offset": 10_000, "site_shift": 0.35},
}


def load_epochs(dataset_ids: Sequence[str], synthetic: SyntheticMIConfig | None = None) -> EpochSet:
    """Load and concatenate the requested datasets.

    Parameters
    ----------
    dataset_ids
        Catalog ids. Must have passed the license gate before calling this function.
    synthetic
        Base configuration for synthetic sources.

    Raises
    ------
    NotImplementedError
        For real datasets, whose adapters are not implemented yet (WP-1.2).
    """
    if not dataset_ids:
        raise ValueError("at least one dataset id is required")
    base = synthetic or SyntheticMIConfig()
    parts: list[EpochSet] = []
    for dataset_id in dataset_ids:
        variant = SYNTHETIC_SOURCES.get(dataset_id)
        if variant is None:
            raise NotImplementedError(
                f"no adapter for dataset {dataset_id!r} yet: real-dataset ingestion is "
                "work package WP-1.2 (docs/plan/work-packages.md)"
            )
        cfg = replace(
            base,
            dataset_id=dataset_id,
            seed=base.seed + int(variant["seed_offset"]),
            site_shift=float(variant["site_shift"]),
        )
        parts.append(generate_synthetic_mi(cfg).epochs)
    return EpochSet.concat(parts)
