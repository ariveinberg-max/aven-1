"""Dataset ids → one harmonized :class:`EpochSet` (Stage 2 entry point).

* Synthetic datasets come straight from the generator (already epoched).
* Real datasets go adapter → transforms → epoching, cached per subject under
  ``data/processed/<pipeline_hash>/<dataset>/`` so reruns are instant and identical.
* All parts are harmonized to one channel set (``"common"`` = intersection) and must
  share a sampling rate (add a ``resample`` transform when mixing datasets).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from neurolayer.core.types import EpochSet
from neurolayer.data.adapters import get_adapter
from neurolayer.data.adapters.base import pseudonymize
from neurolayer.data.catalog import DatasetCard
from neurolayer.data.sources import SYNTHETIC_SOURCES, load_epochs
from neurolayer.data.storage import processed_dir, raw_dir
from neurolayer.data.synthetic import SyntheticMIConfig
from neurolayer.signal.pipeline import PipelineSpec, load_epochs_file, process_subject, save_epochs


@dataclass(slots=True)
class BuildReport:
    """What :func:`build_epochs` did, recorded next to run results."""

    pipeline_hash: str
    cache_hits: int = 0
    cache_misses: int = 0
    epoching: dict[str, dict[str, Any]] = field(default_factory=dict)
    dropped_channels: tuple[str, ...] = ()


def _harmonize(
    parts: Sequence[EpochSet], spec: PipelineSpec
) -> tuple[list[EpochSet], tuple[str, ...]]:
    rates = {p.sfreq for p in parts}
    if len(rates) > 1:
        raise ValueError(
            f"datasets have different sampling rates {sorted(rates)}; add a 'resample' transform"
        )
    if spec.channels == "common":
        shared = set.intersection(*(set(p.ch_names) for p in parts))
        target = tuple(c for c in parts[0].ch_names if c in shared)
        if not target:
            raise ValueError("the datasets share no channels")
    else:
        target = tuple(spec.channels)
    every = set().union(*(p.ch_names for p in parts))
    dropped = tuple(sorted(every - set(target)))
    return [p.select_channels(target) for p in parts], dropped


def build_epochs(
    dataset_ids: Sequence[str],
    *,
    catalog: Mapping[str, DatasetCard],
    data_root: Path,
    pipeline: PipelineSpec,
    synthetic: SyntheticMIConfig | None = None,
    max_subjects: int | None = None,
    use_cache: bool = True,
) -> tuple[EpochSet, BuildReport]:
    """Load, preprocess, epoch and harmonize the requested datasets.

    Callers must have passed the license gate for ``dataset_ids`` already.
    """
    if not dataset_ids:
        raise ValueError("at least one dataset id is required")
    pipeline_hash = pipeline.pipeline_hash()
    report = BuildReport(pipeline_hash=pipeline_hash)
    parts: list[EpochSet] = []
    for dataset_id in dataset_ids:
        if dataset_id in SYNTHETIC_SOURCES:
            parts.append(load_epochs([dataset_id], synthetic))
            continue
        card = catalog[dataset_id]
        adapter = get_adapter(card, raw_dir(data_root, card))
        subjects = adapter.subjects()[:max_subjects] if max_subjects else adapter.subjects()
        cache_dir = processed_dir(data_root, pipeline_hash, dataset_id)
        for subject in subjects:
            cache_path = cache_dir / f"{pseudonymize(subject)}_epochs"
            if use_cache and cache_path.with_suffix(".npz").exists():
                parts.append(load_epochs_file(cache_path))
                report.cache_hits += 1
                continue
            epochs, epoching = process_subject(list(adapter.recordings(subject)), pipeline)
            report.epoching[f"{dataset_id}/{pseudonymize(subject)}"] = epoching.as_dict()
            save_epochs(
                epochs,
                cache_path,
                {
                    "pipeline_hash": pipeline_hash,
                    "pipeline": pipeline.model_dump(mode="json"),
                    "dataset": {"id": card.id, "version": card.version},
                    "epoching": epoching.as_dict(),
                },
            )
            parts.append(epochs)
            report.cache_misses += 1
    harmonized, dropped = _harmonize(parts, pipeline)
    report.dropped_channels = dropped
    return EpochSet.concat(harmonized), report
