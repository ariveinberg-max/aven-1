"""Test doubles shared across test modules (no downloads, no network)."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

PHYSIONET_STYLE_NAMES = ["Fc3.", "Fcz.", "Fc4.", "C3..", "Cz..", "C4..", "Cp3.", "Cpz.", "Cp4."]


def make_moabb_raw(
    n_trials: int = 10, sfreq: float = 160.0, seed: int = 0, with_eog: bool = True
) -> Any:
    """An MNE RawArray shaped like MOABB output: EEG + optional EOG + stim channel."""
    import mne

    rng = np.random.default_rng(seed)
    period = int(4 * sfreq)
    n_samples = n_trials * period + period
    names = [*PHYSIONET_STYLE_NAMES, *(["EOG1"] if with_eog else []), "STI"]
    types = ["eeg"] * len(PHYSIONET_STYLE_NAMES) + (["eog"] if with_eog else []) + ["stim"]
    data = rng.normal(0.0, 1e-5, (len(names), n_samples))
    data[-1] = 0.0
    codes = [2, 3, 4, 7]  # left_hand, right_hand, hands, and an unmapped code 7
    for i in range(n_trials):
        data[-1, i * period + 10] = codes[i % len(codes)]
    info = mne.create_info(names, sfreq, ch_types=types)
    return mne.io.RawArray(data, info, verbose="error")


class FakeMoabbDataset:
    """Implements the slice of the MOABB dataset API the adapter uses."""

    event_id: ClassVar[dict[str, int]] = {"left_hand": 2, "right_hand": 3, "hands": 4, "weird": 7}
    interval = (0.5, 3.5)

    def __init__(self, subjects: tuple[int, ...] = (1, 2)) -> None:
        self.subject_list = list(subjects)
        self.downloads: list[tuple[list[int], str | None]] = []

    def get_data(self, subjects: list[int]) -> dict[int, dict[str, dict[str, Any]]]:
        return {
            s: {
                "1session": {"0run": make_moabb_raw(seed=s), "1run": make_moabb_raw(seed=s + 100)},
                "0session": {"0run": make_moabb_raw(seed=s + 200)},
            }
            for s in subjects
        }

    def download(
        self, subject_list: list[int], path: str | None = None, update_path: bool = False
    ) -> None:
        self.downloads.append((subject_list, path))
