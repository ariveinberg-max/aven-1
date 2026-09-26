"""Canonical EEG channel naming and device montages.

Channels are identified by canonical 10-05 system names everywhere in the pipeline.
Real datasets spell them in many ways (``Fc5.``, ``EEG-C3``, ``EEG FP1-REF``, legacy
``T3``); :func:`normalize_channel_name` maps those spellings onto the canonical form.

Montages are *data*: a :class:`Montage` is a named, ordered tuple of canonical channel
names. CAP-1 regime R3 uses :data:`CONSUMER_MONTAGES` to simulate consumer devices.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

# Canonical spellings of the 10-10 system plus commonly used 10-05 / reference sites.
_STANDARD_NAMES: tuple[str, ...] = (
    # frontal pole / anterior frontal
    "Nz", "Fp1", "Fpz", "Fp2",
    "AF9", "AF7", "AF5", "AF3", "AF1", "AFz", "AF2", "AF4", "AF6", "AF8", "AF10",
    # frontal
    "F9", "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8", "F10",
    # fronto-central / fronto-temporal
    "FT9", "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8", "FT10",
    # central / temporal
    "T9", "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8", "T10",
    # centro-parietal / temporo-parietal
    "TP9", "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8", "TP10",
    # parietal
    "P9", "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8", "P10",
    # parieto-occipital
    "PO9", "PO7", "PO5", "PO3", "PO1", "POz", "PO2", "PO4", "PO6", "PO8", "PO10",
    # occipital / inion
    "O1", "Oz", "O2", "O9", "O10", "I1", "Iz", "I2",
    # references
    "A1", "A2", "M1", "M2",
)  # fmt: skip

# Old 10-20 names that were renamed in the 10-10 system.
_LEGACY_ALIASES: dict[str, str] = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}

_CANONICAL_BY_UPPER: dict[str, str] = {name.upper(): name for name in _STANDARD_NAMES}
_KNOWN_UPPER: frozenset[str] = frozenset(_CANONICAL_BY_UPPER) | frozenset(_LEGACY_ALIASES)

_PREFIXES: tuple[str, ...] = ("EEG-", "EEG_", "EEG ")
_REFERENCE_SUFFIXES: tuple[str, ...] = ("-REF", "-LE", "-AR", "-AVG", "-M1", "-M2", "-A1", "-A2")


class MissingChannelsError(ValueError):
    """Raised when required channels are not present in the available channel list."""

    def __init__(self, missing: Sequence[str], context: str = "") -> None:
        self.missing: tuple[str, ...] = tuple(missing)
        where = f" ({context})" if context else ""
        super().__init__(f"missing required channels{where}: {', '.join(self.missing)}")


def normalize_channel_name(name: str) -> str:
    """Map a raw channel label onto its canonical 10-05 spelling.

    Parameters
    ----------
    name
        Channel label as found in a data file.

    Returns
    -------
    str
        The canonical name (for example ``"FC5"`` for ``"Fc5."``). Names that are not
        recognised as standard electrode sites are returned stripped but otherwise
        unchanged; use :func:`is_standard` to detect them.

    Examples
    --------
    >>> normalize_channel_name("Fc5.")
    'FC5'
    >>> normalize_channel_name("EEG FP1-REF")
    'Fp1'
    >>> normalize_channel_name("T3")
    'T7'
    """
    cleaned = name.strip().rstrip(".")
    upper = cleaned.upper()
    for prefix in _PREFIXES:
        if upper.startswith(prefix):
            cleaned = cleaned[len(prefix) :].strip()
            upper = cleaned.upper()
            break
    for suffix in _REFERENCE_SUFFIXES:
        # Only strip a reference suffix when what remains is a known electrode site,
        # so bipolar derivations such as "C3-C4" are left untouched.
        if upper.endswith(suffix) and upper[: -len(suffix)] in _KNOWN_UPPER:
            cleaned = cleaned[: -len(suffix)]
            upper = cleaned.upper()
            break
    if upper in _LEGACY_ALIASES:
        return _LEGACY_ALIASES[upper]
    return _CANONICAL_BY_UPPER.get(upper, cleaned)


def is_standard(name: str) -> bool:
    """Return ``True`` if ``name`` is already a canonical 10-05 channel name."""
    return name in _CANONICAL_BY_UPPER.values()


def normalize_channel_names(names: Iterable[str]) -> tuple[str, ...]:
    """Normalize a sequence of channel names and check that the result is unique.

    Raises
    ------
    ValueError
        If two raw names collapse onto the same canonical name.
    """
    normalized = tuple(normalize_channel_name(n) for n in names)
    if len(set(normalized)) != len(normalized):
        dupes = sorted({n for n in normalized if normalized.count(n) > 1})
        raise ValueError(f"channel names collide after normalization: {dupes}")
    return normalized


def channel_indices(
    available: Sequence[str], required: Sequence[str], context: str = ""
) -> list[int]:
    """Return the positions of ``required`` channels within ``available``.

    Raises
    ------
    MissingChannelsError
        If any required channel is absent. Never substitutes silently.
    """
    position = {name: i for i, name in enumerate(available)}
    missing = [name for name in required if name not in position]
    if missing:
        raise MissingChannelsError(missing, context)
    return [position[name] for name in required]


@dataclass(frozen=True, slots=True)
class Montage:
    """A named, ordered set of canonical channel names.

    Attributes
    ----------
    name
        Identifier used in configs (for example ``"neurosity_crown"``).
    channels
        Canonical channel names, in device order.
    kind
        ``"device"`` for a shipping product's fixed layout, ``"recommended"`` for a
        placement we recommend on configurable hardware, ``"research"`` for a dataset
        layout used as a stress test.
    description
        Human-readable notes, including where the layout comes from.
    """

    name: str
    channels: tuple[str, ...]
    kind: Literal["device", "recommended", "research"]
    description: str

    def __post_init__(self) -> None:
        non_standard = [c for c in self.channels if not is_standard(c)]
        if non_standard:
            raise ValueError(f"montage {self.name!r} has non-canonical channels: {non_standard}")
        if len(set(self.channels)) != len(self.channels):
            raise ValueError(f"montage {self.name!r} has duplicate channels")


CONSUMER_MONTAGES: dict[str, Montage] = {
    m.name: m
    for m in (
        Montage(
            name="neurosity_crown",
            channels=("CP3", "C3", "F5", "PO3", "PO4", "F6", "C4", "CP4"),
            kind="device",
            description="Neurosity Crown, 8 channels at 256 Hz; covers the sensorimotor strip. "
            "Primary CAP-1 R3 target.",
        ),
        Montage(
            name="muse_s",
            channels=("TP9", "AF7", "AF8", "TP10"),
            kind="device",
            description="Interaxon Muse S (incl. Athena), 4 EEG channels at 256 Hz; forehead and "
            "behind-ear sites, no motor-cortex coverage (stress test).",
        ),
        Montage(
            name="emotiv_epoc_x",
            channels=(
                "AF3",
                "F7",
                "F3",
                "FC5",
                "T7",
                "P7",
                "O1",
                "O2",
                "P8",
                "T8",
                "FC6",
                "F4",
                "F8",
                "AF4",
            ),
            kind="device",
            description="Emotiv EPOC X, 14 channels.",
        ),
        Montage(
            name="emotiv_insight",
            channels=("AF3", "AF4", "T7", "T8", "Pz"),
            kind="device",
            description="Emotiv Insight, 5 channels.",
        ),
        Montage(
            name="openbci_cyton_motor8",
            channels=("FC3", "FC4", "C3", "Cz", "C4", "CP3", "CP4", "Pz"),
            kind="recommended",
            description="Our recommended sensorimotor placement for an 8-channel OpenBCI Cyton.",
        ),
        Montage(
            name="bci_iv_2b",
            channels=("C3", "Cz", "C4"),
            kind="research",
            description="Three-channel layout of BCI Competition IV 2b; extreme low-channel "
            "stress test.",
        ),
    )
}
