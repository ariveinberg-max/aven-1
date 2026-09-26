"""Canonical label vocabularies.

Adapters map dataset-native codes (``769``, ``T1``, ``"left"``) onto these names, and
drop anything unmapped *explicitly* (with a count in the dataset QA report).
"""

from __future__ import annotations

MOTOR_IMAGERY_LABELS: frozenset[str] = frozenset(
    {"left_hand", "right_hand", "both_hands", "feet", "tongue", "rest"}
)
"""Vocabulary for motor imagery and motor execution paradigms."""

CAP1_LABELS: tuple[str, str] = ("left_hand", "right_hand")
"""Label order used by CAP-1 v1 (index 0 = left hand, index 1 = right hand)."""
