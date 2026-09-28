"""Consent registry for data we collect ourselves (Stage 8, WP-8.2; PRIV-2/3/4).

An append-only JSON-lines ledger. Each line is one event: a participant (pseudonym)
granting or revoking consent for a purpose under a versioned consent document. The
current state is derived by replaying the ledger, so history is never lost. Nothing
directly identifying is stored here: names and contact details live in a separate,
access-restricted system keyed by the same pseudonym.

Purposes are separate on purpose: taking part in a study does **not** imply agreement to
commercial model training (PRIV-2).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path


class ConsentPurpose(StrEnum):
    """What a participant can separately agree to."""

    PRODUCT = "product"  # use of the system (calibration, decoding) during their session
    RESEARCH = "research"  # storing recordings for internal research and evaluation
    COMMERCIAL_TRAINING = "commercial_training"  # training models that may ship or be licensed


@dataclass(frozen=True, slots=True)
class ConsentEvent:
    """One grant or revocation."""

    participant: str
    purpose: ConsentPurpose
    action: str  # "grant" | "revoke"
    document_version: str
    document_sha256: str
    timestamp: str
    recorded_by: str


def document_hash(path: Path) -> str:
    """SHA-256 of the consent document the participant actually saw."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ConsentRegistry:
    """Append-only ledger at ``path`` (keep it on encrypted storage: C3 metadata)."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def _append(self, event: ConsentEvent) -> ConsentEvent:
        if not event.participant.startswith("P") or not event.participant[1:].isdigit():
            raise ValueError("participant must be a pseudonym like 'P001', never a name")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(event)) + "\n")
        return event

    def grant(
        self,
        participant: str,
        purposes: list[ConsentPurpose],
        document: Path,
        document_version: str,
        recorded_by: str,
    ) -> list[ConsentEvent]:
        """Record consent for each purpose, bound to the document's hash and version."""
        digest = document_hash(document)
        now = datetime.now(UTC).isoformat(timespec="seconds")
        return [
            self._append(
                ConsentEvent(participant, p, "grant", document_version, digest, now, recorded_by)
            )
            for p in purposes
        ]

    def revoke(self, participant: str, purpose: ConsentPurpose, recorded_by: str) -> ConsentEvent:
        """Record a revocation; downstream use must stop and data be deleted (PRIV-4)."""
        now = datetime.now(UTC).isoformat(timespec="seconds")
        event = ConsentEvent(participant, purpose, "revoke", "-", "-", now, recorded_by)
        return self._append(event)

    def events(self) -> list[ConsentEvent]:
        """All events in ledger order."""
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            raw = json.loads(line)
            raw["purpose"] = ConsentPurpose(raw["purpose"])
            out.append(ConsentEvent(**raw))
        return out

    def allowed(self, participant: str, purpose: ConsentPurpose) -> bool:
        """Return ``True`` if the latest event for (participant, purpose) is a grant."""
        state = False
        for event in self.events():
            if event.participant == participant and event.purpose == purpose:
                state = event.action == "grant"
        return state

    def participants(self, purpose: ConsentPurpose) -> list[str]:
        """Participants whose consent for ``purpose`` is currently granted, sorted.

        Loaders of our own data use this to keep only consented participants for a
        run's purpose (ADR-0012).
        """
        state: dict[str, bool] = {}
        for event in self.events():
            if event.purpose == purpose:
                state[event.participant] = event.action == "grant"
        return sorted(p for p, granted in state.items() if granted)

    def require(self, participant: str, purposes: list[ConsentPurpose]) -> None:
        """Raise ``PermissionError`` unless every purpose is currently granted."""
        missing = [p.value for p in purposes if not self.allowed(participant, p)]
        if missing:
            raise PermissionError(f"{participant} has not consented to: {', '.join(missing)}")
