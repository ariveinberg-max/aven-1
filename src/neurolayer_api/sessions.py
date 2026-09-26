"""Calibration sessions, isolated per tenant, held in memory with a TTL (WP-6.2).

Neural data (calibration trials) lives only in process memory for the session's
lifetime and is deleted on ``DELETE /v1/sessions/{id}`` or after the idle TTL
(privacy requirements PRIV-4, PRIV-9). Replace with an encrypted store before running
more than one replica.
"""

from __future__ import annotations

import copy
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from neurolayer.core.types import FloatArray


@dataclass(slots=True)
class Session:
    """One user's calibration state for one model and one electrode layout."""

    id: str
    tenant_id: str
    model_id: str
    ch_names: tuple[str, ...]
    sfreq: float
    decoder: Any
    created_at: float
    last_used: float
    calibration: list[tuple[FloatArray, int]] = field(default_factory=list)
    unlabeled: list[FloatArray] = field(default_factory=list)
    n_decoded: int = 0

    def calibration_counts(self, label_names: tuple[str, ...]) -> dict[str, int]:
        """Labeled calibration trials per class."""
        counts = np.bincount([y for _, y in self.calibration], minlength=len(label_names))
        return {name: int(c) for name, c in zip(label_names, counts, strict=True)}


class SessionStore:
    """Thread-safe in-memory store; lookups never cross tenants."""

    def __init__(self, ttl_s: int) -> None:
        self.ttl_s = ttl_s
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def create(
        self,
        tenant_id: str,
        model_id: str,
        ch_names: tuple[str, ...],
        sfreq: float,
        base_decoder: Any,
    ) -> Session:
        """Start a session with a private copy of the base decoder."""
        now = time.monotonic()
        session = Session(
            id=secrets.token_urlsafe(16),
            tenant_id=tenant_id,
            model_id=model_id,
            ch_names=ch_names,
            sfreq=sfreq,
            decoder=copy.deepcopy(base_decoder),
            created_at=now,
            last_used=now,
        )
        with self._lock:
            self._evict(now)
            self._sessions[session.id] = session
        return session

    def get(self, tenant_id: str, session_id: str) -> Session:
        """Return the tenant's session; ``KeyError`` for unknown or foreign ids."""
        now = time.monotonic()
        with self._lock:
            self._evict(now)
            session = self._sessions.get(session_id)
            if session is None or session.tenant_id != tenant_id:
                raise KeyError(session_id)
            session.last_used = now
            return session

    def delete(self, tenant_id: str, session_id: str) -> None:
        """Delete a session and all its data."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None or session.tenant_id != tenant_id:
                raise KeyError(session_id)
            del self._sessions[session_id]

    def _evict(self, now: float) -> None:
        expired = [sid for sid, s in self._sessions.items() if now - s.last_used > self.ttl_s]
        for sid in expired:
            del self._sessions[sid]

    def __len__(self) -> int:
        return len(self._sessions)
