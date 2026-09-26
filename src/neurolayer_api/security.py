"""Authentication (JWT) and per-tenant rate limiting (WP-6.3)."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

import jwt

from neurolayer_api.settings import Settings


class AuthError(Exception):
    """Missing or invalid credentials."""


@dataclass(frozen=True, slots=True)
class Principal:
    """The authenticated caller. All data access is scoped to ``tenant_id``."""

    tenant_id: str
    user_id: str


def principal_from_token(token: str | None, settings: Settings) -> Principal:
    """Verify an HS256 bearer token and derive the tenant.

    Tenant resolution: ``tenant_id`` claim, else ``app_metadata.tenant_id`` (Supabase
    custom claims), else the user id (single-user tenant).
    """
    if settings.auth_disabled:
        return Principal(tenant_id="dev", user_id="dev")
    if not token:
        raise AuthError("missing bearer token")
    if not settings.jwt_secret:
        raise AuthError("server has no JWT secret configured")
    try:
        claims: dict[str, Any] = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=["HS256"],
            audience=settings.jwt_audience,
            options={"require": ["exp", "sub"]},
        )
    except jwt.PyJWTError as exc:
        raise AuthError(f"invalid token: {exc}") from exc
    user = str(claims["sub"])
    metadata = claims.get("app_metadata") or {}
    tenant = claims.get("tenant_id") or (
        metadata.get("tenant_id") if isinstance(metadata, dict) else None
    )
    return Principal(tenant_id=str(tenant or user), user_id=user)


def bearer(header: str | None) -> str | None:
    """Extract the token from an ``Authorization: Bearer <token>`` header."""
    if not header:
        return None
    scheme, _, value = header.partition(" ")
    return value.strip() if scheme.lower() == "bearer" and value.strip() else None


class RateLimiter:
    """Thread-safe token bucket per key (in-process; use Redis when running replicas)."""

    def __init__(self, per_minute: int) -> None:
        self.capacity = float(per_minute)
        self.rate = per_minute / 60.0
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str, now: float | None = None) -> bool:
        """Consume one token for ``key`` if available."""
        now = time.monotonic() if now is None else now
        with self._lock:
            tokens, last = self._buckets.get(key, (self.capacity, now))
            tokens = min(self.capacity, tokens + (now - last) * self.rate)
            allowed = tokens >= 1.0
            self._buckets[key] = (tokens - 1.0 if allowed else tokens, now)
            return allowed
