"""Service settings from environment variables (no secrets in code or git)."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


def _flag(env: Mapping[str, str], key: str, default: bool = False) -> bool:
    return env.get(key, "1" if default else "0").strip().lower() in {"1", "true", "yes"}


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime configuration of the API service.

    Attributes
    ----------
    models_dir
        Root of model bundles (``<name>/<version>/bundle.json``).
    enable_docs
        Serve OpenAPI docs (disable in production).
    auth_disabled
        **Local development only**: accept requests without a token as tenant ``dev``.
    jwt_secret, jwt_audience
        HS256 secret and audience of access tokens (Supabase: project JWT secret,
        audience ``authenticated``).
    rate_limit_per_minute
        Token-bucket capacity per tenant.
    max_body_bytes
        Largest accepted request body.
    session_ttl_s
        Idle sessions (and their calibration data) are deleted after this many seconds.
    cors_origins
        Allowed browser origins.
    demo_enabled
        Serve synthetic demo trials (``/v1/demo/trials``) for the dashboard game.
    inspect_timeout_s
        Wall-clock limit of the sandboxed file-inspection worker.
    """

    models_dir: Path = Path("artifacts/models")
    enable_docs: bool = False
    auth_disabled: bool = False
    jwt_secret: str | None = None
    jwt_audience: str | None = "authenticated"
    rate_limit_per_minute: int = 600
    max_body_bytes: int = 8 * 1024 * 1024
    session_ttl_s: int = 3600
    cors_origins: tuple[str, ...] = ("http://localhost:3000",)
    demo_enabled: bool = False
    inspect_timeout_s: float = 30.0

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        """Read ``NEUROLAYER_*`` variables."""
        env = os.environ if env is None else env
        origins = env.get("NEUROLAYER_CORS_ORIGINS", "http://localhost:3000")
        return cls(
            models_dir=Path(env.get("NEUROLAYER_MODELS_DIR", "artifacts/models")),
            enable_docs=_flag(env, "NEUROLAYER_API_DOCS"),
            auth_disabled=_flag(env, "NEUROLAYER_AUTH_DISABLED"),
            jwt_secret=env.get("NEUROLAYER_JWT_SECRET") or None,
            jwt_audience=env.get("NEUROLAYER_JWT_AUDIENCE", "authenticated") or None,
            rate_limit_per_minute=int(env.get("NEUROLAYER_RATE_LIMIT_PER_MINUTE", "600")),
            max_body_bytes=int(env.get("NEUROLAYER_MAX_BODY_BYTES", str(8 * 1024 * 1024))),
            session_ttl_s=int(env.get("NEUROLAYER_SESSION_TTL_S", "3600")),
            cors_origins=tuple(o.strip() for o in origins.split(",") if o.strip()),
            demo_enabled=_flag(env, "NEUROLAYER_DEMO"),
            inspect_timeout_s=float(env.get("NEUROLAYER_INSPECT_TIMEOUT_S", "30")),
        )
