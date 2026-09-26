"""FastAPI application factory.

Stage 6 skeleton: health and service info only. Inference endpoints, authentication
(Supabase JWT), tenant isolation and rate limiting arrive with WP-6.x, per
``docs/architecture/security-and-privacy.md``.
"""

from __future__ import annotations

import os
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel

from neurolayer import __version__


class Health(BaseModel):
    """Liveness response."""

    status: Literal["ok"] = "ok"


class CapabilityStatus(BaseModel):
    """Development status of one platform capability."""

    id: str
    name: str
    status: Literal["planned", "in_development", "gated", "available"]


class ServiceInfo(BaseModel):
    """Service metadata."""

    service: str = "neurolayer-api"
    version: str
    api_version: str = "v1"
    capabilities: list[CapabilityStatus]


CAPABILITIES = [
    CapabilityStatus(
        id="CAP-1", name="Calibration-efficient motor intent", status="in_development"
    ),
    CapabilityStatus(id="CAP-2", name="Reactive selection (SSVEP/c-VEP)", status="planned"),
    CapabilityStatus(id="CAP-3", name="Implicit feedback (error potentials)", status="planned"),
]


def create_app(*, enable_docs: bool | None = None) -> FastAPI:
    """Build the API app.

    Parameters
    ----------
    enable_docs
        Serve interactive OpenAPI docs. Defaults to the ``NEUROLAYER_API_DOCS``
        environment variable (``"1"`` enables); disable in production.
    """
    if enable_docs is None:
        enable_docs = os.environ.get("NEUROLAYER_API_DOCS", "0") == "1"
    app = FastAPI(
        title="neurolayer API",
        version=__version__,
        docs_url="/docs" if enable_docs else None,
        redoc_url=None,
        openapi_url="/openapi.json" if enable_docs else None,
    )

    @app.get("/healthz", response_model=Health, tags=["ops"])
    def healthz() -> Health:
        return Health()

    @app.get("/v1/info", response_model=ServiceInfo, tags=["meta"])
    def info() -> ServiceInfo:
        return ServiceInfo(version=__version__, capabilities=CAPABILITIES)

    return app


app = create_app()
