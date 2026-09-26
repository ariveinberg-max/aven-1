"""FastAPI application: calibration sessions and intent decoding (Stage 6).

Endpoints (all under ``/v1`` require a bearer token unless auth is disabled for local
development):

* ``GET  /healthz``, ``GET /v1/info``: no auth.
* ``GET  /v1/models``: available model bundles.
* ``POST /v1/sessions``: start a calibration session (model + electrode layout).
* ``GET  /v1/sessions/{id}`` / ``DELETE /v1/sessions/{id}``: inspect / delete (data too).
* ``POST /v1/sessions/{id}/calibration``: add labeled (and unlabeled) trials; re-adapts.
* ``POST /v1/sessions/{id}/decode``: batch decoding with class probabilities.
* ``WS   /v1/sessions/{id}/stream``: first message ``{"type": "auth", "token": ...}``,
  then ``{"type": "trial", "data": [[...]]}`` messages, each answered with a prediction.

Security posture: per-tenant session isolation, token-bucket rate limiting, request size
limit, strict response headers, CORS allowlist, no neural data in logs.
"""

from __future__ import annotations

import copy
import json
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Annotated, Any, Literal

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.websockets import WebSocketDisconnect

from neurolayer import __version__
from neurolayer_api.inference import PayloadError, canonical_channels, to_epochs
from neurolayer_api.models import LoadedModel, ModelRegistry
from neurolayer_api.security import AuthError, Principal, RateLimiter, bearer, principal_from_token
from neurolayer_api.sessions import Session, SessionStore
from neurolayer_api.settings import Settings

audit = logging.getLogger("neurolayer_api.audit")


def _configure_audit_logging() -> None:
    """Emit one JSON line per request on stderr unless the host configured handlers.

    Audit records carry method, path, status, latency and tenant, and never neural data.
    """
    if not audit.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        audit.addHandler(handler)
    audit.setLevel(logging.INFO)
    audit.propagate = False


MAX_CHANNELS = 256
MAX_TRIALS = 400


# ----------------------------------------------------------------------------- schemas
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


class ModelInfo(BaseModel):
    """A deployable model bundle."""

    id: str
    labels: list[str]
    sfreq: float
    trial_seconds: float
    trained_channels: list[str]
    synthetic_only: bool


class CreateSession(BaseModel):
    """Start a calibration session for one model and electrode layout."""

    model: str
    ch_names: list[str] = Field(min_length=1, max_length=MAX_CHANNELS)
    sfreq: float = Field(gt=0, le=10_000)


class Trial(BaseModel):
    """One trial window in **microvolts**, channels × samples."""

    data: list[list[float]] = Field(min_length=1, max_length=MAX_CHANNELS)
    label: str | None = None


class CalibrationRequest(BaseModel):
    """Labeled trials (with ``label``) and/or unlabeled trials (``label`` null)."""

    trials: list[Trial] = Field(min_length=1, max_length=MAX_TRIALS)


class DecodeRequest(BaseModel):
    """Trials to decode (labels ignored)."""

    trials: list[Trial] = Field(min_length=1, max_length=MAX_TRIALS)


class SessionInfo(BaseModel):
    """Session state (no neural data)."""

    id: str
    model: str
    ch_names: list[str]
    sfreq: float
    calibration_counts: dict[str, int]
    unlabeled_trials: int
    decoded_trials: int


class Prediction(BaseModel):
    """Decoded intent for one trial."""

    label: str
    probabilities: dict[str, float]


class DecodeResponse(BaseModel):
    """Predictions for a batch of trials."""

    predictions: list[Prediction]
    latency_ms: float


CAPABILITIES = [
    CapabilityStatus(
        id="CAP-1", name="Calibration-efficient motor intent", status="in_development"
    ),
    CapabilityStatus(id="CAP-2", name="Reactive selection (SSVEP/c-VEP)", status="planned"),
    CapabilityStatus(id="CAP-3", name="Implicit feedback (error potentials)", status="planned"),
]


# ----------------------------------------------------------------------------- helpers
def _session_info(session: Session, model: LoadedModel) -> SessionInfo:
    return SessionInfo(
        id=session.id,
        model=session.model_id,
        ch_names=list(session.ch_names),
        sfreq=session.sfreq,
        calibration_counts=session.calibration_counts(model.info.label_names),
        unlabeled_trials=len(session.unlabeled),
        decoded_trials=session.n_decoded,
    )


def _arrays(trials: list[Trial]) -> list[Any]:
    arrays = []
    for trial in trials:
        lengths = {len(row) for row in trial.data}
        if len(lengths) != 1:
            raise PayloadError("all channels of a trial must have the same number of samples")
        arrays.append(np.asarray(trial.data, dtype=np.float64))
    return arrays


def _predict(session: Session, model: LoadedModel, arrays: list[Any]) -> list[Prediction]:
    epochs = to_epochs(arrays, None, session.ch_names, session.sfreq, model.info)
    probabilities = session.decoder.predict_proba(epochs)
    labels = model.info.label_names
    session.n_decoded += len(arrays)
    return [
        Prediction(
            label=labels[int(np.argmax(p))],
            probabilities={name: round(float(v), 6) for name, v in zip(labels, p, strict=True)},
        )
        for p in probabilities
    ]


def principal(request: Request, authorization: Annotated[str | None, Header()] = None) -> Principal:
    """Authenticate the caller and apply the per-tenant rate limit."""
    settings: Settings = request.app.state.settings
    limiter: RateLimiter = request.app.state.limiter
    try:
        who = principal_from_token(bearer(authorization), settings)
    except AuthError as exc:
        raise HTTPException(401, str(exc), headers={"WWW-Authenticate": "Bearer"}) from exc
    if not limiter.allow(who.tenant_id):
        raise HTTPException(429, "rate limit exceeded")
    request.state.tenant = who.tenant_id
    return who


Caller = Annotated[Principal, Depends(principal)]


def create_app(settings: Settings | None = None, *, enable_docs: bool | None = None) -> FastAPI:
    """Build the service. ``enable_docs`` overrides ``settings.enable_docs``."""
    settings = settings or Settings.from_env()
    if settings.jwt_secret is not None and len(settings.jwt_secret.encode()) < 32:
        raise ValueError("NEUROLAYER_JWT_SECRET must be at least 32 bytes (RFC 7518 §3.2)")
    docs = settings.enable_docs if enable_docs is None else enable_docs
    _configure_audit_logging()
    app = FastAPI(
        title="neurolayer API",
        version=__version__,
        docs_url="/docs" if docs else None,
        redoc_url=None,
        openapi_url="/openapi.json" if docs else None,
    )
    registry = ModelRegistry(settings.models_dir)
    store = SessionStore(settings.session_ttl_s)
    limiter = RateLimiter(settings.rate_limit_per_minute)
    app.state.registry, app.state.sessions, app.state.settings = registry, store, settings
    app.state.limiter = limiter

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Authorization", "Content-Type"],
    )

    @app.middleware("http")
    async def guard(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        started = time.perf_counter()
        length = request.headers.get("content-length")
        if length is not None and int(length) > settings.max_body_bytes:
            response: Response = Response(status_code=413, content="request body too large")
        else:
            response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        audit.info(
            json.dumps(
                {
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                    "tenant": getattr(request.state, "tenant", None),
                }
            )
        )
        return response

    def model_or_404(model_id: str) -> LoadedModel:
        try:
            return registry.get(model_id)
        except KeyError as exc:
            raise HTTPException(404, f"unknown model {model_id!r}") from exc

    def session_or_404(who: Principal, session_id: str) -> Session:
        try:
            return store.get(who.tenant_id, session_id)
        except KeyError as exc:
            raise HTTPException(404, "session not found") from exc

    @app.get("/healthz", response_model=Health, tags=["ops"])
    def healthz() -> Health:
        return Health()

    @app.get("/v1/info", response_model=ServiceInfo, tags=["meta"])
    def info() -> ServiceInfo:
        return ServiceInfo(version=__version__, capabilities=CAPABILITIES)

    @app.get("/v1/models", response_model=list[ModelInfo], tags=["models"])
    def models(who: Caller) -> list[ModelInfo]:
        return [
            ModelInfo(
                id=model_id,
                labels=list(i.label_names),
                sfreq=i.sfreq,
                trial_seconds=i.n_times / i.sfreq,
                trained_channels=list(i.trained_channels),
                synthetic_only=bool(i.preprocessing.get("synthetic_only", False)),
            )
            for model_id, i in registry.available().items()
        ]

    @app.post("/v1/sessions", response_model=SessionInfo, status_code=201, tags=["sessions"])
    def create_session(body: CreateSession, who: Caller) -> SessionInfo:
        model = model_or_404(body.model)
        try:
            channels = canonical_channels(body.ch_names)
        except PayloadError as exc:
            raise HTTPException(422, str(exc)) from exc
        session = store.create(who.tenant_id, body.model, channels, body.sfreq, model.decoder)
        return _session_info(session, model)

    @app.get("/v1/sessions/{session_id}", response_model=SessionInfo, tags=["sessions"])
    def get_session(session_id: str, who: Caller) -> SessionInfo:
        session = session_or_404(who, session_id)
        return _session_info(session, model_or_404(session.model_id))

    @app.delete("/v1/sessions/{session_id}", status_code=204, tags=["sessions"])
    def delete_session(session_id: str, who: Caller) -> Response:
        try:
            store.delete(who.tenant_id, session_id)
        except KeyError as exc:
            raise HTTPException(404, "session not found") from exc
        return Response(status_code=204)

    @app.post(
        "/v1/sessions/{session_id}/calibration", response_model=SessionInfo, tags=["sessions"]
    )
    def calibrate(session_id: str, body: CalibrationRequest, who: Caller) -> SessionInfo:
        session = session_or_404(who, session_id)
        model = model_or_404(session.model_id)
        labels = model.info.label_names
        try:
            arrays = _arrays(body.trials)
            unknown = {
                t.label for t in body.trials if t.label is not None and t.label not in labels
            }
            if unknown:
                raise PayloadError(
                    f"unknown labels {sorted(unknown)}; model labels: {list(labels)}"
                )
            for array, trial in zip(arrays, body.trials, strict=True):
                # Validate (and preprocess) now so bad trials never enter the session.
                to_epochs([array], None, session.ch_names, session.sfreq, model.info)
                if trial.label is None:
                    session.unlabeled.append(array)
                else:
                    session.calibration.append((array, labels.index(trial.label)))
            calibration = (
                to_epochs([a for a, _ in session.calibration], [y for _, y in session.calibration],
                          session.ch_names, session.sfreq, model.info)
                if session.calibration else None
            )  # fmt: skip
            unlabeled = (
                to_epochs(session.unlabeled, None, session.ch_names, session.sfreq, model.info)
                if session.unlabeled
                else None
            )
        except PayloadError as exc:
            raise HTTPException(422, str(exc)) from exc
        session.decoder = copy.deepcopy(model.decoder).adapt(calibration, unlabeled)
        return _session_info(session, model)

    @app.post("/v1/sessions/{session_id}/decode", response_model=DecodeResponse, tags=["decode"])
    def decode(session_id: str, body: DecodeRequest, who: Caller) -> DecodeResponse:
        started = time.perf_counter()
        session = session_or_404(who, session_id)
        model = model_or_404(session.model_id)
        try:
            predictions = _predict(session, model, _arrays(body.trials))
        except PayloadError as exc:
            raise HTTPException(422, str(exc)) from exc
        return DecodeResponse(
            predictions=predictions, latency_ms=round((time.perf_counter() - started) * 1000, 3)
        )

    @app.websocket("/v1/sessions/{session_id}/stream")
    async def stream(websocket: WebSocket, session_id: str) -> None:
        await websocket.accept()
        try:
            first = await websocket.receive_json()
            token = first.get("token") if isinstance(first, dict) else None
            who = principal_from_token(token, settings)
            session = store.get(who.tenant_id, session_id)
            model = registry.get(session.model_id)
            await websocket.send_json({"type": "ready", "session": session.id})
            while True:
                message = await websocket.receive_json()
                if not limiter.allow(who.tenant_id):
                    await websocket.send_json({"type": "error", "detail": "rate limit exceeded"})
                    continue
                try:
                    array = np.asarray(message["data"], dtype=np.float64)
                    prediction = _predict(session, model, [array])[0]
                except (KeyError, ValueError, PayloadError) as exc:
                    await websocket.send_json({"type": "error", "detail": str(exc)})
                    continue
                await websocket.send_json({"type": "prediction", **prediction.model_dump()})
        except (AuthError, KeyError):
            await websocket.close(code=4401)
        except WebSocketDisconnect:
            return

    return app


def _default_app() -> FastAPI:
    return create_app()


app = _default_app()
