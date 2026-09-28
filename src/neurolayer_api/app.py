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
* ``POST /v1/recordings/inspect``: summarize an uploaded EDF/BDF file, parsed in a
  resource-limited subprocess (``neurolayer_api.inspect_worker``).
* ``GET  /v1/demo/trials``: synthetic trials for the dashboard calibration game (only
  when ``NEUROLAYER_DEMO=1``).

Security posture: per-tenant session isolation, token-bucket rate limiting, request size
limit, strict response headers, CORS allowlist, no neural data in logs.
"""

from __future__ import annotations

import copy
import json
import logging
import subprocess
import sys
import tempfile
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from starlette.websockets import WebSocketDisconnect

from neurolayer import __version__
from neurolayer.data.synthetic import SyntheticMIConfig, generate_synthetic_mi
from neurolayer.models.bundle import BundleInfo
from neurolayer_api.inference import PayloadError, canonical_channels, to_epochs
from neurolayer_api.models import LoadedModel, ModelRegistry
from neurolayer_api.security import AuthError, Principal, RateLimiter, bearer, principal_from_token
from neurolayer_api.sessions import Session, SessionStore
from neurolayer_api.settings import Settings
from neurolayer_api.warmup import warm_up

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
    window_offset_s: float = Field(description="Trial window start, seconds after the cue.")
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


class DemoTrials(BaseModel):
    """Synthetic trials (microvolts) standing in for a headset in the dashboard demo."""

    synthetic: Literal[True] = True
    ch_names: list[str]
    sfreq: float
    trial_seconds: float
    trials: list[Trial]


class InspectChannel(BaseModel):
    """Per-channel summary of an uploaded recording (band power in dB re 1 µV²/Hz)."""

    name: str
    canonical: str | None
    mu_db: float | None
    beta_db: float | None
    flat: bool
    noisy: bool


class InspectPreview(BaseModel):
    """Downsampled first seconds of up to eight channels, microvolts."""

    sfreq: float
    channels: list[str]
    data_uv: list[list[float]]


class InspectReport(BaseModel):
    """Summary of an uploaded recording (returned to the uploader only; never stored)."""

    format: Literal["edf", "bdf"]
    sfreq: float
    duration_s: float
    n_channels: int
    channels: list[InspectChannel]
    annotations: dict[str, int]
    line_noise_ratio: dict[str, float | None]
    median_abs_amplitude_uv: float
    issues: list[str]
    preview: InspectPreview


CAPABILITIES = [
    CapabilityStatus(
        id="CAP-1", name="Calibration-efficient motor intent", status="in_development"
    ),
    CapabilityStatus(id="CAP-2", name="Reactive selection (SSVEP/c-VEP)", status="planned"),
    CapabilityStatus(id="CAP-3", name="Implicit feedback (error potentials)", status="planned"),
]


# ----------------------------------------------------------------------------- helpers
def _window_offset(info: BundleInfo) -> float:
    pipeline = info.preprocessing.get("pipeline") or {}
    return float(pipeline.get("epoching", {}).get("tmin", 0.5)) if pipeline else 0.0


def _sniff_format(head: bytes) -> Literal["edf", "bdf"]:
    if head[:8] == b"\xffBIOSEMI":
        return "bdf"
    if head[:8] == b"0       ":
        return "edf"
    raise HTTPException(415, "only EDF/EDF+ and BDF files are accepted")


def _run_inspect_worker(content: bytes, file_format: str, timeout_s: float) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="nl-inspect-") as tmp:
        path = Path(tmp) / f"upload.{file_format}"
        path.write_bytes(content)
        try:
            done = subprocess.run(  # noqa: S603  (fixed argv, no shell)
                [sys.executable, "-m", "neurolayer_api.inspect_worker", str(path), file_format],
                capture_output=True,
                timeout=timeout_s,
                check=False,
                env={
                    "PATH": "/usr/bin:/bin",
                    "MNE_DONTWRITE_HOME": "true",
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "HOME": tmp,
                },
            )
        except subprocess.TimeoutExpired as exc:
            raise HTTPException(422, "the file took too long to parse") from exc
    try:
        payload: dict[str, Any] = json.loads(done.stdout.decode() or "{}")
    except json.JSONDecodeError:
        payload = {}
    if done.returncode == 4:
        raise HTTPException(501, payload.get("error", "file inspection is not installed"))
    if done.returncode != 0 or "error" in payload or not payload:
        raise HTTPException(422, payload.get("error", "could not read the file"))
    return payload


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
    registry = ModelRegistry(settings.models_dir)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if settings.warmup:
            app.state.warmup = await run_in_threadpool(warm_up, registry, settings.demo_enabled)
            audit.info(json.dumps({"event": "warmup", "seconds": app.state.warmup}))
        yield

    app = FastAPI(
        title="neurolayer API",
        version=__version__,
        docs_url="/docs" if docs else None,
        redoc_url=None,
        openapi_url="/openapi.json" if docs else None,
        lifespan=lifespan,
    )
    app.state.warmup = {}
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
                window_offset_s=_window_offset(i),
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

    @app.post("/v1/recordings/inspect", response_model=InspectReport, tags=["recordings"])
    async def inspect_recording(request: Request, who: Caller) -> InspectReport:
        # Stream with a hard cap: chunked uploads carry no Content-Length for the middleware.
        chunks, size = [], 0
        async for chunk in request.stream():
            size += len(chunk)
            if size > settings.max_body_bytes:
                raise HTTPException(413, "request body too large")
            chunks.append(chunk)
        content = b"".join(chunks)
        file_format = _sniff_format(content[:8])
        # The worker blocks for up to the timeout: keep it off the event loop.
        report = await run_in_threadpool(
            _run_inspect_worker, content, file_format, settings.inspect_timeout_s
        )
        return InspectReport.model_validate(report)

    @app.get("/v1/demo/trials", response_model=DemoTrials, tags=["demo"])
    def demo_trials(
        who: Caller,
        model: str,
        n_per_class: Annotated[int, Query(ge=1, le=60)] = 30,
        seed: Annotated[int, Query(ge=0, le=10_000)] = 0,
    ) -> DemoTrials:
        if not settings.demo_enabled:
            raise HTTPException(404, "demo data is disabled (set NEUROLAYER_DEMO=1)")
        info = model_or_404(model).info
        config = SyntheticMIConfig(
            n_subjects=1,
            n_trials_per_class=n_per_class,
            sfreq=info.sfreq,
            trial_seconds=info.n_times / info.sfreq,
            channels=info.trained_channels,
            signal_to_noise=0.5,
            efficiencies=(0.8,),
            seed=100_000 + seed,  # never one of the training seeds
            dataset_id="demo",
        )
        epochs = generate_synthetic_mi(config).epochs
        trials = [
            Trial(data=np.round(x / 1e-6, 3).tolist(), label=epochs.label_names[int(y)])
            for x, y in zip(epochs.X, epochs.y, strict=True)
        ]
        return DemoTrials(
            ch_names=list(epochs.ch_names),
            sfreq=epochs.sfreq,
            trial_seconds=info.n_times / info.sfreq,
            trials=trials,
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
