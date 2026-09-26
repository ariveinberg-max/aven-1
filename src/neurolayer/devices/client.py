"""Python client for the neurolayer API, and sliding-window live decoding (WP-7.3).

The client depends only on the standard library. Its transport is injectable, so the
same code talks to a local service, a deployed one, or an in-process test app. Trials
are sent in **microvolts**, which is the API's input unit.

Default deployment is local (``http://localhost:8000``): neural data stays on the
user's machine unless they point the bridge at a remote service (PRIV-9).
"""

from __future__ import annotations

import contextlib
import json
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import numpy as np

from neurolayer.core.types import FloatArray
from neurolayer.devices.output import Intent
from neurolayer.devices.sources import StreamSource

Transport = Callable[[str, str, dict[str, Any] | None], Any]
"""``(method, path, json_body) -> parsed JSON`` (``None`` for empty responses)."""

MAX_TRIALS_PER_REQUEST = 200


class ApiError(RuntimeError):
    """The service answered with an error status."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(f"HTTP {status}: {detail}")
        self.status = status
        self.detail = detail


class UrllibTransport:
    """HTTP(S) transport on :mod:`urllib` with bearer-token auth."""

    def __init__(self, base_url: str, token: str | None = None, timeout_s: float = 30.0) -> None:
        if urlsplit(base_url).scheme not in {"http", "https"}:
            raise ValueError(f"base_url must be http(s), got {base_url!r}")
        self.base_url = base_url.rstrip("/")
        self._token = token
        self._timeout = timeout_s

    def __call__(self, method: str, path: str, body: dict[str, Any] | None) -> Any:
        """Send one request and decode the JSON response."""
        headers = {"Accept": "application/json"}
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        request = urllib.request.Request(  # noqa: S310  (scheme validated in __init__)
            self.base_url + path, data=data, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:  # noqa: S310
                payload = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            with contextlib.suppress(json.JSONDecodeError, AttributeError):
                detail = str(json.loads(detail).get("detail", detail))
            raise ApiError(exc.code, detail) from exc
        return json.loads(payload) if payload else None


@dataclass(frozen=True, slots=True)
class RemoteModel:
    """What the client needs to know about a served model."""

    id: str
    labels: tuple[str, ...]
    trial_seconds: float
    window_offset_s: float
    trained_channels: tuple[str, ...]
    synthetic_only: bool


def _trials(windows: Sequence[FloatArray], labels: Sequence[str | None]) -> list[dict[str, Any]]:
    return [
        {"data": np.round(w, 4).tolist(), "label": label}
        for w, label in zip(windows, labels, strict=True)
    ]


class NeurolayerClient:
    """Typed wrapper over the v1 REST API."""

    def __init__(self, transport: Transport) -> None:
        self._send = transport

    @classmethod
    def connect(cls, base_url: str, token: str | None = None) -> NeurolayerClient:
        """Client for a running service."""
        return cls(UrllibTransport(base_url, token))

    def models(self) -> dict[str, RemoteModel]:
        """Models the caller may use, by id."""
        out = {}
        for raw in self._send("GET", "/v1/models", None):
            out[raw["id"]] = RemoteModel(
                id=raw["id"],
                labels=tuple(raw["labels"]),
                trial_seconds=float(raw["trial_seconds"]),
                window_offset_s=float(raw.get("window_offset_s", 0.5)),
                trained_channels=tuple(raw["trained_channels"]),
                synthetic_only=bool(raw["synthetic_only"]),
            )
        return out

    def create_session(self, model: str, ch_names: Sequence[str], sfreq: float) -> str:
        """Open a calibration session; returns its id."""
        body = {"model": model, "ch_names": list(ch_names), "sfreq": sfreq}
        session_id: str = self._send("POST", "/v1/sessions", body)["id"]
        return session_id

    def calibrate(
        self,
        session_id: str,
        windows: Sequence[FloatArray],
        labels: Sequence[str | None],
    ) -> dict[str, Any]:
        """Send calibration trials (``None`` label = unlabeled); returns session info."""
        info: dict[str, Any] = {}
        for start in range(0, len(windows), MAX_TRIALS_PER_REQUEST):
            batch = slice(start, start + MAX_TRIALS_PER_REQUEST)
            info = self._send(
                "POST",
                f"/v1/sessions/{session_id}/calibration",
                {"trials": _trials(windows[batch], labels[batch])},
            )
        return info

    def decode(self, session_id: str, windows: Sequence[FloatArray]) -> list[Intent]:
        """Decode trial windows into intents."""
        response = self._send(
            "POST",
            f"/v1/sessions/{session_id}/decode",
            {"trials": _trials(windows, [None] * len(windows))},
        )
        return [
            Intent(
                label=p["label"],
                confidence=float(max(p["probabilities"].values())),
                probabilities={k: float(v) for k, v in p["probabilities"].items()},
            )
            for p in response["predictions"]
        ]

    def delete_session(self, session_id: str) -> None:
        """End the session and erase its calibration data on the server."""
        self._send("DELETE", f"/v1/sessions/{session_id}", None)


class LiveDecoder:
    """Sliding-window decoding of a live stream.

    Every ``hop_s`` seconds of new data, the most recent ``window_s`` seconds are sent
    for decoding. Note that CAP-1 models are trained on **cue-locked** windows; this
    asynchronous (self-paced) use is a product prototype, not a measured capability.
    """

    def __init__(
        self,
        source: StreamSource,
        client: NeurolayerClient,
        session_id: str,
        window_s: float,
        hop_s: float = 0.5,
    ) -> None:
        self._source = source
        self._client = client
        self._session = session_id
        self._window = round(window_s * source.sfreq)
        self._hop = max(1, round(hop_s * source.sfreq))
        self._buffer = np.zeros((len(source.ch_names), 0))
        self._since_decode = 0

    def step(self) -> Intent | None:
        """Read new samples; decode if a hop has elapsed and the window is full."""
        chunk = self._source.read()
        if chunk.shape[1]:
            self._buffer = np.concatenate([self._buffer, chunk], axis=1)[:, -self._window :]
            self._since_decode += chunk.shape[1]
        if self._buffer.shape[1] < self._window or self._since_decode < self._hop:
            return None
        self._since_decode = 0
        return self._client.decode(self._session, [self._buffer.copy()])[0]
