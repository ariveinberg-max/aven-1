from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from neurolayer.data.synthetic import SyntheticMIConfig, generate_synthetic_mi
from neurolayer_api.app import create_app
from neurolayer_api.security import RateLimiter
from neurolayer_api.settings import Settings

torch = pytest.importorskip("torch")
jwt = pytest.importorskip("jwt")

from neurolayer.models.bundle import export_bundle  # noqa: E402
from neurolayer.models.proprietary import SpatialFieldDecoder  # noqa: E402

SECRET = "0" * 40  # low-entropy test-only key (not a real secret)
MODEL = "tiny@0.1.0"


@pytest.fixture(scope="module")
def synthetic():  # type: ignore[no-untyped-def]
    return generate_synthetic_mi(
        SyntheticMIConfig(n_subjects=4, n_trials_per_class=30, efficiencies=(0.9,) * 4, seed=5)
    ).epochs


@pytest.fixture(scope="module")
def models_dir(tmp_path_factory: pytest.TempPathFactory, synthetic) -> Path:  # type: ignore[no-untyped-def]
    root = tmp_path_factory.mktemp("models")
    decoder = SpatialFieldDecoder(epochs=5, finetune_epochs=5, device="cpu")
    source = synthetic.subset(synthetic.subject != "sub-000")
    decoder.fit(source)
    export_bundle(
        decoder,
        root / "tiny" / "0.1.0",
        name="tiny",
        version="0.1.0",
        label_names=source.label_names,
        sfreq=source.sfreq,
        n_times=source.n_times,
        trained_channels=source.ch_names,
        preprocessing={"synthetic_only": True},
    )
    return root


def token(tenant: str, user: str = "u1", expires_in: int = 3600, **extra: Any) -> str:
    claims = {
        "sub": user,
        "tenant_id": tenant,
        "aud": "authenticated",
        "exp": int(time.time()) + expires_in,
        **extra,
    }
    return str(jwt.encode(claims, SECRET, algorithm="HS256"))


def headers(tenant: str = "acme") -> dict[str, str]:
    return {"Authorization": f"Bearer {token(tenant)}"}


def make_client(models_dir: Path, **overrides: Any) -> TestClient:
    settings = Settings(models_dir=models_dir, jwt_secret=SECRET, **overrides)
    return TestClient(create_app(settings))


def trials_payload(epochs, idx: list[int], with_labels: bool) -> list[dict[str, Any]]:  # type: ignore[no-untyped-def]
    out = []
    for i in idx:
        item: dict[str, Any] = {"data": (epochs.X[i] * 1e6).tolist()}  # microvolts
        if with_labels:
            item["label"] = epochs.label_names[int(epochs.y[i])]
        out.append(item)
    return out


def new_session(client: TestClient, ch_names: list[str], tenant: str = "acme") -> str:
    response = client.post(
        "/v1/sessions",
        json={"model": MODEL, "ch_names": ch_names, "sfreq": 128.0},
        headers=headers(tenant),
    )
    assert response.status_code == 201, response.text
    return str(response.json()["id"])


def test_auth_is_required_and_verified(models_dir: Path) -> None:
    client = make_client(models_dir)
    assert client.get("/healthz").status_code == 200
    assert client.get("/v1/models").status_code == 401
    assert client.get("/v1/models", headers={"Authorization": "Bearer nonsense"}).status_code == 401
    expired = {"Authorization": f"Bearer {token('acme', expires_in=-10)}"}
    assert client.get("/v1/models", headers=expired).status_code == 401
    wrong_secret = jwt.encode(
        {"sub": "x", "aud": "authenticated", "exp": int(time.time()) + 60},
        "1" * 40,
        algorithm="HS256",
    )
    assert (
        client.get("/v1/models", headers={"Authorization": f"Bearer {wrong_secret}"}).status_code
        == 401
    )
    listing = client.get("/v1/models", headers=headers())
    assert listing.status_code == 200
    assert listing.json()[0]["id"] == MODEL
    assert listing.json()[0]["trial_seconds"] == 2.0


def test_calibrate_then_decode(models_dir: Path, synthetic) -> None:  # type: ignore[no-untyped-def]
    client = make_client(models_dir)
    target = synthetic.for_subject("synthetic_mi", "sub-000")
    session = new_session(client, list(target.ch_names))
    calibrated = client.post(
        f"/v1/sessions/{session}/calibration",
        json={
            "trials": trials_payload(target, list(range(20)), True)
            + trials_payload(target, list(range(20, 30)), False)
        },
        headers=headers(),
    )
    assert calibrated.status_code == 200, calibrated.text
    body = calibrated.json()
    assert sum(body["calibration_counts"].values()) == 20
    assert body["unlabeled_trials"] == 10

    test_idx = list(range(30, len(target)))
    decoded = client.post(
        f"/v1/sessions/{session}/decode",
        json={"trials": trials_payload(target, test_idx, False)},
        headers=headers(),
    )
    assert decoded.status_code == 200
    predictions = decoded.json()["predictions"]
    assert len(predictions) == len(test_idx)
    for p in predictions:
        assert abs(sum(p["probabilities"].values()) - 1.0) < 1e-4
    labels = [target.label_names[int(y)] for y in target.y[test_idx]]
    accuracy = np.mean([p["label"] == t for p, t in zip(predictions, labels, strict=True)])
    assert accuracy > 0.6  # synthetic, well above chance
    info = client.get(f"/v1/sessions/{session}", headers=headers()).json()
    assert info["decoded_trials"] == len(test_idx)


def test_payload_validation(models_dir: Path, synthetic) -> None:  # type: ignore[no-untyped-def]
    client = make_client(models_dir)
    bad_channel = client.post(
        "/v1/sessions",
        json={"model": MODEL, "ch_names": ["C3", "EXG1"], "sfreq": 128.0},
        headers=headers(),
    )
    assert bad_channel.status_code == 422
    unknown_model = client.post(
        "/v1/sessions",
        json={"model": "nope@1", "ch_names": ["C3"], "sfreq": 128.0},
        headers=headers(),
    )
    assert unknown_model.status_code == 404
    target = synthetic.for_subject("synthetic_mi", "sub-000")
    session = new_session(client, list(target.ch_names))
    short = {"trials": [{"data": (target.X[0][:, :100] * 1e6).tolist(), "label": "left_hand"}]}
    response = client.post(f"/v1/sessions/{session}/calibration", json=short, headers=headers())
    assert response.status_code == 422
    assert "2 s long" in response.json()["detail"]
    wrong_label = {"trials": [{"data": (target.X[0] * 1e6).tolist(), "label": "feet"}]}
    assert (
        client.post(
            f"/v1/sessions/{session}/calibration", json=wrong_label, headers=headers()
        ).status_code
        == 422
    )
    ragged = {"trials": [{"data": [[1.0, 2.0], [1.0]]}]}
    assert (
        client.post(f"/v1/sessions/{session}/decode", json=ragged, headers=headers()).status_code
        == 422
    )


def test_tenant_isolation_and_deletion(models_dir: Path, synthetic) -> None:  # type: ignore[no-untyped-def]
    client = make_client(models_dir)
    target = synthetic.for_subject("synthetic_mi", "sub-000")
    session = new_session(client, list(target.ch_names), tenant="acme")
    assert client.get(f"/v1/sessions/{session}", headers=headers("other-co")).status_code == 404
    payload = {"trials": trials_payload(target, [0], False)}
    assert (
        client.post(
            f"/v1/sessions/{session}/decode", json=payload, headers=headers("other-co")
        ).status_code
        == 404
    )
    assert client.delete(f"/v1/sessions/{session}", headers=headers("other-co")).status_code == 404
    assert client.delete(f"/v1/sessions/{session}", headers=headers("acme")).status_code == 204
    assert client.get(f"/v1/sessions/{session}", headers=headers("acme")).status_code == 404


def test_rate_limit_body_limit_and_headers(models_dir: Path) -> None:
    client = make_client(models_dir, rate_limit_per_minute=3)
    codes = [client.get("/v1/models", headers=headers()).status_code for _ in range(4)]
    assert codes == [200, 200, 200, 429]
    assert client.get("/v1/models", headers=headers("another")).status_code == 200  # per tenant
    small = make_client(models_dir, max_body_bytes=100)
    too_big = small.post(
        "/v1/sessions", content=b"x" * 500, headers=headers() | {"Content-Type": "application/json"}
    )
    assert too_big.status_code == 413
    response = client.get("/healthz")
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["Cache-Control"] == "no-store"


def test_websocket_stream(models_dir: Path, synthetic) -> None:  # type: ignore[no-untyped-def]
    client = make_client(models_dir)
    target = synthetic.for_subject("synthetic_mi", "sub-000")
    session = new_session(client, list(target.ch_names))
    with client.websocket_connect(f"/v1/sessions/{session}/stream") as ws:
        ws.send_json({"type": "auth", "token": token("acme")})
        assert ws.receive_json()["type"] == "ready"
        ws.send_json({"type": "trial", "data": (target.X[40] * 1e6).tolist()})
        message = ws.receive_json()
        assert message["type"] == "prediction"
        assert message["label"] in target.label_names
        ws.send_json({"type": "trial", "data": [[1.0]]})
        assert ws.receive_json()["type"] == "error"
    with client.websocket_connect(f"/v1/sessions/{session}/stream") as ws:
        ws.send_json({"type": "auth", "token": token("other-co")})
        with pytest.raises(Exception):  # noqa: B017 - closed with code 4401
            ws.receive_json()


def test_decode_latency_budget(models_dir: Path, synthetic) -> None:  # type: ignore[no-untyped-def]
    """WP-6.2: p95 server-side decode latency for one 2 s window on CPU < 50 ms."""
    client = make_client(models_dir)
    target = synthetic.for_subject("synthetic_mi", "sub-000")
    session = new_session(client, list(target.ch_names))
    payload = {"trials": trials_payload(target, [0], False)}
    latencies = [
        client.post(f"/v1/sessions/{session}/decode", json=payload, headers=headers()).json()[
            "latency_ms"
        ]
        for _ in range(40)
    ]
    assert np.percentile(latencies, 95) < 50.0


def test_settings_from_env_and_rate_limiter_refill() -> None:
    settings = Settings.from_env(
        {
            "NEUROLAYER_MODELS_DIR": "/m",
            "NEUROLAYER_AUTH_DISABLED": "true",
            "NEUROLAYER_CORS_ORIGINS": "https://a.example, https://b.example",
        }
    )
    assert settings.auth_disabled
    assert settings.cors_origins == ("https://a.example", "https://b.example")
    limiter = RateLimiter(per_minute=60)  # one token per second
    assert all(limiter.allow("t", now=0.0) for _ in range(60))
    assert not limiter.allow("t", now=0.0)
    assert limiter.allow("t", now=1.01)


def test_short_jwt_secret_is_rejected_at_startup(models_dir: Path) -> None:
    with pytest.raises(ValueError, match="at least 32"):
        create_app(Settings(models_dir=models_dir, jwt_secret="short"))


def test_dev_mode_without_auth(models_dir: Path) -> None:
    client = TestClient(create_app(Settings(models_dir=models_dir, auth_disabled=True)))
    assert client.get("/v1/models").status_code == 200


def test_audit_log_has_no_neural_data(
    models_dir: Path, synthetic, caplog: pytest.LogCaptureFixture
) -> None:  # type: ignore[no-untyped-def]
    import logging

    client = make_client(models_dir)
    target = synthetic.for_subject("synthetic_mi", "sub-000")
    session = new_session(client, list(target.ch_names))
    audit_logger = logging.getLogger("neurolayer_api.audit")
    audit_logger.addHandler(caplog.handler)
    try:
        payload = {"trials": trials_payload(target, [0], False)}
        client.post(f"/v1/sessions/{session}/decode", json=payload, headers=headers())
    finally:
        audit_logger.removeHandler(caplog.handler)
    lines = [r.getMessage() for r in caplog.records if r.name == "neurolayer_api.audit"]
    assert any('"status": 200' in line and '"tenant": "acme"' in line for line in lines)
    sample = f"{target.X[0][0][0] * 1e6:.6f}"[:8]
    assert not any(sample in line for line in lines)
