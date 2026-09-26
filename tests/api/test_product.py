"""Stage 7 endpoints (file inspection, demo trials) and the device bridge end to end."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from neurolayer.data.synthetic import SyntheticMIConfig, generate_synthetic_mi
from neurolayer.devices.bridge import run_bridge
from neurolayer.devices.client import ApiError, NeurolayerClient, Transport
from neurolayer.devices.output import Intent
from neurolayer.devices.protocol import CueSchedule
from neurolayer.devices.sources import simulated_source
from neurolayer_api.app import create_app
from neurolayer_api.settings import Settings

pytest.importorskip("torch")

from neurolayer.models.bundle import export_bundle
from neurolayer.models.proprietary import SpatialFieldDecoder

MODEL = "demo@0.1.0"


@pytest.fixture(scope="module")
def models_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    epochs = generate_synthetic_mi(
        SyntheticMIConfig(n_subjects=4, n_trials_per_class=30, efficiencies=(0.9,) * 4, seed=8)
    ).epochs
    decoder = SpatialFieldDecoder(epochs=5, finetune_epochs=5, device="cpu")
    decoder.fit(epochs)
    root = tmp_path_factory.mktemp("models")
    export_bundle(
        decoder,
        root / "demo" / "0.1.0",
        name="demo",
        version="0.1.0",
        label_names=epochs.label_names,
        sfreq=epochs.sfreq,
        n_times=epochs.n_times,
        trained_channels=epochs.ch_names,
        preprocessing={"synthetic_only": True},
    )
    return root


def app_client(models_dir: Path, **overrides: Any) -> TestClient:
    return TestClient(create_app(Settings(models_dir=models_dir, auth_disabled=True, **overrides)))


def via(client: TestClient) -> Transport:
    """Route the device client through the in-process app."""

    def send(method: str, path: str, body: dict[str, Any] | None) -> Any:
        response = client.request(method, path, json=body)
        if response.status_code >= 400:
            raise ApiError(response.status_code, str(response.json().get("detail")))
        return response.json() if response.content else None

    return send


# ------------------------------------------------------------------------ model metadata
def test_models_report_window_offset(models_dir: Path) -> None:
    listing = app_client(models_dir).get("/v1/models").json()
    assert listing[0]["window_offset_s"] == 0.0  # synthetic epochs start at the cue


# ------------------------------------------------------------------------------- demo
def test_demo_trials_are_disabled_by_default(models_dir: Path) -> None:
    response = app_client(models_dir).get("/v1/demo/trials", params={"model": MODEL})
    assert response.status_code == 404


def test_demo_trials_calibrate_and_decode(models_dir: Path) -> None:
    client = app_client(models_dir, demo_enabled=True)
    demo = client.get("/v1/demo/trials", params={"model": MODEL, "n_per_class": 20}).json()
    assert demo["synthetic"] is True
    assert len(demo["trials"]) == 40
    assert {t["label"] for t in demo["trials"]} == {"left_hand", "right_hand"}
    assert np.asarray(demo["trials"][0]["data"]).shape == (len(demo["ch_names"]), 256)
    again = client.get("/v1/demo/trials", params={"model": MODEL, "n_per_class": 20}).json()
    assert again["trials"][0]["data"] == demo["trials"][0]["data"]  # seeded

    api = NeurolayerClient(via(client))
    session = api.create_session(MODEL, demo["ch_names"], demo["sfreq"])
    trials = [np.asarray(t["data"]) for t in demo["trials"]]
    labels = [t["label"] for t in demo["trials"]]
    info = api.calibrate(session, trials[:20], labels[:20])
    assert sum(info["calibration_counts"].values()) == 20
    intents = api.decode(session, trials[20:])
    assert len(intents) == 20
    assert all(0.5 <= i.confidence <= 1.0 for i in intents)
    api.delete_session(session)
    with pytest.raises(ApiError, match="404"):
        api.decode(session, trials[:1])


def test_demo_rejects_unknown_model_and_bad_sizes(models_dir: Path) -> None:
    client = app_client(models_dir, demo_enabled=True)
    assert client.get("/v1/demo/trials", params={"model": "nope@1"}).status_code == 404
    bad = client.get("/v1/demo/trials", params={"model": MODEL, "n_per_class": 1000})
    assert bad.status_code == 422


# ---------------------------------------------------------------------------- inspect
@pytest.fixture(scope="module")
def edf_bytes(tmp_path_factory: pytest.TempPathFactory) -> bytes:
    mne = pytest.importorskip("mne")
    pytest.importorskip("edfio")
    rng = np.random.default_rng(0)
    sfreq, seconds = 128.0, 20
    names = ["C3", "Cz", "C4", "Pz"]
    data = rng.standard_normal((len(names), int(sfreq * seconds))) * 10e-6
    data[3] = 0.0  # flat channel
    info = mne.create_info(names, sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose="error")
    raw.set_annotations(mne.Annotations([1.0, 5.0, 9.0], [0.0] * 3, ["T1", "T2", "T1"]))
    path = tmp_path_factory.mktemp("edf") / "upload.edf"
    # MNE exports EEG in microvolts, so the physical range is in µV too.
    mne.export.export_raw(path, raw, fmt="edf", physical_range=(-200, 200), verbose="error")
    return path.read_bytes()


def test_inspect_summarizes_an_edf(models_dir: Path, edf_bytes: bytes) -> None:
    response = app_client(models_dir).post(
        "/v1/recordings/inspect",
        content=edf_bytes,
        headers={"Content-Type": "application/octet-stream"},
    )
    assert response.status_code == 200, response.text
    report = response.json()
    assert report["format"] == "edf"
    assert report["sfreq"] == 128.0
    assert report["n_channels"] == 4
    assert [c["name"] for c in report["channels"]] == ["C3", "Cz", "C4", "Pz"]
    assert report["channels"][3]["flat"] is True
    # White noise of 10 µV at 128 Hz: 100 µV² / 64 Hz ≈ 1.6 µV²/Hz ≈ 1.9 dB.
    assert 0.5 < report["channels"][0]["mu_db"] < 3.5
    assert report["channels"][3]["mu_db"] is None  # flat
    assert report["annotations"] == {"T1": 2, "T2": 1}
    assert any("flat" in issue for issue in report["issues"])
    preview = report["preview"]
    assert len(preview["data_uv"]) == 4
    assert len(preview["data_uv"][0]) <= 500


def test_inspect_rejects_other_files(models_dir: Path, edf_bytes: bytes) -> None:
    client = app_client(models_dir)
    assert client.post("/v1/recordings/inspect", content=b"PK\x03\x04zip").status_code == 415
    corrupt = edf_bytes[:8] + b"\x00" * 300
    response = client.post("/v1/recordings/inspect", content=corrupt)
    assert response.status_code == 422
    assert "could not read" in response.json()["detail"]


def test_inspect_enforces_limits(models_dir: Path, edf_bytes: bytes) -> None:
    slow = app_client(models_dir, inspect_timeout_s=0.001)
    response = slow.post("/v1/recordings/inspect", content=edf_bytes)
    assert response.status_code == 422
    assert "too long" in response.json()["detail"]
    small = app_client(models_dir, max_body_bytes=1000)
    assert small.post("/v1/recordings/inspect", content=edf_bytes).status_code == 413

    def chunked():  # type: ignore[no-untyped-def]  # no Content-Length header
        for start in range(0, len(edf_bytes), 512):
            yield edf_bytes[start : start + 512]

    assert small.post("/v1/recordings/inspect", content=chunked()).status_code == 413


def test_inspect_requires_auth(models_dir: Path, edf_bytes: bytes) -> None:
    client = TestClient(create_app(Settings(models_dir=models_dir, jwt_secret="0" * 40)))
    assert client.post("/v1/recordings/inspect", content=edf_bytes).status_code == 401


# ----------------------------------------------------------------------------- bridge
def test_bridge_calibrates_decodes_and_cleans_up(models_dir: Path) -> None:
    client = app_client(models_dir)
    api = NeurolayerClient(via(client))
    schedule = CueSchedule(n_trials_per_class=10, rest_s=1.0, imagery_s=2.0, seed=3)
    source = simulated_source(schedule, live_trials=6, realtime=False)
    received: list[Intent] = []
    result = run_bridge(
        source, api, MODEL, schedule, received.append, realtime=False, max_decodes=12
    )
    assert result.calibration_trials == 20
    assert result.decoded == len(received) == 12
    assert {i.label for i in received} <= {"left_hand", "right_hand"}
    assert client.get(f"/v1/sessions/{result.session_id}").status_code == 404  # data erased


def test_bridge_unknown_model_opens_no_session(models_dir: Path) -> None:
    api = NeurolayerClient(via(app_client(models_dir)))
    schedule = CueSchedule(n_trials_per_class=1)
    with pytest.raises(KeyError):
        run_bridge(
            simulated_source(schedule, realtime=False),
            api,
            "nope@1",
            schedule,
            print,
            realtime=False,
        )
