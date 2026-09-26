"""``neurolayer consent|pilot|bridge`` end to end (simulated headset, real HTTP server)."""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from neurolayer.cli import main
from neurolayer.data.storage import verify_checksums
from neurolayer.data.synthetic import SyntheticMIConfig, generate_synthetic_mi

FAST = ["--trials-per-class", "1", "--rest-s", "0.2", "--imagery-s", "0.5"]


def _grant(data_root: Path, form: Path, purposes: str) -> int:
    return main(
        [
            "--data-root",
            str(data_root),
            "consent",
            "grant",
            "P001",
            "--purposes",
            purposes,
            "--document",
            str(form),
            "--document-version",
            "v0.1",
            "--recorded-by",
            "tester",
        ]
    )


def test_consent_and_pilot_record(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("mne_bids")
    form = tmp_path / "form.md"
    form.write_text("consent form v0.1", encoding="utf-8")
    base = ["--data-root", str(tmp_path)]
    record = [*base, "pilot", "record", "P001", "--board", "simulated", *FAST]

    assert main(record) == 2
    assert "has not consented" in capsys.readouterr().err
    assert not (tmp_path / "pilot" / "bids").exists()

    assert _grant(tmp_path, form, "research,product") == 0
    assert main([*base, "consent", "check", "P001"]) == 0
    out = capsys.readouterr().out
    assert "research             granted" in out
    assert "commercial_training  no" in out

    assert main(record) == 0
    out = capsys.readouterr().out
    assert "LEFT hand" in out or "RIGHT hand" in out
    assert "written:" in out
    bids = tmp_path / "pilot" / "bids"
    assert list(bids.rglob("*.edf"))
    assert verify_checksums(bids) == []

    assert (
        main([*base, "consent", "revoke", "P001", "--purposes", "research", "--recorded-by", "t"])
        == 0
    )
    assert "delete this participant's recordings" in capsys.readouterr().out
    assert main(record) == 2


def test_consent_rejects_names(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    form = tmp_path / "form.md"
    form.write_text("x", encoding="utf-8")
    args = ["--data-root", str(tmp_path), "consent", "grant", "Alice", "--purposes", "research"]
    extra = ["--document", str(form), "--document-version", "1", "--recorded-by", "t"]
    assert main([*args, *extra]) == 2
    assert "pseudonym" in capsys.readouterr().err


# ----------------------------------------------------------------------------- bridge
def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture(scope="module")
def server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    pytest.importorskip("torch")
    uvicorn = pytest.importorskip("uvicorn")
    from neurolayer.models.bundle import export_bundle
    from neurolayer.models.proprietary import SpatialFieldDecoder
    from neurolayer_api.app import create_app
    from neurolayer_api.settings import Settings

    epochs = generate_synthetic_mi(
        SyntheticMIConfig(n_subjects=3, n_trials_per_class=20, efficiencies=(0.9,) * 3, seed=2)
    ).epochs
    decoder = SpatialFieldDecoder(epochs=3, finetune_epochs=3, device="cpu")
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
    port = _free_port()
    app = create_app(Settings(models_dir=root, auth_disabled=True))
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    instance = uvicorn.Server(config)
    thread = threading.Thread(target=instance.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not instance.started and time.monotonic() < deadline:
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    instance.should_exit = True
    thread.join(timeout=5)


def test_http_client_against_a_live_server(server: str) -> None:
    from neurolayer.devices.client import ApiError, NeurolayerClient

    client = NeurolayerClient.connect(server)
    models = client.models()
    assert models["demo@0.1.0"].trial_seconds == 2.0
    with pytest.raises(ApiError, match="404"):
        client.create_session("nope@1", ["C3", "C4"], 128.0)
    with pytest.raises(ValueError, match="http"):
        NeurolayerClient.connect("file:///etc/passwd")


def test_bridge_command(server: str, capsys: pytest.CaptureFixture[str]) -> None:
    args = ["bridge", "--api", server, "--model", "demo@0.1.0", "--board", "simulated"]
    timing = ["--trials-per-class", "1", "--rest-s", "0.1", "--imagery-s", "2.0"]
    assert main([*args, *timing, "--duration", "3", "--threshold", "0.0"]) == 0
    out = capsys.readouterr().out
    assert "calibration:" in out
    assert "done: 2 calibration trials" in out
    assert "0 decodes" not in out
    assert main([*args[:4], "missing@1", *args[5:], *timing]) == 2
    assert "missing@1" in capsys.readouterr().err
