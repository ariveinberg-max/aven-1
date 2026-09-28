from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from neurolayer import __version__
from neurolayer_api.app import create_app


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(enable_docs=False))


def test_healthz(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_info_lists_capabilities(client: TestClient) -> None:
    body = client.get("/v1/info").json()
    assert body["version"] == __version__
    assert body["api_version"] == "v1"
    assert [c["id"] for c in body["capabilities"]] == ["CAP-1", "CAP-2", "CAP-3"]


def test_docs_disabled_by_default(client: TestClient) -> None:
    assert client.get("/docs").status_code == 404
    assert client.get("/openapi.json").status_code == 404


def test_docs_can_be_enabled() -> None:
    assert TestClient(create_app(enable_docs=True)).get("/openapi.json").status_code == 200
