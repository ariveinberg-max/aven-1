from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from neurolayer.cli import main
from neurolayer.experiments.explain import (
    ClaudeExplainer,
    ExplainerError,
    TemplateExplainer,
    assert_aggregate_only,
    digest_run,
    get_explainer,
)

SUMMARY = {
    "regime": "leave_dataset_out",
    "ks": [0, 5, 20],
    "n_unlabeled": 10,
    "threshold": 0.7,
    "n_subjects_scored": 8,
    "n_subjects_skipped": 1,
    "aucec": 0.71,
    "median_ttc": 5.0,
    "per_budget": [
        {
            "k": k,
            "n_subjects": 8,
            "mean_ba": ba,
            "median_ba": ba,
            "ci_low": ba - 0.05,
            "ci_high": ba + 0.05,
            "usable_user_rate": uur,
        }
        for k, ba, uur in ((0, 0.62, 0.25), (5, 0.70, 0.5), (20, 0.78, 0.75))
    ],
    "skipped": [],
}
MANIFEST = {
    "run_id": "20260926T000000Z-demo-abc123",
    "name": "demo",
    "official": False,
    "git_dirty": True,
    "datasets": [{"id": "synthetic_mi"}, {"id": "synthetic_mi_shifted"}],
    "config": {"decoder": {"name": "nl_spatial_field"}},
}


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    (tmp_path / "summary.json").write_text(json.dumps(SUMMARY), encoding="utf-8")
    (tmp_path / "manifest.json").write_text(json.dumps(MANIFEST), encoding="utf-8")
    # Per-subject results exist in a run directory but must never be read.
    (tmp_path / "results.csv").write_text("SHOULD NOT BE READ", encoding="utf-8")
    return tmp_path


def test_digest_contains_aggregates_only(run_dir: Path) -> None:
    digest = digest_run(run_dir)
    assert digest.synthetic_only
    assert [b.k for b in digest.budgets] == [0, 5, 20]
    payload = digest.payload()
    assert "results" not in json.dumps(payload)
    assert len(json.dumps(payload)) < 2000


@pytest.mark.parametrize(
    "payload",
    [
        {"data": [[1.0, 2.0]]},
        {"meta": {"epochs": 3}},
        {"per_subject": list(range(40))},
        {"blob": "x" * 9000},
    ],
)
def test_guard_rejects_signal_like_payloads(payload: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        assert_aggregate_only(payload)


def test_template_explainer_states_numbers_and_caveats(run_dir: Path) -> None:
    text = TemplateExplainer().explain(digest_run(run_dir)).text
    assert "62.0%" in text
    assert "78.0%" in text
    assert "lab and amplifier never seen" in text
    assert "not a capability claim" in text
    assert "not an official run" in text
    assert "1 people were skipped" in text


class FakeMessages:
    def __init__(self, response: Any) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.response


def fake_client(response: Any) -> tuple[Any, FakeMessages]:
    messages = FakeMessages(response)
    return SimpleNamespace(beta=SimpleNamespace(messages=messages)), messages


def test_claude_explainer_sends_only_the_digest(run_dir: Path) -> None:
    pytest.importorskip("anthropic")
    response = SimpleNamespace(
        stop_reason="end_turn",
        stop_details=None,
        model="claude-opus-5",
        content=[
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text="Accuracy improves with calibration."),
        ],
    )
    client, messages = fake_client(response)
    explanation = ClaudeExplainer(client=client).explain(digest_run(run_dir))
    assert explanation.text == "Accuracy improves with calibration."
    (call,) = messages.calls
    assert call["model"] == "claude-opus-5"
    assert call["thinking"] == {"type": "adaptive"}
    assert call["fallbacks"] == "default"
    assert call["betas"] == ["server-side-fallback-2026-07-01"]
    content = call["messages"][0]["content"]
    sent = json.loads(content.split("<run_digest>")[1].split("</run_digest>")[0])
    assert sent == digest_run(run_dir).payload()


def test_claude_explainer_surfaces_refusals(run_dir: Path) -> None:
    pytest.importorskip("anthropic")
    response = SimpleNamespace(
        stop_reason="refusal",
        stop_details=SimpleNamespace(category="other"),
        model="claude-opus-5",
        content=[],
    )
    client, _ = fake_client(response)
    with pytest.raises(ExplainerError, match="declined"):
        ClaudeExplainer(client=client).explain(digest_run(run_dir))


def test_get_explainer() -> None:
    assert get_explainer("template").name == "template"
    assert get_explainer("claude", "claude-opus-5").name == "claude"
    with pytest.raises(ValueError, match="unknown"):
        get_explainer("gpt")


def test_report_explain_cli(run_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["report", "explain", str(run_dir)]) == 0
    out = capsys.readouterr().out
    assert "Balanced accuracy goes from 62.0%" in out
    assert "(explained by template)" in out
