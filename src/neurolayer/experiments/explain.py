"""Plain-language explanations of CAP-1 runs (WP-7.5).

Provider-agnostic: an :class:`Explainer` turns a :class:`RunDigest` into text. The
default :class:`TemplateExplainer` is deterministic and offline. :class:`ClaudeExplainer`
(``llm`` extra) asks Claude through the Anthropic SDK and is opt-in.

**No neural data ever reaches an LLM** (AGENTS.md rule 9, PRIV-9). A digest is built from
``summary.json`` and ``manifest.json`` only: metrics aggregated over subjects, dataset
ids and run metadata. It never reads per-subject scores, epochs or signals, and
:func:`assert_aggregate_only` re-checks the payload before any provider sees it.
Aggregate results are still confidential (class C2), so calling an external provider
requires an explicit ``--provider claude``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

MAX_PAYLOAD_BYTES = 8_000
MAX_LIST_LENGTH = 16
_FORBIDDEN_KEYS = {"x", "data", "signal", "signals", "epochs", "trials", "samples", "raw"}


@dataclass(frozen=True, slots=True)
class BudgetRow:
    """Aggregate result at one calibration budget (k labeled trials per class)."""

    k: int
    n_subjects: int
    mean_ba: float
    ci_low: float
    ci_high: float
    usable_user_rate: float


@dataclass(frozen=True, slots=True)
class RunDigest:
    """Everything an explainer may see about a run: aggregates and metadata only."""

    run_id: str
    name: str
    decoder: str
    regime: str
    datasets: tuple[str, ...]
    synthetic_only: bool
    official: bool
    git_dirty: bool
    threshold: float
    n_subjects_scored: int
    n_subjects_skipped: int
    aucec: float | None
    median_ttc: float | None
    budgets: tuple[BudgetRow, ...]

    def payload(self) -> dict[str, Any]:
        """JSON-serializable form, checked with :func:`assert_aggregate_only`."""
        out = asdict(self)
        out["datasets"] = list(self.datasets)
        out["budgets"] = [asdict(b) for b in self.budgets]
        assert_aggregate_only(out)
        return out


def assert_aggregate_only(payload: Mapping[str, Any]) -> None:
    """Raise ``ValueError`` if ``payload`` could carry signal-level data.

    Rejects signal-like keys, long numeric lists and oversized payloads. This is a
    guard against future edits to :class:`RunDigest`, not a substitute for building
    digests from aggregates only.
    """
    text = json.dumps(payload)
    if len(text.encode()) > MAX_PAYLOAD_BYTES:
        raise ValueError(f"payload is {len(text)} bytes; explainer payloads are aggregates only")

    def walk(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if str(key).lower() in _FORBIDDEN_KEYS:
                    raise ValueError(f"field {path}.{key} looks like signal data")
                walk(item, f"{path}.{key}")
        elif isinstance(value, list | tuple):
            if len(value) > MAX_LIST_LENGTH:
                raise ValueError(f"{path} has {len(value)} items; only aggregates are allowed")
            for i, item in enumerate(value):
                walk(item, f"{path}[{i}]")

    walk(payload, "$")


def digest_run(run_dir: Path) -> RunDigest:
    """Build a digest from a run's ``summary.json`` and ``manifest.json``."""
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    datasets = tuple(str(d["id"]) for d in manifest.get("datasets", []))
    return RunDigest(
        run_id=str(manifest["run_id"]),
        name=str(manifest.get("name", "")),
        decoder=str(manifest.get("config", {}).get("decoder", {}).get("name", "?")),
        regime=str(summary["regime"]),
        datasets=datasets,
        synthetic_only=bool(datasets) and all(d.startswith("synthetic") for d in datasets),
        official=bool(manifest.get("official", False)),
        git_dirty=bool(manifest.get("git_dirty", True)),
        threshold=float(summary["threshold"]),
        n_subjects_scored=int(summary["n_subjects_scored"]),
        n_subjects_skipped=int(summary["n_subjects_skipped"]),
        aucec=summary.get("aucec"),
        median_ttc=summary.get("median_ttc"),
        budgets=tuple(
            BudgetRow(
                k=int(row["k"]),
                n_subjects=int(row["n_subjects"]),
                mean_ba=float(row["mean_ba"]),
                ci_low=float(row["ci_low"]),
                ci_high=float(row["ci_high"]),
                usable_user_rate=float(row["usable_user_rate"]),
            )
            for row in summary["per_budget"]
        ),
    )


@dataclass(frozen=True, slots=True)
class Explanation:
    """An explanation and who wrote it."""

    text: str
    provider: str


class Explainer(Protocol):
    """Turns a run digest into a plain-language explanation."""

    name: str

    def explain(self, digest: RunDigest) -> Explanation:
        """Explain one run."""
        ...


REGIMES = {
    "within_dataset": "new people from datasets seen in training (R1)",
    "leave_dataset_out": "new people from a lab and amplifier never seen in training (R2/R3)",
}


class TemplateExplainer:
    """Deterministic, offline explanation (the default)."""

    name = "template"

    def explain(self, digest: RunDigest) -> Explanation:
        """Summarize the calibration curve and list the caveats that apply."""
        rows = sorted(digest.budgets, key=lambda b: b.k)
        lines = [
            f"Run {digest.run_id} ({digest.decoder}) was evaluated on "
            f"{REGIMES.get(digest.regime, digest.regime)}: {digest.n_subjects_scored} people "
            f"from {', '.join(digest.datasets)}."
        ]
        if rows:
            first, last = rows[0], rows[-1]
            lines.append(
                f"Balanced accuracy goes from {first.mean_ba:.1%} with {first.k} calibration "
                f"trials per class to {last.mean_ba:.1%} with {last.k} "
                f"(95% CI {last.ci_low:.1%}-{last.ci_high:.1%}); chance is 50%."
            )
            lines.append(
                f"{last.usable_user_rate:.0%} of people reach {digest.threshold:.0%} accuracy "
                f"at {last.k} trials per class"
                + (
                    f"; the median person needs {digest.median_ttc:g} per class."
                    if digest.median_ttc is not None
                    else "; the median person never reaches it within the tested budgets."
                )
            )
        if digest.aucec is not None:
            lines.append(
                f"Area under the calibration-efficiency curve (AUCEC): {digest.aucec:.3f}."
            )
        caveats = []
        if digest.synthetic_only:
            caveats.append(
                "synthetic data only, so this is a pipeline check, not a capability claim"
            )
        if not digest.official or digest.git_dirty:
            caveats.append("not an official run (uncommitted code or no --official flag)")
        if digest.n_subjects_scored < 10:
            caveats.append(f"only {digest.n_subjects_scored} people, so intervals are wide")
        if digest.n_subjects_skipped:
            caveats.append(f"{digest.n_subjects_skipped} people were skipped (too few trials)")
        if caveats:
            lines.append("Caveats: " + "; ".join(caveats) + ".")
        return Explanation(text="\n".join(lines), provider=self.name)


CLAUDE_SYSTEM = (
    "You explain results of a brain-computer-interface benchmark (CAP-1: how quickly a "
    "decoder of imagined left/right hand movement becomes usable for a new person as "
    "calibration trials are added) to a non-specialist founder. You receive one run's "
    "aggregate metrics as JSON. Explain in under 200 words what the numbers mean, how "
    "confident one can be, and what the caveats are. Balanced accuracy chance level is 50%. "
    "Never claim more than the data supports: synthetic data or unofficial runs support no "
    "capability claim. Do not invent numbers that are not in the JSON."
)


class ExplainerError(RuntimeError):
    """The provider could not produce an explanation."""


class ClaudeExplainer:
    """Explanation written by Claude (``llm`` extra; credentials from the environment).

    Sends only :meth:`RunDigest.payload`. Uses adaptive thinking and opts into
    server-side refusal fallbacks (``fallbacks="default"``) so a classifier decline is
    retried on Anthropic's recommended fallback model instead of failing.
    """

    name = "claude"

    def __init__(
        self,
        model: str = "claude-opus-5",
        client: Any | None = None,
        max_tokens: int = 16_000,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._client = client

    def _create(self, payload: str) -> Any:
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError("ClaudeExplainer needs the 'llm' extra (anthropic)") from exc
        client = self._client if self._client is not None else anthropic.Anthropic()
        try:
            return client.beta.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                thinking={"type": "adaptive"},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                system=CLAUDE_SYSTEM,
                messages=[
                    {
                        "role": "user",
                        "content": f"<run_digest>\n{payload}\n</run_digest>\n\nExplain this run.",
                    }
                ],
            )
        except anthropic.AuthenticationError as exc:
            raise ExplainerError(
                "no valid Anthropic credentials (set ANTHROPIC_API_KEY or run `ant auth login`)"
            ) from exc
        except anthropic.RateLimitError as exc:
            raise ExplainerError("rate limited by the Anthropic API; retry later") from exc
        except anthropic.APIStatusError as exc:
            raise ExplainerError(f"Anthropic API error {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise ExplainerError("could not reach the Anthropic API") from exc

    def explain(self, digest: RunDigest) -> Explanation:
        """Ask Claude to explain the digest."""
        response = self._create(json.dumps(digest.payload(), sort_keys=True))
        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            raise ExplainerError(f"the model declined to explain this run (category: {category})")
        text = "".join(block.text for block in response.content if block.type == "text").strip()
        if not text:
            raise ExplainerError(f"empty response (stop_reason: {response.stop_reason})")
        if response.stop_reason == "max_tokens":
            text += "\n[truncated]"
        return Explanation(text=text, provider=f"claude ({response.model})")


def get_explainer(provider: str, model: str | None = None) -> Explainer:
    """``template`` (offline, default) or ``claude``."""
    if provider == "template":
        return TemplateExplainer()
    if provider == "claude":
        return ClaudeExplainer(model=model) if model else ClaudeExplainer()
    raise ValueError(f"unknown explainer provider {provider!r}; use 'template' or 'claude'")
