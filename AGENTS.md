# AGENTS.md: rules for AI coding agents (Codex, Claude Code, etc.)

You are working on **neurolayer**: software that learns patterns in neural signals and turns them into computer interactions. The proprietary core is the neural representation plus calibration-time adaptation. Everything is measured by the **CAP-1 calibration-efficiency protocol** (`docs/product/cap-1-calibration-efficient-intent.md`).

## Before you write code

1. **Find your work package** in `docs/plan/work-packages.md`. Implement exactly that WP. Anything else you notice goes in your PR description under "Follow-ups". Do not implement it.
2. Read the contracts you touch:
   - `docs/architecture/overview.md` (layers)
   - `docs/architecture/data-model.md` (types)
   - `docs/architecture/evaluation-protocol.md` (harness)
   - the relevant ADRs in `docs/adr/`
3. If the WP conflicts with an accepted ADR, **stop and propose a new ADR** in the PR. Never silently change a decision.

## Commands

```bash
make setup     # uv sync (core + dev + api extras) and pre-commit install
make check     # ruff lint + format check, mypy --strict, import-linter, pytest, data guard
make test      # pytest only
make smoke     # end-to-end CAP-1 harness run on synthetic data
```

Python 3.12 with `uv`. Add dependencies with `uv add <pkg>`, or `uv add --optional <extra> <pkg>` for heavy ones. Commit `uv.lock`. The web app is in `apps/web` (`npm ci && npm run lint && npm run typecheck && npm run build`).

## Non-negotiable rules

1. **Never commit data, weights or secrets.**
   - Neural data (`.edf .bdf .gdf .fif .set .mat .xdf …`), arrays (`.npy .npz .h5`), weights (`.pt .pth .ckpt .safetensors .onnx .pkl`) and anything under `data/` or `artifacts/` are blocked by `scripts/check_no_data_files.py`.
   - Never bypass hooks (`--no-verify`).
2. **Respect the layers.**
   - The order is `core ← data ← signal ← representation ← {models | evaluation} ← tracking ← experiments ← cli`.
   - `evaluation` never imports `models`.
   - `import-linter` enforces this in CI.
3. **The evaluation harness is protected.**
   - Do not modify `src/neurolayer/evaluation/` (splits, metrics, protocol loop) without an ADR.
   - Never change how something is measured in order to improve a number.
4. **No leakage.** Decoders:
   - never see test labels
   - never see target subjects during `fit`
   - never compute statistics from the test window (unless it is a declared causal online adaptation)
   - tune hyperparameters on source subjects only
5. **License gate.**
   - Never edit a card's `license` block to make the gate pass. That is a human task (WP-1.1).
   - Never add datasets or pretrained weights without a catalog card (ADR-0005, ADR-0007).
6. **Reproducibility.**
   - Every experiment is a YAML config in `configs/experiments/` run via `neurolayer run`.
   - No hard-coded paths, no unseeded randomness, no results pasted without a run id.
7. **Tests with every change.** Use synthetic data (`neurolayer.data.synthetic`) or tiny in-test fixtures. Never download in default tests; mark network tests `@pytest.mark.network`.
8. **Types and docs.**
   - Full type hints; mypy `--strict` must pass.
   - NumPy-style docstrings on public functions.
   - Signals are float64 **volts**.
   - Channels use canonical 10-05 names.
9. **Security.**
   - Load third-party weights only with `safetensors` or `torch.load(..., weights_only=True)`.
   - No `pickle` for untrusted files.
   - No `shell=True`.
   - No neural data sent to external APIs (including LLM APIs).
10. **Honesty.**
    - Never invent results, citations, dataset facts or license terms.
    - If you cannot verify something, write `TODO(verify)` and say so in the PR.

## Style

- ruff (line length 100) and ruff-format.
- Small pure functions, frozen dataclasses or pydantic models for data, and explicit errors over silent fallbacks.
- Prefer wrapping MNE, MOABB, pyRiemann and Braindecode over re-implementing them.
- Keep heavy imports lazy (inside functions) in modules used by the core.

## Pull requests

Use `.github/pull_request_template.md`. Every PR states:

- the WP id
- what changed
- how it was tested
- any contract or ADR impact
- follow-ups
