# ADR-0002: Monorepo layout and tooling

- **Status:** Accepted
- **Date:** 2026-09-26

## Context
The team is small, and three codebases (ML core, API, web) share contracts. Everything must run on a Windows PC with an NVIDIA GPU (via WSL2), on a Mac, and in CI.

## Decision
- **One repository.**
  - `src/neurolayer` holds the proprietary ML core.
  - `src/neurolayer_api` holds the FastAPI service. It shares the Python project for now, and can be split out when its deployment diverges.
  - `apps/web` holds the Next.js + TypeScript dashboard.
- **Python 3.12** managed by **uv** with a committed `uv.lock`. Heavy dependencies are optional extras:
  - `neuro`: MNE, MOABB, pyRiemann, mne-bids
  - `dl`: PyTorch, Braindecode, safetensors
  - `tracking`: MLflow
  - `api`: FastAPI, uvicorn
  - `devices`: BrainFlow, pylsl
- **Quality gates** (run locally with `make check` and in CI):
  - ruff (lint + format)
  - mypy `--strict` on `src/`
  - pytest (+ hypothesis) with coverage
  - **import-linter** layer contracts
  - data-file guard
- **Node 22 + npm** for the web app; ESLint (next config), `tsc --noEmit`, `next build`.
- `Makefile` as the single entry point for common tasks (works in WSL2, macOS and CI).
- Package code name `neurolayer`. Rename before any public release, after a trademark search.

## Consequences
One place for agents to learn the conventions. CI stays fast because heavy extras are not installed for core checks. Layering is enforced by tooling rather than by review alone.

## Alternatives considered
- Poetry or pip-tools: uv is faster and has first-class lockfiles and workspaces.
- Separate repositories: contracts would drift across repos; overkill at this stage.
- Hydra for configs: powerful, but adds magic. We use typed pydantic + YAML configs instead. Revisit if sweeps become complex.
