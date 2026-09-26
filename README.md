# neurolayer

**A new way for humans to interact with computers.** neurolayer learns patterns in a person's neural signals and turns them into useful interactions for games, computers, AR/VR, accessibility, robotics and everyday AI. The person does not need to know any neuroscience.

```
brain activity → neural sensors → neurolayer (signal processing → neural representation → intent decoding) → API → action
```

> `neurolayer` is a working code name. Rename it before any public release, after a trademark search.

## The technical bet

Neural signals differ between people and between devices. Today every non-invasive brain-computer interface needs per-user calibration, which is the main reason nobody uses one casually.

Our research ([docs/research](docs/research/README.md)) found that one slice of this problem is **unowned by companies and unbenchmarked in academia**:

> **Calibration-efficient intent decoding on low-channel consumer EEG, across people and devices, trained only on commercially clean data.**

Our first measurable capability is **[CAP-1](docs/product/cap-1-calibration-efficient-intent.md)**:

- **Task:** a new person, on an electrode layout we never trained on, gets left/right motor-intent control.
- **Measure:** the *calibration-efficiency curve*, which is accuracy after 0, 5, 10, 20 and 40 calibration trials.
- **Rigor:**
  - held-out subjects and held-out datasets
  - leakage controls
  - a reproduced baseline suite

## What is in the repo today (Stage 0: foundations)

| Area | Status |
|------|--------|
| Research brief: companies, models, datasets, patents, regulation, gap analysis | ✅ [docs/research](docs/research/README.md) |
| Capability spec (CAP-1) and roadmap | ✅ [docs/product](docs/product/) |
| Architecture, data model, evaluation protocol, security, reproducibility, 8 ADRs | ✅ [docs/architecture](docs/architecture/overview.md), [docs/adr](docs/adr/README.md) |
| Canonical data types, channel naming, consumer-device montages | ✅ `src/neurolayer/core` |
| Dataset catalog + **license gate** (exploration / benchmark / training) | ✅ `catalog/datasets`, `src/neurolayer/data/catalog.py` |
| **CAP-1 evaluation harness**: disjoint folds, chronological calibration, metrics | ✅ `src/neurolayer/evaluation` |
| Synthetic motor-imagery generator (tests and CI only) | ✅ `src/neurolayer/data/synthetic.py` |
| Run manifests (git commit, config hash, dataset licenses, seeds) | ✅ `src/neurolayer/tracking` |
| API skeleton (FastAPI) and dashboard skeleton (Next.js) | ✅ `src/neurolayer_api`, `apps/web` |
| CI, pre-commit, data-file guard, secret scanning, Dependabot | ✅ `.github`, `.pre-commit-config.yaml` |
| Real-dataset ingestion → baselines → proprietary model → API → product → real-world | ⏭ [docs/plan/work-packages.md](docs/plan/work-packages.md) |

## Quickstart

Prerequisites:

- [uv](https://docs.astral.sh/uv/)
- Node 22 (for the web app only)
- On the Windows GPU PC, use WSL2 (see [dev setup](docs/guides/dev-setup.md))

```bash
make setup            # uv sync + pre-commit hooks
make check            # lint, format, types, layering, tests, data guard
make smoke            # end-to-end CAP-1 harness run on synthetic data
uv run neurolayer catalog list                 # datasets and their license status
uv run neurolayer catalog check --purpose training
make api              # http://localhost:8000/healthz
make web              # http://localhost:3000
```

Example `make smoke` output (synthetic data and a reference decoder, so this is a pipeline check and **not** a result):

```
 k/class  mean BA            95% CI  median  UUR@70%
       0    0.710    [0.625, 0.779]   0.735     0.67
       5    0.718    [0.611, 0.800]   0.727     0.67
      10    0.765    [0.660, 0.834]   0.790     0.83
      20    0.775    [0.678, 0.850]   0.773     0.83
AUCEC:      0.731
```

## Repository map

```
AGENTS.md            rules for AI coding agents (Codex, Claude Code). Read first
catalog/datasets/    dataset cards (provenance + license) that drive the license gate
configs/experiments/ experiment configs. Every run starts from one
src/neurolayer/      proprietary core: core · data · signal · representation · models · evaluation · tracking · experiments · cli
src/neurolayer_api/  FastAPI service
apps/web/            Next.js + TypeScript dashboard
tests/               unit · integration · api (synthetic data only)
docs/                research · product · architecture · adr · plan · guides · experiments · results
data/, artifacts/    git-ignored: datasets (DVC) and run outputs
```

## How we work

- **Stages with gates:**
  research → architecture → dataset → baseline → proprietary model → API → product → real-world testing ([roadmap](docs/product/roadmap.md)).
- **Work packages:** every change implements one WP from [docs/plan](docs/plan/work-packages.md), with its acceptance criteria and tests. This is how AI agents get well-scoped tasks instead of "build me a brain AI".
- **Experiments are cards + configs + manifests.** Negative results are recorded too ([docs/experiments](docs/experiments/TEMPLATE.md), [results ledger](docs/results/README.md)).
- **Decisions are ADRs.** Agents propose ADRs; they do not silently change decisions.

## Security and IP

- Neural data from people is sensitive. It never goes into git, and never to third-party LLM APIs ([security and privacy](docs/architecture/security-and-privacy.md)).
- **This repository is currently public.** Make it private before proprietary model work begins ([why](docs/research/04-patents-and-ip.md#43-urgent-this-repository-is-public)).

## License

Proprietary. All rights reserved. See [LICENSE](LICENSE).
