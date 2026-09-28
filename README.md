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

## What is in the repo today

Everything below runs end to end on **synthetic data** and is covered by tests. **No capability claim yet.** Every exit gate from Gate 0 on needs real data (waiting on license verification, WP-1.1) or hardware and participants (Gate 3). See the [human and hardware checklist](#what-needs-a-human-or-hardware).

| Stage | What exists | Where |
|-------|-------------|-------|
| 0 Research and foundations | Research brief, CAP-1 spec, architecture, 12 ADRs, CAP-1 harness, run manifests, MLflow mirror, license report | [docs/research](docs/research/README.md), [docs/product](docs/product/), [docs/adr](docs/adr/README.md), `src/neurolayer/{core,evaluation,tracking}` |
| 1 Data ingestion | License-gated MOABB adapters → canonical recordings; BIDS + checksums; dataset audit and QA | `src/neurolayer/data`, `neurolayer data fetch/verify/audit/qa` |
| 2 Signal processing | Versioned transforms, epoching, content-hashed pipelines and cache | `src/neurolayer/signal` |
| 3 Representation | Tangent space, CSP, alignment, PyTorch scaffold, governed pretrained-model loading | `src/neurolayer/representation` |
| 4 Baselines | B0–B4 on regimes R1–R3, paired comparisons, identity probes, multi-seed shuffle control, Gate 0 tooling | `src/neurolayer/models`, `neurolayer report/probe/control/gate0` |
| 5 Proprietary model v0 | Montage-agnostic spatial-field decoder with calibration-time adaptation; experiments H1–H4 (synthetic); safetensors bundles | `src/neurolayer/models/proprietary.py`, [docs/experiments](docs/experiments/), [results](docs/results/README.md) |
| 6 API | Sessions, calibration, decoding, WebSocket streaming, JWT auth, tenancy, limits, audit log | `src/neurolayer_api` |
| 7 Product | Dashboard (calibration game, EDF/BDF inspector), device bridge (BrainFlow/LSL), OS-input sinks, LLM explainer (aggregates only) | `apps/web`, `src/neurolayer/devices`, `neurolayer bridge`, `neurolayer report explain` |
| 8 Real-world tooling | Consent ledger, pilot recorder, pilot protocol, consent-form and device-memo templates | `neurolayer consent/pilot`, [pilot protocol](docs/guides/pilot-protocol.md), [templates](docs/templates/) |

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
make api              # http://localhost:8000/healthz (auth off, demo data on: development only)
make web              # http://localhost:3000
```

Try the product end to end without any hardware:

```bash
uv sync --extra api --extra dl --extra devices   # PyTorch + BrainFlow/LSL
make demo-model                                   # synthetic demo bundle (never shipped)
make api & make web                               # then open http://localhost:3000/calibrate
uv run neurolayer bridge --model nl-synthetic-demo@0.1.0 --board simulated --trials-per-class 5
uv run neurolayer report explain artifacts/runs/<run-id>      # plain-language summary
make e2e                                          # Playwright: browser → Next.js → API → model
```

With a real device, replace `--board simulated` with a BrainFlow board (`--board CYTON_BOARD --serial-port COM3 --channels C3,Cz,C4,...`) or an LSL stream (`--lsl EEG`, e.g. the Neurosity Crown).

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
catalog/models/      third-party model cards (pinned hashes, license gate)
configs/experiments/ experiment configs. Every run starts from one
src/neurolayer/      proprietary core: core · data · signal · representation · models · evaluation · tracking · experiments · devices · cli
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

## What needs a human or hardware

| Item | Why | Where |
|------|-----|-------|
| Make the repo private; branch protection; 2FA (WP-0.8) | Proprietary IP is in a public repo | [security §4](docs/architecture/security-and-privacy.md) |
| Verify dataset licenses (WP-1.1) | Blocks benchmark/training use of every public dataset, and therefore Gates 0–2 | `catalog/datasets/*.yaml` (`license` blocks; humans only) |
| Allow network access to data hosts, then `neurolayer data fetch` | Cloud sessions here cannot reach them. PhysioNet MI already works through its official AWS mirror (`scripts/fetch_physionet_mirror.py`) | openneuro.org, zenodo.org, figshare.com, gigadb.org (ftp.cngb.org), bnci-horizon-2020.eu, bbci.de, huggingface.co |
| Choose the locked holdout (ADR-0009) and fix Gate 2 margins (ADR-0011) | Must be decided on real audits and baselines, before any proprietary run on the holdout | [ADR-0009](docs/adr/0009-locked-holdout-selection.md), [ADR-0011](docs/adr/0011-cap1-gate1-thresholds.md) |
| Pin pretrained weights (B5/B6) | Hashes must come from a reviewed download | `catalog/models/*.yaml` |
| Counsel review of the consent form; accept ADR-0012 | Required before any participant and before training on our own data | [consent form](docs/templates/consent-form.md), [ADR-0012](docs/adr/0012-consent-aware-gate-for-own-data.md) |
| Buy devices; timing test; pilot with ≥ 20 people (Gate 3) | Hardware and people | [pilot protocol](docs/guides/pilot-protocol.md), [device memo](docs/templates/device-selection-memo.md) |
| Verify Apple BCI HID access | Partner/spec question | [research note 07](docs/research/07-os-input-bci-hid.md) |

## License

Proprietary. All rights reserved. See [LICENSE](LICENSE).
