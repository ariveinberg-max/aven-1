# Changelog

All notable changes are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

Stages 0.7–8: tooling for every stage, verified on synthetic data. No capability claim; gates from Gate 0 on need real data, hardware or human decisions.

### Added
- **Stage 0 remainder:** MLflow run mirroring (`neurolayer run --mlflow`); dependency license report in CI.
- **Stage 1:** license-gated MOABB adapters, MNE bridge, BIDS storage with checksums, dataset audit and QA (`neurolayer data fetch|verify|audit|qa`); ADR-0009 (locked-holdout rule).
- **Stage 2:** versioned transforms, epoching, content-hashed pipelines and cache; experiments run real datasets through this path.
- **Stage 3–4:** tangent-space/CSP/log-variance encoders, alignment methods, PyTorch scaffold, governed pretrained-model cards; baselines B0–B4; `neurolayer report compare|probe|control|gate0`; ADR-0010, ADR-0011.
- **Stage 5:** proprietary montage-agnostic spatial-field decoder v0 with calibration-time adaptation; experiments H1–H4 (synthetic); safetensors model bundles (`neurolayer train`).
- **Stage 6:** API with sessions, calibration, decoding, WebSocket streaming, JWT auth, tenancy, rate and size limits, audit log.
- **Stage 7:** dashboard calibration game and EDF/BDF inspector (sandboxed parsing worker, CSP); `window_offset_s` in `/v1/models`; env-gated `/v1/demo/trials`; `neurolayer.devices` (BrainFlow, LSL, replay and simulated sources; cue protocol; recorder; API client; debounced, keyboard and action sinks); `neurolayer bridge`; `neurolayer report explain` (template or Claude, aggregates only); research note on OS input and Apple BCI HID; Playwright e2e in CI.
- **Stage 8:** consent ledger (`neurolayer consent grant|revoke|check`), `neurolayer pilot record`, pilot protocol, consent-form and device-memo templates, pilot dataset-card template; ADR-0012 (proposed).
- Extras: `devices`, `inspect`, `llm`.

## [0.1.0] - 2026-09-26

Stage 0: research and foundations.

### Added
- Research brief (companies, models, datasets, patents and IP, regulation, gap analysis) in `docs/research/`.
- CAP-1 capability spec (calibration-efficient motor-intent decoding) and roadmap.
- Architecture docs, data model, evaluation protocol, security and privacy model, reproducibility guide.
- ADRs 0001–0008.
- `neurolayer` package:
  - core types and channel normalization
  - consumer-device montages
  - dataset catalog with license gate
  - synthetic motor-imagery generator
  - reference decoders
  - CAP-1 evaluation harness
  - run manifests
  - experiment runner
  - CLI
- FastAPI skeleton (`neurolayer_api`) and Next.js dashboard skeleton (`apps/web`).
- CI, pre-commit hooks, data-file guard, Dependabot, issue and PR templates, AGENTS.md.

### Removed
- All content from the previous project in this repository (it remains available in git history).
