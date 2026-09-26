# Changelog

All notable changes are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
