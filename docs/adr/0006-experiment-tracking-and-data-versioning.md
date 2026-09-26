# ADR-0006: Experiment tracking and data versioning

- **Status:** Accepted
- **Date:** 2026-09-26

## Context
We need reproducible experiments from day one. Results and data are IP (C2) or sensitive (C3).

## Decision
- **Run manifests** (`neurolayer.tracking`) are the ground truth. They are plain JSON/CSV in `artifacts/runs/<run_id>/` and need no service.
- **MLflow, self-hosted**, provides the UI and comparisons. A local file or sqlite store runs on the ML PC; a private server comes later if the team grows. The MLflow logger (WP-0.7) mirrors the manifest; it never replaces it.
- **DVC** versions raw datasets and large artifacts. It uses a local cache first, then a private S3-compatible remote (WP-1.3).
- `--official` runs require a clean git tree.

## Consequences
The manifest format must stay backward compatible (versioned field `manifest_version`).

## Alternatives considered
- Weights & Biases: excellent UX, but hosted by a third party by default, which is a mismatch for C2/C3.
- Git LFS for data: poor fit for multi-GB datasets and no pipeline semantics.
