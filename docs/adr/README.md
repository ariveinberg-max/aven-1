# Architecture Decision Records

Write an ADR for any decision that is expensive to reverse or that changes a contract, such as:

- data model or storage
- evaluation protocol or metrics
- dependency or framework choice
- security posture
- license policy

Copy `0000-template.md`, take the next number, and open a PR.

AI agents must **propose** an ADR, never silently change a decided item.

| ADR | Title | Status |
|-----|-------|--------|
| [0001](0001-record-architecture-decisions.md) | Record architecture decisions | Accepted |
| [0002](0002-monorepo-and-tooling.md) | Monorepo layout and tooling (uv, ruff, mypy, pytest, import-linter, Next.js) | Accepted |
| [0003](0003-canonical-data-model-and-storage.md) | Canonical data model and on-disk storage (BIDS + content-addressed processed data) | Accepted |
| [0004](0004-evaluation-protocol.md) | CAP-1 evaluation protocol (calibration-efficiency curve, chronological calibration, disjoint folds) | Accepted |
| [0005](0005-dataset-license-gate.md) | Dataset license gate (exploration / benchmark / training) | Accepted |
| [0006](0006-experiment-tracking-and-data-versioning.md) | Experiment tracking (manifests + self-hosted MLflow) and data versioning (DVC) | Accepted |
| [0007](0007-third-party-model-policy.md) | Third-party foundation models: baselines and initializations only when license-clean | Accepted |
| [0008](0008-first-capability-motor-intent.md) | First capability is calibration-efficient motor intent (CAP-1) | Accepted |
| [0009](0009-locked-holdout-selection.md) | Locked holdout selection rule (dataset choice pending audit) | Proposed |
