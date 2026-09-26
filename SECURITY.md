# Security policy

## Reporting a vulnerability

Do **not** open a public issue. Email the maintainers (see the GitHub profile of the repository owner) with a description, reproduction steps and impact. We aim to acknowledge within 3 business days.

## Handling rules (summary)

The full policy is in [docs/architecture/security-and-privacy.md](docs/architecture/security-and-privacy.md).

- **Neural data from people is sensitive (C3).** It never goes into git, never into third-party LLM APIs, and never into unencrypted storage.
- **Secrets never go into git.** Use `.env` (git-ignored) locally and GitHub Actions secrets or a secret manager in CI and production. gitleaks runs in pre-commit and CI.
- **Datasets and weights never go into git.** They live under `data/` and `artifacts/` (git-ignored, DVC-tracked). `scripts/check_no_data_files.py` enforces this.
- **Third-party weights:** load them only with safetensors or `torch.load(..., weights_only=True)`, pinned by SHA-256 (ADR-0007).
- **Dependencies:** lockfiles are committed; Dependabot, pip-audit and npm audit run.

## Supported versions

Pre-release (0.x). Only `main` is supported.
