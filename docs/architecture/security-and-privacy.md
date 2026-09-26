# Security and privacy architecture

Derived from the [regulatory requirements (PRIV-1…11)](../research/05-regulation-and-privacy.md#52-engineering-requirements-derived-from-the-above) and the IP strategy.

## 1. Data classification

| Class | Examples | Where it may live | Controls |
|-------|----------|-------------------|----------|
| **C0 Public** | Docs meant for publication, open-source dependencies | Anywhere | None |
| **C1 Public datasets** | PhysioNet, MOABB datasets | `data/` on dev machines; private cloud storage | Catalog license gate; never committed to git |
| **C2 Confidential IP** | Source code of proprietary models, training recipes, model weights, evaluation results, internal docs | **Private** repo; private artifact storage | Private repo, 2FA, least-privilege access, no public disclosure before an IP decision |
| **C3 Sensitive personal** | Neural data and metadata from real people we record (Stage 8+), consent records | Encrypted storage only; minimal copies | Consent registry, pseudonymization, encryption at rest and in transit, access logging, deletion workflow, DPIA |
| **C4 Secrets** | API keys, DB passwords, Supabase service keys | Secret manager / GitHub Actions secrets / local `.env` (git-ignored) | Never in git (gitleaks), rotated on exposure |

## 2. Threat model (STRIDE-lite)

| Threat | Asset | Likelihood today | Controls |
|--------|-------|------------------|----------|
| **IP disclosure through the public repo** | C2 | **High: the repo is public** | Make the repo private (action item); CODEOWNERS; branch protection |
| Accidental commit of data or weights | C1–C3 | High (easy mistake, especially for agents) | `.gitignore`; **pre-commit hook `scripts/check_no_data_files.py`** blocks neural-data and weight extensions; the same check runs in CI; `check-added-large-files` |
| Secret leakage | C4 | Medium | gitleaks (pre-commit + CI); `.env` git-ignored; `detect-private-key` |
| Malicious model weights (pickle code execution) | Dev machines | Medium (downloading third-party checkpoints) | Only `torch.load(..., weights_only=True)` or **safetensors**; third-party weights listed and hash-pinned (ADR-0007) |
| Dependency supply chain | All | Medium | Lockfiles (`uv.lock`, `package-lock.json`); Dependabot; `pip-audit` and `npm audit` in CI; new dependencies need justification in the PR |
| AI coding agent misuse (prompt injection via issues, docs or data; exfiltration) | C2–C4 | Medium | Agents run without production secrets; AGENTS.md rules; human review of every PR; no C3 data in agent-accessible environments |
| Re-identification from neural data | C3 | Medium (EEG carries identity; see the "identity trap") | Pseudonymous IDs; no biometric-ID features (PRIV-7); identity probe used as a privacy audit; minimal retention of raw C3 |
| Inference of sensitive states (health, emotion) from user data | C3 | Medium | Purpose limitation; no emotion inference in workplace/education (EU AI Act); consent per purpose |
| API abuse (Stage 6) | Service | Future | Auth (Supabase JWT), rate limiting, tenant isolation (row-level security), request size limits, audit logs |
| Cloud data exposure | C1–C3 | Future | Private buckets, encryption, data processing agreements, region choice, no C3 in third-party LLM calls |

## 3. Controls implemented in this commit

- `.gitignore` excludes `data/`, `artifacts/`, `mlruns/`, `.env*` and all neural-data and weight file extensions.
- `scripts/check_no_data_files.py` runs as a pre-commit hook and in CI. It blocks these extensions:
  - neural data: `.edf .bdf .gdf .fif .set .fdt .vhdr .vmrk .eeg .xdf .cnt .mat`
  - arrays: `.npy .npz .h5 .hdf5 .zarr .parquet`
  - weights: `.pt .pth .ckpt .safetensors .onnx .pkl .joblib`
- Pre-commit hooks: gitleaks, detect-private-key, check-added-large-files (500 KB), nbstripout (notebook outputs can contain data).
- CI: least-privilege `permissions: contents: read`; gitleaks; pip-audit; npm audit; Dependabot for pip/uv, npm and GitHub Actions.
- Dataset license gate (`neurolayer.data.catalog`).
- API skeleton: no secrets, no data endpoints yet; FastAPI docs can be disabled via an environment setting in production.
- Web app: `poweredByHeader: false` and baseline security headers.

## 4. Required repository settings (manual, founder action)

1. **Change visibility to Private.**
2. Branch protection on `main`: require PR, require CI green, require 1 review (CODEOWNERS), no force-push, no deletion.
3. Enable secret scanning and push protection (for a private repo on a personal account, availability depends on plan; gitleaks covers the gap).
4. Enable Dependabot alerts and security updates.
5. Require 2FA on the GitHub account; use fine-grained tokens for automation.

## 5. Stage-gated controls (implemented later)

| Stage | Control |
|-------|---------|
| 6 API | AuthN/AuthZ (Supabase JWT), per-tenant isolation with Postgres row-level security, rate limits, structured audit logs, TLS only, CORS allowlist, OpenAPI schema review, dependency pinning in images, non-root containers |
| 7 Product | Content Security Policy, upload validation (size/type, parse in a sandboxed worker), on-device processing by default (PRIV-9) |
| 8 Real-world | Consent registry, pseudonymization service, encrypted C3 storage, deletion workflow, DPIA, incident response runbook |
