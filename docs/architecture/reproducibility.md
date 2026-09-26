# Reproducibility, experiment tracking and data versioning

## 1. Principles

1. **Configs, not code edits.** Every experiment starts from a YAML file in `configs/experiments/`, validated by `ExperimentConfig`. No hard-coded paths or magic numbers in scripts.
2. **Every run writes a manifest.** `artifacts/runs/<run_id>/manifest.json` records:
   - the git commit and whether the tree was dirty
   - the config and its SHA-256 hash
   - dataset card ids, versions and license SPDX ids
   - the license-gate purpose
   - seeds
   - Python, platform and key package versions
3. **Official vs exploratory runs.** Only `neurolayer run --official`:
   - refuses to run on a dirty git tree
   - requires every dataset to pass the license gate for the configured purpose

   Only official runs may be cited in the [results ledger](../results/README.md), in gate decisions, or outside the company.
4. **Raw data is immutable; derived data is content-addressed** (pipeline hash). See [data-model.md](data-model.md#5-on-disk-layout).
5. **Determinism:** seeds for numpy, torch and Python `random`; deterministic torch algorithms for official runs where available. Nondeterminism that remains (for example some GPU kernels) is reported as variance across ≥ 3 seeds.

## 2. Tools

| Concern | Tool | Status |
|---------|------|--------|
| Code versioning | Git + GitHub (PRs, CI, CODEOWNERS) | ✅ |
| Python env | `uv` + `uv.lock` (Python 3.12) | ✅ |
| JS env | npm + `package-lock.json` (Node 22) | ✅ |
| Run manifests + results | `neurolayer.tracking` (JSON/CSV in `artifacts/runs/`) | ✅ |
| Experiment UI and comparison | **MLflow** (local file or sqlite store on the PC; self-hosted, so neural results stay private) | WP-0.7 |
| Data versioning | **DVC** (initialized in this commit), with a local cache first and private remote storage (S3-compatible or Supabase storage) later | ✅ init, remotes in WP-1.3 |
| Environments | Docker images for the API and ML (CPU); GPU image variant in WP-3.3 | ✅ Dockerfiles (not built in CI yet) |

Why MLflow over Weights & Biases: it can be fully self-hosted, which keeps C2/C3 artifacts in our control, and it is free. W&B is fine for public research, but results are IP here (ADR-0006).

## 3. Experiment lifecycle

1. **Card.** Copy `docs/experiments/TEMPLATE.md` and write the hypothesis and the **prediction before running**.
2. **Config.** Add `configs/experiments/<name>.yaml` in a PR together with the card.
3. **Run.** Exploratory runs are free. The final run is `neurolayer run <config> --official` on a clean, committed tree.
4. **Record.** Add a row to `docs/results/README.md` with the run id, config hash, key metrics and a link to the card.
5. **Decide.** Write the conclusion in the card: supported / refuted / inconclusive, and the next step.

Negative results are recorded too. They prevent re-running dead ends, which matters when AI agents generate many experiments.
