# Work packages

A **work package (WP)** is the unit of work you hand to an engineer or an AI coding agent (Codex, Claude Code). Each WP has:

- a scope
- the contracts it must respect
- the tests it must add
- acceptance criteria that CI or a reviewer can check

Rules for every WP:

- **One WP per branch and PR.** Branch name: `wp/<id>-<slug>`, for example `wp/1.2-physionet-adapter`.
- **`make check` must pass.** No new lint, type or layering violations.
- **Tests use synthetic or tiny fixture data only.** Tests needing downloads are marked `@pytest.mark.network` and are opt-in.
- **No change to `src/neurolayer/evaluation/` or to catalog license fields without an ADR** (ADR-0004, ADR-0005).
- **Out-of-scope improvements go into a new issue,** not into the PR.

How to prompt an agent with a WP: see [working with AI agents](../guides/working-with-ai-agents.md).

Legend: ✅ done · 🔜 next · ⏳ blocked by dependency · 👤 human task

---

## Stage 0: Foundations (this commit)

| WP | Title | Status |
|----|-------|--------|
| 0.1 | Repo tooling: uv, ruff, mypy strict, pytest+hypothesis, import-linter, pre-commit, CI, Dependabot | ✅ |
| 0.2 | Core types (`Recording`, `EpochSet`), channel normalization, consumer montages | ✅ |
| 0.3 | Dataset catalog + license gate + cards for all researched datasets | ✅ |
| 0.4 | CAP-1 evaluation harness (folds, chronological calibration, metrics, protocol runner) | ✅ |
| 0.5 | Synthetic motor-imagery generator; reference decoders | ✅ |
| 0.6 | Run manifests, experiment configs, runner, CLI (`catalog`, `run`, `smoke`) | ✅ |
| 0.7 | MLflow mirror of run manifests | ✅ |
| 0.8 | 👤 Repository settings: **make private**, branch protection, secret scanning, 2FA | 🔜 |
| 0.9 | Dependency license report in CI (fail on GPL/AGPL in the runtime dependency tree) | ✅ |

### WP-0.7: MLflow mirror
- **Scope:** `neurolayer.tracking.mlflow_logger.log_run(manifest, result)`, which logs the config (params), per-budget metrics, AUCEC, UUR and TTC, and attaches `manifest.json`, `results.csv` and `summary.json` as artifacts. Enable it with `neurolayer run --mlflow` (lazy import; `tracking` extra).
- **Contracts:** the manifest stays the source of truth. MLflow failure must not fail the run; log a warning instead.
- **Tests:** a temporary `file:` tracking URI; assert that the run exists with its metrics and artifacts. Skip if MLflow is not installed.
- **Accept when:** `make smoke` followed by `mlflow ui` shows the run with a per-k curve.

### WP-0.9: Dependency license report
- **Scope:** a CI step using `pip-licenses` (runtime extras only) and `license-checker` for npm. Fail on GPL-3.0, AGPL or SSPL in shipped code.
- **Accept when:** CI prints a license table and fails on a deliberately added GPL test dependency (verified locally, then removed).

---

## Stage 1: Data ingestion

### WP-1.1 👤: Verify dataset licenses
- **Scope:** for each card in `catalog/datasets/`:
  - open the license page
  - update the SPDX id and flags
  - quote the evidence
  - set `verified_by` / `verified_on`
- **Start with:** `physionet_mi`, `cho2017`, `lee2019_mi`, `stieger2021`, `dreyer2023`, `bnci2014_001`.
- **Accept when:** `neurolayer catalog check --purpose benchmark --datasets physionet_mi,cho2017,lee2019_mi` exits 0, or any dataset still refused has a documented reason.

### WP-1.2: MOABB/MNE adapter → `Recording` (PhysioNet MI first) ✅
- **Done:** `neurolayer.data.adapters` (protocol, `get_adapter`, `MoabbAdapter`) and `neurolayer.data.mne_bridge`. Tested with an in-memory MOABB double; the PhysioNet download test is opt-in (`pytest -m network`).
- **Scope:**
  - `neurolayer.data.adapters.moabb.MoabbAdapter(card)`, which implements a `DatasetAdapter` protocol (defined in `neurolayer.data.adapters.base`):
    - `subjects() -> list[str]`
    - `recordings(subject) -> Iterator[Recording]`
  - Uses MNE/MOABB (extra `neuro`), imported lazily.
  - Converts to volts, normalizes channel names (`normalize_channel_names`) and maps labels to the canonical vocabulary.
  - Keeps EEG channels only, recording dropped channel types.
  - Pseudonymizes subject ids as `sub-XXX`.
  - Records native → canonical label maps in a QA dict.
- **Contracts:** returns only `Recording` objects; no MNE objects cross the boundary. It must refuse to run unless `evaluate_usage(card, Purpose.EXPLORATION)` allows it.
- **Tests:**
  - unit tests with a tiny MNE `RawArray` built in-test, mocking the MOABB download
  - a `network`-marked test loading PhysioNet subject 1
- **Accept when:** `load_epochs(["physionet_mi"])` works end to end, after WP-2.4 exists. Until then, the adapter yields valid `Recording` objects for subjects 1–3.

### WP-1.3: Raw storage, checksums, DVC ✅
- **Done:** `neurolayer.data.storage` (layout, SHA-256 manifests, BIDS-EDF round trip) and `neurolayer data fetch|verify`. DVC is initialized; run `uvx dvc add data/raw/<id>/<version>` after each fetch.
- **Scope:**
  - `neurolayer data fetch <dataset_id>` downloads into `data/raw/<id>/<version>/`
  - writes `checksums.sha256`
  - converts to BIDS in `data/interim/bids/<id>/` via mne-bids
  - `dvc add`s the raw directory
  - configures a local DVC cache and documents a private remote
- **Tests:** checksum verification detects a modified file; BIDS round trip of a synthetic `Recording`.
- **Accept when:** re-running fetch is a no-op when checksums match.

### WP-1.4: Channel and label audit; choose the locked holdout ✅ tooling · ⏳ decision
- **Done:** `neurolayer.data.audit` and `neurolayer data audit`. The selection rule is ADR-0009. The **choice** waits for downloaded data ([dataset-audit.md](../results/dataset-audit.md)).
- **Scope:** a notebook plus script producing `docs/results/dataset-audit.md`. For each motor-imagery dataset it reports:
  - channels vs canonical, and coverage of each `CONSUMER_MONTAGES` entry
  - trials per class per subject, sessions, sampling rate
  - event timing
- **Decide:** the development pool vs the **locked holdout** (Dreyer2023 or Stieger2021), based on channel coverage and size. Record the decision in ADR-0009.
- **Accept when:** the ADR is merged and card `roles` are updated.

### WP-1.5: Dataset QA report ✅
- **Done:** `neurolayer.data.qa` (flat, noisy, line noise, unit sanity, dropped labels and channels) and `neurolayer data qa [--strict]`.
- **Scope:** per-subject QA metrics:
  - duration
  - flat or noisy channels (robust z-score of variance)
  - line-noise power
  - event counts
  - dropped labels

  Written as JSON next to the BIDS data and summarized in markdown.
- **Accept when:** the report runs on PhysioNet and flags a synthetic bad channel injected in a test.

---

## Stage 2: Signal processing

### WP-2.1: Transform framework
- **Scope:** a `Transform` protocol (`__call__(Recording) -> Recording`, `name`, `version`, `config`) and a `Pipeline` (ordered transforms). `pipeline_hash` = SHA-256 of the canonical JSON of the `(name, version, config)` list.
- **Contracts:** transforms are pure. They never mutate their input, which is guaranteed because arrays are read-only.
- **Tests:** the hash is stable across runs and changes when any version or config changes.

### WP-2.2: Standard transforms
- **Scope:**
  - resample (polyphase)
  - bandpass/highpass (zero-phase FIR, parameters in config)
  - notch (50/60 Hz)
  - re-reference (common average; specified channels)
  - unit sanity check (flags data that looks like µV mislabeled as V)
- **Tests:** golden tests on synthetic sinusoids (the passband is preserved, the stopband is attenuated by at least the spec), plus determinism.

### WP-2.3: Channel harmonization
- **Scope:** select or reorder to a target channel list, driven by config. Optional interpolation is **off by default** and, when used, must be recorded in the manifest. Montage restriction for R3 uses `CONSUMER_MONTAGES`.
- **Tests:** a missing channel raises `MissingChannelsError`, never substitutes silently.

### WP-2.4: Epoching and label harmonization
- **Scope:** `Recording(s) -> EpochSet`:
  - window relative to the cue (config, for example 0.5–2.5 s)
  - label filter (for example CAP-1 L/R)
  - sets `order` chronologically across runs and sessions
  - optional baseline correction
- **Tests:** synthetic `Recording` with known events → exact epoch count, label and order.

### WP-2.5: Artifact handling
- **Scope:** bad-channel detection, amplitude/variance epoch rejection (thresholds in config, report counts), and optional EOG regression when EOG is available. Every rejection is counted in the QA output.
- **Tests:** injected artifacts are rejected; clean epochs are kept.

### WP-2.6: Processed cache + Gate 0
- **Scope:** write and read `data/processed/<pipeline_hash>/<dataset>/sub-XXX_epochs.npz` + `.json`. Then reproduce MOABB within-session CSP+LDA and TS+LR on PhysioNet, Cho2017 and Lee2019 with the same pipeline.
- **Accept when (Gate 0):** mean accuracy within ±3 pp of MOABB's published numbers. Results are recorded in the results ledger with official run ids.

---

## Stage 3: Neural representation

### WP-3.1: Covariance and tangent-space encoders
- **Scope:**
  - `CovarianceEncoder` (shrinkage covariance)
  - `TangentSpaceEncoder` (pyRiemann, reference point configurable)
  - `CSPEncoder`

  All implement `Encoder`.
- **Tests:** shapes, SPD validity, determinism.

### WP-3.2: Alignment methods (calibration-time adaptation components)
- **Scope:**
  - Euclidean alignment
  - Riemannian re-centering
  - Riemannian Procrustes (stretch and rotate)

  Each is a fit-on-subject transform using only **calibration or unlabeled** data.
- **Tests:** aligned target covariance mean ≈ identity; the implementation cannot access test data (checked by an API design review).

### WP-3.3: PyTorch training scaffold
- **Scope:**
  - `neurolayer.representation.torch` with a deterministic trainer: seeds, `torch.use_deterministic_algorithms` for official runs, AMP option, early stopping on **source-only** validation
  - checkpoints via **safetensors**
  - CPU/CUDA/MPS device selection
  - GPU Dockerfile variant
- **Tests:** a tiny model overfits a tiny synthetic set; resume from checkpoint reproduces the loss curve.

### WP-3.4: Third-party foundation-model adapters (license-gated)
- **Scope:** adapters for MIRepNet and LaBraM/CBraMod as `Encoder` implementations, for **benchmark use**. Requirements:
  - weights pinned by SHA-256
  - loaded with `weights_only=True` or safetensors
  - model card in `catalog/models/`

  See ADR-0007.
- **Accept when:** adapters run on synthetic input and are refused for `training` without a legal review.

---

## Stage 4: Baselines (Gate 1)

### WP-4.1: Baseline suite B0–B6 as registered decoders
- **Scope:** implement the baselines in [CAP-1 §4](../product/cap-1-calibration-efficient-intent.md#4-baseline-suite-must-be-reproduced-before-any-proprietary-claim), each registered in `models/registry.py` with an experiment config per regime (R0–R3).
- **Tests:** each decoder passes the harness contract tests (`tests/unit/test_protocol.py` patterns: no label peeking, deterministic given seed).

### WP-4.2: Baseline report and CAP-1 thresholds
- **Scope:** official runs of B0–B6 × R0–R3 on the development pool, plus `docs/results/gate-1-baselines.md`. It contains:
  - CEC plots
  - tables (BA@k, UUR@k, AUCEC, TTC)
  - paired comparisons
- **Decide:** the numeric CAP-1 margins (Δ) in **ADR-0010**, before the locked holdout is touched.

### WP-4.3: Leakage controls
- **Scope:**
  - label-shuffle control run
  - identity probe and dataset-ID probe on encoder outputs
  - an audit checklist in every gate report
- **Accept when:** the shuffle control is at chance (CI includes 0.5) for every baseline.

---

## Stage 5: Proprietary model (Gate 2)

This stage is a research program run as experiment cards (`docs/experiments/`), not a single WP. Initial hypotheses to test, in order:

- **H1:** montage-agnostic encoding (channels as tokens with learned 10-05 position embeddings) + masked-signal pretraining on **license-clean** data beats B3–B6 on R2/R3 at k=0 (U).
- **H2:** calibration-time adaptation (alignment + few-shot head fine-tuning + per-subject portfolio selection) improves AUCEC over H1 alone.
- **H3:** online test-time adaptation (causal, past-only) improves TTC without hurting BA at k=20.
- **H4:** the identity probe on our representation is lower than on the third-party foundation models, with equal or better BA. We learn intent, not identity.

**Gate 2** is evaluated once on the locked holdout per model version, followed by the kill/pivot review ([gap analysis §6.5](../research/06-gap-analysis.md#65-kill-and-pivot-criteria-decide-in-advance-to-avoid-sunk-cost-drift)).

---

## Stage 6: API

### WP-6.1: Model registry and packaging
- **Scope:** a versioned model bundle containing:
  - weights (safetensors)
  - pipeline config and hash
  - the decoder spec
  - the manifest of the training run
  - the license-gate record

  Loaded by the API by version.

### WP-6.2: Inference endpoints
- **Scope:**
  - `POST /v1/sessions` (create a calibration session)
  - `POST /v1/sessions/{id}/calibration` (labeled trials)
  - `POST /v1/sessions/{id}/decode` (batch)
  - `WS /v1/sessions/{id}/stream` (streaming)

  Pydantic schemas. The payload is windows of samples plus channel names and the sampling rate.
- **Accept when:** p95 decode latency is below 50 ms for one 2-second window on CPU; covered by a load test.

### WP-6.3: Auth, tenancy, limits
- **Scope:**
  - Supabase JWT verification
  - per-tenant isolation (Postgres row-level security)
  - rate limiting
  - request size limits
  - structured audit logs
  - CORS allowlist

  See [security](../architecture/security-and-privacy.md).

---

## Stage 7: Product

| WP | Scope |
|----|-------|
| 7.1 | Dashboard: upload an EDF/BDF/FIF file → validate → visualize channels and spectra → run a model → show predictions and confidence. Parsing happens in an isolated worker. |
| 7.2 | Calibration game: a 2–3 minute cue-based flow that produces CAP-1-compatible calibration data |
| 7.3 | Device bridge: BrainFlow/LSL → API streaming (Neurosity Crown, OpenBCI Cyton first) |
| 7.4 | OS input bridge prototype: decoded intents → virtual HID / switch events; investigate the Apple BCI HID profile requirements |
| 7.5 | LLM layer: explain results and let decoded intents trigger agent actions, behind a provider-agnostic interface. **No neural data sent to LLM APIs.** |

## Stage 8: Real-world testing (Gate 3)

| WP | Scope |
|----|-------|
| 8.1 | Device selection memo from the R3 results (Crown vs OpenBCI layout) |
| 8.2 👤 | Consent and data-rights protocol (product / research / commercial training, each separate), with counsel review. Also consider IRB-style ethics review. |
| 8.3 | Pilot collection with ≥ 20 participants; dataset card for our own data (`access: internal`); Gate 3 evaluation |
