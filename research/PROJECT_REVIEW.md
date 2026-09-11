# Project review and implemented improvements

Reviewed 2026-09-09 on branch `main`. Scope: architecture, training/tokenization,
chat/control panel, memory/source storage, preference/reward/RAFT/PPO pipeline,
evaluation, tests, and research coordination notes. Existing local changes in
`make_instructions.py` and `runs/ledger.jsonl` were preserved. No production
checkpoint, tokenizer, training corpus, preference label, or remote run was changed.

## How the system fits together

- `brain.py`: causal Transformer with learned positional embeddings, tied input/output
  weights, pre-normalized attention/MLP blocks, and autoregressive sampling.
- `tokenizer.py` and `train.py`: byte-pair vocabulary, streamed corpus fingerprinting
  and disk token cache, sampled 90/10 validation, AdamW, atomic checkpoints,
  resume identity checks, and local/optional W&B tracking.
- `server.py`, `chat.html`, `ui.html`: loopback HTTP control panel, generation,
  instruction-format conversation history, training subprocess, and storage APIs.
- `memory.py`, `sources.py`, `preferences.py`: independent notes, training text,
  and actual human comparison labels. These are different forms of data.
- `train_reward.py`, `reward_model.py`, `raft.py`, `ppo.py`, `value_model.py`:
  preference reward learning, candidate ranking, and policy optimization.
- `evaluate_capabilities.py`, `eval_heldout.py`: separate exact-answer capability
  evaluation with provenance and external-text next-token scoring.

## Implemented

1. Complete bounded arithmetic parsing replaces substring matching. For example,
   `2+3*4` now yields 14, rather than calculating only `2+3`. Exact fractions avoid
   floating-point rounding. Unsupported syntax falls through to ordinary chat.
2. `recall.py` retrieves existing notes using BM25 and weighted titles, returning
   bounded verbatim excerpts with IDs and explicit no-match responses. It does
   not invent or silently store knowledge.
3. Chat routes tools before checkpoint loading/training restrictions. The interface
   remains usable for tools during training and labels their responses. Invalid
   message objects return a client error. Long sessions send recent history.
4. External-text evaluation now covers every available next-token target, handles
   short inputs, includes partial trailing windows, and weights by target count.

## Remaining findings and priorities

- KV caching accelerates generation before the context fills. Learned positional
  embeddings still require full recomputation when the window shifts.
- Response-only training is implemented and opt-in. Its benefit on the production
  model still requires a separate, independently evaluated experiment.
- Saved notes use lexical retrieval. Paraphrases without shared words may miss.
- Preference validation now separates prompt groups. Small or templated preference
  sets still cannot establish broad assistant quality.
- OS writer locks cover participating processes on the local filesystem. Old code,
  remote writers and distributed/network filesystem semantics need separate care.
  The Windows lock branch is implemented but was not exercised on Windows here.
- CPU dropout resume parity and MPS random-state replay now pass. CUDA replay
  coverage is conditional and was skipped on this machine. Retry chunks preserve
  their original target; independent wrapper launches intentionally start new budgets.
- PPO policies remain separate experimental artifacts; the normal training resume
  path expects optimizer state. Do not overwrite the main checkpoint with one.
- Legacy checkpoints do not bind a tokenizer fingerprint. Their vocabulary size
  can be checked, but original tokenizer identity cannot be established retroactively.
- Existing research notes document poor independent capability results. No new
  broad-intelligence claim follows from tool features or synthetic optimization.

## Verification

`.venv/bin/python -m unittest discover -v`: 37 tests passed, including isolated
small-model training/resume tests and HTTP tool routing while training is active.
No production model was trained or loaded for this verification.
`git diff --check` passed. Existing server is handled separately from remote training.

## Second implementation pass

- Added request-local KV caching with full-window rebuilds. Tests compare cached
  and uncached logits, different attention scales, batched input, greedy output
  through rollover, unchanged weights, and activity output.
- Fixed end markers embedded within a BPE token and separated prompt markers from
  generated markers. Invalid nonfinite/negative temperatures are rejected.
- Added tokenizer-aware complete-turn context selection, latest-message overflow
  errors, and omission metadata displayed in chat.
- Added a reproducible small-model CPU benchmark: five alternating measured runs
  per mode after warmup. Median within-window speedup 2.9025x; full-window speedup
  1.0081x. Greedy outputs match. Production checkpoint speed is not measured.
- Full verification after this pass: 46 tests, including model-route HTTP tests
  against temporary checkpoints. Training forward/optimizer behavior is unchanged.

## Third implementation pass

Training and recovery:
- Full tokenizer identities and cache checksums; malformed merge validation;
  atomic writes with flushed file data and cleanup on failures.
- Local writer locks, independent output directories, and `--init-from` branching.
- Learning-rate inheritance on resume, nonfinite-gradient rejection, fresh phase
  history for new fine-tuning corpora, and response-only loss with correct metrics.
- Retry budget counts consecutive failures; preparation failures and initialization
  flags are handled when moving to resume.
- Export/verification utility, including an atomic-replacement snapshot test.

Preference learning:
- Prompt-group validation, duplicate/contradictory comparison filtering, complete
  context checks, cached encoded pairs, fixed seed, and saved provenance.
- RAFT/PPO verify tokenizer compatibility when identity is recorded.

Measured results:
- Real local checkpoint: SHA-256
  `6fb04450b2289ff78e19c8bb3e7f2fe89ba59235bc0b55e216b8047bccd046d4`,
  step 5000, width 768, 8 layers, context 192, vocab 2048.
- Short-chat CPU decode: 0.274s uncached vs 0.112s cached, **2.44x** speedup.
  Full-window: 0.759s vs 0.774s, about 2% slower. Three measured runs per mode
  after warmup; identical greedy output and activity. See checkpoint benchmark JSON.
- Tiny synthetic response-loss experiment: 100 steps from identical initial
  weights, answer loss 0.345 vs 0.794, answer-token accuracy 85.2% vs 72.7%.
  Same-corpus optimization check only; not held-out generalization.
- A verified local step-5000 bundle was written to
  `work/checkpoint-bundles/local-20260909-step5000`. Source files were not changed.
  The source is a legacy checkpoint without an embedded tokenizer hash, which
  the bundle manifest explicitly records.

The remote CPU training run was not contacted, stopped, or updated. Training-side
changes apply to future launches using this updated code. Production weights were
only read for the local backup and inference benchmark. Existing user changes to
`make_instructions.py` and `runs/ledger.jsonl` remain intact.

## Fourth implementation pass

- Saved a recoverable code snapshot and full test log under
  `work/recovery/verified-83-tests` before starting this pass.
- CPU/backend RNG snapshots, explicit legacy/cross-backend restore metadata, exact
  CPU dropout resume parity, and a real MPS random-state replay test.
- Retry budgets persist through partial-save crashes and final-save retries;
  continuous chunks advance their budget ID only after clean completion.
- Corpus overlap scanning now uses bounded memory while preserving the previous
  normalization/matching protocol, including matches across chunk boundaries.
- External-text scoring streams tokenization, includes all targets, and emits
  versioned reports with source and code hashes. Tokenization boundaries are
  explicitly part of the protocol; old whole-book results are not equivalent.
- Checkpoint loading hashes and loads the same open snapshot. Documentation now
  distinguishes current evaluation-folder exclusion from historical non-exposure.

## Latest verified state

95 tests discovered: 94 passed, one CUDA-only test skipped because CUDA is
unavailable. MPS RNG replay passed on the actual Mac backend. Shell syntax,
chat JavaScript syntax, and `git diff --check` passed. The idle local server
was reloaded and its status, calculator and recall endpoints passed smoke checks.

The complete source, uncommitted patch, file hashes and verification log are saved
under `work/recovery/verified-95-checks`. The earlier verified 83-test snapshot is
also retained. These archives include the pre-existing user edits for recovery;
those edits were not changed or committed by this work.
