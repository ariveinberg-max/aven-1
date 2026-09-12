# Aven-1

A language model built entirely from scratch — no pretrained weights, no API key, no downloaded model. Every piece was written and trained here: a byte-pair tokenizer trained on this project's own text, a causal Transformer, a training loop, and a full RLHF pipeline (reward model, RAFT, PPO, and now DPO) built on real human preference labels, not synthetic ones.

The verified local checkpoint is **58,424,832 parameters** (8 layers, width 768, 192-token context, 2,048-token vocabulary), fine-tuned through multiple iterations of real instruction data. A separate **153M-parameter** pretraining track tests the same pipeline at larger scale. Both are documented in `WRITEUP.md`, including what broke along the way and what fixing it taught — that write-up is the honest, detailed version of this project; this README is the quick-start.

This is not a general assistant, a conscious system, or a model with vision, voice, or web access. It answers from what it was actually trained on, nothing more.

## Try it in 30 seconds

Double-click **Start Brain.command**, keep the Terminal window open, and visit http://127.0.0.1:8765 if your browser doesn't open automatically. Click **Generate text** to try the existing checkpoint immediately — no setup required. Close the server with Control-C; closing the browser tab alone does not stop training.

If macOS won't launch the command file directly:

```sh
cd /Users/ariveinbers/Documents/Codex/my-ai-brain
.venv/bin/python server.py
```

## What's actually in here

- **Pretraining and tokenization** (`brain.py`, `tokenizer.py`, `train.py`) — a from-scratch byte-pair tokenizer and causal Transformer, trained on this project's own corpus.
- **Instruction fine-tuning** (`make_instructions.py`) — a programmatically generated dataset teaching the model to answer instead of just continuing text.
- **RLHF, for real** (`preferences.py`, `reward_model.py`, `train_reward.py`, `raft.py`, `ppo.py`, `dpo.py`) — real human preference collection through the dashboard, a reward model, RAFT (reward-ranked fine-tuning), full PPO with PPO-ptx, and DPO (direct preference optimization) — four different real methods against the same real data, compared honestly rather than assumed to work.
- **A local dashboard and chat interface** (`server.py`, `ui.html`, `chat.html`) — training controls, live loss charts, a preference-labeling panel, a training-data workspace, and note storage kept explicitly separate from the model's learned weights.
- **Real experiment tracking** — Weights & Biases integration (loss, perplexity, gradient/weight norms, sample generations over time) plus a local `runs/ledger.jsonl` for an offline history of every run.

Full narrative, including every real regression and how it was found: `WRITEUP.md`. Ongoing task handoffs and open questions: `research/TASKS.md`.

## Train on your own text

Use UTF-8 plain text you own or have permission to train on (4 KB–20 MB). To start a fresh run while preserving the current one: pause training, stop the server, rename `checkpoints/` to something like `checkpoints-backup`, save your text as `data/training.txt`, and reopen the app.

From the command line:

```sh
.venv/bin/python train.py --data data/training.txt --steps 200 --resume
```

A checkpoint records its corpus hash and refuses a different corpus on resume, preserving model weights, optimizer state, step count, and RNG state. Checkpoints save every 20 steps, on normal completion, and on a handled pause — force-quitting can lose steps since the last save. Run one training process at a time.

## Learned weights vs. training data vs. stored memory

Three things this project keeps explicitly separate, on purpose:

- **Learned weights** (`checkpoints/latest.pt`) — changed only by training, never edited directly.
- **Training-data workspace** (`data/sources/*.txt`) — plain text a *future* run would learn from, managed from the dashboard's workspace panel and compiled into `data/training.txt`.
- **Memory** (`memory.db`) — a plain SQLite note store you write and read directly (`/recall topic` in chat). It's never read by training and never changes a model weight — inspectable, external storage, the opposite of the model's opaque parameters.

## Recreate the environment

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -p "test_*.py"
```

No internet access is required to run or train the model — only the first-time dependency install needs it. No model weights or external datasets are downloaded.

---

## Technical reference

The sections below cover accumulated detail — experiment tracking internals, checkpoint integrity guarantees, inference benchmarks, and evaluation protocol. Skip this on a first read; come back when you need the specifics.

### Tracking training runs

- **Weights & Biases** (cloud, free tier, project `aven-1`): pass `--wandb` to `train.py` for full run tracking at https://wandb.ai/ariveinberg-ari-research/aven-1. Uploads run metrics and config — not your training text:
  - Loss/perplexity: train loss, held-out loss, held-out perplexity, every 20 steps
  - Optimizer health: gradient norm (pre-clip) and total weight norm
  - Gradient/weight histograms per layer (`wandb.watch`, every 100 steps)
  - Throughput (tokens/sec)
  - A live sample-generations table (fixed probe prompts, regenerated every 100 steps)
  - Full run config (architecture, dataset, tokenizer compression ratio, learning rate, batch size)
  - System metrics (CPU/memory), captured automatically

  One-time setup:

  ```sh
  .venv/bin/pip install wandb
  .venv/bin/wandb login
  .venv/bin/python train.py --data data/training.txt --resume --steps 200 --wandb
  ```

  The dashboard's **Resume training** button auto-adds `--wandb` once you're logged in (detected via `~/.netrc`).

- **Run ledger** (`runs/ledger.jsonl`): one line per finished/paused/errored run — architecture, dataset, final perplexity, timestamp. Always written locally regardless of W&B.

See `WANDB.md` for the full metric catalog. `checkpoints/status.json` drives the live dashboard panel.

### Hyperparameter sweeps

`train.py --sweep` runs an isolated, throwaway trial under `sweeps/` — never touches `checkpoints/`, reuses an existing tokenizer, caches the encoded corpus for fast repeated trials. Combined with a W&B Sweep:

```sh
.venv/bin/wandb sweep sweep.yaml
.venv/bin/wandb agent <entity>/aven-1/<sweep-id>
```

`sweep.yaml` searches learning rate, dropout, and batch size on a small architecture over the instruction corpus. Trial files under `sweeps/run-*/` are safe to delete.

### Faster inference

Generation reuses per-layer attention keys/values while there's room in the context window (`Brain.generate(..., use_cache=False)` is the comparison path). Measured 2.90x faster generation within the window on the checked-in CPU benchmark (`research/benchmarks/inference-cpu.json`), with identical greedy output verified bit-for-bit against the non-cached path. This is a local synthetic benchmark, not a measured speedup for the production checkpoint specifically.

```sh
.venv/bin/python benchmark_inference.py --output work/inference-benchmark.json
```

Chat fits recent conversation turns to the checkpoint's context limit, reports omitted history explicitly, and refuses to silently answer a truncated question.

### Isolated response-only fine-tuning

Experiment from an existing checkpoint without touching its directory:

```sh
.venv/bin/python train.py --init-from checkpoints \
  --output-dir work/experiments/response-tuning \
  --data data/instructions.txt --loss-mode response --steps 200 --lr 1e-5
```

Records the parent checkpoint hash; the source run is untouched. The instruction file needs `### Instruction:`, `### Response:`, and `<|end|>` on separate lines — prompt tokens provide context but are excluded from the loss.

Resume the same experiment:

```sh
.venv/bin/python train.py --output-dir work/experiments/response-tuning \
  --data data/instructions.txt --resume --steps 200
```

### Checkpoint integrity and recovery

- Training takes a process lock on its output directory; the dashboard detects external writers and refuses duplicate starts.
- Token caches are checksummed; damaged caches are detected, not silently trusted.
- Checkpoints record tokenizer identity — resuming with a changed tokenizer is rejected.
- Atomic writes: a failed write leaves the previous checkpoint in place. Nonfinite gradients stop before an optimizer update.
- Export a portable, checksummed snapshot without stopping an active writer:

```sh
.venv/bin/python checkpoint_bundle.py export --source checkpoints \
  --destination work/checkpoint-bundles/my-snapshot
.venv/bin/python checkpoint_bundle.py verify work/checkpoint-bundles/my-snapshot
```

### RLHF data quality

Reward/DPO training groups validation by normalized prompt so a prompt never appears in both splits, and excludes duplicate, contradictory, identical-response, and oversized comparisons — without modifying the underlying preference database. At least four usable comparisons across two prompts are required. New reward checkpoints are checked for tokenizer compatibility by RAFT, PPO, and DPO alike.

### Chat tools: calculator and recall

Chat handles complete arithmetic expressions (parentheses, operator precedence, negative numbers, exact decimals — try `Calculate (2+3)*4`) and note recall (`/recall engine` or `search my notes about engine`, ranked with BM25 scoring, missing matches reported explicitly). Both work without a loaded checkpoint and are labeled distinctly from ordinary model answers. Retrieval is local word matching, not semantic embeddings — notes are never created automatically from conversations.

### Reproducible retries

`run_resilient.sh` assigns a stable budget ID to each chunk; checkpoints preserve that chunk's absolute target, so retrying after a saved partial run finishes the remaining steps instead of adding the full requested count again. Checkpoints also save CPU and active-backend random state — moving between backends restores CPU sampling state and records the change, without promising identical floating-point results across hardware.

### External evaluation

```sh
.venv/bin/python eval_heldout.py checkpoints/latest.pt \
  --output work/external-evaluation.json --device cpu
```

Reports include checkpoint/tokenizer/source hashes, exact target count, and loss. Capability corpus-overlap checks stream normalized text rather than loading full corpora into memory.

## Files

- `brain.py` — embeddings, causal attention, Transformer blocks, KV-cached generation.
- `train.py` — data split, optimization, validation, checkpointing, tokenizer training/reuse, W&B + ledger logging, sweep mode.
- `tokenizer.py` — from-scratch byte-pair encoding, trained on this project's own corpus.
- `make_instructions.py` — generates the programmatic instruction-tuning corpus.
- `preferences.py`, `reward_model.py`, `train_reward.py`, `raft.py`, `ppo.py`, `dpo.py`, `value_model.py` — the full RLHF pipeline, four real methods against the same real preference data.
- `server.py`, `ui.html`, `chat.html` — local dashboard, preference-labeling panel, and chat interface.
- `memory.py` — SQLite-backed notes storage, separate from model weights.
- `sources.py` — training-data workspace management and corpus compilation.
- `checkpoint_bundle.py` — portable, checksummed checkpoint export/verify.
- `checkpoints/latest.pt` — trained weights, optimizer state, and tokenizer (`checkpoints/tokenizer.json`).
- `runs/ledger.jsonl` — one-line-per-run summary log.
- `research/TASKS.md` — ongoing handoffs, open bugs, and honest findings, in order.
- `WRITEUP.md` — the full narrative: what was built, what broke, and what fixing it taught.
