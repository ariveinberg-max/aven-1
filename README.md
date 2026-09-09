# My AI Brain

A working first step toward your own AI: a 5,055,360-parameter byte-level Transformer, trained from random weights. No pretrained model, API key, or cloud inference. PyTorch provides tensor math, automatic differentiation, and AdamW; the architecture, training loop, checkpoint handling, and interface are in this project. This is not a general assistant, conscious brain, or a model with voice, vision, web access, or persistent conversational memory.

## Open it

Double-click **Start Brain.command** in this folder. Keep the Terminal window open while using the app. Visit http://127.0.0.1:8765 if the browser does not open. Close the server with Control-C in its Terminal window; closing a browser tab does not stop training.

The first verified checkpoint is already available. Click **Generate text** to try it. Click **Resume training** to add the chosen number of training steps. **Pause & save** finishes the current step before saving. The app runs only on this computer.

If macOS does not launch the command file, run these commands in Terminal:

```sh
cd /Users/ariveinbers/Documents/Codex/my-ai-brain
.venv/bin/python server.py
```

## What it learns

It predicts the next UTF-8 byte from up to 128 preceding bytes. Four causal Transformer blocks use five attention heads and a hidden width of 320. Batch size 8 and float32 are conservative starting defaults for an 8 GB Mac. GPU availability is checked at runtime; training selects MPS when available. Generation uses CPU to keep the interface simple and memory use modest. Close other heavy apps if your Mac experiences memory pressure.

The included `data/demo.txt` contains original, programmatically composed practice stories. Many share a template. They are a pipeline test, not a broad knowledge source. The last 10% of bytes are held out of gradient updates, but the similar templates make this an easy validation set. Falling loss on it does not establish general intelligence, useful conversation, or generalization to unrelated text. Output after a short run is expected to be rough.

The diagram shows actual final-position hidden-state magnitudes after generation, grouped into 32 nodes per block (10 channels per node). Brightness is normalized separately within each block. The connecting lines are a schematic, not a display of individual learned weights. It is not a live scan of thoughts.

## Train on your own text

Use UTF-8 plain text that you own, wrote, or have permission to train on. The starter accepts 4 KB–20 MB. More diverse, clean text is more useful than simply repeating a small document. It reads the entire corpus into memory, so the file-size limit is intentional.

To start a new run while preserving the current one, first pause training, wait for it to finish, and stop the server. Rename `checkpoints` to an unused name such as `checkpoints-demo-backup`. Then save your text as `data/training.txt` and reopen the app. The app will create new random weights for the next training session. The UI also accepts pasted text before a checkpoint exists. Keep backups private if your training text contains personal material.

To continue a CLI run, use the same corpus path every time:

```sh
.venv/bin/python train.py --data data/training.txt --steps 200 --resume
```

For the included demo, omit `--data`. A checkpoint records the corpus hash and rejects a different corpus on resume. It preserves model weights, optimizer state, step count, and CPU random-generator state. Checkpoints save every 20 steps, at normal completion, and on a handled pause. Force-quitting or losing power can lose steps since the previous save. Run one training process at a time; use the panel or CLI, not both simultaneously.

## Training-data workspace vs. stored memory vs. learned weights

The panel now separates three things that are easy to conflate:

- **Learned weights** (`checkpoints/latest.pt`): numbers adjusted by gradient descent. This is the only thing that makes the model's text generation change. You cannot edit these directly; you can only change them by training.
- **Training-data workspace** (`data/sources/*.txt`, managed from the "Training-data workspace" panel): the plain text a *future* run would learn from. Add, edit, or delete named sources here, then click **Compile into training corpus** to concatenate them into `data/training.txt`. Once a checkpoint exists, the corpus is locked (same rule as before) — compiling, saving, or deleting a source is refused until you preserve `checkpoints/` and start a new run.
- **Memory** (`memory.db`, managed from the "Memory" panel): a plain SQLite table of notes you write, edit, and delete directly through the UI. It is not read by training or generation, and saving a note never changes a single model weight. It exists purely as external, inspectable storage — the opposite of the model's opaque learned parameters.

Deleting a memory record or a workspace source is permanent immediately; there is no undo.

## Tracking training runs

- **Weights & Biases** (cloud, free tier, project `aven-1`): pass `--wandb` to `train.py` for full run tracking at https://wandb.ai/ariveinberg-ari-research/aven-1. This uploads run metrics and config — not your training text — to wandb.ai's servers:
  - **Loss/perplexity**: train loss, held-out loss, held-out perplexity, every 20 steps
  - **Optimizer health**: gradient norm (pre-clip) and total weight norm — the two numbers that would have caught an unstable learning rate immediately, instead of discovering it hours later from bad text output
  - **Gradient/weight histograms** per layer (`wandb.watch`, every 100 steps) — see exactly which layer is diverging, not just an aggregate number
  - **Throughput**: tokens/sec
  - **Sample generations table**: a fixed set of probe prompts (greeting, arithmetic, antonym, calendar, identity), regenerated every 100 steps and logged as a table — scroll through training history and watch actual output quality change, instead of manually re-testing after the fact
  - **Run config**: architecture, dataset name/size/token count, tokenizer compression ratio, learning rate, batch size — everything needed to reproduce or compare a run
  - **System metrics** (CPU/memory) — captured automatically by W&B, no code needed

  One-time setup:

  ```sh
  .venv/bin/pip install wandb   # already in requirements.txt
  .venv/bin/wandb login          # opens a browser to create a free account / paste an API key
  .venv/bin/python train.py --data data/training.txt --resume --steps 200 --wandb
  ```

  The panel's **Resume training** button auto-adds `--wandb` once you've logged in (detected via `~/.netrc`), so runs launched from the browser log too.

- **Run ledger** (`runs/ledger.jsonl`): one line per finished, paused, or errored run — architecture, dataset, final perplexity, timestamp. Plain JSON Lines, always written locally regardless of W&B, for a quick offline "what have I actually tried" scan.

`checkpoints/status.json` still drives the live panel in the browser; W&B and the ledger are for comparing runs after the fact.

## Hyperparameter sweeps (Weights & Biases)

`train.py --sweep` runs an isolated, throwaway training trial under `sweeps/` — it never touches `checkpoints/`, reuses an existing tokenizer via `--tokenizer-path` (no re-training BPE per trial), and caches the encoded corpus so many short trials over the same data are fast. Combined with a W&B Sweep, this gives you the multi-run comparison views (parallel coordinates, scatter plots) on the Sweeps tab of your project:

```sh
.venv/bin/wandb sweep sweep.yaml       # prints a sweep ID
.venv/bin/wandb agent <entity>/aven-1/<sweep-id>
```

`sweep.yaml` searches learning rate, dropout, and batch size on a small, fast architecture (128-wide, 2 layers) over the instruction corpus, minimizing held-out perplexity. Stop the agent (Ctrl-C) whenever you have enough runs — each trial's leftover files live under `sweeps/run-*/` and can be deleted freely; they're not your real model.

## Files

- `brain.py`: embeddings, causal attention, Transformer blocks, and generation.
- `train.py`: data split, optimization, validation, checkpointing, tokenizer training/reuse, W&B + ledger logging, sweep mode.
- `tokenizer.py`: from-scratch byte-pair encoding, trained on your own corpus.
- `make_instructions.py`: generates the original, programmatic instruction-tuning corpus (`data/instructions.txt`).
- `server.py` and `ui.html`: local control panel, chat panel, and measured activity display.
- `memory.py`: SQLite-backed notes storage (`memory.db`), separate from model weights.
- `sources.py`: training-data workspace file management (`data/sources/`) and corpus compilation.
- `sweep.yaml`: Weights & Biases sweep config (learning rate / dropout / batch size search).
- `checkpoints/latest.pt`: your trained weights, optimizer state, and tokenizer (`checkpoints/tokenizer.json`).
- `checkpoints/status.json`: progress and loss history for the live panel.
- `runs/ledger.jsonl`: one-line-per-run summary log.
- `sweeps/`: throwaway trial checkpoints and the encoded-corpus cache used by `--sweep`; safe to delete.
- `work/train.log`: messages from training launched in the panel.
- `test_brain.py`: causal-mask, next-byte-target, learning, and generation tests.

## Recreate the environment

Use an Apple Silicon Python 3.11 environment:

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest -v
```

The installed dependencies are pinned in `requirements.txt`. The first setup downloads software packages; running and training the model does not require internet access. No model weights or outside training datasets are downloaded.

GPU implementation reference: https://docs.pytorch.org/docs/stable/notes/mps.html

## Next development stages

Improve the corpus and evaluate on a separate, genuinely different text collection. Then work on tokenization, longer context within the memory budget, and dialogue-specific training/evaluation. Vision and audio would need appropriate encoders, datasets, and objectives; actions require a separate tool-control system. Merely increasing the step count on these demo stories will not add those abilities.


## Expanded W&B analytics

See WANDB.md for the metric catalog and instructions. Add --wandb to your normal training command; --wandb-samples enables generated-text uploads.
