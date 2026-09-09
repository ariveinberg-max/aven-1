# Aven-1 tracking with Weights & Biases

Training analytics are enabled by `--wandb` in train.py. Existing project and
checkpoint files are preserved. The online verification uses a separate tiny
sweep model and the original demo corpus, not your main model.

## Use it

From this project directory, append `--wandb` to your usual training command.
Keep the same data, architecture, resume/finetune settings you normally use.
Do not substitute demo.txt when resuming a checkpoint trained on another corpus.
The terminal prints a clickable W&B run URL. Project defaults to `aven-1`.
Use `--wandb-project NAME` and optionally `--wandb-entity TEAM` to choose another
destination. Existing dashboard training already adds --wandb when its existing
credential check detects W&B login.

If authentication expires, run `.venv/bin/wandb login` in Terminal, not in chat.
`WANDB_MODE=offline` logs locally; use `wandb sync <offline-run-folder>` to upload
later. Offline sessions do not have live cloud charts or W&B resume semantics.

## Metrics

- **Learning:** last training-batch loss, interval-mean training loss, held-out
  loss, train/held-out perplexity, and held-out minus interval-train loss.
- **Prediction:** training-batch token accuracy, held-out token accuracy,
  held-out top-5 accuracy, and predictive entropy in nats.
- **Optimization:** learning rate, pre-clipping gradient norm, interval mean
  gradient norm, fraction of interval steps clipped at norm 1, weight norm,
  component gradient norms after clipping, and component weight norms.
- **Distributions:** W&B watch logs parameter/gradient histograms every 100
  backward passes. Shorter sessions may have no histograms. Hook histograms
  observe gradients during backward, before clipping.
- **Performance:** average session tokens/sec including evaluation/save overhead,
  training-interval tokens/sec, mean training-step milliseconds, evaluation
  seconds, and elapsed session seconds.
- **Progress:** optimizer step, target step, session token count, equivalent
  sampled passes through training data, checkpoint size, and final status.
- **Memory:** process peak RSS; MPS allocated/driver memory on Apple Silicon;
  CUDA allocated/peak memory on CUDA. Unsupported hardware metrics are omitted.
  W&B can additionally collect system metrics available on the host; Apple GPU
  utilization and temperature are not promised.
- **Reproducibility:** architecture, tokenizer vocabulary size, corpus SHA-256,
  token counts/split sizes, optimizer settings, device, Python/PyTorch versions,
  seed, stage, and dataset filename. Corpus contents are not uploaded as an artifact.
- **Optional samples:** add `--wandb-samples` for a table of five fixed prompts
  every 100 steps and on normal completion. Greedy generation avoids consuming
  sampling RNG and keeps prompts comparable. Generated text may reproduce your
  corpus, so sample upload is explicit. No model checkpoint uploads are enabled.

All scalar metrics use `step` as their chart x-axis. Resume keeps the checkpoint's
run identity; fresh runs get distinct IDs. Best held-out loss and best held-out
accuracy appear in summaries. Config values can change on resume; learning rate
is also logged as a time series. Avoid two writers using the same run ID.

Evaluation uses four fixed sampled validation batches. This is a diagnostic,
not a full-corpus benchmark; repeated windows and similar train/validation text
can make it optimistic. Training accuracy is for the last stochastic batch.
The train/validation gap compares an interval of training-mode losses with an
eval-mode held-out sample, not matched full-dataset losses. Perplexity is exp(loss)
capped at exp(20), and should only be compared across the same tokenizer and data.
Equivalent passes count randomly sampled tokens, not complete shuffled epochs.
Metrics do not establish intelligence, factual accuracy, or instruction following.

## Verification

`python -m unittest test_tracking -v` checks metric math and non-mutating diagnostics.
Isolated offline and online tiny-model runs exercise logging, sample tables and
checkpoint summaries. The online test runs for 100 steps to exercise histograms.

Reference: https://docs.wandb.ai/models/ref/python/functions/init
