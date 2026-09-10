# Aven research task board

Goal: develop Aven into a reproducible research project supporting a future AI research company.

| Workstream | Owner | Status | Task |
|---|---|---|---|
| Pretraining | Claude | 58M-param baseline frozen at step 2100 (perplexity 172.07), ready for evaluation; separate 150M run starting | [Baseline and handoff](https://github.com/ariveinberg-max/aven-1/issues/1) |
| Evaluation | Codex | Capability-v1 suite run against 58.4M finetune (step 5000): 0% on all 5 categories, zero training-data overlap confirmed. See WRITEUP.md. | [Independent measurements](https://github.com/ariveinberg-max/aven-1/issues/2) |
| Research results | User, supported by both assistants | Planned; depends on measurements | [First report](https://github.com/ariveinberg-max/aven-1/issues/3) |

[W&B experiments](https://wandb.ai/ariveinberg-ari-research/aven-1)

## Coordination

- **Concurrent-resume hazard (2026-09-09):** Two platforms resumed the same checkpoint/run ID concurrently, causing W&B step-order conflicts and ambiguous lineage. A shared W&B run is not reliable evidence of which platform produced a metric or checkpoint. Before resuming, record the owner/platform, exact checkpoint SHA-256, architecture, starting step and output location; allow only one active writer per run ID/output directory. Parallel continuations must use distinct W&B execution IDs and output folders while recording their common parent checkpoint. Treat folder labels as descriptions, not identity, and verify a durable checkpoint export before ending a cloud session.
- **Rented-GPU persistence hazard (2026-09-09):** A RunPod session cloned the repo and trained to a real checkpoint (153M params, step 20540, perplexity 84.88) on the pod's default container disk, not its persistent `/workspace` volume. Stopping and resuming the pod wiped that disk; the checkpoint was unrecoverable (`find /` across the full filesystem found nothing) even though the pod itself and its billing history still existed. Only the training metrics survived, because they were written into `WRITEUP.md` before the recovery attempt -- the weights did not. **Rule going forward on any rented GPU without a confirmed persistent volume**: download `latest.pt`/`tokenizer.json`/`status.json` periodically *during* a long run (not just at the end), or explicitly clone/train inside the platform's designated persistent-storage path from the start and verify a test file survives a stop/resume cycle before trusting it with a real run.
- **Held-out-eval-wasn't-held-out hazard (2026-09-09):** Every held-out perplexity number reported to date came from `train.py` taking the last 10% of the *same* compiled corpus as "validation" -- same books, same style, just a different slice, not independently-sourced text. Two books (`sign-of-four.txt`, `moby-dick.txt`) were moved into `data/eval-heldout/`, outside `sources.compile_corpus()`'s reach, and `data/training.txt` was recompiled without them; see `eval_heldout.py` and `data/eval-heldout/README.md`. **Rule going forward:** don't report a "held-out" or "validation" metric as evidence of generalization unless the eval data was never compiled into the training corpus at all. `train.py`'s internal 90/10 split is still useful as a training-health signal (is loss dropping), but isn't evidence against overfitting to this specific corpus's style.
- **RLHF pipeline re-verified against the 58.4M finetune (2026-09-09):** `reward_model.py`/`raft.py`/`ppo.py`/`value_model.py` are architecture-agnostic (build from `Config(**saved['config'])`, no hardcoded dims) and were confirmed to construct and load-transfer cleanly against the current `checkpoints/latest.pt`. `server.py`'s `PREFERENCE_PROMPTS` was retested live (two real 0.5/0.9-temperature passes) and trimmed from 19 to 9 prompts — the dropped ones either tied 2/2 (over-drilled, one canned answer) or varied into incoherent non-sequiturs rather than genuine alternative answers. See `WRITEUP.md`'s new RLHF section for full numbers. No preference labels were touched; real human labeling with the refreshed list is still outstanding.
- Read this board and the relevant GitHub issue before working; record the files and branch you intend to change.
- Claude owns the active pretraining changes. Codex owns evaluation and review. Arrange a handoff before changing the other workstream's active files.
- Use separate branches/worktrees for simultaneous edits. Avoid concurrent GPU jobs on the 8 GB Mac.
- Preserve weights, tokenizers, private data, credentials and unrelated uncommitted changes. Do not commit these as part of research reports.
- Record commands, code revision, dataset fingerprints, seed, W&B URL, checkpoint identity and results for each experiment.
- A stored memory is not a learned weight; training loss is not proof of broad intelligence.
- These assignments do not automatically start or message either assistant. The user supplies the issue to the corresponding session.
- Update issue status and this board at handoff. Do not report a task complete until its acceptance criteria are met.

## Handoff format

Owner / issue:
Branch and files changed:
Verified results and commands:
W&B run:
Checkpoint and tokenizer location/hash:
Known limitations:
Next action and owner:
