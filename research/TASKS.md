# Aven research task board

Goal: develop Aven into a reproducible research project supporting a future AI research company.

| Workstream | Owner | Status | Task |
|---|---|---|---|
| Pretraining | Claude | 58M-param baseline frozen at step 2100 (perplexity 172.07), ready for evaluation; separate 150M run starting | [Baseline and handoff](https://github.com/ariveinberg-max/aven-1/issues/1) |
| Evaluation | Codex | Planned; awaiting frozen baseline | [Independent measurements](https://github.com/ariveinberg-max/aven-1/issues/2) |
| Research results | User, supported by both assistants | Planned; depends on measurements | [First report](https://github.com/ariveinberg-max/aven-1/issues/3) |

[W&B experiments](https://wandb.ai/ariveinberg-ari-research/aven-1)

## Coordination

- **Concurrent-resume hazard (2026-09-09):** Two platforms resumed the same checkpoint/run ID concurrently, causing W&B step-order conflicts and ambiguous lineage. A shared W&B run is not reliable evidence of which platform produced a metric or checkpoint. Before resuming, record the owner/platform, exact checkpoint SHA-256, architecture, starting step and output location; allow only one active writer per run ID/output directory. Parallel continuations must use distinct W&B execution IDs and output folders while recording their common parent checkpoint. Treat folder labels as descriptions, not identity, and verify a durable checkpoint export before ending a cloud session.
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
