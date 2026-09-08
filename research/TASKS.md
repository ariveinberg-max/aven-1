# Aven research task board

Goal: develop Aven into a reproducible research project supporting a future AI research company.

| Workstream | Owner | Status | Task |
|---|---|---|---|
| Pretraining | Claude | In progress, reported by user; live state not verified | [Baseline and handoff](https://github.com/ariveinberg-max/aven-1/issues/1) |
| Evaluation | Codex | Planned; awaiting frozen baseline | [Independent measurements](https://github.com/ariveinberg-max/aven-1/issues/2) |
| Research results | User, supported by both assistants | Planned; depends on measurements | [First report](https://github.com/ariveinberg-max/aven-1/issues/3) |

[W&B experiments](https://wandb.ai/ariveinberg-ari-research/aven-1)

## Coordination

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
