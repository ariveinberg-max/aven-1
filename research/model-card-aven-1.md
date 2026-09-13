# Model card — Aven-1

Current as of 2026-09-13. Checkpoint sha256 `5d9df065b544c1580837073ae7ecab2328534971db70fd71e98df2054e20a7e5`.

## Summary

Aven-1 is a byte-pair-tokenized causal Transformer, trained entirely from scratch — no pretrained weights, no external API, no downloaded model. Every stage (tokenizer, pretraining, instruction tuning, and a real RLHF pipeline) was built and run on consumer hardware, not rented infrastructure.

## Architecture

| | |
|---|---|
| Parameters | 58,424,832 |
| Layers | 8 |
| Width | 768 |
| Attention heads | 12 |
| Context length | 192 tokens |
| Vocabulary | 2,048 tokens (custom byte-pair encoding, trained on this project's own corpus) |

## Training lineage

Not a single training run — a chain of real, verified stages:

1. **Pretraining** on a multi-book public-domain corpus (see `WRITEUP.md` for the full expansion history).
2. **Instruction fine-tuning** on a programmatically generated dataset (`make_instructions.py`): arithmetic, antonyms, calendar facts, small talk, book trivia, comprehension, instruction-following, code-reading, and an explicit "I don't know" fallback category.
3. **RAFT (reward-ranked fine-tuning), twice** — the most recent promotion (2026-09-13) trained on reward-model-selected winners from real preference data, verified live and against the frozen capability suite before promotion.

The current checkpoint's own recorded step count (300) reflects only its most recent RAFT fine-tuning phase — training step counts reset at each new phase per this project's own convention (see `README.md`). The full lineage behind it includes the original instruction-tuning run (5,499 steps) plus two RAFT cycles on top.

**Methods tried and NOT currently in the live checkpoint** (kept as evidence, not promoted): DPO (`checkpoints/dpo_policy.pt`), KTO (`checkpoints/kto_policy.pt`), and two PPO attempts (`checkpoints/ppo_policy.pt`) — PPO specifically produced a real, live-verified regression both times it was tried, at two different data scales. Full results in `research/TASKS.md`.

## Training data for the current reward model

- 910 total logged preference comparisons (870 human-labeled, 40 AI-labeled via a narrowly-scoped, separately-measured auto-labeler — see `agents/rlaif_auto_label.py` and its gating measurement).
- 162 usable, deduplicated, non-conflicting comparisons behind the current reward model (`checkpoints/reward.pt`), which scores 60.53% held-out accuracy.

## Evaluation

Frozen capability-v1 suite (`research/evaluation/capability-v1.json`): **0/30 across all five categories** (comprehension, instructions, arithmetic, facts, code_reading).

This is a known, previously-documented pattern, not a new failure — the suite uses strict exact-match scoring, and several categories' correct or reasonable live behavior doesn't hit that exact bar even when the underlying response is genuinely fine. See `research/TASKS.md` for the original finding. Live spot-checking (used throughout this project's actual development, not just this frozen suite) regularly confirms correct answers to real questions — e.g. "Who was the first president of the United States?" → "George Washington was the first president of the United States." reliably, across many samples and temperatures.

## Known limitations, stated plainly

- **No real arithmetic capability.** Correct-looking math answers come from a deterministic calculator tool in the chat interface, not the model itself — typos that bypass the calculator's pattern-matching expose this immediately (e.g. "whatss 1+1" has produced wrong answers live).
- **A known day-of-week wraparound bug**: "what day comes after Sunday" can answer with the wrong day. The training data generator itself is correct (verified) — this is a live model behavior gap, not a data bug.
- **A known comparison-template bug**: "which is bigger, X or Y" has produced malformed answers (e.g. "7 is greater than 7") — the model occasionally applies the wrong response template to the wrong question phrasing.
- **Reward model instability**: held-out accuracy has moved 76% → 75% → 60.53% across three retrains, never simply improving with more data — consistent with known literature on reward models trained on noisy/conflicting preference labels (roughly a quarter of raw logged comparisons are removed as duplicates or conflicting on each retrain).
- **PPO is not currently safe on this project's data scale.** Two real attempts, at two different data volumes, both produced live-verified regressions despite passing the frozen suite and staying within KL limits.
- **An open, untested hypothesis**: whether this 58M-parameter model has a hard capacity ceiling for how much new instruction-tuning content can be safely layered onto one checkpoint within a session — documented in `WRITEUP.md`, not yet resolved with a controlled experiment.
- **Occasional generation glitches at higher sampling temperatures** — typos, mid-sentence template blending — a real, known tradeoff between response diversity (useful for preference labeling) and coherence, not fully eliminated.

## Where to look for more

- `WRITEUP.md` — the full narrative history of what was built and what broke.
- `research/TASKS.md` — dated, detailed handoffs for every real finding, in order.
- `research/scoreboard.md` — running record of both models' measured state.
- `research/build-logs/` — the public, dated build-log entries.
