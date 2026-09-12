# Aven Labs scoreboard

Running record of both models' real, measured state. Append a new dated row
per model each time a real eval sweep is run — don't overwrite history.

## Aven-1 (58.4M, Mac)

| Date | Step | Stage | Capability suite | Reward model | Labeled comparisons | Notes |
|---|---|---|---|---|---|---|
| 2026-09-11 | 5499 | finetune | 0/30 (0% all 5 categories) | — | 500 | Day 1 baseline, pre-RAFT |
| 2026-09-12 | 5499 (RAFT-promoted) | finetune | 0/30 (0% all 5 categories, 0 regressions vs. pre-RAFT) | 112 usable, 75% held-out | 670 | RAFT promoted (see entry2-2026-09-12.md). Checkpoint sha256 84d6a8f2... |

## Aven-2 (153.45M, Windows PC)

| Date | Step | Stage | Held-out perplexity | Capability suite | Notes |
|---|---|---|---|---|---|
| 2026-09-12 | 11260 | pretrain | ~37.9 | 0/30 (0% all 5 categories) | First capability-suite run on Aven-2 this session. 0% is expected and honest — pretrain-only, no instruction fine-tuning, never seen the ### Instruction: format at all. Not comparable to Aven-1's 0% (different cause). |

## Reading this scoreboard honestly

- Both models currently show 0/30 on the frozen capability suite — for **different reasons**. Aven-1's is a known, documented strict-scoring-format issue on categories with real but non-exact-match behavior (see research/TASKS.md). Aven-2's is simply that it has never been instruction-tuned at all. Do not treat these as comparable failures.
- Aven-2's perplexity is a pretraining-loss metric; Aven-1 doesn't have a comparable ongoing pretraining perplexity number since it's past that stage. Compare within a model over time, not perplexity-to-capability-score across models.
- Update this file every time a real eval sweep runs — that's the entire point: a place to see "what have we actually measured," not a place to guess.
