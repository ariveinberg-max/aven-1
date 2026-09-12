# Draft notes — build-log entry 2 (Day 3 RLHF cycle)

Not published yet. Raw material for entry 2, per Day 5's task: "Day 3 RLHF
cycle results, reported as measured numbers."

## Reward model retrain
- Expanded from 88 usable comparisons (prior model) to 112 usable (88 train /
  24 val) after dedup/conflict filtering — 555 raw logged, 266 decided
  (non-tie), 112 survived after removing 94 duplicates + 53 conflicting.
- Held-out accuracy: 54.17% initial → noisy climb (58/87/83/79/67%) → 75%
  final. Essentially flat vs. the prior model's 76.47% despite more data.
- Real takeaway: more raw labels didn't clearly help, because most of the
  growth was duplicates/conflicts, not new signal. Matches the reading-block
  finding that quality/dedup matters more than volume.

## DPO comparison (built same session, not part of "Day 3" per se, but same
## underlying data — worth folding in for contrast)
- Same 112-comparison data, no separate reward model needed.
- Held-out accuracy converged cleanly to 79.17% by step 100, held flat
  through 300 — much more stable than the reward model's noisy run.
- But: 3/4 live-checked generations were byte-identical to pre-DPO; the one
  that changed picked up a minor artifact. Metric stability didn't translate
  to better generations. Not promoted.

## RAFT run (promoted)
- 19 candidates generated from the retrained reward model, from the
  confirmed-stable step-5499 checkpoint.
- Manually inspected all 19 before training — found and hand-fixed one
  broken winner ("Hey there" → nonsensical arithmetic non-answer the reward
  model picked as "least bad" of 4 weak candidates).
- Trained via isolated --init-from/--output-dir (never touched the trusted
  checkpoint during the run itself) — 300 steps, response-only loss.
- Live-checked 7 prompts post-training: 6/7 byte-identical to pre-RAFT, the
  1 that changed was exactly the intended fix.
- compare_capabilities.py vs. frozen v1 suite: 0 regressions, 0 changed
  cases (suite doesn't cover greeting-style prompts — expected null result).
- Promoted to checkpoints/latest.pt. Pre-RAFT state preserved at
  checkpoints-pre-raft-2026-09-12-backup/.

## Side effect worth reporting honestly
- RAFT's confident single answers collapsed the preference-labeling tool's
  pair generation: 29/35 next comparisons came back byte-identical, nothing
  left to label on the affected prompts.
- Fixed in two rounds: (1) swapped the 6 directly-RAFT-trained prompts for
  untested phrasings from the same safe categories, widened/equalized
  sampling temperature; (2) that still tied often in practice, so added
  server-side retry (up to 6 attempts) with a cheap plausibility filter,
  until a pair actually differs and isn't obviously broken.
- Real result after both fixes, tested live: divergence went from ~1/3 to
  5/6 across real test prompts.

## One number for the headline
Before → after this cycle: reward model accuracy roughly flat (76% → 75%),
but the actual live model output changed exactly once, correctly, on a
real live-tested bug — small, verified, honest, not a big before/after
number to lead with.
