# Draft notes — build-log entry 4

Not published yet. Raw material for the next entry.

## RLAIF, measured before deciding
Ari found manual labeling tedious and asked about switching to RLAIF (AI-judged preferences) entirely. Measured it first rather than guessing: ran a local AI judge (llama3.2:3b) against 30 of Ari's own real labels on the same prompt pairs.

- Overall agreement: 53.3% — barely above chance for a binary choice.
- But the disagreements weren't random: they clustered almost entirely on short small-talk prompts (greetings, farewells) where response A/B are subtly similar. On substantive prompts (help requests, open-ended questions), agreement was close to 100% in the same test.

**Real design decision from that measurement**: split the labeling pool. Small-talk prompts stay human-only (`HUMAN_LABEL_PROMPTS`). Substantive prompts get auto-labeled by a new scoped agent (`agents/rlaif_auto_label.py`, `AI_LABEL_PROMPTS`). Added a `source` column to `preferences.db` (migrates existing rows, defaults to `human`) so AI-labeled and human-labeled data are never silently blended without a record of which is which — matches this project's own long-standing stance that faked preference labels defeat the point of RLHF.

Ran it for real: got 18 AI-labeled rows total across two runs. One was caught during a spot-check picking a genuinely garbled response over a clean one and manually removed — honest confirmation the mechanism isn't perfect, just meaningfully better than chance on the bucket it's scoped to.

## Aven-2: full recovery, real progress
Since the crash-loop fix: step 11,260 → 19,080+ (~7,800 more steps), perplexity down from ~38 to **~24.6** — a large, real, sustained improvement, not just "no longer crashing."

## Reading, folded in
- PyTorch's `weights_only` mechanism (the actual cause of the crash-loop): a real, deliberate security feature, default since torch 2.6. Confirmed the fix followed PyTorch's own documented remediation exactly.
- Reward model overoptimization / Goodhart's Law: real, measured scaling laws showing a policy's true quality can peak and then *degrade* with continued optimization against a reward model, even as the proxy score keeps climbing. Directly relevant given this project's reward model is small (~112 comparisons) and likely has a narrower safe-optimization window than the literature's typical setups — validates why every RAFT/DPO round here has been spot-checked against live generations, not just trusted by metric.

## KTO, built and run for the first time
Third method alongside DPO and the reward-model+RAFT pipeline: KTO (Kahneman-Tversky Optimization), which trains on binary desirable/undesirable examples instead of paired comparisons. Derived training data directly from existing preference pairs (each splits into one of each label) — 162 pairs → 324 examples, no new labeling UI needed to get started. Caught and fixed a real bug during review before it ever ran: the first draft's reference-point (z0) calculation just reordered already-computed rewards instead of scoring genuinely mismatched prompt/response combinations, which would have made the whole mechanism meaningless. Fixed properly with real mismatched-pair forward passes.

Result: 62.5–87.5% held-out accuracy, noisier than DPO but in the same range. z0 stayed at exactly 0.000 every step — plausible (unrelated responses scoring at or below the reference floor), but not yet fully understood. Saved separately, not yet live-inspected against generations — an honest open item, not a finished result.

## Second reward-model + RAFT cycle
Retrained the reward model on 162 usable pairs (up from 88) — held-out accuracy landed at 60.53%, *lower* than the prior cycle's 75% despite nearly double the data. Reported as-is, not smoothed over.

RAFT's own candidate diversity collapsed independently: average spread dropped from 7.895 to 1.726, with 16 of 19 prompts showing zero spread at all. Root cause: `raft.py` has its own separate temperature schedule that never received the diversity fix already applied to the labeling tool's pair-generation endpoint — same underlying phenomenon (a confident model stops diverging at low temperature), hitting a second, independent piece of code. Logged as a real, not-yet-fixed gap.

All 19 candidates were clean this round (no hand-fix needed, unlike last time). Trained via the isolated path, live-checked 8 prompts (7/8 identical), and verified 0 regressions — against a freshly-regenerated same-code baseline, since `compare_capabilities.py`'s own code-version check correctly rejected comparing against the prior baseline (KV-cache and numpy fixes had landed in between). Promoted.

## PPO revisited, on paper
Reviewed this project's own prior PPO/ptx-coefficient experiment (documented in `WRITEUP.md`): no coefficient tested there both avoided regression and produced real improvement, blocked by having only 34 real comparisons. Now at 180+ (162 human-usable + 18 AI-labeled at the time) — roughly 5x more data, and `WRITEUP.md`'s own recommended next step was explicitly to revisit PPO once more data existed. Logged as a real, no-longer-blocked candidate for a future attempt — not run yet.

## One number for the headline
Three real RLHF methods now exist side by side on this project (reward-model+RAFT, DPO, KTO), each tested against the same real data, each with honest results reported rather than the best one cherry-picked. The reward model's accuracy went 76% → 75% → 60.53% across three retrains — never simply climbing with more data. That instability, reported plainly every time, is the actual finding worth leading with.
