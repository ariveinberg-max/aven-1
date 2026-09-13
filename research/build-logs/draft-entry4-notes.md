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

## One number for the headline
18 AI-labeled comparisons added with zero manual clicks, gated by an actual measurement (not assumption) of where an AI judge can be trusted — and one of those 18 still needed a human catch. Neither "AI can't help at all" nor "AI can replace this entirely" was true; the real answer was scoped and specific.
