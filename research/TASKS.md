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
- **Recurring memory-pressure hazard (2026-09-09):** A second background job (full Wikipedia extraction) was killed by macOS for system-wide low memory, same root cause as the original pretraining crash: many other apps (especially multiple Chrome helper processes at 500MB-800MB+ each) competing for this Mac's 8GB RAM. Before starting any long-running local job, check for and close unnecessary running apps first, not just verify the job's own memory usage in isolation. Also noted: `xml.etree.ElementTree.iterparse` over a multi-million-element document needs the root element's children released periodically (not just each processed element's own `.clear()`), or the root's growing reference list leaks memory over a long run -- not yet fixed, since discovered only after the crash.
- **Rented-GPU persistence hazard (2026-09-09):** A RunPod session cloned the repo and trained to a real checkpoint (153M params, step 20540, perplexity 84.88) on the pod's default container disk, not its persistent `/workspace` volume. Stopping and resuming the pod wiped that disk; the checkpoint was unrecoverable (`find /` across the full filesystem found nothing) even though the pod itself and its billing history still existed. Only the training metrics survived, because they were written into `WRITEUP.md` before the recovery attempt -- the weights did not. **Rule going forward on any rented GPU without a confirmed persistent volume**: download `latest.pt`/`tokenizer.json`/`status.json` periodically *during* a long run (not just at the end), or explicitly clone/train inside the platform's designated persistent-storage path from the start and verify a test file survives a stop/resume cycle before trusting it with a real run.
- **Held-out-eval-wasn't-held-out hazard (2026-09-09):** Every held-out perplexity number reported to date came from `train.py` taking the last 10% of the *same* compiled corpus as "validation" -- same books, same style, just a different slice, not independently-sourced text. Two books (`sign-of-four.txt`, `moby-dick.txt`) were moved into `data/eval-heldout/`, outside `sources.compile_corpus()`'s reach, and `data/training.txt` was recompiled without them; see `eval_heldout.py` and `data/eval-heldout/README.md`. **Rule going forward:** don't report a "held-out" or "validation" metric as evidence of generalization unless the eval data was never compiled into the training corpus at all. `train.py`'s internal 90/10 split is still useful as a training-health signal (is loss dropping), but isn't evidence against overfitting to this specific corpus's style.
- **RLHF pipeline re-verified against the 58.4M finetune (2026-09-09):** `reward_model.py`/`raft.py`/`ppo.py`/`value_model.py` are architecture-agnostic (build from `Config(**saved['config'])`, no hardcoded dims) and were confirmed to construct and load-transfer cleanly against the current `checkpoints/latest.pt`. `server.py`'s `PREFERENCE_PROMPTS` was retested live (two real 0.5/0.9-temperature passes) and trimmed from 19 to 9 prompts — the dropped ones either tied 2/2 (over-drilled, one canned answer) or varied into incoherent non-sequiturs rather than genuine alternative answers. See `WRITEUP.md`'s new RLHF section for full numbers. No preference labels were touched; real human labeling with the refreshed list is still outstanding.
- **8GB-Mac memory headroom must be checked before any local training run, not just at job start (2026-09-10):** `top -l 1 -s 0` showed only ~81MB unused (7.5GB used, 2.4GB in the compressor) with several Chrome helper processes and the Claude desktop app running — the same starvation pattern that killed two earlier jobs per the entries above. A planned local finetune was deliberately not started under these conditions; see handoff below. Rule: run `top -l 1 -s 0 | grep PhysMem` (or equivalent) immediately before starting any local train.py invocation, not only before long unattended jobs — a low-headroom system will fail the same way whether the job is short or long.
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

## Latest handoff (2026-09-10)

Owner / issue: Claude / RLHF preference-labeling gap (issue #3 support work)
Branch and files changed: main, `make_instructions.py` (commit 0c88002), `.claude/agents/aven-researcher.md` (commit 6664102)
Verified results and commands: Added `OPEN_ENDED` category (9 opinion/explanation topics x 3 hand-written multi-sentence answers x 4 phrasing wrappers) to fix the tie/incoherence problem documented in `WRITEUP.md`'s RLHF section — the model had zero trained multi-reply substance on open-ended prompts. Also fixed `continuation_examples()`, which was calling `.read_text()` on the full 6.95GB `data/training.txt` (safe at the old 354MB size, not at current size) — bounded to a 20MB read. Ran `python3 make_instructions.py` locally: 17,400 instruction blocks, 1.89MB, peak RSS ~340MB, 0.58s. `checkpoints/` backed up to `checkpoints-finetune-v5-backup` in preparation for a finetune run.
W&B run: none this session (no training run yet)
Checkpoint and tokenizer location/hash: `checkpoints/latest.pt` unchanged (step 5000, data_sha256 `56fc5282...`) — NOT yet finetuned on the new `data/instructions.txt`
Known limitations: The new open-ended training data exists but has not been trained into the model yet — a local finetune (`train.py --data data/instructions.txt --finetune --loss-mode response`) was deliberately deferred because this Mac had only ~81MB free RAM at the time (see hazard log above). `PREFERENCE_PROMPTS` in `server.py` also still excludes open-ended prompts (trimmed out 2026-09-09 for lacking trained variety) — should be revisited once the model is actually finetuned on the new category and verified live to produce coherent, varied answers.
Next action and owner: (1) User/Claude: once this Mac has real free memory, run the finetune above, verify live generation on a few OPEN_ENDED-style prompts is coherent, then add those prompts back into `PREFERENCE_PROMPTS`. (2) User: resume RLHF preference labeling (12 decided / 18 tied so far) once real trained variety exists to label. (3) User: check the Windows PC (192.168.68.65) 153M pretraining run once back on its network — unreachable from outside the LAN as of this session.

## Follow-up handoff (2026-09-10, same day)

Owner / issue: Claude / live chat qualitative bug found by hand testing
Branch and files changed: main, `make_instructions.py`, `WRITEUP.md` (uncommitted at time of writing — see next commit after this entry)
Verified results and commands: Live-tested the dashboard Chat tab (screenshot review) and found a real fact-splicing bug: "who was the first president" answered "George Water freezes at 0 degrees Celsius." — the model blended two unrelated memorized facts mid-sentence after 3 unrelated turns of context (greeting, declined math request, wrong arithmetic). Traced root cause: NOT prompt truncation (`prompting.py`'s `build_chat_prompt()` never cuts a turn mid-sentence), but that `make_instructions.py`'s multi-turn training data only ever generated exactly 2-turn conversations before this fix — near-zero training exposure to answering correctly with 3+ turns of unrelated clutter in context, which is the real shape of live chat usage. Also found (but did not fix, see limitations): the `whatss 1+1` typo bypassed `tools/calculator.py`'s exact-phrase regex and fell through to the untrained model, which then answered "1 plus 1 is 14" — confirming (again) no real arithmetic capability, and separately exposing a tool-routing fragility. Full detail and the exact prompt/reply table in `WRITEUP.md`'s new "Live chat qualitative failures" section.
W&B run: none (no training run this entry either)
Checkpoint and tokenizer location/hash: unchanged, `checkpoints/latest.pt` step 5000 — this fix has NOT been trained in yet
Known limitations: `build_two_turn()` generalized to `build_multi_turn()` (2-4 turns instead of fixed 2, `target_multi` 2200 -> 4000) as a hypothesis-driven data change — **unverified**. It has not been finetuned or re-tested against the exact four prompts that exposed the bug; do not report this as fixed until that happens. The calculator-typo-routing gap (`tools/calculator.py`/`server.py`) was deliberately left untouched since both files had unrelated uncommitted work in progress at the time — needs its own owner and pass.
Next action and owner: (1) User/Claude: once real free memory exists on this Mac, run the finetune, then manually re-send the exact four prompts from the qualitative table in `WRITEUP.md` and confirm whether the splicing is actually reduced — this is a real test, not a formality. (2) Whoever owns `server.py`/`tools/calculator.py` next: consider loosening `expression_from_message()`'s prefix regex to tolerate typos like "whatss", or fuzzy-match known prefixes, so a typo doesn't silently skip the deterministic calculator.
