# Building Aven-1: a language model from random weights

Aven-1 is a causal Transformer trained from scratch — no pretrained weights, no external API, no downloaded model — on an 8 GB M2 MacBook Air. This is a write-up of what was actually built and, more usefully, what broke along the way and what fixing it taught.

## What's in the pipeline

Every stage a real language model goes through, built by hand at a scale a laptop can run:

- **A from-scratch byte-pair encoding tokenizer** (`tokenizer.py`) — the same merge-the-most-frequent-pair algorithm GPT-2 uses, trained on the project's own corpus, not downloaded.
- **Pretraining** on six public-domain books (~2.2 MB: *Alice in Wonderland*, *Pride and Prejudice*, *Sherlock Holmes*, *Frankenstein*, *The Time Machine*, *A Christmas Carol*), teaching general English structure.
- **Instruction fine-tuning** on a programmatically generated dataset (`make_instructions.py`) — arithmetic, antonyms, calendar facts, small talk, book trivia, sentence continuation, and an explicit "I don't know" fallback category — teaching the model to answer instead of just continuing text.
- **A chat interface** (`chat.html`) and a full training dashboard (`ui.html`) — sidebar navigation, live loss charts, a training-data workspace, and SQLite-backed notes storage kept explicitly separate from the model's learned weights.
- **Real experiment tracking**: Weights & Biases logging loss, held-out perplexity, gradient norm, weight norm, per-layer gradient/weight histograms, throughput, and a live table of sample generations that updates during training.
- **A hyperparameter sweep** (`sweep.yaml`) — 31 automated runs searching learning rate, dropout, and batch size, with W&B's parallel-coordinates view showing learning rate as the dominant factor for held-out perplexity.
- **A free cloud GPU path** (`Aven-1-Colab.ipynb`) — the identical training code running on a Colab T4 GPU instead of the M2's CPU/MPS, verified end to end: trained remotely, downloaded, and continued locally without incident.
- **RLHF, in progress** (`preferences.py`, `reward_model.py`, `train_reward.py`, `raft.py`) — real human preference collection, a reward model, and RAFT (reward-ranked fine-tuning); see below for what building it against real, insufficient data revealed immediately.

Current model: 19.8M parameters (512-wide, 6 layers, 8 heads, 192-token context, 1,536-token vocabulary), fine-tuned through five iterations on progressively larger and more diverse instruction data (6,000 → 17,400 examples).

## What actually broke, and what it taught

The interesting part isn't that this worked — it's the failures, because each one is a real, generalizable lesson rather than a tutorial step.

**Hyperparameters don't transfer across model scale.** The sweep found a strong learning rate (~0.003) on a small 601K-parameter proxy model used for fast search. Applying that same rate directly to the real 19.8M-parameter model caused visible instability — perplexity climbed instead of falling, and the resulting checkpoint generated garbled text. The fix wasn't more tuning; it was recognizing that a learning rate optimal for one model size can actively destabilize a larger one, and reverting to a conservative, previously-validated rate. This is a known phenomenon in the literature (motivating techniques like µP for hyperparameter transfer) — encountering it firsthand, by breaking a model and diagnosing why, is a different kind of understanding than reading about it.

**A model that "doesn't know" an answer is a data problem, not a model problem.** Early on, asking the model anything outside its training patterns caused it to hallucinate a garbled, malformed response — it had no way to express uncertainty. The fix was adding an explicit fallback category to the training data: varied unfamiliar-sounding questions paired with an honest "I don't have a reliable answer for that" response. This is the same principle behind why real assistant models are trained to refuse or hedge — it has to be taught, not assumed.

**Category imbalance silently caps what a model learns, even with correct data.** Antonym questions kept returning the fallback answer instead of a real one, despite antonym examples being present in the training set. The cause wasn't a model limitation — it was that antonyms made up roughly 1.5% of the training corpus against arithmetic's ~29%, meaning the model saw them far too rarely to learn the pattern reliably. Reweighting the category (not changing the model at all) fixed it completely.

**Formal training data does not generalize to how people actually type.** The model handled "What is 4 plus 9?" correctly but broke completely on "whats 1+1" or "hello introduce yourself" — real casual phrasing (lowercase, no punctuation, contractions, combined requests) that never appeared in the clean, grammatically formal training examples. The fix was systematic: a `casualize()` transform applied to a majority of training categories, teaching the same correct response regardless of formal or informal phrasing.

**Small models memorize arithmetic; they don't compute it.** After all fixes, formal-phrasing arithmetic ("What is 4 plus 9?") is reliably correct, but compact notation ("1+1") produces plausible-looking, consistently wrong answers. This is an honest architectural ceiling at this scale, not a bug — the model is pattern-matching against memorized examples, not performing arithmetic. Real arithmetic capability in language models is understood to be an emergent property of much larger scale, not something a 20M-parameter model trained on a few thousand examples can be expected to have.

## RLHF, in progress

The fine-tuning above teaches the model to imitate one written example per prompt. RLHF goes further: it optimizes the model directly against actual human preference, which is closer to why models like ChatGPT feel helpful rather than just "trained on good examples." Three pieces, built and tested against real data, not simulated:

- **Preference collection** (`preferences.py`, a Preferences tab in the dashboard): the model generates two responses to the same prompt at different sampling temperatures, a real person picks which is actually better or calls it a tie. No synthetic or self-labeled preferences — that would defeat the entire premise.
- **A reward model** (`reward_model.py`, `train_reward.py`): the same Transformer architecture, initialized from the fine-tuned checkpoint's own weights via transfer learning, with a scalar "how good is this" head replacing the next-token head. Trained on real comparisons via the Bradley-Terry pairwise loss.
- **RAFT** (`raft.py`) — Reward-rAnked FineTuning: a real, published simplification of full RLHF. Full RLHF optimizes via PPO (a value network, a clipped policy objective, a KL penalty against a reference model) — meaningful additional machinery for a 20M-parameter model. RAFT instead generates several candidates per prompt, keeps only the one the reward model scores highest, and fine-tunes on the winners — same direction, without PPO's complexity.

**What labeling only 51 real comparisons revealed, immediately:** 44 of them (86%) came back as ties. Not because the tool was broken — because most of the prompts hit categories the model already answers with extremely peaked confidence (arithmetic, antonyms, calendar facts), so both sampled responses were identical regardless of temperature, even at an extreme 0.4-vs-1.4 gap. There was no preference signal to collect on those. The fix was picking prompts genuinely outside anything the model was drilled on — creative writing, opinions, open-ended questions — where real uncertainty exists.

**Running the reward model and RAFT on only 6 real comparisons confirmed the obvious risk, concretely.** Held-out accuracy read "100%" — meaningless with a single held-out example and a 19.8M-parameter model trivially memorizing 6 training pairs (loss went to exactly 0.000). Running RAFT with that reward model didn't just produce a weak result — it confidently selected outputs like *"on the barrow window"* and *"The month after Tue is Aprise."* as the "best" response to real prompts. The pipeline was working exactly as built; the data feeding it wasn't sufficient yet, and the model happily optimized toward noise. That's now a hard guardrail in `raft.py`: it refuses to run against a reward model trained on fewer than 20 comparisons without an explicit `--force`.

This stage is intentionally left mid-flight rather than pushed to a fake finish: RLHF's entire premise depends on real, sufficient human preference data, and that can't be rushed or synthesized without defeating the point.

## Honest scope

This is not a ChatGPT competitor and was never intended to be. It's a demonstration — and a real one, not a toy that fakes the stages — of what building a language model from first principles actually involves: a tokenizer trained from scratch, a real pretrain→fine-tune pipeline, honest evaluation via held-out perplexity, a genuine hyperparameter search with reproducible findings, and a debugging process where every fix came from a diagnosed cause, not guesswork. Closing the remaining gap to something like ChatGPT is a difference of roughly four to five orders of magnitude in parameters, data, and compute — a gap of infrastructure and resources, not of understanding the underlying method.

## What's next

- Label enough real preference comparisons (aiming for 100+ decided, non-tie pairs) to make the reward model and RAFT results actually meaningful, rather than confidently wrong
- Continued data diversity work, particularly multi-turn conversation quality
- Deeper study of the foundational papers (*Attention Is All You Need*, the GPT series, scaling laws) now that every mechanism they describe has been implemented and debugged firsthand
