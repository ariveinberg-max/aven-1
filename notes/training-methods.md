# Training methods — reference notes

Personal reference for transcribing into the physical notebook. Covers every
RLHF/fine-tuning method mentioned in Day 1-2 reading and in this project's own
code. Not project documentation — a study aid.

## The five methods, in plain terms

**SFT (Supervised Fine-Tuning)** — the baseline. Show the model correct
(prompt, response) pairs, train it to predict the response with normal
cross-entropy loss. This is what `train.py --loss-mode response` does. No
notion of "better vs. worse" — every example is just "correct."

**Reward Model + RAFT/PPO (classic RLHF)** — three separate stages:
1. Collect human preference pairs (prompt, response_a, response_b, winner).
2. Train a *reward model* to predict which response a human would prefer
   (Bradley-Terry pairwise loss — literally what `reward_model.py` /
   `train_reward.py` implement).
3. Use the reward model's scores to improve the policy:
   - **RAFT**: generate several candidates per prompt, keep only the
     reward model's favorite, supervised-fine-tune on the winners.
     Simple, stable, but only as good as "pick the best of N."
   - **PPO**: full reinforcement learning — roll out responses, score them,
     compute advantages, take clipped policy-gradient steps, with a KL
     penalty against a frozen reference model to stop the policy drifting
     into "reward-hacked" gibberish. This is the actual algorithm behind
     InstructGPT/ChatGPT's original RLHF. Needs a value network too.

Downside of this whole family: the reward model is a *third, separately
noisy* model sitting between the preference data and the policy. Every
error the reward model makes gets baked into whatever RAFT/PPO does next.

**DPO (Direct Preference Optimization)** — skips the reward model entirely.
Trains the policy directly on (chosen, rejected) pairs: increase how much
more the policy prefers the chosen response over the reference model,
relative to the rejected one. One loss function, no rollouts, no value
network, no separate reward model. `dpo.py` in this repo implements this.
Cheaper and (in this project's own first test) more numerically stable
than the reward-model route — but stability in the metric didn't clearly
mean better real answers, at this project's current data scale.

**ORPO (Odds Ratio Preference Optimization)** — folds supervised fine-tuning
and preference optimization into *one* training step instead of two.
The actual loss (arXiv 2403.07691):

    L = L_SFT - λ · log(sigmoid(log(odds(chosen) / odds(rejected))))
    where odds(y|x) = P(y|x) / (1 - P(y|x))

No reference model at all — everything is computed from the policy being
trained right now, unlike DPO which needs a frozen pre-training snapshot
just for the comparison. Weak penalty on the rejected response, strong
reinforcement on the chosen one, in one pass. Pitched as well-suited to
small datasets — exactly this project's situation. Comparative studies are
mixed (won on one benchmark, lost to DPO on others — not a settled "ORPO
is better" result). Not yet implemented here — a real candidate for a
future `orpo.py`, same house conventions as `dpo.py`.

**KTO (Kahneman-Tversky Optimization)** — doesn't need *ranked* pairs at
all, just a binary thumbs-up/thumbs-down per response. Much easier to
collect in a live product (no need to generate two responses and compare
them), at some cost in signal quality per label. Also not yet implemented.

## The one real lesson so far, across all of them

More than which algorithm: **data quality and quantity dominate the
result.** The reading block source cited 500–2,000 curated examples
beating 50,000 scraped ones. This project's own first DPO run used only
112 usable comparisons after removing duplicates/conflicts (out of 555
raw logged) — every method (reward model, DPO) showed real instability at
that scale, regardless of which one was used. The honest next lever is
more, cleaner labeled data — not a fifth method.

## Sources
- Day 1 reading: "Fine-Tuning LLMs in 2026" (bigdataboutique.com) — introduced
  DPO/ORPO/KTO as the field's shift away from classic reward-model RLHF.
- This project's own `reward_model.py`, `raft.py`, `ppo.py`, `dpo.py`
  docstrings — the actual, real implementations these notes describe.
