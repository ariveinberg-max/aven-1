"""Full PPO (Proximal Policy Optimization) — the actual algorithm behind
RLHF in InstructGPT/ChatGPT, not RAFT's simplification.

Four models involved, all the same Transformer architecture:
  - policy: the model being trained (starts as a copy of the fine-tuned checkpoint)
  - reference: a FROZEN copy of the policy from before PPO started. Without
    this, the policy can drift into gibberish that scores high on the reward
    model but means nothing — a well-known PPO failure mode called reward
    hacking. The KL penalty below is what prevents it.
  - reward model: frozen, scores a full (prompt, response) pair
  - value model: trained alongside the policy, estimates expected future
    reward at each token so the advantage estimate (GAE) has lower variance
    than raw rewards would

Per iteration: generate responses from the current policy (rollout), score
them, compute advantages via GAE, then take several gradient steps on the
clipped surrogate objective — the actual mechanism, not a stand-in for it.

Also implements PPO-ptx, the fix InstructGPT's own paper uses for a real
failure mode: pure PPO, rolled out only on a narrow slice of prompts,
updates the entire shared backbone and can quietly degrade capabilities
outside that slice (verified here — a first run without ptx broke
arithmetic entirely, a category PPO never touched). PPO-ptx mixes the
original fine-tuning data's plain language-modeling loss into every
update, anchoring everything PPO isn't actively reinforcing.

This is known to be finicky even with abundant preference data; with the
small amount collected here, expect instability to be visible, not hidden.
That's reported honestly, not smoothed over.
"""
import argparse
import copy
from pathlib import Path
import torch
import torch.nn.functional as F
from brain import Brain, Config
from tokenizer import Tokenizer
from value_model import ValueModel
import preferences

ROOT = Path(__file__).resolve().parent

PPO_PROMPTS = [
    'Introduce yourself', 'Who are you?', 'Tell me about yourself', 'What kind of AI are you?',
    'Hello!', 'Hey there', 'Good morning', 'How are you doing?', "How's it going?",
    'Thank you', 'Thanks for the help', 'Can you help me?', 'I have a question',
    'Goodbye', 'See you later',
]


@torch.no_grad()
def rollout(policy, tokenizer, prompt, context, max_new_tokens=60, temperature=0.8):
    """Generate a response from the policy, returning the full (prompt+response)
    token ids and where the response starts. Mirrors Brain.generate's sampling
    logic directly so PPO trains on exactly what the model would actually produce."""
    policy.eval()
    prompt_text = f'### Instruction:\n{prompt}\n\n### Response:\n'
    prompt_ids = tokenizer.encode(prompt_text) or [10]
    ids = torch.tensor([prompt_ids], dtype=torch.long)
    prompt_len = ids.shape[1]
    end_ids = tokenizer.encode('<|end|>')
    for _ in range(max_new_tokens):
        logits, _, _ = policy(ids[:, -context:])
        logits = logits[:, -1] / temperature
        cutoff = torch.topk(logits, min(40, logits.shape[-1])).values[:, -1:]
        logits = logits.masked_fill(logits < cutoff, float('-inf'))
        next_id = torch.multinomial(F.softmax(logits, dim=-1), 1)
        ids = torch.cat([ids, next_id], dim=1)
        if end_ids and ids.shape[1] - prompt_len >= len(end_ids) and ids[0, -len(end_ids):].tolist() == end_ids:
            break
    return ids, prompt_len


def sequence_logprobs(model, ids, context):
    """Per-position log-prob of the actual next token, teacher-forced. Position i
    holds log p(ids[i+1] | ids[:i+1]). Truncates to the model's context window."""
    ids = ids[:, -context:]
    logits, _, _ = model(ids)
    log_probs = F.log_softmax(logits[:, :-1], dim=-1)
    return log_probs.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)  # (batch, seq_len-1)


def compute_gae(rewards, values, gamma=1.0, lam=0.95):
    """Generalized Advantage Estimation over a single response's token positions."""
    T = rewards.shape[0]
    advantages = torch.zeros(T)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_value = values[t + 1] if t + 1 < T else 0.0
        delta = rewards[t] + gamma * next_value - values[t]
        last_gae = delta + gamma * lam * last_gae
        advantages[t] = last_gae
    returns = advantages + values[:T]
    return advantages, returns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', type=int, default=20, help='Rollout + PPO-update cycles')
    parser.add_argument('--rollout-batch', type=int, default=6, help='Prompts sampled per iteration')
    parser.add_argument('--ppo-epochs', type=int, default=4, help='Gradient passes over each rollout batch')
    parser.add_argument('--lr-policy', type=float, default=1e-5)
    parser.add_argument('--lr-value', type=float, default=1e-4)
    parser.add_argument('--kl-coef', type=float, default=0.1, help='Weight of the per-token KL penalty against the reference model')
    parser.add_argument('--clip-eps', type=float, default=0.2)
    parser.add_argument('--vf-coef', type=float, default=0.5)
    parser.add_argument('--ent-coef', type=float, default=0.01)
    parser.add_argument('--max-new-tokens', type=int, default=60)
    parser.add_argument('--kl-limit', type=float, default=8.0, help='Abort if mean per-response KL exceeds this (policy collapsing)')
    parser.add_argument('--ptx-coef', type=float, default=1.0,
                         help='Weight of the PPO-ptx language-modeling loss (mixes in the original fine-tuning '
                              'data to prevent PPO from degrading capabilities it never rolls out on). 0 disables it.')
    parser.add_argument('--ptx-data', type=Path, default=ROOT/'data/instructions.txt')
    parser.add_argument('--ptx-batch-size', type=int, default=4)
    parser.add_argument('--force', action='store_true', help='Skip the reward-model data-sufficiency guardrail')
    args = parser.parse_args()

    checkpoint_path = ROOT/'checkpoints/latest.pt'
    reward_path = ROOT/'checkpoints/reward.pt'
    if not checkpoint_path.exists():
        raise SystemExit('No fine-tuned checkpoint found. Train and fine-tune Aven-1 first.')
    if not reward_path.exists():
        raise SystemExit('No reward model found. Run train_reward.py first.')

    reward_saved = torch.load(reward_path, map_location='cpu', weights_only=True)
    train_count = reward_saved.get('train_count', 0)
    if train_count < 20 and not args.force:
        raise SystemExit(f'The reward model was trained on only {train_count} comparisons — too few to trust '
                          f'for direct policy optimization (PPO will happily exploit a weak reward model). '
                          f'Label more in the Preferences tab, retrain train_reward.py, then retry. '
                          f'Pass --force to proceed anyway (not recommended).')

    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if saved.get('stage') != 'finetune':
        raise SystemExit('PPO needs a fine-tuned checkpoint, not just pretraining.')
    config = Config(**saved['config'])
    tokenizer = Tokenizer().load(ROOT/'checkpoints/tokenizer.json')

    ptx_data = None
    if args.ptx_coef > 0:
        if not args.ptx_data.exists():
            raise SystemExit(f'--ptx-data {args.ptx_data} not found. Pass --ptx-coef 0 to disable PPO-ptx (not recommended).')
        raw = args.ptx_data.read_bytes()
        ptx_ids = tokenizer.encode_ids(raw)
        if len(ptx_ids) < config.context + 1:
            raise SystemExit(f'--ptx-data is too short ({len(ptx_ids)} tokens) for the context window ({config.context}).')
        ptx_data = torch.tensor(ptx_ids, dtype=torch.long)
        print(f'PPO-ptx enabled: mixing in {args.ptx_data.name} ({len(ptx_ids):,} tokens) to anchor capabilities '
              f'PPO does not roll out on.', flush=True)

    def ptx_batch():
        starts = torch.randint(len(ptx_data) - config.context, (args.ptx_batch_size,))
        x = torch.stack([ptx_data[i:i + config.context] for i in starts])
        y = torch.stack([ptx_data[i + 1:i + config.context + 1] for i in starts])
        return x, y

    policy = Brain(config)
    policy.load_state_dict(saved['model'])
    reference = copy.deepcopy(policy)
    for p in reference.parameters():
        p.requires_grad_(False)
    reference.eval()

    from reward_model import RewardModel
    reward_model = RewardModel(Config(**reward_saved['config']))
    reward_model.load_state_dict(reward_saved['model'])
    reward_model.eval()
    for p in reward_model.parameters():
        p.requires_grad_(False)
    print(f'Reward model trained on {train_count} real comparisons ({reward_saved.get("val_count", "?")} held out).', flush=True)

    value_model = ValueModel(config)
    copied = value_model.load_backbone(saved['model'])
    print(f'Value model initialized from the policy checkpoint ({copied} matching tensors copied).', flush=True)

    policy_opt = torch.optim.AdamW(policy.parameters(), lr=args.lr_policy, weight_decay=0.0)
    value_opt = torch.optim.AdamW(value_model.parameters(), lr=args.lr_value, weight_decay=0.01)

    import random
    random.seed(7)

    for it in range(1, args.iterations + 1):
        prompts = [random.choice(PPO_PROMPTS) for _ in range(args.rollout_batch)]
        batch = []
        for prompt in prompts:
            ids, prompt_len = rollout(policy, tokenizer, prompt, config.context, args.max_new_tokens)
            if ids.shape[1] <= prompt_len:
                continue  # generated nothing before context truncation; skip
            if ids.shape[1] > config.context:
                continue  # would misalign prompt_len against a left-truncated window; skip rather than risk it
            with torch.no_grad():
                old_logprobs = sequence_logprobs(policy, ids, config.context)[:, prompt_len - 1:]
                ref_logprobs = sequence_logprobs(reference, ids, config.context)[:, prompt_len - 1:]
                values = value_model(ids[:, -config.context:])[0, prompt_len - 1:-1]
                rm_score = reward_model(ids).item()
            kl = (old_logprobs - ref_logprobs)[0]
            rewards = -args.kl_coef * kl.clone()
            rewards[-1] += rm_score
            advantages, returns = compute_gae(rewards, values)
            adv_std = advantages.std()
            advantages = (advantages - advantages.mean()) / (adv_std + 1e-8) if adv_std > 1e-6 else advantages
            batch.append(dict(ids=ids, prompt_len=prompt_len, old_logprobs=old_logprobs.detach(),
                               advantages=advantages.detach(), returns=returns.detach(),
                               kl_mean=kl.mean().item(), rm_score=rm_score))

        if not batch:
            print(f'Iteration {it}: no usable rollouts (all hit context limit), skipping.', flush=True)
            continue

        mean_kl = sum(b['kl_mean'] for b in batch) / len(batch)
        mean_reward = sum(b['rm_score'] for b in batch) / len(batch)
        if mean_kl > args.kl_limit:
            print(f'Iteration {it}: mean KL {mean_kl:.2f} exceeds --kl-limit {args.kl_limit} — '
                  f'the policy is drifting from the reference too fast (reward hacking risk). Stopping.', flush=True)
            break

        policy.train()
        for epoch in range(args.ppo_epochs):
            total_policy_loss = total_value_loss = total_entropy = total_ptx_loss = 0.0
            for i, b in enumerate(batch):
                ids, prompt_len = b['ids'], b['prompt_len']
                logits, _, _ = policy(ids[:, -config.context:])
                log_probs_full = F.log_softmax(logits[:, :-1], dim=-1)
                new_logprobs = log_probs_full.gather(-1, ids[:, -config.context:][:, 1:].unsqueeze(-1)).squeeze(-1)[:, prompt_len - 1:]
                probs = F.softmax(logits[:, prompt_len - 1:-1], dim=-1)
                entropy = -(probs * log_probs_full[:, prompt_len - 1:]).sum(-1).mean()

                new_values = value_model(ids[:, -config.context:])[0, prompt_len - 1:-1]

                ratio = torch.exp(new_logprobs[0] - b['old_logprobs'][0])
                surr1 = ratio * b['advantages']
                surr2 = torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps) * b['advantages']
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(new_values, b['returns'])
                loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy

                ptx_loss = None
                if ptx_data is not None and i == 0:  # one ptx gradient contribution per epoch, not per rollout
                    ptx_x, ptx_y = ptx_batch()
                    _, ptx_loss, _ = policy(ptx_x, ptx_y)
                    loss = loss + args.ptx_coef * ptx_loss

                policy_opt.zero_grad(set_to_none=True)
                value_opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(value_model.parameters(), 1.0)
                policy_opt.step()
                value_opt.step()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                if ptx_loss is not None:
                    total_ptx_loss += ptx_loss.item()
            n = len(batch)
            if epoch == args.ppo_epochs - 1:
                print(f'Iteration {it:3d} | reward {mean_reward:+.3f} | KL {mean_kl:.3f} | '
                      f'policy_loss {total_policy_loss/n:.4f} | value_loss {total_value_loss/n:.4f} | '
                      f'ptx_loss {total_ptx_loss:.4f} | '
                      f'entropy {total_entropy/n:.3f}', flush=True)

    out = ROOT/'checkpoints/ppo_policy.pt'
    torch.save(dict(config=saved['config'], model=policy.state_dict(), stage='ppo',
                     data_sha256=saved['data_sha256'], dataset=saved['dataset'],
                     tokenizer_note='reuses checkpoints/tokenizer.json'), out)
    print(f'\nSaved PPO-trained policy to {out} — NOT automatically promoted to checkpoints/latest.pt. '
          f'Inspect its generations first (it is a real risk that PPO degraded quality with this little '
          f'preference data); promote manually by copying it over latest.pt only if it actually looks better.', flush=True)


if __name__ == '__main__':
    main()
