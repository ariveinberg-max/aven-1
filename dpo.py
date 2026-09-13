"""DPO (Direct Preference Optimization) — trains the policy directly on
preference pairs, without a separate reward model. RAFT and PPO both depend
on one; DPO skips that step entirely.

Real motivation for adding this alongside RAFT/PPO, not replacing them: this
project's own history already shows the reward model staying noisy at this
data scale (held-out accuracy swings between retrains with more than the
usual sampling noise would explain — see train_reward.py's most recent run),
plus RAFT/PPO's own documented instability (a RAFT run that regressed an
unrelated capability, PPO's known forgetting risk). DPO removes one whole
noise source — the reward model — by optimizing straight from
preferences.db's (prompt, response_a, response_b, winner) rows.

Per DPO's original formulation (Rafailov et al., 2023): for each (chosen,
rejected) pair, compare how much MORE the policy now prefers the chosen
response over the reference (a frozen copy of the policy from before DPO
started) than it prefers the rejected one. Increasing that margin is the
entire training signal — no rollouts, no value network, no reward model.

Usage:
    .venv/bin/python dpo.py --steps 300
"""
import argparse
import copy
import math
import random
from pathlib import Path
import torch
import torch.nn.functional as F
from brain import Brain, Config
from tokenizer import Tokenizer
from artifact_io import allow_safe_rng_globals, atomic_torch_save, file_sha256, validate_tokenizer
from run_lock import WriterLock, WriterBusy
import preferences
from preference_data import split_comparisons

ROOT = Path(__file__).resolve().parent


def encode_pair(tokenizer, prompt, response, context):
    """Returns (full token ids, index where the response starts) so the loss
    can be restricted to response tokens only — same convention as
    train.py's --loss-mode response, applied to a single (prompt, response)
    pair instead of a streamed corpus."""
    prefix = f'### Instruction:\n{prompt}\n\n### Response:\n'
    prefix_ids = tokenizer.encode(prefix)
    full_ids = tokenizer.encode(prefix + response)
    if len(full_ids) > context:
        raise ValueError('Preference comparison exceeds checkpoint context; shorten it before labeling.')
    if len(full_ids) <= len(prefix_ids):
        full_ids = full_ids + [10]  # guarantee at least one response token
    return torch.tensor([full_ids], dtype=torch.long), len(prefix_ids)


def response_logprob(model, ids, prefix_len):
    """Sum of log p(token) over the response span only, under this model."""
    logits, _, _ = model(ids)
    response_logits = logits[0, prefix_len - 1:-1]
    response_targets = ids[0, prefix_len:]
    log_probs = F.log_softmax(response_logits, dim=-1)
    return log_probs.gather(1, response_targets.unsqueeze(1)).squeeze(1).sum()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--beta', type=float, default=0.1,
                         help='DPO temperature — higher trusts the preference margin more, lower is more conservative.')
    parser.add_argument('--val-fraction', type=float, default=0.2)
    args = parser.parse_args()
    if (args.steps < 1 or not math.isfinite(args.lr) or args.lr <= 0
            or not math.isfinite(args.beta) or args.beta <= 0 or not 0 < args.val_fraction < 1):
        parser.error('Use positive steps/learning rate/beta and a validation fraction between zero and one.')
    try:
        with WriterLock(ROOT/'checkpoints'):
            train_dpo(args)
    except WriterBusy as exc:
        parser.error(str(exc))


def train_dpo(args):
    checkpoint_path = ROOT/'checkpoints/latest.pt'
    if not checkpoint_path.exists():
        raise SystemExit('No fine-tuned checkpoint found. Train and fine-tune Aven-1 first.')
    parent_sha256 = file_sha256(checkpoint_path)
    allow_safe_rng_globals()
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if saved.get('stage') != 'finetune':
        raise SystemExit('DPO needs a fine-tuned checkpoint, not just pretraining.')
    config = Config(**saved['config'])

    tokenizer = Tokenizer().load(ROOT/'checkpoints/tokenizer.json')
    validate_tokenizer(saved, tokenizer, 'Policy checkpoint')

    data = preferences.all_decided()
    usable = []
    skipped_context = 0
    encoded = {}
    for row in data:
        chosen_resp = row['response_a'] if row['winner'] == 'a' else row['response_b']
        rejected_resp = row['response_b'] if row['winner'] == 'a' else row['response_a']
        try:
            chosen_ids, chosen_prefix = encode_pair(tokenizer, row['prompt'], chosen_resp, config.context)
            rejected_ids, rejected_prefix = encode_pair(tokenizer, row['prompt'], rejected_resp, config.context)
        except ValueError:
            skipped_context += 1
            continue
        usable.append(row)
        encoded[(row['prompt'], row['response_a'], row['response_b'])] = (
            chosen_ids, chosen_prefix, rejected_ids, rejected_prefix)

    train_data, val_data, data_report = split_comparisons(usable, args.val_fraction, seed=7)
    data_report['skipped_context'] = skipped_context
    rng = random.Random(7)
    torch.manual_seed(7)
    torch.set_num_threads(2)
    print(f'{len(train_data)} train comparisons, {len(val_data)} held out; '
          f"{data_report['train_prompt_count']} train prompts, {data_report['val_prompt_count']} validation prompts.", flush=True)
    print(f'Preference preparation: {data_report}', flush=True)

    policy = Brain(config)
    policy.load_state_dict(saved['model'])
    reference = copy.deepcopy(policy)
    reference.eval()
    for p in reference.parameters():
        p.requires_grad_(False)

    def pair_tensors(row):
        return encoded[(row['prompt'], row['response_a'], row['response_b'])]

    print('Precomputing reference log-probabilities (frozen model, computed once)...', flush=True)
    ref_logprobs = {}
    with torch.no_grad():
        for row in train_data + val_data:
            chosen_ids, chosen_prefix, rejected_ids, rejected_prefix = pair_tensors(row)
            key = (row['prompt'], row['response_a'], row['response_b'])
            ref_logprobs[key] = (response_logprob(reference, chosen_ids, chosen_prefix).item(),
                                  response_logprob(reference, rejected_ids, rejected_prefix).item())

    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=0.01)

    def evaluate(rows):
        policy.eval()
        correct = 0
        with torch.no_grad():
            for row in rows:
                chosen_ids, chosen_prefix, rejected_ids, rejected_prefix = pair_tensors(row)
                key = (row['prompt'], row['response_a'], row['response_b'])
                ref_chosen, ref_rejected = ref_logprobs[key]
                pol_chosen = response_logprob(policy, chosen_ids, chosen_prefix).item()
                pol_rejected = response_logprob(policy, rejected_ids, rejected_prefix).item()
                margin = (pol_chosen - ref_chosen) - (pol_rejected - ref_rejected)
                correct += int(margin > 0)
        policy.train()
        return correct / max(len(rows), 1)

    print(f'Initial held-out accuracy: {evaluate(val_data):.2%} (random guessing is 50%)', flush=True)
    policy.train()
    for step in range(1, args.steps + 1):
        row = rng.choice(train_data)
        chosen_ids, chosen_prefix, rejected_ids, rejected_prefix = pair_tensors(row)
        key = (row['prompt'], row['response_a'], row['response_b'])
        ref_chosen, ref_rejected = ref_logprobs[key]
        pol_chosen = response_logprob(policy, chosen_ids, chosen_prefix)
        pol_rejected = response_logprob(policy, rejected_ids, rejected_prefix)
        margin = (pol_chosen - ref_chosen) - (pol_rejected - ref_rejected)
        loss = -F.logsigmoid(args.beta * margin)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step % 50 == 0 or step == args.steps:
            acc = evaluate(val_data)
            print(f'Step {step} | loss {loss.item():.3f} | held-out accuracy {acc:.2%}', flush=True)

    out = ROOT/'checkpoints/dpo_policy.pt'
    atomic_torch_save(out, dict(config=saved['config'], model=policy.state_dict(), stage='dpo',
                     step=saved.get('step', 0),
                     data_sha256=saved.get('data_sha256'), dataset=saved.get('dataset'),
                     parent_checkpoint_sha256=parent_sha256, tokenizer_sha256=tokenizer.fingerprint(),
                     train_count=len(train_data), val_count=len(val_data),
                     heldout_accuracy=evaluate(val_data), data_report=data_report,
                     beta=args.beta, seed=7, steps=args.steps, lr=args.lr))
    print(f'\nSaved DPO-trained policy to {out} — NOT automatically promoted to checkpoints/latest.pt. '
          f'Inspect its generations first (it is a real risk that DPO degraded quality with this little '
          f'preference data, same as RAFT/PPO); promote manually by copying it over latest.pt only if it '
          f'actually looks better.', flush=True)


if __name__ == '__main__':
    main()
