"""KTO (Kahneman-Tversky Optimization) -- trains the policy on binary
desirable/undesirable labels instead of paired comparisons, per Ethayarajh
et al. 2024 ("KTO: Model Alignment as Prospect Theoretic Optimization").

Derives training data from this project's existing preferences.db pairs:
each (chosen, rejected) comparison splits into one desirable example (the
chosen response) and one undesirable example (the rejected response). This
uses all data already collected without needing a new labeling UI -- the
real advantage of KTO (labeling a single response as good/bad, no pairing
required) only matters for *future* labeling sessions, which this script
does not change.

Per the paper's value function (using a human-utility model from prospect
theory, not direct preference likelihood):
    reward(x, y) = beta * (log pi_policy(y|x) - log pi_reference(y|x))
    z_0 (reference point) = mean reward over MISMATCHED (x, y') pairs in
        the same mini-batch -- an estimate of what a "typical" response's
        reward looks like under the current policy, clamped at 0
    desirable:   value = lambda_D * sigmoid(reward - z_0)
    undesirable: value = lambda_U * sigmoid(z_0 - reward)
    loss = mean(1 - value)

Usage:
    .venv/bin/python kto.py --steps 300
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
from artifact_io import atomic_torch_save, file_sha256, validate_tokenizer
from run_lock import WriterLock, WriterBusy
import preferences
from preference_data import split_comparisons
from dpo import encode_pair, response_logprob

ROOT = Path(__file__).resolve().parent


def derive_kto_examples(pairs):
    """Each (chosen, rejected) comparison becomes one desirable and one
    undesirable single-response example."""
    examples = []
    for row in pairs:
        chosen = row['response_a'] if row['winner'] == 'a' else row['response_b']
        rejected = row['response_b'] if row['winner'] == 'a' else row['response_a']
        examples.append(dict(prompt=row['prompt'], response=chosen, label='desirable'))
        examples.append(dict(prompt=row['prompt'], response=rejected, label='undesirable'))
    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=8, help='Mini-batch size -- needed to estimate the reference point z_0 from mismatched pairs within the batch.')
    parser.add_argument('--val-fraction', type=float, default=0.2)
    args = parser.parse_args()
    if (args.steps < 1 or not math.isfinite(args.lr) or args.lr <= 0
            or not math.isfinite(args.beta) or args.beta <= 0
            or args.batch_size < 2 or not 0 < args.val_fraction < 1):
        parser.error('Use positive steps/learning rate/beta, a batch size of at least 2, and a validation fraction between zero and one.')
    try:
        with WriterLock(ROOT/'checkpoints'):
            train_kto(args)
    except WriterBusy as exc:
        parser.error(str(exc))


def train_kto(args):
    checkpoint_path = ROOT/'checkpoints/latest.pt'
    if not checkpoint_path.exists():
        raise SystemExit('No fine-tuned checkpoint found. Train and fine-tune Aven-1 first.')
    parent_sha256 = file_sha256(checkpoint_path)
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if saved.get('stage') != 'finetune':
        raise SystemExit('KTO needs a fine-tuned checkpoint, not just pretraining.')
    config = Config(**saved['config'])

    tokenizer = Tokenizer().load(ROOT/'checkpoints/tokenizer.json')
    validate_tokenizer(saved, tokenizer, 'Policy checkpoint')

    pairs = preferences.all_decided()
    train_pairs, val_pairs, data_report = split_comparisons(pairs, args.val_fraction, seed=7)
    train_data = derive_kto_examples(train_pairs)
    val_data = derive_kto_examples(val_pairs)

    encoded = {}
    skipped_context = 0
    usable_train, usable_val = [], []
    for bucket, dest in ((train_data, usable_train), (val_data, usable_val)):
        for ex in bucket:
            try:
                ids, prefix_len = encode_pair(tokenizer, ex['prompt'], ex['response'], config.context)
            except ValueError:
                skipped_context += 1
                continue
            encoded[(ex['prompt'], ex['response'])] = (ids, prefix_len)
            dest.append(ex)
    if len(usable_train) < args.batch_size or not usable_val:
        raise SystemExit(f'Need at least {args.batch_size} usable training examples and 1 validation example; '
                          f'have {len(usable_train)} train / {len(usable_val)} val. Label more comparisons first.')

    print(f'{len(usable_train)} train examples ({len(train_pairs)} pairs x2), {len(usable_val)} val examples '
          f'({len(val_pairs)} pairs x2); skipped {skipped_context} for exceeding context.', flush=True)
    print(f'Preference preparation: {data_report}', flush=True)

    policy = Brain(config)
    policy.load_state_dict(saved['model'])
    reference = copy.deepcopy(policy)
    reference.eval()
    for p in reference.parameters():
        p.requires_grad_(False)

    print('Precomputing reference log-probabilities (frozen model, computed once)...', flush=True)
    ref_logprob = {}
    with torch.no_grad():
        for ex in usable_train + usable_val:
            ids, prefix_len = encoded[(ex['prompt'], ex['response'])]
            ref_logprob[(ex['prompt'], ex['response'])] = response_logprob(reference, ids, prefix_len).item()

    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=0.01)
    rng = random.Random(7)
    torch.manual_seed(7)
    torch.set_num_threads(2)

    def rewards_for(batch, use_no_grad):
        out = []
        ctx = torch.no_grad() if use_no_grad else torch.enable_grad()
        with ctx:
            for ex in batch:
                ids, prefix_len = encoded[(ex['prompt'], ex['response'])]
                logp = response_logprob(policy, ids, prefix_len)
                r = args.beta * (logp - ref_logprob[(ex['prompt'], ex['response'])])
                out.append(r)
        return out

    def mismatched_z0(batch):
        # Real mismatched pairs: prompt i with response (i+1) -- a genuinely
        # different (prompt, response) combination not in the training data,
        # giving an estimate of "what reward does a typical, unrelated
        # response get right now" as the contrastive reference point. Always
        # no_grad: z_0 is a constant reference, never backpropped through.
        prompts = [ex['prompt'] for ex in batch]
        responses = [ex['response'] for ex in batch]
        rotated = responses[1:] + responses[:1]
        rewards = []
        with torch.no_grad():
            for prompt, response in zip(prompts, rotated):
                try:
                    ids, prefix_len = encode_pair(tokenizer, prompt, response, config.context)
                except ValueError:
                    continue  # mismatched combo happens to exceed context -- skip, batch is still valid
                policy_logp = response_logprob(policy, ids, prefix_len).item()
                ref_logp = response_logprob(reference, ids, prefix_len).item()
                rewards.append(args.beta * (policy_logp - ref_logp))
        if not rewards:
            return 0.0
        return max(0.0, sum(rewards) / len(rewards))

    def evaluate(data):
        policy.eval()
        batch = data if len(data) <= args.batch_size else rng.sample(data, args.batch_size)
        rewards = [r.item() for r in rewards_for(batch, use_no_grad=True)]
        z0 = mismatched_z0(batch)
        correct = 0
        for ex, r in zip(batch, rewards):
            agrees = (r > z0) if ex['label'] == 'desirable' else (r < z0)
            correct += int(agrees)
        policy.train()
        return correct / len(batch)

    print(f'Initial held-out accuracy: {evaluate(usable_val):.2%} (random guessing is 50%)', flush=True)
    policy.train()
    for step in range(1, args.steps + 1):
        batch = rng.sample(usable_train, args.batch_size)
        rewards = rewards_for(batch, use_no_grad=False)
        z0 = mismatched_z0(batch)
        losses = []
        for ex, r in zip(batch, rewards):
            if ex['label'] == 'desirable':
                value = torch.sigmoid(r - z0)
            else:
                value = torch.sigmoid(z0 - r)
            losses.append(1 - value)
        loss = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step % 50 == 0 or step == args.steps:
            acc = evaluate(usable_val)
            print(f'Step {step} | loss {loss.item():.3f} | z0 {z0:.3f} | held-out accuracy {acc:.2%}', flush=True)

    out = ROOT/'checkpoints/kto_policy.pt'
    atomic_torch_save(out, dict(config=saved['config'], model=policy.state_dict(), stage='kto',
                     data_sha256=saved.get('data_sha256'), dataset=saved.get('dataset'),
                     parent_checkpoint_sha256=parent_sha256, tokenizer_sha256=tokenizer.fingerprint(),
                     train_count=len(usable_train), val_count=len(usable_val), data_report=data_report,
                     beta=args.beta, batch_size=args.batch_size, seed=7, steps=args.steps, lr=args.lr))
    print(f'\nSaved KTO-trained policy to {out} — NOT automatically promoted to checkpoints/latest.pt. '
          f'Inspect its generations first, then promote manually by copying it over latest.pt only if it '
          f'actually looks better.', flush=True)


if __name__ == '__main__':
    main()
