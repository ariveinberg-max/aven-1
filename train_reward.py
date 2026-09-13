"""Trains a reward model on real preference comparisons (preferences.py).

Requires a fine-tuned Aven-1 checkpoint to initialize the backbone from, and
at least a handful of decided (non-tie) comparisons. With very few
comparisons the result is a real but weak/noisy signal — that's an honest
outcome to report, not a bug. Label more via the dashboard's Preferences
tab to improve it.
"""
import argparse
import random
import math
from pathlib import Path
import torch
from brain import Config
from tokenizer import Tokenizer
from reward_model import RewardModel
import preferences
from preference_data import split_comparisons
from artifact_io import allow_safe_rng_globals, atomic_torch_save, file_sha256
from run_lock import WriterLock, WriterBusy

ROOT = Path(__file__).resolve().parent


def encode_pair(tokenizer, prompt, response, context):
    text = f'### Instruction:\n{prompt}\n\n### Response:\n{response}'
    ids = tokenizer.encode(text)
    if len(ids) > context:
        raise ValueError('Preference comparison exceeds checkpoint context; shorten it before labeling.')
    if len(ids) < 2:
        ids = ids + [10]
    return torch.tensor([ids], dtype=torch.long)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--val-fraction', type=float, default=0.2)
    args = parser.parse_args()
    if args.steps < 1 or not math.isfinite(args.lr) or args.lr <= 0 or not 0 < args.val_fraction < 1:
        parser.error('Use positive steps/learning rate and a validation fraction between zero and one.')
    try:
        with WriterLock(ROOT/'checkpoints'):
            train_reward(args)
    except WriterBusy as exc:
        parser.error(str(exc))


def train_reward(args):
    checkpoint_path = ROOT/'checkpoints/latest.pt'
    if not checkpoint_path.exists():
        raise SystemExit('No fine-tuned checkpoint found. Train and fine-tune Aven-1 first.')
    parent_sha256 = file_sha256(checkpoint_path)
    allow_safe_rng_globals()
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if saved.get('stage') != 'finetune':
        raise SystemExit('Reward model needs a fine-tuned checkpoint, not just pretraining.')
    config = Config(**saved['config'])

    tokenizer = Tokenizer().load(ROOT/'checkpoints/tokenizer.json')

    if tokenizer.vocab_size != config.vocab or saved.get('tokenizer_sha256', tokenizer.fingerprint()) != tokenizer.fingerprint():
        raise ValueError('Tokenizer does not match the policy checkpoint.')
    data = preferences.all_decided()
    usable = []
    skipped_context = 0
    encoded = {}
    for row in data:
        try:
            a = encode_pair(tokenizer, row['prompt'], row['response_a'], config.context)
            b = encode_pair(tokenizer, row['prompt'], row['response_b'], config.context)
        except ValueError:
            skipped_context += 1
            continue
        usable.append(row)
        encoded[(row['prompt'], row['response_a'], row['response_b'])] = (a, b)
    train_data, val_data, data_report = split_comparisons(usable, args.val_fraction, seed=7)
    data_report['skipped_context'] = skipped_context
    rng = random.Random(7)
    torch.manual_seed(7)
    torch.set_num_threads(2)
    print(f'{len(train_data)} train comparisons, {len(val_data)} held out; '
          f"{data_report['train_prompt_count']} train prompts, {data_report['val_prompt_count']} validation prompts.", flush=True)
    print(f'Preference preparation: {data_report}', flush=True)

    model = RewardModel(config)
    copied = model.load_backbone(saved['model'])
    print(f'Initialized reward model from the fine-tuned checkpoint ({copied} matching tensors copied).', flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    def pair_tensors(row):
        a, b = encoded[(row['prompt'], row['response_a'], row['response_b'])]
        return (a, b) if row['winner'] == 'a' else (b, a)

    def evaluate(rows):
        model.eval()
        correct = 0
        with torch.no_grad():
            for row in rows:
                chosen_ids, rejected_ids = pair_tensors(row)
                r_chosen, r_rejected = model(chosen_ids), model(rejected_ids)
                correct += int((r_chosen > r_rejected).item())
        model.train()
        return correct / max(len(rows), 1)

    print(f'Initial held-out accuracy: {evaluate(val_data):.2%} (random guessing is 50%)', flush=True)
    model.train()
    for step in range(1, args.steps + 1):
        row = rng.choice(train_data)
        chosen_ids, rejected_ids = pair_tensors(row)
        r_chosen, r_rejected = model(chosen_ids), model(rejected_ids)
        loss = -torch.nn.functional.logsigmoid(r_chosen - r_rejected).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step % 50 == 0 or step == args.steps:
            acc = evaluate(val_data)
            print(f'Step {step} | loss {loss.item():.3f} | held-out accuracy {acc:.2%}', flush=True)

    out = ROOT/'checkpoints/reward.pt'
    atomic_torch_save(out, dict(config=saved['config'], model=model.state_dict(),
                     train_count=len(train_data), val_count=len(val_data),
                     heldout_accuracy=evaluate(val_data), data_report=data_report,
                     parent_checkpoint_sha256=parent_sha256, tokenizer_sha256=tokenizer.fingerprint(),
                     seed=7, steps=args.steps, lr=args.lr))
    print(f'Saved reward model to {out}', flush=True)


if __name__ == '__main__':
    main()
