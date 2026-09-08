"""Trains a reward model on real preference comparisons (preferences.py).

Requires a fine-tuned Aven-1 checkpoint to initialize the backbone from, and
at least a handful of decided (non-tie) comparisons. With very few
comparisons the result is a real but weak/noisy signal — that's an honest
outcome to report, not a bug. Label more via the dashboard's Preferences
tab to improve it.
"""
import argparse
import random
from pathlib import Path
import torch
from brain import Config
from tokenizer import Tokenizer
from reward_model import RewardModel
import preferences

ROOT = Path(__file__).resolve().parent


def encode_pair(tokenizer, prompt, response, context):
    text = f'### Instruction:\n{prompt}\n\n### Response:\n{response}'
    ids = tokenizer.encode(text)
    ids = ids[-context:] if len(ids) > context else ids
    if len(ids) < 2:
        ids = ids + [10]
    return torch.tensor([ids], dtype=torch.long)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--val-fraction', type=float, default=0.2)
    args = parser.parse_args()

    checkpoint_path = ROOT/'checkpoints/latest.pt'
    if not checkpoint_path.exists():
        raise SystemExit('No fine-tuned checkpoint found. Train and fine-tune Aven-1 first.')
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if saved.get('stage') != 'finetune':
        raise SystemExit('Reward model needs a fine-tuned checkpoint, not just pretraining.')
    config = Config(**saved['config'])

    tokenizer = Tokenizer().load(ROOT/'checkpoints/tokenizer.json')

    data = preferences.all_decided()
    if len(data) < 4:
        raise SystemExit(f'Only {len(data)} decided (non-tie) comparisons available. '
                          'Label more in the Preferences tab (aim for at least ~20-30) before training.')
    random.seed(7)
    random.shuffle(data)
    n_val = max(1, int(len(data) * args.val_fraction))
    val_data, train_data = data[:n_val], data[n_val:]
    print(f'{len(train_data)} train comparisons, {len(val_data)} held out '
          f'(total {len(data)} decided; {preferences.count() - len(data)} ties excluded)', flush=True)

    model = RewardModel(config)
    copied = model.load_backbone(saved['model'])
    print(f'Initialized reward model from the fine-tuned checkpoint ({copied} matching tensors copied).', flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    def pair_tensors(row):
        if row['winner'] == 'a':
            chosen, rejected = row['response_a'], row['response_b']
        else:
            chosen, rejected = row['response_b'], row['response_a']
        return (encode_pair(tokenizer, row['prompt'], chosen, config.context),
                encode_pair(tokenizer, row['prompt'], rejected, config.context))

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
        row = random.choice(train_data)
        chosen_ids, rejected_ids = pair_tensors(row)
        r_chosen, r_rejected = model(chosen_ids), model(rejected_ids)
        loss = -torch.nn.functional.logsigmoid(r_chosen - r_rejected).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 50 == 0 or step == args.steps:
            acc = evaluate(val_data)
            print(f'Step {step} | loss {loss.item():.3f} | held-out accuracy {acc:.2%}', flush=True)

    out = ROOT/'checkpoints/reward.pt'
    torch.save(dict(config=saved['config'], model=model.state_dict(),
                     train_count=len(train_data), val_count=len(val_data)), out)
    print(f'Saved reward model to {out}', flush=True)


if __name__ == '__main__':
    main()
