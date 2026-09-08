"""RAFT (Reward-rAnked FineTuning): step 3 of RLHF, simplified for laptop scale.

Full RLHF optimizes the policy against a reward model via PPO — a clipped
policy-gradient objective with a value network and a KL penalty against a
reference model. That's a lot of additional machinery for a 20M-parameter
model. RAFT is a real, published simplification: generate several candidate
responses per prompt, score them with the reward model, keep only the
best one per prompt, and supervised-fine-tune the policy on those winners.
Same direction as full RLHF (optimize the policy toward what the reward
model scores highly) without PPO's complexity — reuses this project's
existing --finetune mechanism as the actual optimization step.

Usage:
    .venv/bin/python raft.py --candidates 4
    .venv/bin/python train.py --data data/raft.txt --finetune --steps 500 --wandb
"""
import argparse
from pathlib import Path
import torch
from brain import Brain, Config
from reward_model import RewardModel
from tokenizer import Tokenizer

ROOT = Path(__file__).resolve().parent

RAFT_PROMPTS = [
    'Write a short story about a robot who wants to learn to paint.',
    "What's the best way to spend a weekend?",
    'Describe what a city on the moon might look like.',
    'What do you think about school?',
    'Write a poem about the ocean.',
    'If you could change one thing about yourself, what would it be?',
    'Explain why the sky is blue.',
    'What makes a good friend?',
    'Tell me a story about a dragon who is afraid of fire.',
    'What would you do with a million dollars?',
    'Describe your perfect day.',
    'Write the beginning of a mystery novel.',
    'What is the strangest animal you can imagine?',
    'How do you think computers will change in the future?',
    'Give me a recipe for something creative, even if it sounds silly.',
    'Introduce yourself', 'How are you doing?', 'Can you help me?',
    'What is the meaning of life?', 'Tell me something interesting.',
]


def encode_pair(tokenizer, prompt, response, context):
    ids = tokenizer.encode(f'### Instruction:\n{prompt}\n\n### Response:\n{response}')
    ids = ids[-context:] if len(ids) > context else (ids + [10] if len(ids) < 2 else ids)
    return torch.tensor([ids], dtype=torch.long)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidates', type=int, default=4, help='Responses generated per prompt before picking the best')
    parser.add_argument('--out', type=Path, default=ROOT/'data/raft.txt')
    parser.add_argument('--force', action='store_true',
                         help='Proceed even if the reward model was trained on very few comparisons (not recommended — '
                              'it will confidently pick garbage as "best").')
    args = parser.parse_args()
    if not 2 <= args.candidates <= 16:
        parser.error('--candidates must be 2-16.')

    checkpoint_path = ROOT/'checkpoints/latest.pt'
    reward_path = ROOT/'checkpoints/reward.pt'
    if not checkpoint_path.exists():
        raise SystemExit('No fine-tuned checkpoint found. Train and fine-tune Aven-1 first.')
    if not reward_path.exists():
        raise SystemExit('No reward model found. Run train_reward.py first.')

    reward_saved_check = torch.load(reward_path, map_location='cpu', weights_only=True)
    train_count = reward_saved_check.get('train_count', 0)
    if train_count < 20 and not args.force:
        raise SystemExit(f'The reward model was trained on only {train_count} comparisons — too few to trust. '
                          f'With this little data it will confidently rank garbage as "best" (verified: it does). '
                          f'Label more in the Preferences tab (aim for 100+), retrain train_reward.py, then retry. '
                          f'Pass --force to proceed anyway (not recommended).')

    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if saved.get('stage') != 'finetune':
        raise SystemExit('RAFT needs a fine-tuned checkpoint, not just pretraining.')
    config = Config(**saved['config'])
    policy = Brain(config)
    policy.load_state_dict(saved['model'])

    reward_saved = reward_saved_check
    reward_model = RewardModel(Config(**reward_saved['config']))
    reward_model.load_state_dict(reward_saved['model'])
    reward_model.eval()
    print(f'Reward model trained on {reward_saved.get("train_count", "?")} comparisons '
          f'({reward_saved.get("val_count", "?")} held out).', flush=True)

    tokenizer = Tokenizer().load(ROOT/'checkpoints/tokenizer.json')

    blocks = []
    reward_gaps = []
    for prompt in RAFT_PROMPTS:
        wrapped = f'### Instruction:\n{prompt}\n\n### Response:\n'
        candidates = []
        for i in range(args.candidates):
            temp = 0.6 + 0.6 * (i / max(args.candidates - 1, 1))  # spread from 0.6 to 1.2
            text, _ = policy.generate(wrapped, count=120, temperature=temp, tokenizer=tokenizer, stop_text='<|end|>')
            response = text[len(wrapped):].strip() if text.startswith(wrapped) else text.strip()
            if response:
                candidates.append(response)
        if not candidates:
            continue
        with torch.no_grad():
            scores = [reward_model(encode_pair(tokenizer, prompt, r, config.context)).item() for r in candidates]
        best_idx = max(range(len(candidates)), key=lambda i: scores[i])
        best, worst = candidates[best_idx], candidates[scores.index(min(scores))]
        reward_gaps.append(max(scores) - min(scores))
        blocks.append(f'### Instruction:\n{prompt}\n\n### Response:\n{best}\n<|end|>\n')
        print(f'{prompt[:50]:50s} | best score {max(scores):+.3f} | spread {max(scores)-min(scores):.3f}', flush=True)

    if not blocks:
        raise SystemExit('No candidates generated — check the checkpoint and tokenizer.')
    args.out.write_text('\n'.join(blocks), encoding='utf-8')
    avg_gap = sum(reward_gaps) / len(reward_gaps)
    print(f'\nWrote {len(blocks)} reward-ranked examples to {args.out}', flush=True)
    print(f'Average reward spread within a prompt\'s candidates: {avg_gap:.3f} '
          f'(near zero means the reward model isn\'t distinguishing candidates yet — '
          f'label more preference data and retrain it).', flush=True)
    print(f'Next: .venv/bin/python train.py --data {args.out} --finetune --steps 500 --wandb', flush=True)


if __name__ == '__main__':
    main()
