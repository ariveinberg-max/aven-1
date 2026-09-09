"""Genuine held-out perplexity: score a checkpoint on data/eval-heldout/,
books that are never compiled into data/training.txt by sources.compile_corpus().

This is deliberately separate from train.py's internal "held-out" split,
which is really just the last 10% of the same training corpus (same books,
same style, just a different slice -- see the note in WRITEUP.md). The files
here are excluded from data/sources/ entirely, so a checkpoint has never seen
a single token of them during pretraining or fine-tuning.

Usage:
    python3 eval_heldout.py checkpoints/latest.pt
    python3 eval_heldout.py checkpoints/latest.pt --context 256 --batch-size 8

Does not train, fine-tune, or write anything back to the checkpoint -- read-only.
"""
import argparse
import math
from pathlib import Path

import torch

from brain import Brain, Config
from tokenizer import Tokenizer

ROOT = Path(__file__).resolve().parent
HELDOUT_DIR = ROOT / 'data/eval-heldout'


def load_heldout_ids(tokenizer):
    files = sorted(p for p in HELDOUT_DIR.glob('*.txt') if p.is_file())
    if not files:
        raise SystemExit(f'No .txt files found in {HELDOUT_DIR}')
    per_file = {}
    for path in files:
        raw = path.read_text(encoding='utf-8', errors='replace')
        ids = tokenizer.encode_ids(raw.encode('utf-8'))
        per_file[path.name] = torch.tensor(ids, dtype=torch.long)
    return per_file


@torch.no_grad()
def score(model, ids, context, batch_size, device):
    """Full, non-overlapping pass over `ids` (unlike train.py's random-batch
    sampling) so the number is reproducible and covers every token once."""
    model.eval()
    if len(ids) <= context:
        return None  # too short to form even one window
    starts = list(range(0, len(ids) - context - 1, context))
    losses, weights = [], []
    for i in range(0, len(starts), batch_size):
        chunk = starts[i:i + batch_size]
        x = torch.stack([ids[s:s + context] for s in chunk]).to(device)
        y = torch.stack([ids[s + 1:s + context + 1] for s in chunk]).to(device)
        _, loss, _ = model(x, y)
        losses.append(loss.item() * len(chunk))
        weights.append(len(chunk))
    total_loss = sum(losses) / sum(weights)
    return total_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('checkpoint', type=Path, help='Path to a latest.pt-style checkpoint')
    parser.add_argument('--context', type=int, default=None, help='Override context window (defaults to the checkpoint\'s)')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    saved = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    tokenizer_path = args.checkpoint.parent / 'tokenizer.json'
    if not tokenizer_path.exists():
        raise SystemExit(f'No tokenizer.json next to {args.checkpoint}; cannot encode held-out text consistently.')
    tok = Tokenizer()
    tok.load(tokenizer_path)

    config = Config(**saved['config'])
    context = args.context or config.context
    model = Brain(config).to(args.device)
    model.load_state_dict(saved['model'])

    per_file = load_heldout_ids(tok)
    print(f'Checkpoint: {args.checkpoint} (step {saved.get("step", "?")}, stage {saved.get("stage", "?")})')
    print(f'Held-out set: {len(per_file)} file(s) from {HELDOUT_DIR}, context={context}\n')

    overall_loss_weighted, overall_tokens = 0.0, 0
    for name, ids in per_file.items():
        loss = score(model, ids, context, args.batch_size, args.device)
        if loss is None:
            print(f'  {name}: skipped ({len(ids)} tokens, shorter than context {context})')
            continue
        ppl = math.exp(min(loss, 20))
        print(f'  {name}: {len(ids):,} tokens, loss={loss:.4f}, perplexity={ppl:.2f}')
        overall_loss_weighted += loss * len(ids)
        overall_tokens += len(ids)

    if overall_tokens:
        combined_loss = overall_loss_weighted / overall_tokens
        print(f'\nCombined held-out: {overall_tokens:,} tokens, loss={combined_loss:.4f}, perplexity={math.exp(min(combined_loss, 20)):.2f}')


if __name__ == '__main__':
    main()
