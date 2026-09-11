"""External-text next-token evaluation with bounded tokenization memory.

Directory membership does not prove a historical checkpoint never saw these
texts. Training exposure must be checked against the complete checkpoint lineage.
This evaluator never trains or modifies checkpoint/source files.
"""
import argparse
import math
import hashlib
import json
from pathlib import Path

import torch

from brain import Brain, Config
from tokenizer import Tokenizer
from artifact_io import file_sha256, load_checkpoint_snapshot, validate_tokenizer, atomic_json

ROOT = Path(__file__).resolve().parent
HELDOUT_DIR = ROOT / 'data/eval-heldout'


@torch.no_grad()
def score(model, ids, context, batch_size, device):
    """Score every next-token target once, including the final partial window."""
    if not 1 <= context <= model.config.context or batch_size < 1:
        raise ValueError('Context must fit the model and batch size must be positive.')
    model.eval()
    if len(ids) < 2:
        return None
    total_loss, total_targets = 0.0, 0
    full_end = ((len(ids) - 1) // context) * context
    for start in range(0, full_end, batch_size * context):
        starts = range(start, min(full_end, start + batch_size * context), context)
        x = torch.stack([ids[s:s + context] for s in starts]).to(device)
        y = torch.stack([ids[s + 1:s + context + 1] for s in starts]).to(device)
        _, loss, _ = model(x, y)
        total_loss += loss.item() * y.numel()
        total_targets += y.numel()
    if full_end < len(ids) - 1:
        x = ids[full_end:-1].unsqueeze(0).to(device)
        y = ids[full_end + 1:].unsqueeze(0).to(device)
        _, loss, _ = model(x, y)
        total_loss += loss.item() * y.numel()
        total_targets += y.numel()
    return total_loss / total_targets


def score_file(model, path, tokenizer, context, batch_size, device, chunk_bytes=65536):
    """Score fixed-size byte-chunk tokenization without loading the whole book.

    BPE merges across encoding chunk boundaries are omitted. Chunk size is part
    of the protocol and recorded so comparisons use the same token sequence.
    """
    if chunk_bytes < 1 or batch_size < 1 or not 1 <= context <= model.config.context:
        raise ValueError('Invalid scoring window or chunk size.')
    buffer = []
    total_loss, targets, tokens, byte_count = 0.0, 0, 0, 0
    digest = hashlib.sha256()
    window = context * batch_size
    with Path(path).open('rb') as source:
        while chunk := source.read(chunk_bytes):
            digest.update(chunk)
            byte_count += len(chunk)
            encoded = tokenizer.encode_ids(chunk)
            tokens += len(encoded)
            buffer.extend(encoded)
            while len(buffer) > window:
                loss = score(model, torch.tensor(buffer[:window+1]), context, batch_size, device)
                total_loss += loss * window
                targets += window
                buffer = buffer[window:]
    if len(buffer) > 1:
        loss = score(model, torch.tensor(buffer), context, batch_size, device)
        total_loss += loss * (len(buffer) - 1)
        targets += len(buffer) - 1
    loss = total_loss / targets if targets else None
    return dict(file=Path(path).name, sha256=digest.hexdigest(), bytes=byte_count,
                tokens=tokens, targets=targets, loss=loss,
                perplexity=math.exp(min(loss, 20)) if loss is not None else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('checkpoint', type=Path, help='Path to a latest.pt-style checkpoint')
    parser.add_argument('--data-dir', type=Path, default=HELDOUT_DIR)
    parser.add_argument('--output', type=Path, default=None, help='Optional new JSON report; never overwritten.')
    parser.add_argument('--chunk-bytes', type=int, default=65536)
    parser.add_argument('--context', type=int, default=None, help='Override context window (defaults to the checkpoint\'s)')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    if args.output is not None and args.output.exists():
        parser.error('Output exists; choose a new report path.')
    if args.chunk_bytes < 1:
        parser.error('Chunk size must be positive.')
    torch.set_num_threads(2)
    saved, checkpoint_sha = load_checkpoint_snapshot(args.checkpoint)
    tokenizer_path = args.checkpoint.parent / 'tokenizer.json'
    if not tokenizer_path.exists():
        raise SystemExit(f'No tokenizer.json next to {args.checkpoint}; cannot encode held-out text consistently.')
    tok = Tokenizer()
    tokenizer_sha = file_sha256(tokenizer_path)
    tok.load(tokenizer_path)
    validate_tokenizer(saved, tok)
    if file_sha256(tokenizer_path) != tokenizer_sha:
        parser.error('Tokenizer changed during setup.')

    config = Config(**saved['config'])
    context = config.context if args.context is None else args.context
    if not 1 <= context <= config.context or args.batch_size < 1:
        parser.error('Context must fit the checkpoint and batch size must be positive.')
    model = Brain(config).to(args.device)
    model.load_state_dict(saved['model'])

    files = sorted(args.data_dir.glob('*.txt'))
    if not files:
        parser.error('No evaluation text files found.')
    print(f'Checkpoint step {saved.get("step", "?")}; external-text evaluation, exposure not established.')
    results = []
    for path in files:
        result = score_file(model, path, tok, context, args.batch_size, args.device, args.chunk_bytes)
        results.append(result)
        print(json.dumps(result))
    targets = sum(row['targets'] for row in results)
    combined_loss = sum(row['loss'] * row['targets'] for row in results if row['targets']) / targets if targets else None
    report = dict(protocol='external-text-v2', perplexity_loss_cap=20, checkpoint_sha256=checkpoint_sha,
                  tokenizer_sha256=tokenizer_sha, checkpoint_step=saved.get('step'), config=saved['config'],
                  context=context, batch_size=args.batch_size, device=args.device,
                  chunk_bytes=args.chunk_bytes, torch=str(torch.__version__),
                  training_exposure='NOT ESTABLISHED; current directory exclusion is not historical provenance',
                  evaluator_sha256=file_sha256(__file__),
                  code_sha256={name: file_sha256(ROOT/name) for name in ('brain.py', 'tokenizer.py', 'artifact_io.py')},
                  results=results, targets=targets, loss=combined_loss,
                  perplexity=math.exp(min(combined_loss, 20)) if combined_loss is not None else None)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(args.output, report, overwrite=False)
    print(json.dumps({key: report[key] for key in ('targets', 'loss', 'perplexity')}))


if __name__ == '__main__':
    main()
