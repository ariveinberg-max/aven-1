"""A hands-on test of one specific claim from Attention Is All You Need:
without dividing by sqrt(d_k), the dot products in QK^T grow large as the
per-head dimension grows, pushing softmax into a region with vanishingly
small gradients and destabilizing training.

Trains two otherwise-identical small models from the same random init on
the same real data and the same batches, differing only in whether the
attention scaling factor is applied. Uses a single large head (heads=1,
width=256, so d_k = 256) specifically to make the effect as visible as
possible -- with many small heads the per-head dimension is small and the
effect is easy to miss.
"""
import time
import torch
from pathlib import Path
from brain import Brain, Config
from tokenizer import Tokenizer

ROOT = Path(__file__).resolve().parent


def batch(data, size, context, device, generator):
    starts = torch.randint(len(data) - context, (size,), generator=generator)
    x = torch.stack([data[i:i + context] for i in starts])
    y = torch.stack([data[i + 1:i + context + 1] for i in starts])
    return x.to(device), y.to(device)


def main():
    tokenizer = Tokenizer().load(ROOT / 'checkpoints/tokenizer.json')
    raw = (ROOT / 'data/instructions.txt').read_bytes()
    ids = tokenizer.encode_ids(raw)
    data = torch.tensor(ids, dtype=torch.long)
    split = int(len(data) * 0.9)
    training, validation = data[:split], data[split:]
    print(f'Corpus: {len(data):,} tokens (reusing your real fine-tuning data and tokenizer)', flush=True)

    base_config = dict(width=256, layers=2, heads=1, context=64, vocab=tokenizer.vocab_size)
    print(f'Config: {base_config} -- a single head, so d_k = width = 256 (large, to make the effect visible)', flush=True)

    torch.manual_seed(42)
    scaled = Brain(Config(**base_config, attn_scale=None))  # the paper's formula: 1/sqrt(d_k)
    torch.manual_seed(42)
    unscaled = Brain(Config(**base_config, attn_scale=1.0))  # no scaling at all

    scaled_opt = torch.optim.AdamW(scaled.parameters(), lr=3e-4, weight_decay=0.01)
    unscaled_opt = torch.optim.AdamW(unscaled.parameters(), lr=3e-4, weight_decay=0.01)

    steps = 300
    batch_gen = torch.Generator().manual_seed(7)
    eval_gen = torch.Generator().manual_seed(123)

    def evaluate(model):
        model.eval()
        with torch.no_grad():
            x, y = batch(validation, 8, 64, 'cpu', eval_gen)
            loss = model(x, y)[1].item()
        model.train()
        return loss

    print(f'\n{"step":>5} | {"scaled train":>13} | {"unscaled train":>15} | {"scaled val":>11} | {"unscaled val":>13}', flush=True)
    start = time.monotonic()
    for step in range(1, steps + 1):
        x, y = batch(training, 8, 64, 'cpu', batch_gen)  # identical batch for both models

        scaled_opt.zero_grad(set_to_none=True)
        scaled_loss = scaled(x, y)[1]
        scaled_loss.backward()
        scaled_grad_norm = torch.nn.utils.clip_grad_norm_(scaled.parameters(), float('inf'))
        scaled_opt.step()

        unscaled_opt.zero_grad(set_to_none=True)
        unscaled_loss = unscaled(x, y)[1]
        unscaled_loss.backward()
        unscaled_grad_norm = torch.nn.utils.clip_grad_norm_(unscaled.parameters(), float('inf'))
        unscaled_opt.step()

        if step % 20 == 0 or step == 1:
            sv, uv = evaluate(scaled), evaluate(unscaled)
            print(f'{step:5d} | {scaled_loss.item():13.4f} | {unscaled_loss.item():15.4f} | {sv:11.4f} | {uv:13.4f}', flush=True)

    print(f'\nDone in {time.monotonic()-start:.1f}s', flush=True)
    print(f'Final gradient norm this last step -- scaled: {scaled_grad_norm:.4f} | unscaled: {unscaled_grad_norm:.4f}', flush=True)
    print('\nIf the paper\'s claim holds here: unscaled loss should be visibly higher/noisier, or its\n'
          'gradient norm anomalously small (vanishing gradients from softmax saturation) despite\n'
          'starting from the exact same weights and seeing the exact same training batches.', flush=True)


if __name__ == '__main__':
    main()
