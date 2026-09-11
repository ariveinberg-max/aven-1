"""Controlled tiny-model test of response-only loss; not a capability benchmark."""
import argparse
import copy
import json
from pathlib import Path
import tempfile
import time
import numpy as np
import torch
from brain import Brain, Config
from tokenizer import Tokenizer
from train import batch, encode_corpus_to_cache
from response_mask import build_response_mask
from artifact_io import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    seed = 73
    torch.manual_seed(seed)
    config = Config(width=64, layers=2, heads=4, context=96, vocab=256)
    initial = Brain(config)
    corpus = ''.join(f'### Instruction:\nRepeat the color {color}.\n\n### Response:\n{color}\n<|end|>\n\n'
                     for _ in range(16) for color in ['red', 'blue', 'green', 'gold'])
    results = []
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source, cache, labels = [root / name for name in ['corpus.txt', 'tokens.bin', 'mask.bin']]
        source.write_text(corpus)
        tok = Tokenizer()
        count = encode_corpus_to_cache(source, tok, cache)
        build_response_mask(source, cache, tok, labels)
        data = np.memmap(cache, dtype=np.uint16, mode='r')
        mask = np.memmap(labels, dtype=np.uint8, mode='r')
        # A fixed probe on the same synthetic corpus measures optimization only.
        # It is intentionally not described as held-out generalization.
        probe_x, probe_y = batch(data, 16, config.context, 'cpu', torch.Generator().manual_seed(101), target_mask=mask)
        for mode in ['all', 'response']:
            model = copy.deepcopy(initial)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
            rng = torch.Generator().manual_seed(202)
            with torch.no_grad():
                initial_loss = model(probe_x, probe_y)[1].item()
            started = time.perf_counter()
            for _ in range(100):
                x, y = batch(data, 4, config.context, 'cpu', rng, target_mask=mask if mode == 'response' else None)
                optimizer.zero_grad(set_to_none=True)
                loss = model(x, y)[1]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
            with torch.no_grad():
                logits, final_loss, _ = model(probe_x, probe_y)
                valid = probe_y != -100
                accuracy = (logits.argmax(-1)[valid] == probe_y[valid]).float().mean().item()
            results.append(dict(loss_mode=mode, initial_response_loss=initial_loss,
                                final_response_loss=final_loss.item(), response_token_accuracy=accuracy,
                                seconds=time.perf_counter()-started))
        del data, mask
    report = dict(seed=seed, config=vars(config), steps=100, batch_size=4, device='cpu', threads=1,
                  learning_rate=0.002, torch=str(torch.__version__), token_count=count,
                  purpose='Optimization sanity check on a repeated synthetic corpus; not held-out generalization.',
                  results=results)
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
