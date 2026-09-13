"""Read-only CPU decoding parity/timing check against a checkpoint bundle."""
import argparse
import json
from pathlib import Path
import statistics
import time
import torch
from brain import Brain, Config
from tokenizer import Tokenizer
from checkpoint_bundle import verify_bundle
from artifact_io import allow_safe_rng_globals, atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = verify_bundle(args.bundle)
    torch.set_num_threads(1)
    allow_safe_rng_globals()
    saved = torch.load(args.bundle/'latest.pt', map_location='cpu', weights_only=True, mmap=True)
    model = Brain(Config(**saved['config'])).eval()
    model.load_state_dict(saved['model'])
    tokenizer = Tokenizer().load(args.bundle/'tokenizer.json')
    results = []
    for case, prompt in [('chat', '### Instruction:\nHello!\n\n### Response:\n'),
                         ('full_window', 'A small model reads text. ' * model.config.context)]:
        timings = {False: [], True: []}
        outputs = {}
        for iteration in range(4):
            for cached in ([False, True] if iteration % 2 == 0 else [True, False]):
                start = time.perf_counter()
                result = model.generate(prompt, count=16, temperature=0, tokenizer=tokenizer, use_cache=cached)
                duration = time.perf_counter() - start
                outputs[cached] = result
                if iteration:
                    timings[cached].append(duration)
            if outputs[False] != outputs[True]:
                raise RuntimeError('Cached and uncached generation or activity differs.')
        plain, cached = [statistics.median(timings[key]) for key in [False, True]]
        results.append(dict(case=case, prompt_tokens=len(tokenizer.encode(prompt)), new_tokens=16,
                            uncached_seconds=plain, cached_seconds=cached, speedup=plain/cached,
                            identical_output_and_activity=True))
    report = dict(checkpoint_sha256=manifest['files']['latest.pt']['sha256'],
                  checkpoint_step=saved['step'], config=saved['config'],
                  device='cpu', threads=1, torch=str(torch.__version__), repetitions=3,
                  statistic='median; alternating order after warmup; includes tokenization and activity inspection',
                  results=results)
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
