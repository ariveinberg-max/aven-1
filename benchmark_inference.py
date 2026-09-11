"""Small CPU inference benchmark; never loads or changes production checkpoints."""
import argparse
import json
import platform
import statistics
import time
from pathlib import Path
import torch
from brain import Brain, Config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(19)
    config = Config(width=128, layers=4, heads=4, context=128)
    model = Brain(config).eval()
    results = []
    for name, prompt, count in [('within_context', 'a' * 32, 64), ('full_context', 'a' * 128, 32)]:
        timings = {False: [], True: []}
        outputs = {}
        for iteration in range(6):
            for cached in ([False, True] if iteration % 2 == 0 else [True, False]):
                start = time.perf_counter()
                output = model.generate(prompt, count=count, temperature=0, use_cache=cached)[0]
                elapsed = time.perf_counter() - start
                outputs[cached] = output
                if iteration:
                    timings[cached].append(elapsed)
        assert outputs[False] == outputs[True], 'Greedy output differs'
        plain, cached = [statistics.median(timings[k]) for k in [False, True]]
        results.append(dict(case=name, new_tokens=count, uncached_seconds=plain,
                            cached_seconds=cached, speedup=plain/cached, identical_output=True))
    report = dict(device='cpu', threads=1, seed=19, torch=str(torch.__version__),
                  platform=platform.platform(), config=vars(config),
                  repetitions=5, statistic='median; alternating order; one warmup per mode', results=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
