"""Response-only labels for the project's line-delimited instruction format."""
from pathlib import Path
import numpy as np
from artifact_io import atomic_write


def labeled_spans(path):
    """Stream byte counts and labels; include end markers as response targets."""
    in_response = False
    with Path(path).open('rb') as source:
        while True:
            line = source.readline(1_000_001)
            if not line:
                break
            if len(line) > 1_000_000:
                raise ValueError('Response training requires lines under 1 MB.')
            marker = line.rstrip(b'\r\n')
            if marker == b'### Instruction:':
                in_response = False
                label = False
            elif marker == b'### Response:':
                in_response = True
                label = False
            elif marker == b'<|end|>':
                label = in_response
                in_response = False
            else:
                label = in_response
            yield len(line), label


def build_response_mask(corpus, token_cache, tokenizer, output):
    """Align token IDs to source byte spans without retokenizing the corpus.

    A token crossing a prompt/response boundary is excluded; it would otherwise
    teach prompt bytes. Memory use is bounded by a token chunk and one text line.
    """
    spans = iter(labeled_spans(corpus))
    remaining, active = next(spans, (0, False))
    targets = 0
    total = 0
    def write(destination):
        nonlocal remaining, active, targets, total
        with Path(token_cache).open('rb') as tokens:
            while True:
                block = tokens.read(65536 * 2)
                if not block:
                    break
                if len(block) % 2:
                    raise ValueError('Incomplete token cache.')
                ids = np.frombuffer(block, dtype=np.uint16)
                labels = np.empty(len(ids), dtype=np.uint8)
                for index, token in enumerate(ids):
                    needed = len(tokenizer.vocab[int(token)])
                    supervised = True
                    while needed:
                        if not remaining:
                            remaining, active = next(spans, (0, False))
                            if not remaining:
                                raise ValueError('Token cache extends beyond the corpus.')
                        take = min(needed, remaining)
                        supervised = supervised and active
                        needed -= take
                        remaining -= take
                    labels[index] = supervised
                targets += int(labels.sum())
                total += len(labels)
                destination.write(labels.tobytes())
        if remaining or next(spans, None) is not None:
            raise ValueError('Token cache does not cover the corpus.')
        if not targets:
            raise ValueError('No response targets found; use ### Response: and <|end|> on separate lines.')
    atomic_write(output, write)
    return dict(tokens=total, response_tokens=targets)
