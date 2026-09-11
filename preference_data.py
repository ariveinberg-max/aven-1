"""Prepare human comparisons without editing the preference database."""
import hashlib
import json
import random
from collections import defaultdict


def normalized(text):
    return ' '.join(text.split())


def split_comparisons(rows, val_fraction=0.2, seed=7):
    """Deduplicate comparisons, exclude conflicting labels, split by prompt.

    Reversed A/B ordering is canonicalized. The same prompt never appears in
    both training and validation. Raw source decisions remain untouched.
    """
    if not 0 < val_fraction < 1:
        raise ValueError('Validation fraction must be between zero and one.')
    groups = defaultdict(list)
    report = dict(input_count=len(rows), invalid=0, identical=0, duplicates=0, conflicting=0)
    for row in rows:
        if (row.get('winner') not in ('a', 'b') or
                any(not isinstance(row.get(k), str) or not row[k].strip() for k in ('prompt', 'response_a', 'response_b'))):
            report['invalid'] += 1
            continue
        a, b = normalized(row['response_a']), normalized(row['response_b'])
        if a == b:
            report['identical'] += 1
            continue
        key = (normalized(row['prompt']).casefold(), *sorted((a, b)))
        chosen = a if row['winner'] == 'a' else b
        groups[key].append((row, chosen))
    cleaned = []
    for key in sorted(groups):
        comparisons = groups[key]
        if len({chosen for _, chosen in comparisons}) > 1:
            report['conflicting'] += len(comparisons)
            continue
        report['duplicates'] += len(comparisons) - 1
        cleaned.append(dict(comparisons[0][0]))
    by_prompt = defaultdict(list)
    for row in cleaned:
        by_prompt[normalized(row['prompt']).casefold()].append(row)
    prompts = sorted(by_prompt)
    if len(cleaned) < 4 or len(prompts) < 2:
        raise ValueError('Need at least four distinct usable comparisons across at least two prompts after filtering.')
    random.Random(seed).shuffle(prompts)
    count = min(len(prompts) - 1, max(1, int(len(prompts) * val_fraction)))
    val_prompts = set(prompts[:count])
    train_rows = [row for row in cleaned if normalized(row['prompt']).casefold() not in val_prompts]
    val_rows = [row for row in cleaned if normalized(row['prompt']).casefold() in val_prompts]
    fingerprint = hashlib.sha256(json.dumps(cleaned, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    report.update(train_count=len(train_rows), val_count=len(val_rows),
                  train_prompt_count=len(prompts) - count, val_prompt_count=count,
                  data_sha256=fingerprint, seed=seed, split='normalized prompt groups')
    return train_rows, val_rows, report
