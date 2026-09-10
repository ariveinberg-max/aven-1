"""Compare capability reports only when their recorded protocols match."""
import argparse
import json
from pathlib import Path

MATCH_FIELDS = ('suite_version', 'suite_sha256', 'tokenizer_sha256', 'evaluator_sha256',
                'code_sha256', 'prompt_format', 'max_new_tokens', 'decoding',
                'device', 'torch_version', 'python_version')


def compare(before, after):
    for key in MATCH_FIELDS:
        if key not in before or key not in after:
            raise ValueError(f'Missing comparison metadata: {key}')
        if before[key] != after[key]:
            raise ValueError(f'Incompatible reports: {key} differs')
    def index(report):
        rows = report['results']
        result = {r['id']: r for r in rows}
        if not result or len(result) != len(rows):
            raise ValueError('Empty suite or duplicate case IDs')
        for r in rows:
            if r['status'] != 'evaluated':
                raise ValueError(f"Unevaluated case: {r['id']}")
            if not isinstance(r['response'], str) or not isinstance(r['correct'], bool):
                raise ValueError(f"Invalid result: {r['id']}")
            if r['correct'] != (r['response'].strip() in r['answers']):
                raise ValueError(f"Stored score does not match response: {r['id']}")
        return result
    left, right = index(before), index(after)
    if left.keys() != right.keys():
        raise ValueError('Case IDs differ')
    categories = {}
    changes = []
    for key, old in left.items():
        new = right[key]
        for field in ('prompt', 'answers', 'category', 'prompt_tokens'):
            if old[field] != new[field]:
                raise ValueError(f'Case {key}: {field} differs')
        c = categories.setdefault(old['category'], {'total': 0, 'before_correct': 0, 'after_correct': 0})
        c['total'] += 1
        c['before_correct'] += int(old['correct'])
        c['after_correct'] += int(new['correct'])
        if old['correct'] != new['correct']:
            changes.append({'id': key, 'change': 'improved' if new['correct'] else 'regressed',
                            'before_response': old['response'], 'after_response': new['response']})
    for c in categories.values():
        c['accuracy_change_percentage_points'] = 100 * (c['after_correct'] - c['before_correct']) / c['total']
    return {'before_checkpoint_sha256': before['checkpoint_sha256'],
            'after_checkpoint_sha256': after['checkpoint_sha256'],
            'categories': categories, 'changed_cases': changes,
            'overlap_evidence': {'before': before.get('overlap_check', 'NOT CHECKED'),
                                 'after': after.get('overlap_check', 'NOT CHECKED')},
            'limitations': 'Matched recorded protocol only. Overlap heuristics do not establish unseen training data; this small suite is not a general intelligence measure.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('before', type=Path)
    p.add_argument('after', type=Path)
    a = p.parse_args()
    try:
        result = compare(json.loads(a.before.read_text()), json.loads(a.after.read_text()))
    except (ValueError, KeyError, TypeError) as exc:
        p.error(str(exc))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
