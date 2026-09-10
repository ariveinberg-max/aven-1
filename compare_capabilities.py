"""Compare capability reports only when their recorded protocols match."""
import argparse
import json
import re
from pathlib import Path

MATCH_FIELDS = ('suite_version', 'suite_sha256', 'tokenizer_sha256', 'evaluator_sha256',
                'code_sha256', 'prompt_format', 'max_new_tokens', 'decoding',
                'device', 'torch_version', 'python_version')


def validate_report(report):
    if not isinstance(report, dict):
        raise ValueError('Report must be a JSON object')
    for key in MATCH_FIELDS:
        if key not in report or report[key] is None or report[key] == '':
            raise ValueError(f'Missing comparison metadata: {key}')
    for key in ('checkpoint_sha256', 'suite_sha256', 'tokenizer_sha256', 'evaluator_sha256'):
        if not isinstance(report.get(key), str) or not re.fullmatch(r'[0-9a-f]{64}', report[key]):
            raise ValueError(f'Invalid SHA-256: {key}')
    sources = report['code_sha256']
    if not isinstance(sources, dict) or not sources:
        raise ValueError('Missing supporting-code hashes')
    for name, digest in sources.items():
        if not isinstance(name, str) or not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest):
            raise ValueError('Invalid supporting-code hash')
    if type(report['max_new_tokens']) is not int or report['max_new_tokens'] < 1:
        raise ValueError('Invalid token budget')
    rows = report.get('results')
    if not isinstance(rows, list) or not rows:
        raise ValueError('Results must be a nonempty list')
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('Each result must be an object')
        for field in ('id', 'prompt', 'category', 'status'):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f'Invalid result field: {field}')
        answers = row.get('answers')
        if not isinstance(answers, list) or not answers or any(not isinstance(x, str) or not x.strip() for x in answers):
            raise ValueError('Answers must be a nonempty list of nonempty strings')
        if type(row.get('prompt_tokens')) is not int or row['prompt_tokens'] < 1:
            raise ValueError('Invalid prompt token count')


def compare(before, after):
    validate_report(before)
    validate_report(after)
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
            'regression_count': sum(c['change'] == 'regressed' for c in changes),
            'improvement_count': sum(c['change'] == 'improved' for c in changes),
            'same_checkpoint': before['checkpoint_sha256'] == after['checkpoint_sha256'],
            'overlap_evidence': {'before': before.get('overlap_check', 'NOT CHECKED'),
                                 'after': after.get('overlap_check', 'NOT CHECKED')},
            'limitations': 'Matched recorded protocol only. Overlap heuristics do not establish unseen training data; this small suite is not a general intelligence measure.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('before', type=Path)
    p.add_argument('after', type=Path)
    p.add_argument('--fail-on-regression', action='store_true', help='Exit 1 if any previously correct case becomes incorrect')
    a = p.parse_args()
    try:
        result = compare(json.loads(a.before.read_text()), json.loads(a.after.read_text()))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        p.error(str(exc))
    print(json.dumps(result, indent=2))
    if a.fail_on_regression and result['regression_count']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
