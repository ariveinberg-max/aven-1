"""Turn one capability report into an auditable Markdown error worksheet."""
import argparse
from collections import Counter
import html
import json
from pathlib import Path
from compare_capabilities import validate_report


def cell(value):
    # Keep untrusted model text from creating Markdown rows, links or HTML.
    text = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
    return '<code>' + html.escape(text).replace('|', '&#124;').replace('\n', '<br>') + '</code>'


def worksheet(report):
    validate_report(report)
    ids = [r['id'] for r in report['results']]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate case IDs')
    rows = []
    counts = Counter()
    categories = {}
    for r in report['results']:
        if r['status'] == 'prompt_too_long':
            label = 'Not evaluated: prompt exceeds context'
            if r.get('correct') is not False or r.get('response') is not None:
                raise ValueError('Invalid skipped-case record')
        elif r['status'] == 'evaluated':
            answer = r.get('response')
            if not isinstance(answer, str) or type(r.get('correct')) is not bool:
                raise ValueError('Invalid response or score')
            if r['correct'] != (answer.strip() in r['answers']):
                raise ValueError('Stored score conflicts with response')
            if r['correct']:
                label = 'Correct'
            elif not answer.strip():
                label = 'Empty response'
            elif answer.strip().casefold() in [a.casefold() for a in r['answers']]:
                label = 'Case-only mismatch'
            else:
                label = 'Incorrect: manual review needed'
        else:
            raise ValueError('Unknown result status')
        counts[label] += 1
        c = categories.setdefault(r['category'], dict(correct=0, total=0, evaluated=0))
        c['total'] += 1
        c['evaluated'] += r['status'] == 'evaluated'
        c['correct'] += r.get('correct') is True
        if label != 'Correct':
            rows.append('| ' + ' | '.join(cell(x) for x in (r['id'],r['category'],label,r['prompt'],r['answers'],r['response'])) + ' |')
    lines = ['# Capability error worksheet', '',
             'Checkpoint: ' + cell(report['checkpoint_sha256']),
             '', 'Suite: ' + cell(report['suite_version']) + ' / ' + cell(report['suite_sha256']),
             '', 'Tokenizer: ' + cell(report['tokenizer_sha256']),
             '', 'Overlap evidence: ' + cell(report.get('overlap_check','NOT CHECKED')),
             '', 'Scores are recomputed from recorded responses. These labels describe output shape, not the cause of failure. Do not turn these test items into training examples.',
             '', '| Category | Correct | Total | Evaluated |', '|---|---:|---:|---:|']
    for name,c in sorted(categories.items()):
        lines.append(f"| {cell(name)} | {c['correct']} | {c['total']} | {c['evaluated']} |")
    lines += ['', '## Observed outcomes', '']
    lines += [f'- {label}: {count}' for label,count in sorted(counts.items())]
    lines += ['', '## Cases to review', '', '| ID | Category | Observation | Prompt | Accepted answers | Response |', '|---|---|---|---|---|---|']
    lines += rows or ['| — | — | No failed or skipped cases | — | — | — |']
    lines += ['', 'Manual review: distinguish factual/reasoning errors from formatting failures, prompt misunderstanding, repetition and context limits. A correct answer embedded in an otherwise wrong response still fails the frozen exact-match protocol. Do not change scoring after examining results.', '']
    return '\n'.join(lines)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('report',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    try:
        result=worksheet(json.loads(a.report.read_text()))
        # Exclusive creation prevents replacing an existing report or checkpoint.
        with a.output.open('x',encoding='utf-8') as f: f.write(result)
    except (OSError,ValueError,KeyError,TypeError) as exc:
        p.error(str(exc))
    print(a.output)


if __name__=='__main__': main()
