"""Read-only checkpoint evaluation; never trains or executes generated code."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import torch
from brain import Brain, Config
from tokenizer import Tokenizer


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def score(response, answers):
    # Strict whole-response scoring: whitespace at the edges is ignored, nothing else.
    return response.strip() in answers


def overlap(cases, paths):
    findings = []
    for path in paths:
        text = ' '.join(Path(path).read_text().casefold().split())
        for case in cases:
            words = case['prompt'].casefold().split()
            for i in range(max(0, len(words) - 7)):
                span = ' '.join(words[i:i+8])
                if span in text:
                    findings.append(dict(case_id=case['id'], corpus=str(path), matching_span=span))
                    break
    return findings


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--expect-sha256', required=True)
    p.add_argument('--expect-tokenizer-sha256', required=True)
    p.add_argument('--suite', type=Path, default=Path(__file__).parent/'research/evaluation/capability-v1.json')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--corpus', type=Path, action='append', default=[])
    p.add_argument('--device', choices=['cpu','mps','cuda'], default='cpu')
    p.add_argument('--format', choices=['plain','instruction'], default='plain')
    p.add_argument('--max-new-tokens', type=int, default=32)
    a = p.parse_args()
    if a.output.exists(): p.error('Output already exists; choose a new report path.')
    if a.max_new_tokens < 1: p.error('--max-new-tokens must be positive.')
    checkpoint_hash, tokenizer_hash = sha(a.checkpoint), sha(a.tokenizer)
    if checkpoint_hash != a.expect_sha256: p.error('Checkpoint SHA-256 mismatch.')
    if tokenizer_hash != a.expect_tokenizer_sha256: p.error('Tokenizer SHA-256 mismatch.')
    suite = json.loads(a.suite.read_text())
    cases = suite['cases']
    if len({c['id'] for c in cases}) != len(cases): p.error('Duplicate case IDs.')
    if len({c['prompt'] for c in cases}) != len(cases): p.error('Duplicate prompts.')
    hits = overlap(cases, a.corpus)
    if hits: p.error('Evaluation/corpus overlap detected: ' + json.dumps(hits))
    torch.manual_seed(42)
    torch.set_num_threads(4)
    saved = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    model = Brain(Config(**saved['config']))
    model.load_state_dict(saved['model'], strict=True)
    model.to(a.device).eval()
    tok = Tokenizer().load(a.tokenizer)
    if tok.vocab_size != model.config.vocab: p.error('Tokenizer/model vocabulary mismatch.')
    results = []
    for case in cases:
        prompt = case['prompt'] + '\nAnswer:' if a.format == 'plain' else f"### Instruction:\n{case['prompt']}\n\n### Response:\n"
        tokens = tok.encode(prompt)
        if len(tokens) > model.config.context:
            results.append(dict(**case, status='prompt_too_long', prompt_tokens=len(tokens), response=None, correct=False))
            continue
        # Decode generated tokens only; stop at newline for the fixed short-answer protocol.
        ids = torch.tensor([tokens], device=a.device)
        generated = []
        with torch.inference_mode():
            for _ in range(a.max_new_tokens):
                logits, _, _ = model(ids[:, -model.config.context:])
                next_id = logits[:, -1].argmax(-1, keepdim=True)
                generated.append(next_id.item())
                ids = torch.cat([ids, next_id], dim=1)
                if '\n' in tok.decode(generated) or '<|end|>' in tok.decode(generated): break
        response = tok.decode(generated).split('\n',1)[0].split('<|end|>',1)[0]
        results.append(dict(**case, status='evaluated', prompt_tokens=len(tokens), response=response, correct=score(response,case['answers'])))
    categories = defaultdict(lambda: dict(correct=0,total=0,evaluated=0))
    for row in results:
        c = categories[row['category']]
        c['total'] += 1
        c['correct'] += int(row['correct'])
        c['evaluated'] += int(row['status']=='evaluated')
    for c in categories.values(): c['accuracy'] = c['correct']/c['total']
    report = dict(suite_version=suite['version'], suite_sha256=sha(a.suite), checkpoint_sha256=checkpoint_hash,
                  tokenizer_sha256=tokenizer_hash, checkpoint_step=saved['step'], config=saved['config'],
                  parameters=sum(p.numel() for p in model.parameters()), device=a.device,
                  torch_version=str(torch.__version__), python_version=platform.python_version(),
                  code_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).parent,text=True).strip(),
                  evaluator_sha256=sha(__file__), prompt_format=a.format, max_new_tokens=a.max_new_tokens,
                  decoding='greedy; first newline or <|end|>; strict case-sensitive exact match after stripping edge whitespace',
                  overlap_check='8-word exact normalized spans; no hits' if a.corpus else 'NOT CHECKED',
                  corpora=[dict(path=str(x),sha256=sha(x)) for x in a.corpus], categories=dict(categories),results=results)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f: json.dump(report,f,indent=2)
    print(json.dumps(report['categories'],indent=2))
    print('Report:',a.output)


if __name__ == '__main__': main()
