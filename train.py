"""Bounded training with held-out evaluation, atomic saves, and resumable AdamW."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import signal
import time
import torch
from brain import Brain, Config, device_name
from tokenizer import Tokenizer

ROOT = Path(__file__).resolve().parent


def append_ledger(record):
    path = ROOT/'runs/ledger.jsonl'
    path.parent.mkdir(exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(record) + '\n')


def atomic_json(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj), encoding='utf-8')
    tmp.replace(path)


def batch(data, size, context, device, generator=None):
    starts = torch.randint(len(data)-context, (size,), generator=generator)
    x = torch.stack([data[i:i+context] for i in starts])
    y = torch.stack([data[i+1:i+context+1] for i in starts])
    return x.to(device), y.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=ROOT/'data/demo.txt')
    parser.add_argument('--steps', type=int, default=200, help='Additional optimizer steps')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default=device_name())
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--vocab', type=int, default=2048, help='BPE vocabulary size for a fresh run (ignored on --resume)')
    parser.add_argument('--width', type=int, default=Config().width, help='Hidden width for a fresh run (ignored on --resume)')
    parser.add_argument('--layers', type=int, default=Config().layers, help='Transformer blocks for a fresh run (ignored on --resume)')
    parser.add_argument('--heads', type=int, default=Config().heads, help='Attention heads for a fresh run (ignored on --resume)')
    parser.add_argument('--context', type=int, default=Config().context, help='Context length in tokens for a fresh run (ignored on --resume)')
    parser.add_argument('--finetune', action='store_true',
                         help='Continue an existing checkpoint on a new corpus (e.g. instruction data) instead of raw pretraining text. Reuses weights and tokenizer; deliberately skips the corpus-match check.')
    parser.add_argument('--wandb', action='store_true',
                         help='Also log this run to Weights & Biases (cloud, free tier). Requires `wandb login` once beforehand. TensorBoard logging always happens locally regardless of this flag.')
    parser.add_argument('--lr', type=float, default=3e-4, help='AdamW learning rate. Applies even on --resume/--finetune (overrides the saved optimizer state).')
    parser.add_argument('--dropout', type=float, default=None,
                         help='Dropout. On a fresh run, defaults to 0. On --resume/--finetune, leaves the checkpoint\'s existing dropout untouched unless explicitly passed (safe to change any time — dropout adds no parameters).')
    parser.add_argument('--sweep', action='store_true',
                         help='Hyperparameter-search mode: always a fresh, isolated run under sweeps/ that never touches checkpoints/. Reuses a tokenizer via --tokenizer-path and caches encoded corpora for speed across many trials.')
    parser.add_argument('--tokenizer-path', type=Path, default=None, help='Reuse an existing tokenizer.json instead of training a new one (sweep mode).')
    args = parser.parse_args()
    if not 1 <= args.steps <= 10000 or not 1 <= args.batch_size <= 32:
        parser.error('Steps must be 1–10000 and batch size 1–32.')
    if not 256 <= args.vocab <= 16384:
        parser.error('Vocab size must be 256–16384.')
    if args.width % args.heads != 0:
        parser.error('Width must be divisible by heads.')
    if args.dropout is not None and not 0 <= args.dropout < 1:
        parser.error('Dropout must be in [0, 1).')
    if args.sweep:
        args.resume = False
        args.finetune = False
        out = ROOT/'sweeps'/f'run-{time.strftime("%Y%m%d-%H%M%S")}-{__import__("os").getpid()}'
        out.mkdir(parents=True)
    else:
        out = ROOT/'checkpoints'
        out.mkdir(exist_ok=True)
    checkpoint = out/'latest.pt'
    tokenizer_path = out/'tokenizer.json'
    if checkpoint.exists() and not args.resume and not args.finetune:
        parser.error('A checkpoint exists. Use --resume, --finetune, or move checkpoints/ to preserve it before starting over.')
    if args.finetune and not checkpoint.exists():
        parser.error('--finetune continues an existing checkpoint; none exists yet. Pretrain first.')
    torch.set_num_threads(4)
    torch.manual_seed(42)
    raw = args.data.read_bytes()
    if len(raw) < 4096 or len(raw) > 20_000_000:
        parser.error('Use a UTF-8 text corpus between 4 KB and 20 MB for this starter.')
    raw.decode('utf-8')
    fingerprint = hashlib.sha256(raw).hexdigest()
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True) if (args.resume or args.finetune) and checkpoint.exists() else None
    if saved and not args.finetune and saved['data_sha256'] != fingerprint:
        parser.error('The corpus changed. Preserve the old checkpoints/ folder and start a new run, or pass --finetune if this is deliberate.')
    if args.finetune and saved['data_sha256'] == fingerprint and saved.get('stage') == 'finetune':
        args.finetune = False  # same corpus as last time: this is really just a --resume of the finetune run
    tok = Tokenizer()
    if saved:
        if not tokenizer_path.exists():
            parser.error('No tokenizer.json next to this checkpoint; it predates BPE tokenization. Preserve checkpoints/ and start a new run.')
        tok.load(tokenizer_path)
    elif args.tokenizer_path:
        tok.load(args.tokenizer_path)
    elif tokenizer_path.exists():
        print('Reusing the tokenizer already saved in checkpoints/ for this corpus.', flush=True)
        tok.load(tokenizer_path)
    else:
        print(f'Training a byte-pair tokenizer (target vocab {args.vocab})…', flush=True)
        start_tok = time.monotonic()
        tok.train(raw, args.vocab)
        tok.save(tokenizer_path)
        print(f'Tokenizer ready: {tok.vocab_size} tokens in {time.monotonic()-start_tok:.1f}s', flush=True)
    cache_path = ROOT/'sweeps/cache'/f'{fingerprint[:16]}-{tok.vocab_size}.pt'
    if args.sweep and cache_path.exists():
        ids = torch.load(cache_path, weights_only=True).tolist()
    else:
        ids = tok.encode_ids(raw)
        if args.sweep:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(torch.tensor(ids, dtype=torch.long), cache_path)
    print(f'Corpus: {len(raw):,} bytes -> {len(ids):,} tokens ({len(raw)/max(len(ids),1):.2f} bytes/token)', flush=True)
    if len(ids) < 512:
        parser.error('Too few tokens after encoding for a useful context window; add more training text.')
    data = torch.tensor(ids, dtype=torch.long)
    split = int(len(data)*0.9)
    training, validation = data[:split], data[split:]
    model = Brain(Config(**saved['config']) if saved else
                  Config(width=args.width, layers=args.layers, heads=args.heads, context=args.context,
                         vocab=tok.vocab_size, dropout=args.dropout if args.dropout is not None else Config().dropout)).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    step = 0
    history = []
    stage = 'sweep' if args.sweep else 'pretrain'
    if saved:
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        for group in optimizer.param_groups:
            group['lr'] = args.lr  # otherwise the reloaded state silently overrides --lr
        if args.dropout is not None and args.dropout != model.config.dropout:
            print(f'Overriding dropout {model.config.dropout} -> {args.dropout} (shape-compatible, safe on resume).', flush=True)
            model.config.dropout = args.dropout
            for module in model.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.p = args.dropout
        entering_finetune = args.finetune and saved.get('stage', 'pretrain') != 'finetune'
        if entering_finetune:
            print('Entering fine-tuning stage: weights carried over, step count and loss history reset for this new phase.', flush=True)
            stage = 'finetune'
        else:
            step = saved['step']
            history = saved['history']
            torch.set_rng_state(saved['rng'])
            stage = saved.get('stage', 'pretrain')
    stop = False
    def request_stop(*_):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    start, target = time.monotonic(), step+args.steps
    params = sum(p.numel() for p in model.parameters())
    if args.sweep:
        run_id = f'sweep-{fingerprint[:6]}-lr{args.lr:g}-do{model.config.dropout:g}-bs{args.batch_size}-{__import__("os").getpid()}'
    elif saved and saved.get('run_id') and not entering_finetune:
        run_id = saved['run_id']  # keep the same W&B line across resumes of the same logical run
    else:
        run_id = f'{stage}-{args.data.stem}-{fingerprint[:8]}'
    run_config = dict(stage=stage, dataset=args.data.name, parameters=params, lr=args.lr,
                       batch_size=args.batch_size, data_bytes=len(raw), token_count=len(ids),
                       bytes_per_token=round(len(raw)/max(len(ids), 1), 3), **asdict(model.config))
    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(project='aven-1', id=run_id, name=run_id, resume='allow', config=run_config)
        wandb_run.define_metric('*', step_metric='step')
        wandb.watch(model, log='all', log_freq=100, log_graph=False)
        sample_prompts = ['Hello!', 'What is 6 plus 7?', 'What is the opposite of fast?',
                           'What day comes after Friday?', 'Introduce yourself']
    state = dict(status='training', step=step, target=target, parameters=params, device=args.device,
                 data_bytes=len(raw), token_count=len(ids), vocab=tok.vocab_size,
                 bytes_per_token=round(len(raw)/max(len(ids), 1), 3), stage=stage, run_id=run_id,
                 history=history, elapsed=0, dataset=args.data.name)
    def perplexity(loss):
        return round(math.exp(min(loss, 20)), 2)
    def evaluate():
        model.eval()
        values = []
        rng = torch.Generator().manual_seed(123)
        with torch.no_grad():
            for _ in range(4):
                x, y = batch(validation, args.batch_size, model.config.context, args.device, rng)
                values.append(model(x, y)[1].item())
        model.train()
        return sum(values)/len(values)
    def save():
        tmp = out/'latest.tmp'
        torch.save(dict(config=asdict(model.config), model=model.state_dict(), optimizer=optimizer.state_dict(),
                        step=step, history=history, data_sha256=fingerprint, rng=torch.get_rng_state(),
                        stage=stage, dataset=args.data.name, run_id=run_id), tmp)
        tmp.replace(checkpoint)
    if not history:
        v = evaluate()
        history.append(dict(step=0, train_loss=None, val_loss=v, val_perplexity=perplexity(v)))
    atomic_json(out/'status.json', state)
    print(f'{params:,} parameters | vocab {tok.vocab_size} | {args.device} | initial held-out perplexity {history[-1]["val_perplexity"]}', flush=True)
    last_loss = None
    try:
        while step < target and not stop:
            model.train()
            x, y = batch(training, args.batch_size, model.config.context, args.device)
            optimizer.zero_grad(set_to_none=True)
            loss = model(x, y)[1]
            if not torch.isfinite(loss):
                raise RuntimeError('Loss became nonfinite. Training stopped.')
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            last_loss = loss.item()
            if step % 20 == 0 or step == target or stop:
                val = evaluate()
                history.append(dict(step=step, train_loss=last_loss, val_loss=val, val_perplexity=perplexity(val)))
                save()
                elapsed = time.monotonic() - start
                tokens_per_sec = (step - (target - args.steps)) * args.batch_size * model.config.context / max(elapsed, 1e-6)
                weight_norm = sum(p.data.float().norm()**2 for p in model.parameters()).sqrt().item()
                if wandb_run:
                    wandb_run.log({'loss/train': last_loss, 'loss/held_out': val,
                                    'perplexity/held_out': perplexity(val),
                                    'throughput/tokens_per_sec': tokens_per_sec,
                                    'optim/grad_norm': grad_norm.item(), 'optim/weight_norm': weight_norm,
                                    'optim/lr': optimizer.param_groups[0]['lr'], 'step': step}, step=step)
                    if step % 100 == 0 or step == target:
                        rows = []
                        for p in sample_prompts:
                            wrapped = f'### Instruction:\n{p}\n\n### Response:\n' if stage == 'finetune' else p
                            text, _ = model.generate(wrapped, count=50, temperature=0.8, tokenizer=tok,
                                                      stop_text='<|end|>' if stage == 'finetune' else None)
                            rows.append([step, p, text[len(wrapped):] if text.startswith(wrapped) else text])
                        wandb_run.log({'samples': wandb.Table(columns=['step', 'prompt', 'response'], data=rows)}, step=step)
                print(f'Step {step} | train {last_loss:.3f} | held-out {val:.3f} | perplexity {perplexity(val)} | grad_norm {grad_norm:.3f}', flush=True)
            state.update(step=step, train_loss=last_loss, elapsed=round(time.monotonic()-start, 1))
            atomic_json(out/'status.json', state)
        save()
        state.update(status='paused' if stop else 'complete', step=step)
        atomic_json(out/'status.json', state)
        append_ledger(dict(run_id=run_id, stage=stage, dataset=args.data.name, parameters=params,
                            vocab=tok.vocab_size, step=step, target=target,
                            final_val_loss=history[-1]['val_loss'], final_perplexity=history[-1]['val_perplexity'],
                            status=state['status'], config=asdict(model.config),
                            timestamp=time.strftime('%Y-%m-%dT%H:%M:%S')))
    except Exception as exc:
        state.update(status='error', error=str(exc), step=step)
        atomic_json(out/'status.json', state)
        append_ledger(dict(run_id=run_id, stage=stage, dataset=args.data.name, parameters=params,
                            vocab=tok.vocab_size, step=step, target=target, status='error', error=str(exc),
                            config=asdict(model.config), timestamp=time.strftime('%Y-%m-%dT%H:%M:%S')))
        raise
    finally:
        if wandb_run:
            wandb_run.finish()


if __name__ == '__main__':
    main()
