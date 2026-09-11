"""Bounded training with held-out evaluation, atomic saves, and resumable AdamW."""
import argparse
import codecs
from dataclasses import asdict
import hashlib
import json
import math
import os
import uuid
import platform
import numpy as np
from tracking import prediction_metrics, memory_metrics, diagnostics
from pathlib import Path
import signal
import time
import torch
from brain import Brain, Config, device_name
from tokenizer import Tokenizer
from run_lock import WriterLock, WriterBusy
from artifact_io import atomic_json, atomic_torch_save, file_sha256
from response_mask import build_response_mask
from rng_state import capture_rng, restore_rng

ROOT = Path(__file__).resolve().parent


def torch_device(name):
    """Resolve a --device string to what .to() actually needs. 'dml' (DirectML,
    for AMD/Intel GPUs on Windows -- no CUDA equivalent exists for them) isn't a
    plain torch device string like 'cpu'/'cuda'/'mps': it needs torch_directml's
    own device object. Lazily imported so machines without torch-directml
    installed (e.g. this Mac) are unaffected by --device never being 'dml' here.
    args.device itself stays the plain string everywhere else (metadata, logging,
    checkpoint fields) -- only actual tensor .to() calls need this resolved form.
    """
    if name == 'dml':
        import torch_directml
        return torch_directml.device()
    return name


def append_ledger(record):
    path = ROOT/'runs/ledger.jsonl'
    path.parent.mkdir(exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(record) + '\n')


def batch(data, size, context, device, generator=None, target_mask=None):
    """data may be a torch tensor (tests, small in-memory corpora) or a numpy
    memmap (real training -- see the streaming pipeline in main()); as_tensor
    handles either uniformly without loading the whole backing store."""
    for _ in range(50 if target_mask is not None else 1):
        starts = torch.randint(len(data)-context, (size,), generator=generator).tolist()
        if torch.is_tensor(data):
            x = torch.stack([data[i:i+context].to(torch.long) for i in starts])
            y = torch.stack([data[i+1:i+context+1].to(torch.long) for i in starts])
        else:
            # np.array() copies -- trivial cost for one context-length window,
            # but avoids writing through to the on-disk memmap cache.
            x = torch.stack([torch.from_numpy(np.array(data[i:i+context])).long() for i in starts])
            y = torch.stack([torch.from_numpy(np.array(data[i+1:i+context+1])).long() for i in starts])
        if target_mask is not None:
            mask = torch.stack([torch.as_tensor(np.array(target_mask[i+1:i+context+1]), dtype=torch.bool) for i in starts])
            y = y.masked_fill(~mask, -100)
            if not mask.any():
                continue
        return x.to(device), y.to(device)
    raise ValueError('Could not sample response targets; use denser instruction data or a larger context.')


def stream_fingerprint_and_validate(path, chunk_size=8 * 1024 * 1024):
    """Stream-hash a corpus file (sha256) and validate it's real UTF-8,
    without loading it into memory -- needed once a corpus reaches hundreds
    of MB or more (a full read_bytes()+decode() would itself use as much RAM
    as the file is large, on top of everything else)."""
    digest = hashlib.sha256()
    decoder = codecs.getincrementaldecoder('utf-8')()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            digest.update(chunk)
            try:
                decoder.decode(chunk)
            except UnicodeDecodeError as e:
                raise ValueError(f'Corpus is not valid UTF-8: {e}')
        decoder.decode(b'', final=True)
    return digest.hexdigest()


TOKENIZER_TRAIN_SAMPLE_BYTES = 20_000_000  # bounded sample for BPE merge learning on huge corpora
ENCODE_CHUNK_BYTES = 8 * 1024 * 1024


def encode_corpus_to_cache(data_path, tok, cache_path, chunk_size=ENCODE_CHUNK_BYTES):
    """Stream-encode a corpus to a token-ID cache file on disk (uint16,
    since vocab is capped at 16384) and return the total token count,
    without ever holding the full token array in memory. Chunk boundaries
    (every chunk_size bytes) mean a handful of merges that would span a
    boundary are missed -- a small, accepted imprecision, same tradeoff most
    streaming tokenizers make; on a real corpus this affects a negligible
    fraction of tokens."""
    tmp = cache_path.with_name(cache_path.name + f'.{uuid.uuid4().hex}.tmp')
    token_count = 0
    try:
        with data_path.open('rb') as f, open(tmp, 'wb') as out_f:
            for chunk in iter(lambda: f.read(chunk_size), b''):
                ids = tok.encode_ids(chunk)
                np.asarray(ids, dtype=np.uint16).tofile(out_f)
                token_count += len(ids)
        tmp.replace(cache_path)
    finally:
        tmp.unlink(missing_ok=True)
    return token_count


def valid_token_cache(path, tokenizer_fingerprint):
    """A name match alone cannot detect a truncated or corrupted cache."""
    try:
        metadata = json.loads(path.with_suffix('.json').read_text())
        size = path.stat().st_size
        return (size > 0 and size % 2 == 0 and metadata['bytes'] == size
                and metadata['tokenizer_sha256'] == tokenizer_fingerprint
                and metadata['sha256'] == file_sha256(path))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=ROOT/'data/demo.txt')
    parser.add_argument('--init-from', type=Path, default=None, help='Initialize an isolated fine-tune from another checkpoint directory; requires --output-dir.')
    parser.add_argument('--loss-mode', choices=['all', 'response'], default=None, help='Response-only instruction loss; default inherits checkpoint or all tokens.')
    parser.add_argument('--output-dir', type=Path, default=None, help='Isolated checkpoint directory; defaults to checkpoints/.')
    parser.add_argument('--steps', type=int, default=200, help='Additional optimizer steps')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda', 'dml'], default=device_name())
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--expect-params', type=int, help='Require this parameter count when resuming or fine-tuning a checkpoint')
    for field in ('vocab', 'width', 'layers', 'heads', 'context'):
        parser.add_argument(f'--{field}', type=int, default=None,
                            help=f'{field} for a fresh run; if explicitly passed on resume/finetune, must match saved config')
    parser.add_argument('--finetune', action='store_true',
                         help='Continue an existing checkpoint on a new corpus (e.g. instruction data) instead of raw pretraining text. Reuses weights and tokenizer; deliberately skips the corpus-match check.')
    parser.add_argument('--wandb', action='store_true',
                         help='Log analytics to Weights & Biases. Use WANDB_MODE=offline for local-only logging; online mode requires wandb login.')
    parser.add_argument('--wandb-project', default='aven-1')
    parser.add_argument('--wandb-entity', default=None, help='Optional W&B account or team')
    parser.add_argument('--wandb-samples', action='store_true', help='Log generated text samples, which may reproduce training text')
    parser.add_argument('--lr', type=float, default=None, help='Explicit AdamW learning-rate override. Otherwise inherit the checkpoint rate, or use 3e-4 for a new run.')
    parser.add_argument('--dropout', type=float, default=None,
                         help='Dropout. On a fresh run, defaults to 0. On --resume/--finetune, leaves the checkpoint\'s existing dropout untouched unless explicitly passed (safe to change any time — dropout adds no parameters).')
    parser.add_argument('--sweep', action='store_true',
                         help='Hyperparameter-search mode: always a fresh, isolated run under sweeps/ that never touches checkpoints/. Reuses a tokenizer via --tokenizer-path and caches encoded corpora for speed across many trials.')
    parser.add_argument('--tokenizer-path', type=Path, default=None, help='Reuse an existing tokenizer.json instead of training a new one (sweep mode).')
    args = parser.parse_args()
    if not 1 <= args.steps <= 10000 or not 1 <= args.batch_size <= 32:
        parser.error('Steps must be 1–10000 and batch size 1–32.')
    if args.lr is not None and (not math.isfinite(args.lr) or args.lr <= 0):
        parser.error('Learning rate must be finite and positive.')
    if args.init_from is not None:
        if args.resume or args.output_dir is None or args.sweep:
            parser.error('--init-from requires --output-dir and cannot be combined with --resume or --sweep.')
        if args.init_from.resolve() == args.output_dir.resolve():
            parser.error('--init-from and --output-dir must be different directories.')
        args.finetune = True
    if args.sweep and args.output_dir is not None:
        parser.error('--sweep creates its own isolated output directory; omit --output-dir.')
    if args.dropout is not None and not 0 <= args.dropout < 1:
        parser.error('Dropout must be in [0, 1).')
    if args.sweep:
        args.resume = False
        args.finetune = False
        out = ROOT/'sweeps'/f'run-{time.strftime("%Y%m%d-%H%M%S")}-{__import__("os").getpid()}'
    else:
        out = args.output_dir.resolve() if args.output_dir else ROOT/'checkpoints'
    try:
        with WriterLock(out):
            run_training(args, parser, out)
    except WriterBusy as exc:
        parser.error(str(exc))


def run_training(args, parser, out):
    checkpoint = out/'latest.pt'
    tokenizer_path = out/'tokenizer.json'
    load_checkpoint = args.init_from.resolve()/'latest.pt' if args.init_from else checkpoint
    load_tokenizer = load_checkpoint.parent/'tokenizer.json'
    if args.init_from and checkpoint.exists():
        parser.error('--init-from requires an output directory without a checkpoint; use --resume for an existing child.')
    if checkpoint.exists() and not args.resume and not args.finetune:
        parser.error('A checkpoint exists. Use --resume, --finetune, or move checkpoints/ to preserve it before starting over.')
    if (args.resume or args.finetune) and not load_checkpoint.exists():
        parser.error('--resume/--finetune requires an existing checkpoint; none exists.')
    torch.set_num_threads(4)
    torch.manual_seed(42)
    data_size = args.data.stat().st_size
    if data_size < 4096:
        parser.error('Use a UTF-8 text corpus of at least 4 KB.')
    try:
        fingerprint = stream_fingerprint_and_validate(args.data)
    except ValueError as e:
        parser.error(str(e))
    saved = None
    parent_checkpoint_sha256 = None
    if args.resume or args.finetune:
        # Hash and load the same open file even if another process replaces its path.
        with load_checkpoint.open('rb') as source:
            digest = hashlib.sha256()
            for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
            parent_checkpoint_sha256 = digest.hexdigest()
            source.seek(0)
            saved = torch.load(source, map_location='cpu', weights_only=True)
    if saved:
        # Brain ties output.weight to token.weight; count that parameter only once.
        saved_params = sum(value.numel() for name, value in saved['model'].items()
                           if name != 'output.weight')
        if args.expect_params is not None and args.expect_params != saved_params:
            parser.error(f'--expect-params mismatch: expected {args.expect_params}, checkpoint has {saved_params} parameters.')
        for field in ('width', 'layers', 'heads', 'context', 'vocab'):
            explicit = getattr(args, field)
            actual = saved['config'].get(field)
            if explicit is not None and explicit != actual:
                parser.error(f'--{field} mismatch: requested {explicit}, checkpoint has {actual}.')
            setattr(args, field, actual)
    else:
        for field in ('width', 'layers', 'heads', 'context', 'vocab'):
            if getattr(args, field) is None:
                setattr(args, field, 2048 if field == 'vocab' else getattr(Config(), field))
    if not 256 <= args.vocab <= 16384:
        parser.error('Vocab size must be 256–16384.')
    if args.heads <= 0 or args.width <= 0 or args.width % args.heads != 0:
        parser.error('Width and heads must be positive; width must be divisible by heads.')
    if args.layers <= 0 or args.context <= 0:
        parser.error('Layers and context must be positive.')
    loss_mode = args.loss_mode or (saved.get('loss_mode', 'all') if saved else 'all')
    if saved and not args.finetune and loss_mode != saved.get('loss_mode', 'all'):
        parser.error('Changing loss mode requires --finetune, not --resume.')
    if saved and not args.finetune and saved['data_sha256'] != fingerprint:
        parser.error('The corpus changed. Preserve the old checkpoints/ folder and start a new run, or pass --finetune if this is deliberate.')
    if args.finetune and not args.init_from and saved['data_sha256'] == fingerprint and saved.get('stage') == 'finetune' and loss_mode == saved.get('loss_mode', 'all'):
        args.finetune = False  # same corpus as last time: this is really just a --resume of the finetune run
    out.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer()
    if saved:
        if not load_tokenizer.exists():
            parser.error('No tokenizer.json next to this checkpoint; it predates BPE tokenization. Preserve checkpoints/ and start a new run.')
        tok.load(load_tokenizer)
    elif args.tokenizer_path:
        tok.load(args.tokenizer_path)
    elif tokenizer_path.exists():
        print('Reusing the tokenizer already saved in checkpoints/ for this corpus.', flush=True)
        tok.load(tokenizer_path)
    else:
        print(f'Training a byte-pair tokenizer (target vocab {args.vocab})…', flush=True)
        start_tok = time.monotonic()
        with args.data.open('rb') as f:
            sample = f.read(TOKENIZER_TRAIN_SAMPLE_BYTES)
        tok.train(sample, args.vocab)
        tok.save(tokenizer_path)
        print(f'Tokenizer ready: {tok.vocab_size} tokens in {time.monotonic()-start_tok:.1f}s', flush=True)
    tokenizer_fingerprint = tok.fingerprint()
    if args.init_from and tokenizer_path.exists() and Tokenizer().load(tokenizer_path).fingerprint() != tokenizer_fingerprint:
        parser.error('Output directory already contains a different tokenizer; choose a new --output-dir.')
    if saved and tok.vocab_size != args.vocab:
        parser.error('Tokenizer vocabulary does not match the checkpoint.')
    if saved and saved.get('tokenizer_sha256', tokenizer_fingerprint) != tokenizer_fingerprint:
        parser.error('Tokenizer merge rules changed since this checkpoint was saved.')
    # Reused sweep tokenizers must also travel with their resulting checkpoint.
    if not tokenizer_path.exists():
        tok.save(tokenizer_path)
    cache_dir = ROOT/'data/.cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    token_cache_path = cache_dir/f'v2-{fingerprint}-{tokenizer_fingerprint}-{ENCODE_CHUNK_BYTES}.bin'
    if valid_token_cache(token_cache_path, tokenizer_fingerprint):
        token_count = token_cache_path.stat().st_size // 2  # uint16 = 2 bytes/token
        print(f'Reusing cached token encoding ({token_count:,} tokens).', flush=True)
    else:
        print('Encoding corpus to tokens…', flush=True)
        start_enc = time.monotonic()
        token_count = encode_corpus_to_cache(args.data, tok, token_cache_path)
        atomic_json(token_cache_path.with_suffix('.json'), dict(tokenizer_sha256=tokenizer_fingerprint,
                    bytes=token_count * 2, sha256=file_sha256(token_cache_path)))
        print(f'Encoded in {time.monotonic()-start_enc:.1f}s', flush=True)
    print(f'Corpus: {data_size:,} bytes -> {token_count:,} tokens ({data_size/max(token_count,1):.2f} bytes/token)', flush=True)
    if token_count < 512:
        parser.error('Too few tokens after encoding for a useful context window; add more training text.')
    data = np.memmap(token_cache_path, dtype=np.uint16, mode='r', shape=(token_count,))
    split = int(len(data)*0.9)
    training, validation = data[:split], data[split:]
    training_mask = validation_mask = None
    if loss_mode == 'response':
        mask_path = out/'response-mask.bin'
        try:
            mask_info = build_response_mask(args.data, token_cache_path, tok, mask_path)
        except ValueError as exc:
            parser.error(str(exc))
        masks = np.memmap(mask_path, dtype=np.uint8, mode='r', shape=(token_count,))
        training_mask, validation_mask = masks[:split], masks[split:]
        if not training_mask[1:].any() or not validation_mask[1:].any():
            parser.error('Both training and validation splits need response targets.')
        print(f"Response-only loss: {mask_info['response_tokens']:,}/{token_count:,} tokens supervised.", flush=True)
    device_obj = torch_device(args.device)
    model = Brain(Config(**saved['config']) if saved else
                  Config(width=args.width, layers=args.layers, heads=args.heads, context=args.context,
                         vocab=tok.vocab_size, dropout=args.dropout if args.dropout is not None else Config().dropout)).to(device_obj)
    explicit_lr = args.lr
    if args.lr is None:
        args.lr = saved['optimizer']['param_groups'][0]['lr'] if saved else 3e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    step = 0
    history = []
    rng_resume = 'new_phase'
    stage = 'sweep' if args.sweep else 'finetune' if loss_mode == 'response' else 'pretrain'
    if saved:
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        if explicit_lr is not None:
            for group in optimizer.param_groups:
                group['lr'] = args.lr
        if args.dropout is not None and args.dropout != model.config.dropout:
            print(f'Overriding dropout {model.config.dropout} -> {args.dropout} (shape-compatible, safe on resume).', flush=True)
            model.config.dropout = args.dropout
            for module in model.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.p = args.dropout
        entering_finetune = args.finetune
        if entering_finetune:
            print('Entering fine-tuning stage: weights carried over, step count and loss history reset for this new phase.', flush=True)
            stage = 'finetune'
        else:
            step = saved['step']
            history = saved['history']
            rng_resume = restore_rng(saved, args.device)
            stage = saved.get('stage', 'pretrain')
    stop = False
    def request_stop(*_):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    start = time.monotonic()
    session_start_step = step
    budget_id = os.environ.get('AVEN_BUDGET_ID')
    target = step + args.steps
    if budget_id and saved and not args.finetune and saved.get('budget_id') == budget_id:
        if saved.get('budget_steps') != args.steps:
            parser.error('Retry step budget changed within the same supervisor chunk.')
        target = saved['budget_target']
        if not isinstance(target, int) or target < step:
            parser.error('Invalid saved retry target.')
    params = sum(p.numel() for p in model.parameters())
    run_id = f'{stage}-{fingerprint[:8]}-{uuid.uuid4().hex}'
    # Older checkpoints have only run_id. Siblings from the same legacy parent
    # must still share a lineage even though each execution gets a new run ID.
    lineage_id = ((saved.get('lineage_id') or saved.get('run_id') or
                   f'lineage-{uuid.uuid5(uuid.NAMESPACE_URL, parent_checkpoint_sha256).hex}')
                  if saved else run_id)
    provenance = dict(parent_checkpoint_sha256=parent_checkpoint_sha256,
                      lineage_id=lineage_id, platform=args.device,
                      starting_step=saved['step'] if saved else 0,
                      tokenizer_sha256=tokenizer_fingerprint, loss_mode=loss_mode,
                      rng_resume=rng_resume, budget_id=budget_id, budget_target=target, budget_steps=args.steps)
    run_config = dict(stage=stage, dataset=args.data.name, parameters=params, lr=args.lr,
                       batch_size=args.batch_size, data_bytes=data_size, token_count=token_count,
                       bytes_per_token=round(data_size/max(token_count, 1), 3), **asdict(model.config))
    run_config.update(provenance)
    run_config.update(device=args.device, torch_version=str(torch.__version__), python_version=platform.python_version(), seed=42,
                      data_sha256=fingerprint, train_tokens=len(training), validation_tokens=len(validation),
                      optimizer='AdamW', weight_decay=0.01, grad_clip=1.0, evaluation_batches=4,
                      validation_seed=123, samples_enabled=args.wandb_samples)
    if min(len(training), len(validation)) <= model.config.context:
        parser.error('Each data split must contain more tokens than the model context.')
    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(project=args.wandb_project, entity=args.wandb_entity, id=run_id, name=run_id, resume='never', group=lineage_id, tags=[lineage_id], config=run_config, save_code=False)
        wandb_run.define_metric('step')
        wandb_run.define_metric('*', step_metric='step')
        wandb_run.define_metric('loss/held_out', summary='min')
        wandb_run.define_metric('accuracy/held_out', summary='max')
        wandb_run.summary['status'] = 'training'
        print(f'W&B tracking: {wandb_run.url or "offline local run"}', flush=True)
        wandb.watch(model, log='all', log_freq=100, log_graph=False)
        sample_prompts = ['Hello!', 'What is 6 plus 7?', 'What is the opposite of fast?',
                           'What day comes after Friday?', 'Introduce yourself']
    state = dict(status='training', step=step, target=target, parameters=params, device=args.device,
                 data_bytes=data_size, token_count=token_count, vocab=tok.vocab_size,
                 bytes_per_token=round(data_size/max(token_count, 1), 3), stage=stage, run_id=run_id,
                 history=history, elapsed=0, dataset=args.data.name, writer_pid=os.getpid(), **provenance)
    def perplexity(loss):
        return round(math.exp(min(loss, 20)), 2)
    eval_metrics = {}
    def evaluate():
        model.eval()
        values = []
        measurements = []
        rng = torch.Generator().manual_seed(123)
        with torch.no_grad():
            for _ in range(4):
                x, y = batch(validation, args.batch_size, model.config.context, device_obj, rng, target_mask=validation_mask)
                logits, eval_loss, _ = model(x, y)
                values.append((eval_loss.item(), int((y != -100).sum().item())))
                if wandb_run:
                    measurements.append(prediction_metrics(logits, y))
        if measurements:
            eval_metrics.update({k: sum(m[k] * count for m, (_, count) in zip(measurements, values)) / sum(count for _, count in values) for k in measurements[0]})
        model.train()
        return sum(value * count for value, count in values) / sum(count for _, count in values)
    def save():
        atomic_torch_save(checkpoint, dict(config=asdict(model.config), model=model.state_dict(), optimizer=optimizer.state_dict(),
                        step=step, history=history, data_sha256=fingerprint, rng=torch.get_rng_state(), rng_state=capture_rng(args.device),
                        stage=stage, dataset=args.data.name, run_id=run_id, **provenance))
    if not history:
        v = evaluate()
        history.append(dict(step=0, train_loss=None, val_loss=v, val_perplexity=perplexity(v)))
    atomic_json(out/'status.json', state)
    print(f'{params:,} parameters | vocab {tok.vocab_size} | {args.device} | initial held-out perplexity {history[-1]["val_perplexity"]}', flush=True)
    if wandb_run and step == 0:
        wandb_run.log({'step': 0, 'loss/held_out': history[-1]['val_loss'],
                       'perplexity/held_out': history[-1]['val_perplexity'],
                       'accuracy/held_out': eval_metrics.get('accuracy', 0)})
    last_loss = None
    interval_losses, interval_grads = [], []
    interval_start = time.monotonic()
    try:
        while step < target and not stop:
            model.train()
            x, y = batch(training, args.batch_size, model.config.context, device_obj, target_mask=training_mask)
            optimizer.zero_grad(set_to_none=True)
            logits, loss, _ = model(x, y)
            if not torch.isfinite(loss):
                raise RuntimeError('Loss became nonfinite. Training stopped.')
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            step += 1
            last_loss = loss.item()
            interval_losses.append(last_loss)
            interval_grads.append(grad_norm.item())
            if step % 20 == 0 or step == target or stop:
                training_seconds = max(time.monotonic()-interval_start, 1e-6)
                eval_start = time.monotonic()
                val = evaluate()
                eval_seconds = time.monotonic()-eval_start
                history.append(dict(step=step, train_loss=last_loss, val_loss=val, val_perplexity=perplexity(val)))
                save()
                elapsed = time.monotonic() - start
                tokens_per_sec = (step - session_start_step) * args.batch_size * model.config.context / max(elapsed, 1e-6)
                weight_norm = sum(p.data.float().norm()**2 for p in model.parameters()).sqrt().item()
                if wandb_run:
                    train_predictions = prediction_metrics(logits, y)
                    interval_mean = sum(interval_losses)/len(interval_losses)
                    metrics = {'loss/train': last_loss, 'loss/train_interval_mean': interval_mean,
                               'loss/held_out': val, 'loss/held_out_minus_train': val-interval_mean,
                               'perplexity/train': perplexity(interval_mean), 'perplexity/held_out': perplexity(val),
                               'accuracy/train_batch': train_predictions['accuracy'],
                               'accuracy/held_out': eval_metrics['accuracy'],
                               'accuracy/held_out_top5': eval_metrics['top5_accuracy'],
                               'prediction/held_out_entropy_nats': eval_metrics['entropy_nats'],
                               'throughput/tokens_per_sec': tokens_per_sec,
                               'throughput/train_interval_tokens_per_sec': len(interval_losses)*args.batch_size*model.config.context/training_seconds,
                               'timing/train_step_ms': 1000*training_seconds/len(interval_losses),
                               'timing/evaluation_seconds': eval_seconds, 'timing/session_seconds': elapsed,
                               'progress/session_tokens': (step-session_start_step)*args.batch_size*model.config.context,
                               'progress/session_equivalent_passes': (step-session_start_step)*args.batch_size*model.config.context/len(training),
                               'progress/target_step': target, 'optim/grad_norm': grad_norm.item(),
                               'optim/grad_norm_interval_mean': sum(interval_grads)/len(interval_grads),
                               'optim/clipped_step_fraction': sum(g>1 for g in interval_grads)/len(interval_grads),
                               'optim/weight_norm': weight_norm, 'optim/lr': optimizer.param_groups[0]['lr'],
                               'checkpoint/size_mb': checkpoint.stat().st_size/2**20, 'step': step}
                    metrics.update(memory_metrics(args.device))
                    metrics.update(diagnostics(model))
                    wandb_run.log(metrics, step=step, commit=False)
                    if args.wandb_samples and (step % 100 == 0 or step == target):
                        rows = []
                        for p in sample_prompts:
                            wrapped = f'### Instruction:\n{p}\n\n### Response:\n' if stage == 'finetune' else p
                            text, _ = model.generate(wrapped, count=50, temperature=0.0, tokenizer=tok,
                                                      stop_text='<|end|>' if stage == 'finetune' else None)
                            rows.append([step, p, text[len(wrapped):] if text.startswith(wrapped) else text])
                        wandb_run.log({'samples': wandb.Table(columns=['step', 'prompt', 'response'], data=rows)}, step=step, commit=False)
                    wandb_run.log({'step': step}, step=step)
                interval_losses.clear()
                interval_grads.clear()
                interval_start = time.monotonic()
                print(f'Step {step} | train {last_loss:.3f} | held-out {val:.3f} | perplexity {perplexity(val)} | grad_norm {grad_norm:.3f}', flush=True)
            state.update(step=step, train_loss=last_loss, elapsed=round(time.monotonic()-start, 1))
            atomic_json(out/'status.json', state)
        save()
        state.update(status='paused' if stop else 'complete', step=step)
        atomic_json(out/'status.json', state)
        if wandb_run:
            wandb_run.summary.update({'status': state['status'], 'final_step': step, 'final_val_loss': history[-1]['val_loss']})
        append_ledger(dict(run_id=run_id, **provenance, stage=stage, dataset=args.data.name, parameters=params,
                            vocab=tok.vocab_size, step=step, target=target,
                            final_val_loss=history[-1]['val_loss'], final_perplexity=history[-1]['val_perplexity'],
                            status=state['status'], config=asdict(model.config),
                            timestamp=time.strftime('%Y-%m-%dT%H:%M:%S')))
    except Exception as exc:
        state.update(status='error', error=str(exc), step=step)
        atomic_json(out/'status.json', state)
        if wandb_run:
            wandb_run.summary.update({'status': state['status'], 'final_step': step, 'final_val_loss': history[-1]['val_loss']})
        append_ledger(dict(run_id=run_id, **provenance, stage=stage, dataset=args.data.name, parameters=params,
                            vocab=tok.vocab_size, step=step, target=target, status='error', error=str(exc),
                            config=asdict(model.config), timestamp=time.strftime('%Y-%m-%dT%H:%M:%S')))
        raise
    finally:
        if wandb_run:
            wandb_run.finish(exit_code=1 if state['status'] == 'error' else 0)


if __name__ == '__main__':
    main()
