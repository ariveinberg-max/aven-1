"""Low-frequency diagnostics for Aven training. No raw corpus or checkpoint uploads."""
import math
import resource
import sys
import torch


def prediction_metrics(logits, targets):
    with torch.no_grad():
        logp = logits.detach().float().log_softmax(-1)
        return {
            'accuracy': (logits.argmax(-1) == targets).float().mean().item(),
            'top5_accuracy': (logits.topk(min(5, logits.shape[-1]), dim=-1).indices == targets[..., None]).any(-1).float().mean().item(),
            'entropy_nats': -(logp.exp() * logp).sum(-1).mean().item(),
        }


def memory_metrics(device):
    # ru_maxrss is bytes on macOS and KiB on Linux; this is a peak, not current RSS.
    scale = 1 if sys.platform == 'darwin' else 1024
    values = {'memory/process_peak_rss_mb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale / 2**20}
    if device == 'mps':
        values.update({'memory/mps_allocated_mb': torch.mps.current_allocated_memory()/2**20,
                       'memory/mps_driver_mb': torch.mps.driver_allocated_memory()/2**20})
    elif device == 'cuda':
        values.update({'memory/cuda_allocated_mb': torch.cuda.memory_allocated()/2**20,
                       'memory/cuda_peak_allocated_mb': torch.cuda.max_memory_allocated()/2**20})
    return values


def diagnostics(model):
    values = {}
    with torch.no_grad():
        for name, module in model.named_children():
            parameters = list(module.parameters())
            if not parameters:
                continue
            values[f'layers/{name}/weight_norm'] = math.sqrt(sum(p.detach().float().square().sum().item() for p in parameters))
            values[f'layers/{name}/grad_norm_after_clip'] = math.sqrt(sum(p.grad.detach().float().square().sum().item() for p in parameters if p.grad is not None))
    return values
