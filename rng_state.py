"""Save CPU sampling state and the active training backend's random state."""
import torch


def capture_rng(device):
    state = dict(device=device, cpu=torch.get_rng_state())
    if device == 'cuda':
        state['accelerator'] = torch.cuda.get_rng_state()
    elif device == 'mps':
        state['accelerator'] = torch.mps.get_rng_state()
    return state


def restore_rng(checkpoint, device):
    state = checkpoint.get('rng_state')
    if state is None:
        torch.set_rng_state(checkpoint['rng'])
        return 'legacy_cpu_only'
    torch.set_rng_state(state['cpu'])
    if state['device'] != device:
        return 'cpu_restored_backend_changed'
    if device == 'cuda':
        torch.cuda.set_rng_state(state['accelerator'])
    elif device == 'mps':
        torch.mps.set_rng_state(state['accelerator'])
    return 'full_backend_state'
