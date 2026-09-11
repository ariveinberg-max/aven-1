"""Checkpoint-compatible KV caching for Brain's learned-position Transformer."""
import torch
from torch.nn import functional as F


@torch.no_grad()
def cached_logits(model, ids, cache=None):
    """Prefill a window, or append one token. Cache belongs to one request only.

    Sliding the window changes every learned position embedding, so callers must
    rebuild the cache rather than evicting its first entry when context is full.
    """
    if model.training:
        raise ValueError('Cached decoding requires eval mode.')
    offset = 0 if cache is None else cache[0][0].shape[2]
    if ids.ndim != 2 or ids.shape[1] < 1 or offset + ids.shape[1] > model.config.context:
        raise ValueError('Token window must fit the model context.')
    if cache is not None and ids.shape[1] != 1:
        raise ValueError('Append exactly one token to an existing cache.')
    x = model.token(ids) + model.position(torch.arange(offset, offset + ids.shape[1], device=ids.device))
    updated = []
    for index, block in enumerate(model.blocks):
        b, t, d = x.shape
        q, k, v = block.qkv(block.norm1(x)).chunk(3, dim=-1)
        q, k, v = [a.reshape(b, t, block.heads, d // block.heads).transpose(1, 2) for a in (q, k, v)]
        if cache is not None:
            k = torch.cat((cache[index][0], k), dim=2)
            v = torch.cat((cache[index][1], v), dim=2)
        # A single appended query can attend to every cached key. A triangular
        # mask here would incorrectly expose only the first key.
        a = F.scaled_dot_product_attention(q, k, v, is_causal=cache is None, scale=block.attn_scale)
        x = x + block.drop1(block.proj(a.transpose(1, 2).contiguous().reshape(b, t, d)))
        x = x + block.drop2(block.mlp(block.norm2(x)))
        updated.append((k, v))
    return model.output(model.norm(x[:, -1:])), updated
