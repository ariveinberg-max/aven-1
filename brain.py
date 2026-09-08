"""A byte-level causal Transformer. No downloaded models or pretrained weights."""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Config:
    width: int = 320
    layers: int = 4
    heads: int = 5
    context: int = 128
    vocab: int = 256
    dropout: float = 0.0


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.heads = c.heads
        self.norm1 = nn.LayerNorm(c.width)
        self.qkv = nn.Linear(c.width, 3 * c.width)
        self.proj = nn.Linear(c.width, c.width)
        self.drop1 = nn.Dropout(c.dropout)
        self.norm2 = nn.LayerNorm(c.width)
        self.mlp = nn.Sequential(nn.Linear(c.width, 4*c.width), nn.GELU(), nn.Linear(4*c.width, c.width))
        self.drop2 = nn.Dropout(c.dropout)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        q, k, v = [a.reshape(b, t, self.heads, d//self.heads).transpose(1, 2) for a in (q, k, v)]
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.drop1(self.proj(a.transpose(1, 2).contiguous().reshape(b, t, d)))
        return x + self.drop2(self.mlp(self.norm2(x)))


class Brain(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or Config()
        c = self.config
        self.token = nn.Embedding(c.vocab, c.width)
        self.position = nn.Embedding(c.context, c.width)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.layers)])
        self.norm = nn.LayerNorm(c.width)
        self.output = nn.Linear(c.width, c.vocab, bias=False)
        self.apply(self._init)
        self.output.weight = self.token.weight

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, ids, targets=None, inspect=False):
        x = self.token(ids) + self.position(torch.arange(ids.shape[1], device=ids.device))
        activity = []
        for block in self.blocks:
            x = block(x)
            if inspect:
                # Actual last-position hidden state, grouped into 32 channels for display.
                vals = x[0, -1].detach().float().cpu()
                activity.append([float(a.abs().mean()) for a in torch.tensor_split(vals, 32)])
        logits = self.output(self.norm(x))
        loss = None if targets is None else F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        return logits, loss, activity

    @torch.no_grad()
    def generate(self, prompt, count=160, temperature=0.8, tokenizer=None, stop_text=None):
        """tokenizer=None assumes raw UTF-8 bytes (vocab=256), matching older checkpoints.
        stop_text: if the decoded output ends with this string, stop early (used by fine-tuned checkpoints)."""
        self.eval()
        raw = tokenizer.encode(prompt) if tokenizer else (list(prompt.encode('utf-8')) or [10])
        raw = raw or [10]
        ids = torch.tensor([raw], device=next(self.parameters()).device)
        for _ in range(count):
            logits, _, _ = self(ids[:, -self.config.context:])
            logits = logits[:, -1] / temperature
            cutoff = torch.topk(logits, min(40, logits.shape[-1])).values[:, -1:]
            logits = logits.masked_fill(logits < cutoff, float('-inf'))
            ids = torch.cat([ids, torch.multinomial(F.softmax(logits, dim=-1), 1)], dim=1)
            if stop_text:
                tail = ids[0, -len(stop_text)-4:].tolist()
                decoded = tokenizer.decode(tail) if tokenizer else bytes(tail).decode('utf-8', errors='replace')
                if decoded.endswith(stop_text):
                    break
        _, _, activity = self(ids[:, -self.config.context:], inspect=True)
        out_ids = ids[0].tolist()
        text = tokenizer.decode(out_ids) if tokenizer else bytes(out_ids).decode('utf-8', errors='replace')
        if stop_text and text.endswith(stop_text):
            text = text[:-len(stop_text)]
        return text, activity


def device_name():
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'
