"""A value model for PPO: same architecture as the reward model, but scores
every position in the sequence, not just the last.

PPO needs this to estimate "how much total reward do I expect from here
onward" at each token, which is what turns a single end-of-response reward
score into a lower-variance per-step learning signal (via GAE). Initialized
from the fine-tuned Aven-1 checkpoint, same as the reward model.
"""
import torch
from torch import nn
from brain import Block, Config


class ValueModel(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or Config()
        c = self.config
        self.token = nn.Embedding(c.vocab, c.width)
        self.position = nn.Embedding(c.context, c.width)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.layers)])
        self.norm = nn.LayerNorm(c.width)
        self.value_head = nn.Linear(c.width, 1)
        self.apply(self._init)

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def load_backbone(self, brain_state_dict):
        own = self.state_dict()
        copied = 0
        for key, value in brain_state_dict.items():
            if key in own and key != 'output.weight' and own[key].shape == value.shape:
                own[key] = value
                copied += 1
        self.load_state_dict(own)
        return copied

    def forward(self, ids):
        x = self.token(ids) + self.position(torch.arange(ids.shape[1], device=ids.device))
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.value_head(x).squeeze(-1)  # one value per position: (batch, seq_len)
