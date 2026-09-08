"""A reward model: the same Transformer architecture as Aven-1, but with a
scalar "how good is this response" head instead of a next-token head.

Initialized from the fine-tuned Aven-1 checkpoint's weights (the standard
real-world approach — a reward model starts from a model that already
understands the domain, then a fresh scalar head learns to score it against
actual human preferences). Trained on preferences.py's real comparison data
via the Bradley-Terry pairwise loss: score(chosen) should exceed
score(rejected).
"""
import torch
from torch import nn
from brain import Block, Config


class RewardModel(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or Config()
        c = self.config
        self.token = nn.Embedding(c.vocab, c.width)
        self.position = nn.Embedding(c.context, c.width)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.layers)])
        self.norm = nn.LayerNorm(c.width)
        self.reward_head = nn.Linear(c.width, 1)
        self.apply(self._init)

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def load_backbone(self, brain_state_dict):
        """Copy the transformer body (embeddings + blocks + norm) from a fine-tuned
        Aven-1 checkpoint. The reward head is left freshly initialized — it has no
        equivalent in the language-modeling checkpoint."""
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
        return self.reward_head(x[:, -1]).squeeze(-1)  # scalar score from the final position
