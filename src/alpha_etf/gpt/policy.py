"""Small Transformer policy for Phase 3b formula generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class TransformerPolicyConfig:
    model_vocab_size: int
    max_sequence_len: int
    d_model: int = 64
    num_layers: int = 2
    num_heads: int = 4
    ff_dim: int = 128
    dropout: float = 0.1
    use_critic_head: bool = False

    def to_dict(self) -> dict[str, int | float | bool]:
        return asdict(self)


class TransformerFormulaPolicy(nn.Module):
    """A minimal GPT-style policy that predicts the next formula token."""

    def __init__(self, config: TransformerPolicyConfig):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.model_vocab_size, config.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.max_sequence_len, config.d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=config.num_layers, enable_nested_tensor=False)
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head_actor = nn.Linear(config.d_model, config.model_vocab_size)
        self.head_critic = nn.Linear(config.d_model, 1) if config.use_critic_head else None

    def forward(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if idx.ndim != 2:
            raise ValueError(f"idx must have shape [batch, seq], got {tuple(idx.shape)}")
        _, seq_len = idx.shape
        if seq_len > self.config.max_sequence_len:
            raise ValueError(f"sequence length {seq_len} exceeds max {self.config.max_sequence_len}")

        x = self.token_emb(idx) + self.pos_emb[:, :seq_len, :]
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=idx.device)
        x = self.blocks(x, mask=mask, is_causal=True)
        x = self.ln_f(x)
        last = x[:, -1, :]
        logits = self.head_actor(last)
        value = self.head_critic(last).squeeze(-1) if self.head_critic is not None else None
        return logits, value