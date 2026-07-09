"""Small Transformer policy for Phase 3b formula generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
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
    use_rmsnorm: bool = False
    use_swiglu: bool = False

    def to_dict(self) -> dict[str, int | float | bool]:
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, ff_dim: int):
        super().__init__()
        self.w = nn.Linear(d_model, ff_dim * 2)
        self.out = nn.Linear(ff_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.w(x).chunk(2, dim=-1)
        return self.out(value * F.silu(gate))


class RMSNormSwiGLUBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ffn = SwiGLU(d_model, ff_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        attn_in = self.norm1(x)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, attn_mask=mask, need_weights=False)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class TransformerFormulaPolicy(nn.Module):
    """A minimal GPT-style policy that predicts the next formula token."""

    def __init__(self, config: TransformerPolicyConfig):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.model_vocab_size, config.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.max_sequence_len, config.d_model))
        self.enhanced_blocks = bool(config.use_rmsnorm or config.use_swiglu)
        if self.enhanced_blocks:
            if not (config.use_rmsnorm and config.use_swiglu):
                raise ValueError("RMSNorm and SwiGLU are enabled together in the enhanced policy block")
            self.blocks = nn.ModuleList(
                RMSNormSwiGLUBlock(config.d_model, config.num_heads, config.ff_dim, config.dropout)
                for _ in range(config.num_layers)
            )
            self.ln_f = RMSNorm(config.d_model)
        else:
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
        if self.enhanced_blocks:
            for block in self.blocks:
                x = block(x, mask)
        else:
            x = self.blocks(x, mask=mask, is_causal=True)
        x = self.ln_f(x)
        last = x[:, -1, :]
        logits = self.head_actor(last)
        value = self.head_critic(last).squeeze(-1) if self.head_critic is not None else None
        return logits, value