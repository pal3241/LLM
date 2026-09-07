from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


@dataclass
class CausalLMOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None


class RMSNorm(nn.Module):
    def __init__(self, size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (normalized * self.weight.float()).to(x.dtype)


def _rope_frequencies(config: ModelConfig, device: torch.device) -> torch.Tensor:
    positions = torch.arange(config.max_seq_len, device=device, dtype=torch.float32)
    dimensions = torch.arange(0, config.head_dim, 2, device=device, dtype=torch.float32)
    inv_freq = 1.0 / (config.rope_theta ** (dimensions / config.head_dim))
    return torch.outer(positions, inv_freq)


def apply_rope(x: torch.Tensor, frequencies: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to a [batch, heads, sequence, head_dim] tensor."""
    seq_len = x.size(-2)
    freq = frequencies[:seq_len].to(device=x.device)
    cos = freq.cos()[None, None, :, :]
    sin = freq.sin()[None, None, :, :]
    even, odd = x[..., 0::2], x[..., 1::2]
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
    return rotated.flatten(-2).to(x.dtype)


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        q_size = config.num_attention_heads * config.head_dim
        kv_size = config.num_kv_heads * config.head_dim
        self.q_proj = nn.Linear(config.hidden_size, q_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, kv_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, kv_size, bias=False)
        self.o_proj = nn.Linear(q_size, config.hidden_size, bias=False)
        self.dropout = config.dropout

    def forward(self, x: torch.Tensor, frequencies: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.config.num_attention_heads, -1).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.config.num_kv_heads, -1).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.config.num_kv_heads, -1).transpose(1, 2)
        q, k = apply_rope(q, frequencies), apply_rope(k, frequencies)

        repeats = self.config.num_attention_heads // self.config.num_kv_heads
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)
        attended = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(attended)


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attention = GroupedQueryAttention(config)
        self.feed_forward = SwiGLU(config)

    def forward(self, x: torch.Tensor, frequencies: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x), frequencies)
        return x + self.feed_forward(self.ffn_norm(x))


class MiniLLMForCausalLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(TransformerBlock(config) for _ in range(config.num_layers))
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        self.register_buffer("rope_frequencies", _rope_frequencies(config, torch.device("cpu")), persistent=False)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None) -> CausalLMOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids harus berbentuk [batch, sequence]")
        if input_ids.size(1) > self.config.max_seq_len:
            raise ValueError("sequence melebihi max_seq_len")
        x = self.token_embedding(input_ids)
        for layer in self.layers:
            x = layer(x, self.rope_frequencies)
        logits = self.lm_head(self.norm(x))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)
        return CausalLMOutput(logits=logits, loss=loss)

    @property
    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
