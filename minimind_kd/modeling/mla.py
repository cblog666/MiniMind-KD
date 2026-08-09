from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .normalization import RMSNorm


class GatedMLA(nn.Module):
    """NoPE Multi-head Latent Attention with a full-rank output gate."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_key_value_heads: int,
        qk_nope_head_dim: int,
        kv_lora_rank: int,
        q_lora_rank: int = 0,
        qk_direct_head_dim: int = 0,
        v_head_dim: int | None = None,
        norm_eps: float = 1e-6,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if qk_nope_head_dim <= 0:
            raise ValueError("qk_nope_head_dim must be positive")
        if qk_direct_head_dim < 0:
            raise ValueError("qk_direct_head_dim must be non-negative")
        if num_heads % num_key_value_heads:
            raise ValueError("num_heads must be divisible by num_key_value_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_direct_head_dim = qk_direct_head_dim
        self.q_head_dim = qk_nope_head_dim + qk_direct_head_dim
        self.v_head_dim = qk_nope_head_dim if v_head_dim is None else v_head_dim
        if self.v_head_dim <= 0:
            raise ValueError("v_head_dim must be positive")
        self.query_projection_size = num_heads * self.q_head_dim
        self.output_projection_size = num_heads * self.v_head_dim
        self.groups = num_heads // num_key_value_heads
        self.dropout_p = dropout

        self.q_lora_rank = q_lora_rank
        if q_lora_rank > 0:
            self.q_a = nn.Linear(hidden_size, q_lora_rank, bias=False)
            self.q_norm = RMSNorm(q_lora_rank, norm_eps)
            self.q_b = nn.Linear(q_lora_rank, self.query_projection_size, bias=False)
            self.q_b.weight._muon_num_heads = num_heads  # type: ignore[attr-defined]
        else:
            self.q_proj = nn.Linear(hidden_size, self.query_projection_size, bias=False)
            self.q_proj.weight._muon_num_heads = num_heads  # type: ignore[attr-defined]

        self.kv_a = nn.Linear(hidden_size, kv_lora_rank, bias=False)
        self.kv_norm = RMSNorm(kv_lora_rank, norm_eps)
        self.kv_b = nn.Linear(
            kv_lora_rank,
            num_key_value_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )
        self.kv_b.weight._muon_num_heads = num_key_value_heads  # type: ignore[attr-defined]
        self.k_direct_proj = (
            nn.Linear(hidden_size, qk_direct_head_dim, bias=False) if qk_direct_head_dim else None
        )
        self.output_gate = nn.Linear(hidden_size, self.output_projection_size, bias=False)
        self.output_proj = nn.Linear(self.output_projection_size, hidden_size, bias=False)

    def _queries(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.q_lora_rank > 0:
            return self.q_b(self.q_norm(self.q_a(hidden_states)))
        return self.q_proj(hidden_states)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, sequence, _ = hidden_states.shape
        q = self._queries(hidden_states).view(batch, sequence, self.num_heads, self.q_head_dim)
        q = q.transpose(1, 2)
        compressed_kv = self.kv_norm(self.kv_a(hidden_states))
        kv = self.kv_b(compressed_kv).view(
            batch,
            sequence,
            self.num_key_value_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        k, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if self.groups > 1:
            k = torch.repeat_interleave(k, self.groups, dim=1)
            v = torch.repeat_interleave(v, self.groups, dim=1)
        if self.k_direct_proj is not None:
            direct = self.k_direct_proj(hidden_states).view(batch, sequence, 1, self.qk_direct_head_dim)
            direct = direct.transpose(1, 2).expand(-1, self.num_heads, -1, -1)
            k = torch.cat((k, direct), dim=-1)

        scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) / math.sqrt(self.q_head_dim)
        causal = torch.ones(sequence, sequence, device=scores.device, dtype=torch.bool).tril()
        valid = causal.view(1, 1, sequence, sequence)
        if attention_mask is not None:
            key_valid = attention_mask.to(torch.bool).view(batch, 1, 1, sequence)
            valid = valid & key_valid
        scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        probabilities = torch.softmax(scores, dim=-1).to(q.dtype)
        probabilities = F.dropout(probabilities, p=self.dropout_p, training=self.training)
        output = torch.matmul(probabilities, v).transpose(1, 2).contiguous()
        output = output.view(batch, sequence, self.output_projection_size)
        output = output * torch.sigmoid(self.output_gate(hidden_states))
        output = self.output_proj(output)
        if attention_mask is not None:
            output = output * attention_mask.unsqueeze(-1).to(output.dtype)
        return output
