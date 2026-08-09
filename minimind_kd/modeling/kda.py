from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .normalization import RMSNorm


class CausalDepthwiseConv1d(nn.Module):
    """Readable ShortConv equivalent used before KDA q/k/v projections."""

    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        if kernel_size <= 0:
            raise ValueError("kernel_size must be positive")
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.empty(channels, 1, kernel_size))
        nn.init.normal_(self.weight, std=kernel_size**-0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_channels_first = x.transpose(1, 2)
        padded = F.pad(x_channels_first, (self.kernel_size - 1, 0))
        convolved = F.conv1d(padded, self.weight, groups=x.shape[-1])
        return F.silu(convolved.transpose(1, 2))


class KimiDeltaAttention(nn.Module):
    """Reference KDA recurrence from the Kimi K3 report.

    This implementation is deliberately kernel-free and loops over sequence
    positions. It is mathematically faithful and easy to inspect, but production
    training should replace it with a fused/chunkwise KDA kernel.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        decay_rank: int,
        conv_kernel_size: int,
        log_decay_min: float = -5.0,
        norm_eps: float = 1e-6,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if head_dim <= 0:
            raise ValueError("head_dim must be positive")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.projection_size = num_heads * head_dim
        self.log_decay_min = log_decay_min

        self.q_proj = nn.Linear(hidden_size, self.projection_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.projection_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.projection_size, bias=False)
        self.q_conv = CausalDepthwiseConv1d(self.projection_size, conv_kernel_size)
        self.k_conv = CausalDepthwiseConv1d(self.projection_size, conv_kernel_size)
        self.v_conv = CausalDepthwiseConv1d(self.projection_size, conv_kernel_size)

        self.decay_down = nn.Linear(hidden_size, decay_rank, bias=False)
        self.decay_up = nn.Linear(decay_rank, self.projection_size, bias=True)
        self.decay_log_scale = nn.Parameter(torch.zeros(num_heads))
        self.beta_proj = nn.Linear(hidden_size, num_heads, bias=False)

        self.output_norm = RMSNorm(self.head_dim, norm_eps)
        self.output_gate = nn.Linear(hidden_size, self.projection_size, bias=False)
        self.output_proj = nn.Linear(self.projection_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Consumed by the per-head Muon optimizer builder.
        for parameter in (self.q_proj.weight, self.k_proj.weight, self.v_proj.weight):
            parameter._muon_num_heads = num_heads  # type: ignore[attr-defined]

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        batch, sequence, _ = x.shape
        return x.view(batch, sequence, self.num_heads, self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, sequence, _ = hidden_states.shape
        q = F.normalize(self._shape(self.q_conv(self.q_proj(hidden_states))), p=2.0, dim=-1)
        k = F.normalize(self._shape(self.k_conv(self.k_proj(hidden_states))), p=2.0, dim=-1)
        v = self._shape(self.v_conv(self.v_proj(hidden_states)))

        decay_logits = self._shape(self.decay_up(self.decay_down(hidden_states)))
        decay_scale = self.decay_log_scale.exp().view(1, 1, self.num_heads, 1)
        log_decay = self.log_decay_min * torch.sigmoid(decay_scale * decay_logits.float())
        alpha = log_decay.exp()
        beta = torch.sigmoid(self.beta_proj(hidden_states).float()).unsqueeze(-1)

        state = hidden_states.new_zeros(
            (batch, self.num_heads, self.head_dim, self.head_dim), dtype=torch.float32
        )
        outputs: list[torch.Tensor] = []
        if attention_mask is None:
            attention_mask = torch.ones(batch, sequence, device=hidden_states.device, dtype=torch.bool)
        else:
            attention_mask = attention_mask.to(torch.bool)

        for position in range(sequence):
            key = k[:, position].float()
            value = v[:, position].float()
            query = q[:, position].float()
            decayed_state = alpha[:, position].unsqueeze(-1) * state
            predicted_value = torch.einsum("bhd,bhdv->bhv", key, decayed_state)
            delta = value - predicted_value
            proposed_state = decayed_state + beta[:, position].unsqueeze(-1) * torch.einsum(
                "bhd,bhv->bhdv", key, delta
            )
            valid = attention_mask[:, position].view(batch, 1, 1, 1)
            state = torch.where(valid, proposed_state, state)
            output = torch.einsum("bhd,bhdv->bhv", query, state)
            output = output * attention_mask[:, position].view(batch, 1, 1)
            outputs.append(output)

        recurrent_output = torch.stack(outputs, dim=1).to(hidden_states.dtype)
        recurrent_output = self.output_norm(recurrent_output)
        gate = torch.sigmoid(self._shape(self.output_gate(hidden_states)))
        gated = (recurrent_output * gate).reshape(batch, sequence, self.projection_size)
        return self.dropout(self.output_proj(gated))
